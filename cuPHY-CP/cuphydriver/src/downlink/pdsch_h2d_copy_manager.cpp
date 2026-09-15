/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 49) // "DRV.H2D_MGR"

#include "pdsch_h2d_copy_manager.hpp"
#include "mps.hpp"
#include "nvlog.hpp"

PdschH2DCopyManager::PdschH2DCopyManager(MpsCtx*  mpsCtx,
                                         bool     useBatchedMemcpy,
                                         uint32_t copyWaitThNs,
                                         bool     threadEnable)
    : m_mpsCtx(mpsCtx),
      m_useBatchedMemcpy(useBatchedMemcpy),
      m_batchedMemcpyHelper(DL_MAX_CELLS_PER_SLOT,
                            batchedMemcpySrcHint::srcIsHost,
                            batchedMemcpyDstHint::dstIsDevice,
                            useBatchedMemcpy && (CUPHYDRIVER_PDSCH_USE_BATCHED_COPY == 1)),
      m_threadEnable(threadEnable),
      m_copyWaitThNs(copyWaitThNs)
{
    m_mpsCtx->setCtx();

    CUDA_CHECK(cudaStreamCreateWithPriority(&m_stream, cudaStreamNonBlocking, -4));

    for (int i = 0; i < MAX_PDSCH_TB_CPY_CUDA_EVENTS; i++)
    {
        CUDA_CHECK(cudaEventCreate(&m_startEvents[i]));
        CUDA_CHECK(cudaEventCreate(&m_completeEvents[i]));
    }

    std::fill(m_doneCurSlotIdx.begin(), m_doneCurSlotIdx.end(), -1);
    std::fill(m_cudaEventRecDone.begin(), m_cudaEventRecDone.end(), false);

    resetPreponeInfo();

    NVLOGD_FMT(TAG,
               "PdschH2DCopyManager: created — stream={} threadEnable={} "
               "useBatchedMemcpy={} copyWaitThNs={} maxEvents={}",
               static_cast<void*>(m_stream), m_threadEnable,
               m_useBatchedMemcpy, m_copyWaitThNs, MAX_PDSCH_TB_CPY_CUDA_EVENTS);
}

PdschH2DCopyManager::~PdschH2DCopyManager()
{
    NVLOGD_FMT(TAG,
               "PdschH2DCopyManager: destroying — stream={} threadEnable={}",
               static_cast<void*>(m_stream), m_threadEnable);

    // Stop and join the copy worker BEFORE tearing down CUDA stream/events.
    // The worker references those handles every loop iteration; detaching
    // (the prior behaviour) left it free to use destroyed handles → UAF.
    stopThread();

    try {
        for (int i = 0; i < MAX_PDSCH_TB_CPY_CUDA_EVENTS; i++)
        {
            // Null handles occur in the null-object construction path
            // (minimal PhyDriverCtx). cudaEventDestroy(nullptr) would return
            // cudaErrorInvalidResourceHandle and trigger spurious error logs.
            if (m_startEvents[i] != nullptr) {
                CUDA_CHECK(cudaEventDestroy(m_startEvents[i]));
            }
            if (m_completeEvents[i] != nullptr) {
                CUDA_CHECK(cudaEventDestroy(m_completeEvents[i]));
            }
        }
    } catch (const std::exception& e) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "EXCEPTION in cudaEventDestroy for H2D copy events: {}", e.what());
    }

    try {
        if (m_stream != nullptr) {
            cudaStreamDestroy(m_stream);
        }
    } catch (const std::exception& e) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "EXCEPTION in cudaStreamDestroy for H2D stream: {}", e.what());
    }
}

void PdschH2DCopyManager::setCtx()
{
    // Null-object instances have no MPS context — nothing to set.
    if (m_mpsCtx == nullptr) [[unlikely]] {
        return;
    }
    m_mpsCtx->setCtx();
}

void PdschH2DCopyManager::updateBatchedMemcpyInfo(void* dst, const void* src, std::size_t count)
{
    // Null-object instances have no CUDA stream; the helper's fallback path
    // would otherwise issue a real cudaMemcpyAsync on the default stream.
    if (m_stream == nullptr) [[unlikely]] {
        return;
    }
    // const_cast is safe: this is an H2D-only path (cudaMemcpyHostToDevice is
    // hardcoded below) and the downstream cudaMemcpyAsync signature already
    // takes `const void* src`. cuphyBatchedMemcpyHelper::updateMemcpy stores
    // src in std::vector<void*> purely as an internal forwarding mechanism —
    // it never writes through the pointer. If updateMemcpy is ever extended
    // to mutate src (or to support D2H/H2H copies that could), this cast
    // must be revisited.
    m_batchedMemcpyHelper.updateMemcpy(dst, const_cast<void*>(src), count,
                                       cudaMemcpyHostToDevice, m_stream);
}

cuphyStatus_t PdschH2DCopyManager::performBatchedMemcpy()
{
    return m_batchedMemcpyHelper.launchBatchedMemcpy(m_stream);
}

void PdschH2DCopyManager::resetBatchedMemcpyBatches()
{
    m_batchedMemcpyHelper.reset();
}

void PdschH2DCopyManager::resetPreponeInfo()
{
    std::fill(m_preponeRing.begin(), m_preponeRing.end(), h2d_copy_prepone_info_t{});
}

void PdschH2DCopyManager::stopThread()
{
    if (!m_threadEnable)
    {
        return;
    }
    // Request cooperative cancellation via jthread's built-in stop_source,
    // then join. The worker observes st.stop_requested() on its next loop
    // iteration and returns; ~jthread()'s implicit request_stop + join is
    // therefore a no-op (idempotent — already stopped and joined here).
    m_copyThread.request_stop();
    try {
        if (m_copyThread.joinable()) {
            m_copyThread.join();
        }
    } catch (const std::exception& e) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "EXCEPTION joining H2D copy thread: {}", e.what());
    }
    m_threadEnable = false;
}

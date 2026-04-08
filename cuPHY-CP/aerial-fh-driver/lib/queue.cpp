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

#include "queue.hpp"

#include "dpdk.hpp"
#include "fronthaul.hpp"
#include "nic.hpp"
#include "time.hpp"
#include "utils.hpp"
#include "gpu_comm.hpp"
#include <doca_gpunetio.h>
#include <atomic>
#define TAG "FH.QUEUE"

// TODO FIXME remove when DOCA warnings are fixed
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

namespace aerial_fh
{
static std::atomic<uint32_t> g_next_dpdk_rx_queue_idx{MLX5_EXTERNAL_RX_QUEUE_ID_MIN};    
Queue::Queue(Nic* nic, uint16_t id, uint16_t size) :
    nic_{nic},
    id_{id},
    size_{size},
    doca_tx_h{},
    doca_rx_h{}
{
}

Queue::~Queue()
{
}

uint16_t Queue::get_id() const
{
    return id_;
}

doca_rx_items_t* Queue::get_doca_rx_items()
{
    return &doca_rx_h;
}
doca_tx_items_t* Queue::get_doca_tx_items()
{
    return &doca_tx_h;
}

Txq::Txq(Nic* nic, uint16_t id, bool gpu_m) :
    Queue{nic, id, nic->get_info().txq_size},
    gpu_m_{gpu_m}, wqe_pi{nullptr}, cqe_ci{nullptr}
{
    auto port_id   = nic_->get_port_id();
    auto socket_id = rte_eth_dev_socket_id(port_id);
    auto name      = nic_->get_name();
    int ret        = 0;

    NVLOGI_FMT(TAG, "Setting up TXQ #{} on NIC {} with {} descriptors for GPU-init comm {}", id_, nic_->get_info().name.c_str(), size_, gpu_m_);

    if (is_gpu()) {
		Fronthaul* fh = nic_->get_fronthaul();
        doca_error_t ret;
        if(fh->get_info().enable_gpu_comm_via_cpu==1)
        {
            ret = doca_create_tx_queue(&(doca_tx_h), fh->get_docaGpuParams()->gpu, nic->get_doca_dev(), QUEUE_DESC, id_, DOCA_GPU_MEM_TYPE_CPU_GPU);
        }
        else
        {
            ret = doca_create_tx_queue(&(doca_tx_h), fh->get_docaGpuParams()->gpu, nic->get_doca_dev(), QUEUE_DESC, id_, DOCA_GPU_MEM_TYPE_GPU);
        }
        if (ret != DOCA_SUCCESS)
                THROW_FH(ret, StringBuilder() << "Failed to setup DOCA GPU TxQ #" << id_ << " on NIC " << nic_->get_info().name << " because doca_create_tx_queue was a failure");

        NVLOGI_FMT(TAG,"Doca Ethernet TxQ created! gpu_addr {} cpu_addr {}", (void *)doca_tx_h.eth_txq_gpu, (void *)doca_tx_h.eth_txq_cpu);

    } else {
        ret = rte_eth_tx_queue_setup(port_id, id_, size_, socket_id, nullptr);
        if(ret != 0)
        {
            THROW_FH(ret, StringBuilder() << "Failed to setup TXQ #" << id_ << " on NIC " << nic_->get_info().name << ": " << rte_strerror(-ret));
        }
    }
}

Txq::~Txq()
{
    if (is_gpu()) {
        Gpu* gpu_ = nic_->get_fronthaul()->gpus()[nic_->get_info().cuda_device].get();
		// Destroy TXQ
    }
}

size_t Txq::send(rte_mbuf** mbufs, size_t mbuf_count, TxqSendTiming* timing)
{
    auto                              port_id = nic_->get_port_id();
    size_t                            nb_tx   = 0;
    Fronthaul* fh = nic_->get_fronthaul();

    if(is_gpu())
    {
        THROW_FH(ENOTSUP, StringBuilder() << "Failed to send from TXQ #" << id_ << " on NIC " << nic_->get_info().name << " because it's a GPU-init comm queue");
    }

    const auto t0 = Time::now_ns();
    while(nb_tx < mbuf_count)
    {
        nb_tx += rte_eth_tx_burst(port_id, id_, &mbufs[nb_tx], mbuf_count - nb_tx);
        if(Time::now_ns() - t0 > kTxBurstTimeout)
        {
            THROW_FH(ETIMEDOUT, StringBuilder() << "rte_eth_tx_burst timeout");
        }
    }
    const uint64_t after_burst = timing ? Time::now_ns() : 0;

    if(timing)
    {
        timing->lock_wait_ns = 0;  // No lock in send()
        timing->tx_burst_loop_ns = after_burst - t0;
    }

    return nb_tx;
}

size_t Txq::send_lock(rte_mbuf** mbufs, size_t mbuf_count, TxqSendTiming* timing)
{
    auto                              port_id = nic_->get_port_id();
    size_t                            nb_tx   = 0;
    Fronthaul* fh = nic_->get_fronthaul();

    const uint64_t before_lock = timing ? Time::now_ns() : 0;
    const std::lock_guard<aerial_fh::FHMutex> lock(mtx_);
    const uint64_t after_lock = timing ? Time::now_ns() : 0;

    if(is_gpu())
    {
        THROW_FH(ENOTSUP, StringBuilder() << "Failed to send from TXQ #" << id_ << " on NIC " << nic_->get_info().name << " because it's a GPU-init comm queue");
    }

    const auto t0 = Time::now_ns();
    while(nb_tx < mbuf_count)
    {
        nb_tx += rte_eth_tx_burst(port_id, id_, &mbufs[nb_tx], mbuf_count - nb_tx);
        if(Time::now_ns() - t0 > kTxBurstTimeout)
        {
            THROW_FH(ETIMEDOUT, StringBuilder() << "rte_eth_tx_burst timeout");
        }
    }
    const uint64_t after_burst = timing ? Time::now_ns() : 0;

    if(timing)
    {
        timing->lock_wait_ns = after_lock - before_lock;
        timing->tx_burst_loop_ns = after_burst - after_lock;
    }

    return nb_tx;
}

size_t Txq::send_lock(rte_mbuf** mbufs, size_t mbuf_count, Ns tx_window_start, TxqSendTiming* timing)
{
    auto                              port_id = nic_->get_port_id();
    size_t                            nb_tx   = 0;
    Fronthaul* fh = nic_->get_fronthaul();

    const uint64_t before_lock = timing ? Time::now_ns() : 0;
    const std::lock_guard<aerial_fh::FHMutex> lock(mtx_);
    const uint64_t after_lock = timing ? Time::now_ns() : 0;

    if(is_gpu())
    {
        THROW_FH(ENOTSUP, StringBuilder() << "Failed to send from TXQ #" << id_ << " on NIC " << nic_->get_info().name << " because it's a GPU-init comm queue");
    }

    if((tx_window_start > last_tx_ts_on_txq_))
    {
        mbufs[0]->ol_flags |= fh->get_timestamp_mask_();
        *RTE_MBUF_DYNFIELD(mbufs[0], fh->get_timestamp_offset(), uint64_t*) = tx_window_start;
        last_tx_ts_on_txq_                                                  = tx_window_start;
    }

    const auto t0 = Time::now_ns();
    while(nb_tx < mbuf_count)
    {
        nb_tx += rte_eth_tx_burst(port_id, id_, &mbufs[nb_tx], mbuf_count - nb_tx);
        if(Time::now_ns() - t0 > kTxBurstTimeout)
        {
            THROW_FH(ETIMEDOUT, StringBuilder() << "rte_eth_tx_burst timeout");
        }
    }
    const uint64_t after_burst = timing ? Time::now_ns() : 0;

    if(timing)
    {
        timing->lock_wait_ns = after_lock - before_lock;
        timing->tx_burst_loop_ns = after_burst - after_lock;
    }

    return nb_tx;
}



void Txq::poll_complete()
{
    auto                              port_id = nic_->get_port_id();
    const std::lock_guard<aerial_fh::FHMutex> lock(mtx_);
    rte_eth_tx_descriptor_status(port_id, id_, 0);
}

void Txq::warm_up()
{
    constexpr size_t   kWarmUpPacketCount = 1024;//Does the number matter? What's the reasonable value?
    constexpr uint16_t kDataLen           = 64;
    rte_mbuf*          mbufs[kWarmUpPacketCount];

    auto name    = nic_->get_name();
    auto port_id = nic_->get_port_id();
    auto mp      = nic_->get_cpu_tx_mbuf_pool();

    if(is_gpu()) {
        THROW_FH(ENOTSUP, StringBuilder() << "Failed to warmup TXQ #" << id_ << " on NIC " << nic_->get_info().name << " because it's a GPU-init comm queue");
    }

    NVLOGD_FMT(TAG, "Warming up TXQ #{} on NIC {}", id_, name.c_str());

    if(0 != rte_mempool_get_bulk(mp, reinterpret_cast<void**>(mbufs), kWarmUpPacketCount))
    {
        NVLOGW_FMT(TAG, "Failed to allocate {} mbufs to warm-up TXQ #{} on NIC {}", kWarmUpPacketCount, id_, name.c_str());
        return;
    }

    for(int i = 0; i < kWarmUpPacketCount; i++)
    {
        auto mbuf      = mbufs[i];
        mbuf->data_len = kDataLen;
        mbuf->pkt_len  = kDataLen;
        auto data      = rte_pktmbuf_mtod(mbuf, void*);
        memset(data, 0, kDataLen);
    }

    size_t nb_tx      = 0;
    size_t mbuf_count = kWarmUpPacketCount;
    auto   t0         = Time::now_ns();
    while(nb_tx < mbuf_count)
    {
        nb_tx += rte_eth_tx_burst(port_id, id_, &mbufs[nb_tx], mbuf_count - nb_tx);
        if(Time::now_ns() - t0 > kTxBurstTimeout)
        {
            THROW_FH(ETIMEDOUT, StringBuilder() << "Failed to warmup TXQ #" << id_ << " on NIC " << name);
        }
    }
}

bool Txq::is_gpu() const {
    return gpu_m_;
}

Rxq::Rxq(Nic* nic, uint16_t id) :
    Queue{nic, id, nic->get_info().rxq_size},
    doca_rxq_items{}  // Zero-initialize all members of doca_rxq_items
{
    auto port_id   = nic_->get_port_id();
    auto socket_id = rte_eth_dev_socket_id(port_id);
    auto name      = nic_->get_name();
    Fronthaul* fh = nic_->get_fronthaul();
    auto mp=(rte_mempool*)nullptr;
    if(!(fh->get_info().cuda_device_ids.empty()))
    {
        mp = nic_->get_rx_mbuf_pool(false);
    }
    else
    {
        if(nic->get_info().per_rxq_mempool)
        {
            nic_->create_cpu_rx_mbuf_per_queue_pool(cpu_mbuf_pool_queue_,id);
            mp = cpu_mbuf_pool_queue_.get();
        }
        else
        {
            if(!fh->get_info().cuda_device_ids_for_compute.empty()) //CPU Init-comms
                mp = nic_->get_rx_mbuf_pool(true);
            else
                mp = nic_->get_rx_mbuf_pool(false);
        }
    }

    NVLOGI_FMT(TAG, "Setting up RXQ #{} on NIC {} with {} descriptors", id_, nic_->get_info().name.c_str(), size_);

    if(!(fh->get_info().cuda_device_ids.empty()) && !fh->get_info().cpu_rx_only) {
        doca_error_t ret;
        // Assign a unique incremental DPDK queue index before creating the DOCA RX queue
        // Use atomic fetch_add to prevent overflow above UINT16_MAX
        const uint32_t current_idx = g_next_dpdk_rx_queue_idx.fetch_add(1, std::memory_order_relaxed);
        if (current_idx >= UINT16_MAX) {
            THROW_FH(EOVERFLOW, StringBuilder() << "Exceeded maximum DPDK RX queue index");
        }
        doca_rx_h.dpdk_queue_idx = static_cast<uint16_t>(current_idx);
        ret = doca_create_rx_queue(&(doca_rx_h), fh->get_docaGpuParams()->gpu, nic->get_doca_dev(), QUEUE_DESC, 
                                    (fh->get_info().enable_gpu_comm_via_cpu == 1) ? DOCA_GPU_MEM_TYPE_CPU_GPU : DOCA_GPU_MEM_TYPE_GPU,
                                    MAX_PKT_SIZE, MAX_PKT_NUM, fh->get_info().enable_gpu_comm_via_cpu);
        if (ret != DOCA_SUCCESS)
            THROW_FH(ret, StringBuilder() << "Failed to setup DOCA GPU RxQ #" << id_ << " on NIC " << nic_->get_info().name << " because doca_gpu_rxq_create was a failure");

        if(rte_pmd_mlx5_external_rx_queue_id_map(port_id,doca_rx_h.dpdk_queue_idx, doca_rx_h.hw_queue_idx)<0)
        {
            THROW_FH(rte_errno, StringBuilder() << "Failed to map DPDK queue index to HW queue index");
        }

        NVLOGI_FMT(TAG,"Doca Ethernet RxQ created! gpu_addr {} cpu_addr {}",
                (void *)doca_rx_h.eth_rxq_gpu, (void *)doca_rx_h.eth_rxq_cpu);
    } else {
        auto ret = rte_eth_rx_queue_setup(port_id, id_, size_, socket_id, nullptr, mp);
        if(ret != 0)
            THROW_FH(ret, StringBuilder() << "Failed to setup RXQ #" << id_ << " on NIC " << name << ": " << rte_strerror(-ret));
    }
}

size_t Rxq::receive(MsgReceiveInfo* info, size_t* num_msgs)
{
    auto      port_id = nic_->get_port_id();
    rte_mbuf* mbufs[kRxPktBurst];

    {
        const std::lock_guard<aerial_fh::FHMutex> lock(mtx_);
        *num_msgs = rte_eth_rx_burst(port_id, id_, mbufs, std::min(*num_msgs, kRxPktBurst));
    }

    return fill_rx_msg_info(mbufs, info, *num_msgs);
}

size_t Rxq::receive_until(MsgReceiveInfo* info, size_t* num_msgs, Ns timeout)
{
    auto      port_id = nic_->get_port_id();
    rte_mbuf* mbufs[kRxPktBurst];
    size_t    msgs_recv     = 0;
    size_t    msgs_recv_max = std::min(*num_msgs, kRxPktBurst);

    {
        const std::lock_guard<aerial_fh::FHMutex> lock(mtx_);
        do
        {
            msgs_recv += rte_eth_rx_burst(port_id, id_, mbufs + msgs_recv, msgs_recv_max - msgs_recv);
        } while((msgs_recv < msgs_recv_max) && (Time::now_ns() < timeout));
    }

    size_t num_bytes = fill_rx_msg_info(mbufs, info, msgs_recv);
    *num_msgs        = msgs_recv;

    return num_bytes;
}

size_t Rxq::fill_rx_msg_info(rte_mbuf** mbufs, MsgReceiveInfo* info, size_t num_msgs)
{
    size_t rx_bytes            = 0;
    auto   rx_timestamp_offset = nic_->get_fronthaul()->get_timestamp_offset();

    for(size_t i = 0; i < num_msgs; ++i)
    {
        //ASSERT_MBUF_UNCHAINED(mbufs[i]);
        info[i].buffer        = rte_pktmbuf_mtod_offset(mbufs[i], void*, 0);
        info[i].buffer_length = mbufs[i]->pkt_len;
        info[i].opaque        = mbufs[i];
        info[i].rx_timestamp  = *RTE_MBUF_DYNFIELD(mbufs[i], rx_timestamp_offset, Ns*);
        rx_bytes += info[i].buffer_length;
    }

    return rx_bytes;
}

RxqPcap::RxqPcap(Nic* nic, uint16_t id) :
    Queue{nic, id, kRxqPcapSize}
{
    auto port_id   = nic_->get_port_id();
    auto socket_id = rte_eth_dev_socket_id(port_id);
    auto name      = nic_->get_name();
    auto mp        = nic_->get_cpu_mbuf_pool();

    NVLOGD_FMT(TAG, "Setting up PCAP capture RXQ #{} on NIC {} with {} descriptors", id_ , name.c_str(), size_);

    auto ret = rte_eth_rx_queue_setup(port_id, id_, size_, socket_id, nullptr, mp);
    if(ret != 0)
    {
        THROW_FH(ret, StringBuilder() << "Failed to setup PCAP capture RXQ #" << id_ << " on NIC " << name << ": " << rte_strerror(-ret));
    }
}

void RxqPcap::receive()
{
    auto      port_id = nic_->get_port_id();
    rte_mbuf* mbufs[kRxPktBurst];
    size_t    num_pkts = rte_eth_rx_burst(port_id, id_, mbufs, kRxPktBurst);
    rte_pktmbuf_free_bulk(&mbufs[0], num_pkts);
}

} // namespace aerial_fh

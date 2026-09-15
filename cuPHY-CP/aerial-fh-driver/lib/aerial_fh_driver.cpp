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

#include "aerial-fh-driver/api.hpp"

#include "flow.hpp"
#include "fronthaul.hpp"
#include "memreg.hpp"
#include "nic.hpp"
#include "peer.hpp"
#include "ring_buffer.hpp"
#include "time.hpp"
#include "utils.hpp"
#include "cuda_driver_utils/cuda_driver_utils.hpp"

#define TAG "FH.LIB"

// Assumes fhi is in scope
#define FH_CATCH_EXCEPTIONS()                      \
    catch(aerial_fh::FronthaulException const& e)  \
    {                                              \
        NVLOGE_FMT(TAG,AERIAL_ORAN_FH_EVENT, "Exception! {}", e.what()); \
        return e.err_code();                       \
    }                                              \
    catch(std::exception const& e)                 \
    {                                              \
        NVLOGE_FMT(TAG,AERIAL_ORAN_FH_EVENT, "Exception! {}", e.what()); \
        return EINVAL;                             \
    }                                              \
    catch(...)                                     \
    {                                              \
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Uncaught exception!");     \
        return EINVAL;                             \
    }

#define FH_CATCH_EXCEPTIONS_TX()                   \
    catch(aerial_fh::FronthaulException const& e)  \
    {                                              \
        NVLOGE_FMT(TAG,AERIAL_ORAN_FH_EVENT, "Exception! {}", e.what()); \
        return 0;                                  \
    }                                              \
    catch(std::exception const& e)                 \
    {                                              \
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Exception! {}", e.what()); \
        return 0;                                  \
    }                                              \
    catch(...)                                     \
    {                                              \
        NVLOGE_FMT(TAG,AERIAL_ORAN_FH_EVENT,"Uncaught exception!");     \
        return 0;                                  \
    }

#define FH_CATCH_EXCEPTIONS_ALLOC()                \
    catch(aerial_fh::FronthaulException const& e)  \
    {                                              \
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Exception! {}", e.what()); \
        return nullptr;                            \
    }                                              \
    catch(std::exception const& e)                 \
    {                                              \
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Exception! {}" ,e.what()); \
        return nullptr;                            \
    }                                              \
    catch(...)                                     \
    {                                              \
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Uncaught exception!");     \
        return nullptr;                            \
    }


int aerial_fh::open(FronthaulInfo* info, FronthaulHandle* handle)
{
    try
    {
        *handle = new Fronthaul(info);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::close(FronthaulHandle handle)
{
    try
    {
        auto fhi = static_cast<Fronthaul*>(handle);
        delete fhi;
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::add_nic(FronthaulHandle handle, NicInfo const* info, NicHandle* output_handle)
{
    try
    {
        if (info->cuda_device < 0 && info->txq_count_gpu > 0) {
            NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Exception! Can't create {} GPU TXQs with invalid CUDA device id {}",info->txq_count_gpu,info->cuda_device);
            return -ENOTSUP;
        }

        auto fhi       = static_cast<Fronthaul*>(handle);
        *output_handle = new Nic(fhi, info);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::remove_nic(NicHandle handle)
{
    try
    {
        auto nic = static_cast<Nic*>(handle);
        delete nic;
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::print_stats(NicHandle handle, bool extended)
{
    try
    {
        auto nic = static_cast<Nic*>(handle);
        nic->print_stats();
        if(extended)
        {
            nic->print_extended_stats();
        }
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::is_cx6_nic(NicHandle handle,bool& cx6)
{
    try
    {
        auto nic = static_cast<Nic*>(handle);
        cx6 = nic->is_cx6();
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::reset_stats(NicHandle handle)
{
    try
    {
        auto nic = static_cast<Nic*>(handle);
        nic->reset_stats();
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::add_peer(NicHandle handle, PeerInfo const* info, PeerHandle* output_handle,std::vector<uint16_t>& eAxC_list_ul,std::vector<uint16_t>& eAxC_list_srs, std::vector<uint16_t>& eAxC_list_dl)
{
    try
    {
        auto nic       = static_cast<Nic*>(handle);
        *output_handle = new Peer(nic, info,eAxC_list_ul,eAxC_list_srs,eAxC_list_dl);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

size_t aerial_fh::get_gpu_regular_size(PeerHandle handle) {
    auto peer = static_cast<Peer*>(handle);
    return  peer->getGpuRegularSize();
}

size_t aerial_fh::get_gpu_pinned_size(PeerHandle handle) {
    auto peer = static_cast<Peer*>(handle);
    return  peer->getGpuPinnedSize();
}

size_t aerial_fh::get_cpu_regular_size(PeerHandle handle) {
    auto peer = static_cast<Peer*>(handle);
    return  peer->getCpuRegularSize();
}

size_t aerial_fh::get_cpu_pinned_size(PeerHandle handle) {
    auto peer = static_cast<Peer*>(handle);
    return  peer->getCpuPinnedSize();
}

int aerial_fh::remove_peer(PeerHandle handle)
{
    try
    {
        auto peer = static_cast<Peer*>(handle);
        delete peer;
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::update_peer(PeerHandle handle, MacAddr dst_mac_addr,std::vector<uint16_t>& eAxC_list_ul,std::vector<uint16_t>& eAxC_list_srs)
{
    try
    {
        auto peer = static_cast<Peer*>(handle);
        peer->update(dst_mac_addr,eAxC_list_ul,eAxC_list_srs);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::update_rx_metrics(PeerHandle handle, size_t rx_packets, size_t rx_bytes)
{
    try
    {
        auto peer = static_cast<Peer*>(handle);
        peer->update_rx_metrics(rx_packets, rx_bytes);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::update_tx_metrics(PeerHandle handle, size_t tx_packets, size_t tx_bytes)
{
    try
    {
        auto peer = static_cast<Peer*>(handle);
        peer->update_rx_metrics(tx_packets, tx_bytes);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::update_peer(PeerHandle handle, UserDataCompressionMethod dl_comp_meth, uint8_t dl_bit_width)
{
    try
    {
        auto peer = static_cast<Peer*>(handle);
        peer->update(dl_comp_meth, dl_bit_width);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::reset_peer_flow_registration(const PeerHandle handle)
{
    try
    {
        auto peer = static_cast<Peer*>(handle);
        peer->get_eaxcid_idx_mp().clear();
        peer->get_dlu_eaxcid_idx_mp().clear();
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::begin_peer_flow_registration(const PeerHandle handle)
{
    try
    {
        auto peer = static_cast<Peer*>(handle);
        peer->set_defer_flow_hdr_sync(true);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::end_peer_flow_registration(const PeerHandle handle)
{
    try
    {
        auto peer = static_cast<Peer*>(handle);
        peer->set_defer_flow_hdr_sync(false);
        // Flushes the deferred flow_hdr H2D and drains setup_stream_ (including
        // earlier template/pkt-hdr copies from flow registration).
        peer->sync_flow_hdr_size_info_to_device();
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::update_peer(PeerHandle handle, MacAddr dst_mac_addr, uint16_t vlan_tci,std::vector<uint16_t>& eAxC_list_ul,std::vector<uint16_t>& eAxC_list_srs)
{
    try
    {
        auto peer = static_cast<Peer*>(handle);
        peer->update(dst_mac_addr, vlan_tci,eAxC_list_ul,eAxC_list_srs);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::update_peer_max_num_prbs_per_symbol(PeerHandle handle,uint16_t max_num_prbs_per_symbol)
{
    try
    {
        auto peer = static_cast<Peer*>(handle);
        peer->update_max_num_prbs_per_symbol(max_num_prbs_per_symbol);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::get_uplane_txqs(PeerHandle handle, TxqHandle* txqs, size_t* num_txqs)
{
    try
    {
        auto peer          = static_cast<Peer*>(handle);
        auto txqs_internal = reinterpret_cast<Txq**>(txqs);
        peer->get_uplane_txqs(txqs_internal, num_txqs);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::get_doca_rxq_items(PeerHandle handle,void* rxq_items)
{
    try
    {
        auto peer          = static_cast<Peer*>(handle);
        *(static_cast<doca_rx_items_t*>(rxq_items))= *(peer->get_rxq()->get_doca_rx_items());
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::get_doca_rxq_items_srs(PeerHandle handle,void* rxq_items)
{
    try
    {
        auto peer          = static_cast<Peer*>(handle);
        *(static_cast<doca_rx_items_t*>(rxq_items))= *(peer->get_rxq_srs()->get_doca_rx_items());
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::add_flow(PeerHandle handle, FlowInfo* info, FlowHandle* output_handle)
{
    try
    {
        auto peer      = static_cast<Peer*>(handle);
        *output_handle = new Flow(peer, info);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::remove_flow(FlowHandle handle)
{
    try
    {
        auto flow = static_cast<Flow*>(handle);
        delete flow;
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::update_flow(FlowHandle handle, FlowInfo const* info)
{
    try
    {
        auto flow = static_cast<Flow*>(handle);
        flow->update(info);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

void* aerial_fh::allocate_memory(size_t size, unsigned align)
{
    try
    {
        auto ptr = rte_malloc(NULL, size, align);
        NVLOGI_FMT(TAG,"Allocated {} bytes from hugepage area with {} -byte alignment @{}" ,size , align, ptr);
        return ptr;
    }
    FH_CATCH_EXCEPTIONS_ALLOC();
    return nullptr;
}

int aerial_fh::free_memory(void* ptr)
{
    try
    {
        NVLOGI_FMT(TAG,"Freeing hugepage memory @{}" , ptr);
        rte_free(ptr);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::register_memory(FronthaulHandle handle, MemRegInfo const* info, MemRegHandle* output_handle)
{
    try
    {
        auto fhi       = static_cast<Fronthaul*>(handle);
        *output_handle = new MemReg(fhi, info);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::unregister_memory(MemRegHandle handle)
{
    try
    {
        auto memreg = static_cast<MemReg*>(handle);
        delete memreg;
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::prepare_cplane(PeerHandle handle, CPlaneMsgSendInfo const* info, size_t num_msgs, TxRequestHandle* output_handle)
{
    try
    {
        auto peer = static_cast<Peer*>(handle);
        TxRequestCplane* tx_request;
        peer->prepare_cplane(info, num_msgs, &tx_request);
        *output_handle = tx_request;
    }
    FH_CATCH_EXCEPTIONS_TX();
    return 0;
}

size_t aerial_fh::send_cplane(PeerHandle handle, CPlaneMsgSendInfo const* info, size_t num_msgs)
{
    try
    {
        auto peer = static_cast<Peer*>(handle);

        if(num_msgs > 0)
        {
            return peer->send_cplane(info, num_msgs);
        }
    }
    FH_CATCH_EXCEPTIONS_TX();
    return 0;
}

size_t aerial_fh::send_cplane_mmimo(PeerHandle handle, CPlaneMsgSendInfo const* info, size_t num_msgs)
{
    try
    {
        auto peer = static_cast<Peer*>(handle);

        if(num_msgs > 0)
        {
            return peer->send_cplane_mmimo(info, num_msgs);
        }
    }
    FH_CATCH_EXCEPTIONS_TX();
    return 0;
}

int aerial_fh::prepare_uplane(PeerHandle handle, UPlaneMsgSendInfo const* info, size_t num_msgs, UPlaneTxCompleteNotification notification, TxRequestHandle* output_handle, int txq_index /* = 0 */)
{
    try
    {
        auto             peer = static_cast<Peer*>(handle);
        TxRequestUplane* tx_request;
        peer->prepare_uplane(info, num_msgs, notification, &tx_request, txq_index);
        *output_handle = tx_request;
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}


int aerial_fh::prepare_uplane_with_preallocated_tx_request(PeerHandle handle, UPlaneMsgMultiSectionSendInfo const* info, UPlaneTxCompleteNotification notification, TxRequestHandle* output_handle, int txq_index /* = 0 */)
{
    try
    {
        auto             peer = static_cast<Peer*>(handle);
        TxRequestUplane* tx_request =  static_cast<TxRequestUplane*>(*output_handle);
        peer->prepare_uplane_with_preallocated_tx_request(info, notification, &tx_request, txq_index);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::preallocate_mbufs(PeerHandle handle, TxRequestHandle* output_handle, int num_mbufs)
{
    try
    {
        auto             peer = static_cast<Peer*>(handle);
        TxRequestUplane* tx_request =  static_cast<TxRequestUplane*>(*output_handle);
        peer->preallocate_mbufs(&tx_request, num_mbufs);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::free_preallocated_mbufs(PeerHandle handle, TxRequestHandle* output_handle)
{
    try
    {
        auto             peer = static_cast<Peer*>(handle);
        TxRequestUplane* tx_request =  static_cast<TxRequestUplane*>(*output_handle);
        peer->free_preallocated_mbufs(&tx_request);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}


int aerial_fh::alloc_tx_request(PeerHandle handle, TxRequestHandle* output_handle)
{
    try
    {
        auto             peer = static_cast<Peer*>(handle);
        TxRequestUplane* tx_request;
        peer->alloc_tx_request(&tx_request);
        *output_handle = tx_request;
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::free_tx_request(TxRequestHandle tx_req_handle)
{
    try
    {
        auto tx_request = static_cast<TxRequestUplane*>(tx_req_handle);
        auto peer       = tx_request->peer;
        peer->free_tx_request(tx_request);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

size_t aerial_fh::prepare_and_send_uplane(PeerHandle handle, UPlaneMsgSendInfo const* info, size_t num_msgs, UPlaneTxCompleteNotification notification, TxqHandle txq_handle, int txq_index /* = 0 */)
{
    try
    {
        auto             peer = static_cast<Peer*>(handle);
        TxRequestUplane* tx_request;
        peer->prepare_uplane(info, num_msgs, notification, &tx_request, txq_index);
        auto txq = static_cast<Txq*>(txq_handle);
        return peer->send_uplane(tx_request, txq);
    }
    FH_CATCH_EXCEPTIONS_TX();
    return 0;
}

size_t aerial_fh::send_uplane(TxRequestHandle handle, TxqHandle txq_handle)
{
    try
    {
        auto tx_request = static_cast<TxRequestUplane*>(handle);
        auto peer       = tx_request->peer;
        auto txq        = static_cast<Txq*>(txq_handle);
        return peer->send_uplane(tx_request, txq);
    }
    FH_CATCH_EXCEPTIONS_TX();
    return 0;
}

size_t aerial_fh::send_uplane_without_freeing_tx_request(TxRequestHandle handle, TxqHandle txq_handle, TxqSendTiming* timing)
{
    try
    {
        auto tx_request = static_cast<TxRequestUplane*>(handle);
        auto peer       = tx_request->peer;
        auto txq        = static_cast<Txq*>(txq_handle);
        return peer->send_uplane_without_freeing_tx_request(tx_request, txq, timing);
    }
    FH_CATCH_EXCEPTIONS_TX();
    return 0;
}

int aerial_fh::poll_tx_complete(PeerHandle handle)
{
    try
    {
        auto peer = static_cast<Peer*>(handle);
        peer->poll_tx_complete();
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::receive(PeerHandle handle, MsgReceiveInfo* info, size_t* num_msgs, bool srs)
{
    try
    {
        auto peer = static_cast<Peer*>(handle);
        peer->receive(info, num_msgs, srs);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::receive_flow(FlowHandle handle, MsgReceiveInfo* info, size_t* num_msgs)
{
    try
    {
        auto flow = static_cast<Flow*>(handle);
        flow->receive(info, num_msgs);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::receive_until(PeerHandle handle, MsgReceiveInfo* info, size_t* num_msgs, Ns timeout)
{
    try
    {
        auto peer = static_cast<Peer*>(handle);
        peer->receive_until(info, num_msgs, timeout);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::receive_flow_until(FlowHandle handle, MsgReceiveInfo* info, size_t* num_msgs, Ns timeout)
{
    try
    {
        auto flow = static_cast<Flow*>(handle);
        flow->receive_until(info, num_msgs, timeout);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::free_rx_messages(MsgReceiveInfo const* info, size_t num_msgs)
{
    try
    {
        while(num_msgs > 0)
        {
            rte_mbuf* mbufs[kRxPktBurst];
            size_t    bulk_size = std::min(num_msgs, kRxPktBurst);

            for(size_t i = 0; i < bulk_size; ++i)
            {
                mbufs[i] = static_cast<rte_mbuf*>(info[i].opaque);
            }
            rte_pktmbuf_free_bulk(mbufs, bulk_size);
            num_msgs -= bulk_size;
        }
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::ring_create(FronthaulHandle handle, RingBufferInfo const* info, RingBufferHandle* output_handle)
{
    try
    {
        auto fhi       = static_cast<Fronthaul*>(handle);
        *output_handle = new RingBuffer(fhi, info);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::ring_destroy(RingBufferHandle handle)
{
    try
    {
        auto ring = static_cast<RingBuffer*>(handle);
        delete ring;
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::ring_enqueue(RingBufferHandle handle, void* obj)
{
    try
    {
        auto ring = static_cast<RingBuffer*>(handle);
        return ring->enqueue(obj);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::ring_enqueue_bulk_tx_request_cplane_mbufs(RingBufferHandle handle, TxRequestHandle txrequest, PeerHandle peer_handle, size_t count)
{
    try
    {
        auto ring = static_cast<RingBuffer*>(handle);
        auto tx_request = static_cast<TxRequestCplane*>(txrequest);
        auto peer = static_cast<Peer*>(peer_handle);
        auto ret = ring->enqueue_bulk((void**)tx_request->mbufs, count);
        peer->enqueue_cplane_tx_request_descriptor(tx_request);
        return ret;
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}


size_t aerial_fh::ring_enqueue_bulk(RingBufferHandle handle, void* const* objs, size_t count)
{
    try
    {
        auto ring = static_cast<RingBuffer*>(handle);
        return ring->enqueue_bulk(objs, count);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

size_t aerial_fh::ring_enqueue_burst(RingBufferHandle handle, void* const* objs, size_t count)
{
    try
    {
        auto ring = static_cast<RingBuffer*>(handle);
        return ring->enqueue_burst(objs, count);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::ring_dequeue(RingBufferHandle handle, void** obj)
{
    try
    {
        auto ring = static_cast<RingBuffer*>(handle);
        return ring->dequeue(obj);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

size_t aerial_fh::ring_dequeue_bulk(RingBufferHandle handle, void** objs, size_t count)
{
    try
    {
        auto ring = static_cast<RingBuffer*>(handle);
        return ring->dequeue_bulk(objs, count);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

size_t aerial_fh::ring_dequeue_burst(RingBufferHandle handle, void** objs, size_t count)
{
    try
    {
        auto ring = static_cast<RingBuffer*>(handle);
        return ring->dequeue_burst(objs, count);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

size_t aerial_fh::ring_dequeue_burst_mbufs_payload_offset(RingBufferHandle handle, void** objs, MsgReceiveInfo* info, size_t count)
{
    try
    {
        auto ring = static_cast<RingBuffer*>(handle);
        auto dequeue_count = ring->dequeue_burst(objs, count);
        for(int i = 0; i < dequeue_count; ++i)
        {
            info[i].opaque = objs[i];
            info[i].buffer = (void*)rte_pktmbuf_mtod_offset((rte_mbuf*)objs[i], void*, 0);
        }
        return dequeue_count;
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}


size_t aerial_fh::ring_free_count(RingBufferHandle handle)
{
    try
    {
        auto ring = static_cast<RingBuffer*>(handle);
        return ring->free_count();
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

bool aerial_fh::ring_full(RingBufferHandle handle)
{
    try
    {
        auto ring = static_cast<RingBuffer*>(handle);
        return ring->full();
    }
    FH_CATCH_EXCEPTIONS();
    return false;
}

bool aerial_fh::ring_empty(RingBufferHandle handle)
{
    try
    {
        auto ring = static_cast<RingBuffer*>(handle);
        return ring->empty();
    }
    FH_CATCH_EXCEPTIONS();
    return false;
}

int aerial_fh::update_metrics(FronthaulHandle handle)
{
    try
    {
        auto fhi = static_cast<Fronthaul*>(handle);
        fhi->update_metrics();
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}


int aerial_fh::gpu_comm_update_tx_metrics(PeerHandle handle, TxRequestGpuCommHandle tx_request_handle)
{
    try
    {
        auto peer = static_cast<Peer*>(handle);
        auto tx_request = static_cast<TxRequestUplaneGpuComm*>(tx_request_handle);
        peer->gpu_comm_update_tx_metrics(tx_request);
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

/**
 * Stamp the stable per-cell index into a TxRequestUplaneGpuComm (see api.hpp).
 *
 * @param[in] tx_request_handle GpuComm TX request handle
 * @param[in] cell_idx Logical cell index in [0, API_MAX_NUM_CELLS)
 * @return 0 on success, -EINVAL or -ERANGE on failure
 */
[[nodiscard]] int aerial_fh::set_gpu_request_cell_idx(TxRequestGpuCommHandle tx_request_handle, uint8_t cell_idx) noexcept
{
    if(tx_request_handle == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "set_gpu_request_cell_idx: null tx_request_handle (cell_idx={})", cell_idx);
        return -EINVAL;
    }
    if(cell_idx >= API_MAX_NUM_CELLS)
    {
        // Out-of-range index would alias another cell's per-cell state; reject.
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "set_gpu_request_cell_idx: cell_idx={} >= API_MAX_NUM_CELLS={}", cell_idx, API_MAX_NUM_CELLS);
        return -ERANGE;
    }
    static_cast<TxRequestUplaneGpuComm*>(tx_request_handle)->m_cell_idx = cell_idx;
    return 0;
}

int aerial_fh::setup_uplane_gpu_comm(PeerHandle handle, uint8_t frame_id, uint16_t subframe_id, uint16_t slot_id,
    uint16_t max_num_prb_per_symbol, std::chrono::nanoseconds cell_start_time,
    std::chrono::nanoseconds symbol_duration, bool commViaCpu, TxRequestGpuCommHandle* output_handle,
    PartialUplaneSlotInfo_t** out_partial_info,
    UplaneConversionParams* out_params,
    const uint16_t* eaxcid_list,
    uint16_t num_eaxcids)
{
    try
    {
        auto peer = static_cast<Peer*>(handle);
        TxRequestUplaneGpuComm* tx_request;
        peer->gpu_comm_setup_tx_request(frame_id, subframe_id, slot_id, max_num_prb_per_symbol,
            cell_start_time, symbol_duration, commViaCpu, &tx_request, out_partial_info,
            out_params, eaxcid_list, num_eaxcids);
        *output_handle = tx_request;
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

int aerial_fh::prepare_uplane_gpu_comm(PeerHandle handle, UPlaneMsgSendInfo const* info, size_t num_msgs, TxRequestGpuCommHandle* output_handle,
	std::chrono::nanoseconds cell_start_time,  std::chrono::nanoseconds symbol_duration,bool commViaCpu)
{
//printf("cell_start_time %lu and symbol duration %lu\n", cell_start_time.count(), symbol_duration.count());
    try
    {
        auto peer = static_cast<Peer*>(handle);
        TxRequestUplaneGpuComm* tx_request;
        peer->gpu_comm_prepare_uplane(info, num_msgs, &tx_request, cell_start_time, symbol_duration,commViaCpu);
        *output_handle = tx_request;
    }
    FH_CATCH_EXCEPTIONS();
    return 0;
}

// Can delete once the relevant function is removed from cuphydriver
int aerial_fh::prepare_uplane_gpu_comm_v2(PeerHandle handle, UPlaneMsgSendInfo const* info, size_t num_msgs, TxRequestGpuCommHandle* output_handle,
		std::chrono::nanoseconds cell_start_time,  std::chrono::nanoseconds symbol_duration)
{
    return prepare_uplane_gpu_comm(handle, info, num_msgs, output_handle, cell_start_time, symbol_duration,false);
}


int aerial_fh::ring_cpu_doorbell(NicHandle handle, TxRequestGpuPercell *pTxRequestGpuPercell, PreparePRBInfo &prb_info, PacketTimingInfo &packet_timing_info) {
    try
    {
        auto nic = static_cast<Nic*>(handle);
        return nic->ring_cpu_doorbell(pTxRequestGpuPercell, prb_info, packet_timing_info);
    }
    FH_CATCH_EXCEPTIONS_TX();
    return 0;
}

int aerial_fh::send_uplane_gpu_comm(NicHandle handle, TxRequestGpuPercell *pTxRequestGpuPercell, PreparePRBInfo &prb_info)
{
    try
    {
        auto nic = static_cast<Nic*>(handle);
        return nic->gpu_comm_send_uplane(pTxRequestGpuPercell, prb_info);
    }
    FH_CATCH_EXCEPTIONS_TX();
    return 0;
}

int aerial_fh::set_TriggerTs_GpuComm(NicHandle handle,uint32_t slot_idx,uint64_t trigger_ts)
{
    try
    {
        auto nic = static_cast<Nic*>(handle);
        return nic->gpu_comm_set_trigger_ts(slot_idx,trigger_ts);
    }
    FH_CATCH_EXCEPTIONS_TX();
    return 0;
}

int aerial_fh::trigger_cqe_tracer_cb(NicHandle handle, TxRequestGpuPercell *pTxRequestGpuPercell)
{
    try
    {
        auto nic = static_cast<Nic*>(handle);
        return nic->gpu_comm_trigger_cqe_tracer_cb(pTxRequestGpuPercell);
    }
    FH_CATCH_EXCEPTIONS_TX();
    return 0;
}


int aerial_fh::print_max_delays(NicHandle handle)
{
    try
    {
        auto nic = static_cast<Nic*>(handle);
        nic->log_max_delays();
    }
    FH_CATCH_EXCEPTIONS_TX();
    return 0;
}

std::optional<aerial_fh::CplaneSender> aerial_fh::make_cplane_sender(NicHandle nic_handle, oran_pkt_dir direction)
{
    auto* nic = static_cast<Nic*>(nic_handle);
    if (!nic) {
        return std::nullopt;
    }

    auto& qm = nic->get_queue_manager();
    Txq* txq = nullptr;
    bool txq_assigned = false;
    try {
        txq = qm.assign_txq(false);
        txq_assigned = true;
    } catch (...) {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT,
            "make_cplane_sender: failed to assign TX queue for NIC {}", nic->get_name());
        return std::nullopt;
    }

    auto* pool = nic->get_cpu_cplane_mbuf_pool();
    if (!pool) {
        if (txq_assigned && txq) {
            qm.reclaim(txq);
        }
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT,
            "make_cplane_sender: no C-plane mempool for NIC {}", nic->get_name());
        return std::nullopt;
    }

    rte_mbuf* test_mbuf = nullptr;
    if (rte_pktmbuf_alloc_bulk(pool, &test_mbuf, 1) == 0) {
        uint16_t capacity = rte_pktmbuf_tailroom(test_mbuf);
        uint16_t mtu = nic->get_mtu();
        rte_pktmbuf_free(test_mbuf);
        if (mtu > capacity) {
            if (txq_assigned && txq) {
                qm.reclaim(txq);
            }
            NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT,
                "make_cplane_sender: MTU {} exceeds mbuf capacity {}", mtu, capacity);
            return std::nullopt;
        }
    }

    return CplaneSender(static_cast<TxqHandle>(txq), pool);
}

int aerial_fh::CplaneSender::alloc(rte_mbuf** mbufs, unsigned count)
{
    return rte_pktmbuf_alloc_bulk(mempool_, mbufs, count);
}

void aerial_fh::CplaneSender::free(rte_mbuf** mbufs, unsigned count)
{
    rte_pktmbuf_free_bulk(mbufs, count);
}

std::error_code aerial_fh::CplaneSender::send(rte_mbuf** mbufs, std::size_t count)
{
    if (count == 0) {
        return {};
    }
    if ((mbufs == nullptr) || (txq_handle_ == nullptr)) {
        return std::make_error_code(std::errc::invalid_argument);
    }
    auto* txq = static_cast<Txq*>(txq_handle_);
    try {
        const std::size_t sent_packets = use_lock_
            ? txq->send_lock(mbufs, count)
            : txq->send(mbufs, count);
        if (sent_packets != count) {
            return std::make_error_code(std::errc::io_error);
        }
    } catch (const aerial_fh::FronthaulException& e) {
        return std::error_code(e.err_code(), std::generic_category());
    } catch (...) {
        return std::make_error_code(std::errc::io_error);
    }
    return {};
}

size_t aerial_fh::prepare_cplane_count_packets(PeerHandle handle, CPlaneMsgSendInfo const* info, size_t num_msgs)
{
    try
    {
        auto peer = static_cast<Peer*>(handle);
        return peer->prepare_cplane_count_packets(info, num_msgs);
    }
    FH_CATCH_EXCEPTIONS_TX();
    return 0;
}

size_t aerial_fh::prepare_uplane_count_packets(PeerHandle handle, UPlaneMsgMultiSectionSendInfo const* info)
{
    try
    {
        auto peer = static_cast<Peer*>(handle);
        return peer->prepare_uplane_count_packets(info);
    }
    FH_CATCH_EXCEPTIONS_TX();
    return 0;
}

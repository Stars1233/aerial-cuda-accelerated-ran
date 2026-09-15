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

#if !defined(NV_PHY_MODULE_HPP_INCLUDED_)
#define NV_PHY_MODULE_HPP_INCLUDED_

#include "app_config.hpp"
#include "yaml.hpp"
#include "nv_phy_instance.hpp"
#include "nv_phy_mac_transport.hpp"
#include "nv_phy_epoll_context.hpp"
#include "slot_command/csirs_lookup.hpp"
#include "nv_phy_tick_updater.hpp"
#include "nv_tick_generator.hpp"
#include "nvlog.hpp"
#include "stat_log.h"
#include "nv_phy_driver_proxy.hpp"
#include "nv_phy_config_option.hpp"
#include "nv_utils.hpp"
#include "cuphyoam.hpp"
#include "nv_phy_limit_errors.hpp"
#include "nv_fapi_message_storage.hpp"
#include "active_cell_mask.hpp"
#ifdef ENABLE_FAPI_STORE_REPLAY
#include "nv_phy_non_slot_dispatch.hpp"
#include "nv_slot_telemetry.hpp"  // SlotTelemetrySnapshot, stamp_ring_telemetry
#endif
#include "nv_slot_task_pool.hpp"      // SlotTaskPool, SLOT_STORAGE_DEPTH, slot_index_from_u32
#include "nv_tx_data_ring_buffer.hpp" // TxDataRingBuffer, TxDataSlotState

#include <array>
#include <atomic>
#include <bit>
#include <concepts>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <chrono>
#include <queue>
#include <map>
#include <utility>

#ifdef ENABLE_FAPI_STORE_REPLAY
#include <optional>
#endif
namespace nv
{

namespace detail
{
/**
 * L1 mMIMO enable flag from PHYDriverProxy.
 *
 * Queries PHYDriverProxy on every call; the flag is stable within a slot.
 * Call once at the slot boundary and cache the result (e.g. PhyModuleView::mmimo_enabled_)
 * rather than invoking this in a per-PDU loop.
 *
 * @return  true when @c PHYDriverProxy::getInstance().l1_mMIMO_enable_info() reports a
 *          non-zero value (L1 mMIMO is enabled); false when it reports zero.
 *          Return value must be checked — callers should cache it at the slot boundary
 *          instead of invoking this function in per-PDU loops.
 */
[[nodiscard]] inline bool query_l1_mmimo_enabled()
{
    uint8_t m{};
    std::ignore = PHYDriverProxy::getInstance().l1_mMIMO_enable_info(&m);
    return !!m;
}

/**
 * L1 SRS enable flag from PHYDriverProxy.
 *
 * Same caching contract as @ref query_l1_mmimo_enabled — call once at the slot
 * boundary and cache the result (e.g. @c PhyModuleView::srs_enabled_) rather
 * than invoking this in a per-PDU loop.
 *
 * @return  true when @c PHYDriverProxy::getInstance().l1_enable_srs_info() reports a
 *          non-zero value; false when it reports zero. Return value must be checked.
 */
[[nodiscard]] inline bool query_l1_srs_enabled()
{
    uint8_t s{};
    std::ignore = PHYDriverProxy::getInstance().l1_enable_srs_info(&s);
    return !!s;
}
} // namespace detail

#if 1
#ifdef SCF_FAPI_10_04
typedef enum
{
    RX_DATA_IND_IDX = 0,
    CRC_DATA_IND_IDX,
    UCI_DATA_IND_IDX,
    RACH_DATA_IND_IDX,
    SRS_DATA_IND_IDX,
    DL_TTI_RSP_IDX,
    MAX_IND_INDEX
} indication_index;
typedef enum
{
    NO_CHANGE,
    ONE_MSG_INSTANCE_PER_SLOT,
    MULTI_MSG_INSTANCE_PER_SLOT,
    RESERVED
} indication_instance_val;
struct PHY_config
{
    //Value, for each entry: • 0: no (change in) configuration • 1: limit to one message instance per slot & numerology • 2: allow generation of more than one message instance per slot &numerology • other values are reserved
    //Scope of each entry: [0]: Rx_Data.indication [1]: CRC.indication [2]: UCI.indication [3]: RACH.indication [4]: SRS.indication [5]: DL_TTI.response
    uint8_t indication_instances_per_slot[6];
    PHY_config()
    {
        for(uint8_t& indication : indication_instances_per_slot)
            indication = 1;
    }
};
#endif
#endif

using namespace std::chrono;
typedef std::reference_wrapper<PHY_instance> PHY_instance_ref;
struct phy_mac_msg_desc;

enum dl_tb_loc
{
    TB_LOC_INLINE       = 0,
    TB_LOC_EXT_HOST_BUF = 1,
    TB_LOC_EXT_GPU_BUF  = 2
};

#define MAX_PER_CH_PROC_SEGMENTS 4

// Verify that CPLANE_BATCH_CELL_LIMIT (defined in nv_cplane_batch_tasks.hpp
// without MAX_CELLS_PER_SLOT access) is large enough to cover the actual cell array size.
static_assert(nv::CPLANE_BATCH_CELL_LIMIT >= MAX_CELLS_PER_SLOT,
              "CPLANE_BATCH_CELL_LIMIT must be >= MAX_CELLS_PER_SLOT");
// SlotTaskPool, SLOT_STORAGE_DEPTH, and slot_index_from_u32 are provided by nv_slot_task_pool.hpp.

using ch_segment       = pair<uint16_t, uint16_t>;
using ch_seg_timelines = array<ch_segment, MAX_PER_CH_PROC_SEGMENTS>;
using ch_indexes       = array<uint, 8>;

enum sync_mode_t : uint8_t
{
    SYNC_MODE_PER_CELL,
    SYNC_MODE_PER_SLOT
};

class PHY_module;

#ifdef ENABLE_FAPI_STORE_REPLAY
struct NonSlotDispatchState final {
    std::array<FapiNonSlotMessageStorage, MAX_CELLS_PER_SLOT> message_storage{};
    std::optional<NonSlotLpExecutor> executor{};
    nv::ActiveCellMask active_cell_mask_{};

    NonSlotDispatchState() = default;

    NonSlotDispatchState(std::array<FapiNonSlotMessageStorage, MAX_CELLS_PER_SLOT>&& moved_message_storage,
                         uint64_t moved_staged_active_cell_bitmap,
                         uint64_t moved_produced_active_cell_bitmap,
                         uint64_t moved_committed_active_cell_bitmap) :
        message_storage(std::move(moved_message_storage)),
        active_cell_mask_(moved_staged_active_cell_bitmap, moved_produced_active_cell_bitmap, moved_committed_active_cell_bitmap)
    {
    }

    NonSlotDispatchState(NonSlotDispatchState&& other) noexcept :
        NonSlotDispatchState(std::move(other.message_storage),
                             other.active_cell_mask_.staged(),
                             other.active_cell_mask_.produced(),
                             other.active_cell_mask_.committed())
    {
    }

    NonSlotDispatchState& operator=(NonSlotDispatchState&&) = delete;
    NonSlotDispatchState(const NonSlotDispatchState&) = delete;
    NonSlotDispatchState& operator=(const NonSlotDispatchState&) = delete;

    [[nodiscard]] uint64_t active_cell_bitmap() const noexcept;
    [[nodiscard]] uint32_t active_cell_count() const noexcept;
    [[nodiscard]] uint64_t activate_cell_immediate(uint16_t cell_id) noexcept;
    [[nodiscard]] uint64_t deactivate_cell_immediate(uint16_t cell_id) noexcept;
    void stage_cell_started(uint16_t cell_id) noexcept;
    void stage_cell_stopped(uint16_t cell_id) noexcept;
    [[nodiscard]] uint64_t commit_staged_active_cell_bitmap() noexcept;
    void mark_produced(uint16_t cell_id) noexcept;
    void install_worker(PHY_module& owner, NonSlotLpThreadConfig config);
    void start_worker();
    void stop_worker();
    void move_worker_from(PHY_module& owner, NonSlotDispatchState& other);
    [[nodiscard]] bool enqueue_work(phy_mac_msg_desc& smsg);
};
#endif

/**
 * @brief Callback function for PHY module reset
 * @param transp Pointer to the PHY-MAC transport
 * @param phy_module Pointer to the PHY module instance
 * @return Status code (0 for success)
 */
int phy_module_reset_callback(phy_mac_transport* transp, PHY_module* phy_module);

/**
 * @brief Concept constraining the @p Policy template argument of
 *        PHY_module::fire_cplane_batch().
 *
 * A conforming Policy is a tag struct with three static member functions that
 * together implement the direction-specific (DL or UL) side effects of one
 * C-plane batch dispatch cycle.  The concept is checked at namespace scope
 * because C++20 concepts are not permitted inside class bodies.
 *
 * @par Required static members
 *
 * @code{.cpp}
 * static nv::cplane_process_cell_fn_t process_cell_fn();
 * @endcode
 * Returns the per-cell processing callback stored in
 * `CplaneBatchTaskArg::process_cell`.  The returned pointer is written into
 * the task argument before the batch worker runs, so it must match the exact
 * function-pointer type `nv::cplane_process_cell_fn_t`.
 *
 * @code{.cpp}
 * static void on_drop(PHY_module* m, uint32_t ring_idx, uint32_t slot_u32);
 * @endcode
 * Called when `push_task()` fails (e.g. task-pool exhausted).  Must undo the
 * `tasks_in_flight` increment that was performed before the push attempt,
 * without triggering slot release (that is the caller's responsibility).
 * @param m         The owning PHY_module instance.
 * @param ring_idx  Ring-buffer index for the current slot.
 * @param slot_u32  Packed SFN/slot word for the current slot.
 *
 * @code{.cpp}
 * static bool push_task(uint64_t ts, l1_task_work_fn_t fn, void* arg);
 * @endcode
 * Enqueues the task onto the direction-specific worker pool.
 * @param ts   Execution timestamp in nanoseconds (from system_clock).
 * @param fn   Work function to execute on the worker thread.
 * @param arg  Opaque argument forwarded to @p fn.
 * @return     @c true on success, @c false if the push failed (pool full or
 *             driver null); the caller will invoke @c on_drop() on failure.
 *
 * @par Implementers
 * - `PHY_module::DlcBatchPolicy` — routes tasks to the DL worker pool via
 *   `PHYDriverProxy::l1_push_new_dl_task()`.
 * - `PHY_module::UlcBatchPolicy` — routes tasks to the UL worker pool via
 *   `PHYDriverProxy::l1_push_new_ul_task()`.
 *
 * @par Usage
 * @code{.cpp}
 * // Inside PHY_module::fire_cplane_batch (simplified):
 * template <CplaneBatchPolicy Policy>
 * void PHY_module::fire_cplane_batch(uint32_t ring_idx, uint32_t slot_u32, uint64_t ts)
 * {
 *     arg.process_cell = Policy::process_cell_fn();
 *     if (!Policy::push_task(ts, task_work_fn_cplane_batch, &arg))
 *         Policy::on_drop(this, ring_idx, slot_u32);
 * }
 * @endcode
 */
template <typename Policy>
concept CplaneBatchPolicy = requires(
    PHY_module* m,
    uint32_t    ring_idx,
    uint32_t    slot_u32,
    uint64_t    ts,
    uint8_t     cell_id,
    l1_task_work_fn_t fn,
    void*       arg) {
    { Policy::process_cell_fn()   } -> std::same_as<nv::cplane_process_cell_fn_t>;
    { Policy::post_batch_fn()     } -> std::same_as<nv::post_batch_fn_t>;
    { Policy::on_batch_error_fn() } -> std::same_as<nv::cplane_on_batch_error_fn_t>;
    { Policy::get_slot_map(m, ring_idx, cell_id) } -> std::same_as<void*>;
    { Policy::on_drop(m, ring_idx, slot_u32) } -> std::same_as<void>;
    { Policy::push_task(ts, fn, arg) } -> std::same_as<bool>;
};

/**
 * @brief Main PHY module class managing PHY instances and message processing
 *
 * The PHY_module class is responsible for managing multiple PHY instances,
 * handling message routing, tick generation, and coordinating communication
 * between MAC and PHY layers.
 */
class PHY_module {
public:
    /**
     * @brief Destructor
     */
    ~PHY_module();

    /**
     * @brief Constructor
     * @param node_config YAML configuration node containing module settings
     */
    PHY_module(yaml::node node_config);

    PHY_module(const PHY_module&)            = delete;
    PHY_module& operator=(const PHY_module&) = delete;

    /**
     * @brief Move constructor
     * @param other PHY_module instance to move from
     *
     * NOTE: task_pool_ is intentionally omitted from the initializer list.
     * nv::SlotTaskPool contains std::atomic members, which are not moveable,
     * so std::move(other.task_pool_) would not compile. The default-constructed
     * state (all counters zero, bitmaps clear) is the correct initial state for
     * a freshly moved-to instance. PHY_module must only be move-constructed
     * before any tasks are in-flight; moving a live instance would leave the new
     * object with stale-zero counters while the source retains the real state.
     * The low-priority non-slot executor follows the same rule: the move path
     * preserves its configuration and recreates an empty executor bound to the
     * moved-to module, but moving while that worker owns queued descriptors is
     * not supported.
     */
    PHY_module(PHY_module&& other) :
        transport_wrapper_(std::move(other.transport_wrapper_)),
        thread_(std::move(other.thread_)),
        phy_instances_(std::move(other.phy_instances_)),
        phy_refs_(std::move(other.phy_refs_)),
        epoll_ctx_p(std::move(other.epoll_ctx_p)),
        thread_cfg_(std::move(other.thread_cfg_)),
        dl_tbs_queue_(std::move(other.dl_tbs_queue_)),
        tick_updater_(std::move(other.tick_updater_)),
        callbacks_(std::move(other.callbacks_)),
        tti_module_(std::move(other.tti_module_)),
        dl_tb_location_(std::move(other.dl_tb_location_)),
        lbrm_(std::move(other.lbrm_)),
        gps_alpha_(std::move(other.gps_alpha_)),
        gps_beta_(std::move(other.gps_beta_)),
        ul_cqi_(std::move(other.ul_cqi_)),
        test_type(std::move(other.test_type)),
        ss_curr(std::move(other.ss_curr)),
        tti_event_count(std::move(other.tti_event_count)),
        allowed_tick_error(std::move(other.allowed_tick_error)),
        prepone_h2d_copy_(std::move(other.prepone_h2d_copy_)),
        rssi_(std::move(other.rssi_)),
        cell_group_(std::move(other.cell_group_)),
        dtx_thresholds_(std::move(other.dtx_thresholds_)),
        dtx_thresholds_pusch_(std::move(other.dtx_thresholds_pusch_)),
        server_addr(std::move(other.server_addr)),
        target_node(std::move(other.target_node)),
        enable_se_sync_cmd(std::move(other.enable_se_sync_cmd)),
        current_tick_list_(std::move(other.current_tick_list_)),
        l1_slot_ind_tick_(std::move(other.l1_slot_ind_tick_)),
        l2a_start_tick_(std::move(other.l2a_start_tick_)),
        l2a_end_tick_(std::move(other.l2a_end_tick_)),
        last_fapi_msg_tick_(std::move(other.last_fapi_msg_tick_)),
        new_slot_(std::move(other.new_slot_)),
        startup_boot_complete_(std::move(other.startup_boot_complete_)),
        is_ul_slot_(std::move(other.is_ul_slot_)),
        is_dl_slot_(std::move(other.is_dl_slot_)),
        is_csirs_slot_(std::move(other.is_csirs_slot_)),
        slot_message_storage_(std::move(other.slot_message_storage_)),
#ifdef ENABLE_FAPI_STORE_REPLAY
        nonslot_dispatch_state_(std::move(other.nonslot_dispatch_state_)),
        ul_order_scratch_(std::move(other.ul_order_scratch_)),
#endif
        // group_command_(std::move(other.group_command_))
        current_slot_cmd_index(other.current_slot_cmd_index),
        slot_command_array(std::move(other.slot_command_array)),
        num_cells_active(other.num_cells_active),
        total_cell_num(other.total_cell_num),
        curr_slot_fapi_num(other.curr_slot_fapi_num),
        next_slot_fapi_num(other.next_slot_fapi_num),
#ifdef ENABLE_L2_SLT_RSP
        active_cell_bitmap(std::move(other.active_cell_bitmap)),
        fapi_eom_rcvd_bitmap(std::move(other.fapi_eom_rcvd_bitmap)),
        cached_cell_bitmap(other.cached_cell_bitmap),
#else
        slot_msgs_received(std::move(other.slot_msgs_received)),
        num_msgs(other.num_msgs),
        allowed_fapi_latency(std::move(other.allowed_fapi_latency)),
        ss_last(std::move(other.ss_last)),
#endif
        cell_update_cb_fn(std::move(other.cell_update_cb_fn)),
        current_tick_(other.current_tick_),
        ipc_sync_mode(std::move(other.ipc_sync_mode)),
        first_dl_slot_(std::move(other.first_dl_slot_)),
        first_ul_slot_(std::move(other.first_ul_slot_)),
        config_options_(std::move(other.config_options_)),
        bfwCoeff_mem_info(std::move(other.bfwCoeff_mem_info)),
        timer_thread_wakeup_threshold_(std::move(other.timer_thread_wakeup_threshold_)),
        l2a_allowed_latency_(std::move(other.l2a_allowed_latency_)),
        fapi_config_check_mask_(std::move(other.fapi_config_check_mask_)),
        ch_proc_seg_indexes(std::move(other.ch_proc_seg_indexes)),
        ch_proc_seg_timelines(std::move(other.ch_proc_seg_timelines)),
#ifdef ENABLE_L2_SLT_RSP
        cell_limit_errors_(std::move(other.cell_limit_errors_)),
        group_limit_errors_(std::move(other.group_limit_errors_))
#endif
    {
        // Replace pointer to parent module in PHY instances, since that
        // module no longer exists...
        for(auto& phy : phy_instances_)
        {
            phy->set_module(*this);
        }

        // Old module does not exist, remap the transport fd event callback.
        std::unique_ptr<member_event_callback<PHY_module>> mcb_p(new member_event_callback<PHY_module>(this, &PHY_module::msg_processing));
        for(phy_mac_transport* ptransport : transport_wrapper_.get_transports())
        {
            ptransport->set_reset_callback(phy_module_reset_callback, this);
            epoll_ctx_p->add_fd(ptransport->get_fd(), mcb_p.get());
        }
        msg_mcb_p = std::move(mcb_p);
        tti_module_.set_module(*this);
#ifdef ENABLE_FAPI_STORE_REPLAY
        move_nonslot_lp_worker_from(other);
#endif

        tick_logger        = other.tick_logger;
        slot_latency       = other.slot_latency;
        other.slot_latency = nullptr;
        other.tick_logger  = nullptr;
    }

    /**
     * @brief Get references to all PHY instances
     * @return Vector of PHY instance references
     */
    std::vector<PHY_instance_ref>& PHY_instances() { return phy_refs_; }

    /**
     * @brief Start the PHY module thread
     *
     * Initiates the PHY_module processing thread that handles message
     * routing and coordination between MAC and PHY instances.
     */
    void start();

    /**
     * Stop the PHY module
     *
     * Signals the module thread and tick generator to stop.
     */
    void stop();

    /**
     * @brief Join the PHY module thread
     *
     * Blocks until all PHY instances have completed their execution.
     */
    void join();

    /**
     * @brief Get transport instance for a specific cell
     * @param cell_id Cell identifier
     * @return Reference to the PHY-MAC transport for the specified cell
     */
    phy_mac_transport& transport(int cell_id) { return transport_wrapper_.get_transport(cell_id); }

    /**
     * @brief Get the transport wrapper instance
     * @return Reference to the transport wrapper managing all transports
     */
    phy_mac_transport_wrapper& transport_wrapper() { return transport_wrapper_; }

    /**
     * @brief Reset a transport connection
     * @param transp Pointer to the transport to reset
     * @return Status code (0 for success)
     */
    int reset_transport(phy_mac_transport* transp);

    /**
     * @brief Store one message into non-slot or slot storage.
     * @param msg Message descriptor to store.
     * @param slot_u32 Absolute slot for slot-scoped messages, or
     * `SFN_SLOT_INVALID` for non-slot messages.
     * @return true if the message is accepted by the target storage, false otherwise.
     * @note For non-slot messages, replaced buffers can be released via transport.
     */
    [[nodiscard]] bool store_message(const phy_mac_msg_desc& msg, uint32_t slot_u32);

private:
    /**
     * @brief Resolve the ring entry for an absolute SFN/slot.
     * @param slot_u32 Absolute slot encoded as SFN/slot u32.
     * @return Reference to the mapped slot-message storage entry.
     * @note Uses modulo mapping with `SLOT_STORAGE_DEPTH`; different absolute
     * slots can map to the same ring entry across time.
     * @thread_safety Called from the msg-processing thread only.
     */
    [[nodiscard]] FapiSlotMessageStorage& slot_message_storage(const uint32_t slot_u32);
    /**
     * @brief Convert a packed slot_u32 to its ring-buffer index.
     *
     * Combines `slot_index_from_u32()` and the modulo-SLOT_STORAGE_DEPTH step
     * that every call site repeats.  The current numerology is read from
     * `tick_updater_.mu_highest_` each call, matching the existing per-site pattern.
     *
     * @param slot_u32 Packed SFN/slot value.
     * @return Ring index in [0, SLOT_STORAGE_DEPTH).
     */
    [[nodiscard]] uint32_t ring_idx_from_slot(uint32_t slot_u32) const noexcept;
    /**
     * @brief Resolve the non-slot message storage for a given cell.
     * @param cell_id Zero-based cell index; must be less than the configured
     *                cell count (bounds-checked via `at()`).
     * @return Reference to the `FapiNonSlotMessageStorage` owned by that cell.
     * @thread_safety Called from the msg-processing thread only.
     */
#ifdef ENABLE_FAPI_STORE_REPLAY
    [[nodiscard]] FapiNonSlotMessageStorage& non_slot_message_storage(const uint16_t cell_id)
    {
        return nonslot_dispatch_state_.message_storage.at(cell_id);
    }
#endif

    /**
     * @brief Main thread function for PHY module processing
     */
    void thread_func();

    /**
     * @brief Process incoming messages from transport layer
     */
    void msg_processing();

    /**
     * @brief Thread function for cell configuration updates
     * @param arg Pointer to PHY_module instance
     * @return void pointer (unused)
     */
    static void* cell_update_thread_func(void* arg);

    /**
     * @brief Thread function for SFN/slot synchronization commands
     * @param arg Pointer to PHY_module instance
     * @return void pointer (unused)
     */
    static void* sfn_slot_sync_cmd_thread_func(void* arg);

    /**
     * @brief Receive a message from transport
     * @return true if message was received successfully, false otherwise
     */
    bool recv_msg();
    /**
     * @brief Drain the transport receive queue, classify each message, and
     *        place it in the appropriate storage (slot or non-slot).
     * @note  Replaces recv_msg() when ENABLE_FAPI_STORE_REPLAY=ON.
     *        Triggers EOM-complete and per-slot task dispatch internally.
     * @thread_safety Called exclusively from the msg-processing thread.
     */
    void process_fapi_messages();
    /**
     * @brief Path B ring-buffer-aware ERROR.ind recovery.
     *
     * Resets ring slot state for @p ss_msg, releases all stored FAPI payload
     * messages for that slot, resets the slot boundary (targeting ring_idx(N)
     * before ss_curr advances), then advances ss_curr to N+2 so that any
     * stale payload messages arriving after the reset are dropped by the
     * lag/lead window check in store_slot_message().
     *
     * @note Must be called BEFORE set_curr_sfn_slot() — see ordering comment
     *       in implementation.
     * @param cell_id Cell that sent the ERROR.ind.
     * @param ss_msg  SFN/slot of the errored slot.
     * @thread_safety Called from the msg-processing thread only.
     */
    void handle_error_ind_path_b(std::size_t cell_id, sfn_slot_t ss_msg);
    /**
     * @brief Store a non-slot FAPI message (e.g. CONFIG.request, START.request)
     *        in per-cell non-slot storage for deferred processing.
     * @param smsg   Message descriptor received from the transport layer.
     * @param ss_msg Slot identifier associated with the message; expected to
     *               equal SFN_SLOT_INVALID for non-slot messages.
     * @return true  if the message was accepted and stored successfully;
     *         false if the cell index is out of range or the message type is
     *               not recognised as a non-slot message.
     * @thread_safety Called from the msg-processing thread only.
     */
    [[nodiscard]] bool store_non_slot_message(phy_mac_msg_desc& smsg, sfn_slot_t ss_msg);
    /**
     * @brief Store a slot-associated FAPI message (e.g. DL_TTI.request,
     *        UL_TTI.request) in the ring-buffer slot storage for task handling
     *        after EOM-complete.
     * @param smsg   Message descriptor received from the transport layer.
     * @param ss_msg Packed SFN/slot value identifying the target slot entry.
     * @return true  if the message was stored in the correct slot ring entry;
     *         false if the slot entry is not ready to accept the message
     *               (e.g. ring conflict or storage full).
     * @thread_safety Called from the msg-processing thread only.
     */
    [[nodiscard]] bool store_slot_message(phy_mac_msg_desc& smsg, sfn_slot_t ss_msg);
    /**
     * @brief Dispatch and clear stored non-slot messages for one cell.
     * @param cell_id Cell identifier whose non-slot storage is consumed.
     * @note Dispatches via `on_msg()` and clears storage after this pass to
     * avoid duplicate handling in subsequent iterations.
     * @thread_safety Called from the msg-processing thread.
     */
    void process_pending_non_slot_messages(uint16_t cell_id);

#ifdef ENABLE_FAPI_STORE_REPLAY
    /**
     * @brief Transfer a CONFIG/START/STOP descriptor to the low-priority non-slot worker.
     * @param smsg RX descriptor owned by the caller on entry.
     * @return true when ownership is transferred; false when caller still owns @p smsg.
     * @note Called from the message-processing thread's dispatch path for the parallel
     *       non-slot messages (CONFIG/START/STOP).
     */
    [[nodiscard]] bool enqueue_nonslot_lp_work(phy_mac_msg_desc& smsg);
    /**
     * @brief Start the low-priority non-slot worker.
     * @note Called from @c start() only when @c ENABLE_FAPI_STORE_REPLAY is enabled.
     */
    void start_nonslot_lp_worker();
    /**
     * @brief Stop the low-priority non-slot worker and release queued descriptors.
     * @note Called after msg-processing quiesces, and defensively from the destructor.
     */
    void stop_nonslot_lp_worker();
    /**
     * @brief Recreate the non-slot executor after @c PHY_module move construction.
     * @param other Source module that still owns the pre-move executor configuration.
     * @note The executor is intentionally non-movable, so the moved-to module
     *       constructs a fresh empty executor bound to itself.
     */
    void move_nonslot_lp_worker_from(PHY_module& other);
    /**
     * @brief Parse and install low-priority non-slot worker configuration.
     * @param node_config L2 adapter YAML node containing message/timer/nonslot thread config.
     */
    void init_nonslot_lp_thread_config(yaml::node node_config);
    /**
     * @brief Publish staged active-cell state for future slots.
     */
    void commit_staged_active_cell_bitmap() noexcept;
    /**
     * @brief Snapshot committed active-cell state into the ring slot being armed.
     * @param slot_u32 Packed SFN/slot for the ring entry being prepared.
     */
    void snapshot_active_cell_state(uint32_t slot_u32) noexcept;
#endif
    /**
     * @brief Process and clear all slot-message ring entries marked ready.
     * @note Iterates the slot-message ring and consumes only entries whose
     *       `ready()` flag is set.  Dispatches each stored descriptor to
     *       `on_msg()`, releases the NVIPC buffer when ownership is returned,
     *       and clears the ring entry metadata for reuse.
     * @thread_safety Called from the msg-processing thread only.
     */
    /**
     * @brief Execute slot-boundary side effects common to SLOT.ind and
     *        EOM-complete events.
     * @param slot_end_rcvd true when triggered by an EOM-complete event
     *                      (calls process_phy_commands() with true and resets
     *                      the EOM bitmap and slot-state flags);
     *                      false when triggered by a SLOT.indication.
     * @note Resets fapi_eom_rcvd_bitmap, tti_event_count, and the
     *       new_slot/is_ul_slot/is_dl_slot/is_csirs_slot flags.
     * @thread_safety Called from the msg-processing thread only.
     */
    void run_slot_boundary_reset(bool slot_end_rcvd);
    /**
     * @brief Dispatch a message that bypasses slot storage entirely.
     * @param msg Message descriptor received from the transport layer;
     *            identified by is_immediate_non_slot_msg() before this call.
     * @note Adjusts tti_event_count in non-per-slot sync mode, validates
     *       cell_id, invokes on_msg(), and releases the NVIPC buffer when
     *       the PHY does not retain ownership.
     * @thread_safety Called from the msg-processing thread only.
     */
    void process_immediate_non_slot_msg(phy_mac_msg_desc& msg);

    /**
     * @brief Check if time threshold has been exceeded
     * @param time_ns Time in nanoseconds
     * @param slot_num Slot number
     * @param is_ul true for uplink, false for downlink
     * @return true if threshold exceeded, false otherwise
     */
    bool check_time_threshold(std::chrono::nanoseconds, uint16_t, bool);

    // =========================================================================
    // nvphy FAPI tasks — shared
    // =========================================================================

    /**
     * @brief Guard the ring slot at the start of each slot.
     * Checks @c task_pool_.tasks_in_flight[ring_idx] and stores
     * @c SlotTaskPool::SLOT_TASK_SENTINEL to arm the counter when the ring slot is idle.
     * A short burst of live-task ring reuse is dropped with warnings; persistent reuse
     * remains fatal to avoid hiding a stuck task.
     * @param slot_u32 Packed SFN/slot for this slot.
     */
    void reset_for_slot(uint32_t slot_u32);

    /**
     * @brief Completion callback for C-plane batch tasks (all-tasks counter only).
     *
     * Decrements @c task_pool_.tasks_in_flight[ring_idx]. When it reaches 0 every task
     * (channel-aggr + C-plane) is done, so @c finalize_slot_cleanup(ring_idx) releases the
     * stored FAPI messages. Never touches @c channel_tasks_in_flight — publish is driven by
     * the channel path (@c on_slot_channel_aggr_complete).
     *
     * @param ring_idx Ring slot index.
     * @param slot_u32 Packed SFN/slot for message release.
     */
    void on_slot_channel_task_complete(uint32_t ring_idx, uint32_t slot_u32);

    /**
     * @brief Completion callback for channel-aggr tasks (two-counter path).
     *
     * Decrements BOTH @c channel_tasks_in_flight[ring_idx] and @c tasks_in_flight[ring_idx].
     * When the channel counter reaches 0 (all channel-aggr tasks done, independent of
     * C-plane) it calls @c publish_slot_command(ring_idx). When the all-tasks counter reaches
     * 0 it calls @c finalize_slot_cleanup(ring_idx). C-plane batches use
     * @c on_slot_channel_task_complete instead and never touch the channel counter.
     *
     * @param ring_idx Ring slot index.
     * @param slot_u32 Packed SFN/slot for message release.
     */
    void on_slot_channel_aggr_complete(uint32_t ring_idx, uint32_t slot_u32);

    /**
     * @brief Publish the U-plane slot command to cuphydriver (fires at channel-done).
     *
     * The l1_enqueue_phy_work + telemetry half of the slot publish.
     * Safe to run while a C-plane batch is still live: the redundant SlotRefTs write was
     * removed from the direct-mode enqueue, so this no longer races the C-plane send.
     * Handles the empty-slot (channel_array_size == 0) case itself.
     *
     * @param ring_idx Ring slot index used to look up slot_cmd_idx and slot_u32.
     */
    void publish_slot_command(uint32_t ring_idx);

    /**
     * @brief Per-slot cleanup, deferred to all-tasks-done (channel + C-plane).
     *
     * Per-PHY reset_slot(false) + release_stored_slot_messages. Runs only after every
     * task (incl. C-plane, the last stored-FAPI reader) has finished.
     *
     * @param ring_idx Ring slot index.
     * @param slot_u32 Packed SFN/slot for message release.
     */
    void finalize_slot_cleanup(uint32_t ring_idx, uint32_t slot_u32);

#ifdef ENABLE_FAPI_STORE_REPLAY
    /**
     * @brief Capture EOM-derived fields into @c slot_telemetry_[ring_idx] and
     *        stamp @c slot_command_array[slot_cmd_idx].tick_original. Msg
     *        thread only.
     * @param[in] ring_idx     Per-ring snapshot index to populate.
     * @param[in] ss_curr      SFN/slot reaching EOM.
     * @param[in] slot_cmd_idx Slot-command-array index for this slot.
     */
    void stash_slot_telemetry_for_worker_publish(uint32_t   ring_idx,
                                                 sfn_slot_t ss_curr,
                                                 uint32_t   slot_cmd_idx);

    /**
     * @brief Per-ring telemetry stamp for one payload msg at ingestion.
     *
     * Sets slot-type flags per @p msg_id and @c is_csirs_slot from @p has_csirs
     * (caller reads the store-time sidecar — no payload walk here).
     *
     * @param[in] ss_msg    SFN/slot of the message.
     * @param[in] msg_id    FAPI msg_id.
     * @param[in] has_csirs True when this DL_TTI carries CSI-RS (sidecar).
     */
    void stamp_ring_telemetry_on_ingest(sfn_slot_t ss_msg,
                                        int32_t    msg_id,
                                        bool       has_csirs = false);

    /**
     * @brief Rebind a ring entry to @p slot_u32, resetting the storage binding and
     *        the coupled per-ring telemetry snapshot together in one owner so the
     *        two cannot desync.
     * @param[in] slot_u32  Absolute packed slot (upper 16 bits SFN, lower 16 slot).
     * @return The storage bind outcome; the telemetry snapshot is cleared on
     *         FreshBind / Rebound (i.e. any outcome other than AlreadyBound).
     */
    [[nodiscard]] FapiSlotMessageStorage::BindOutcome bind_ring_slot(uint32_t slot_u32);
#endif

    /**
     * @brief Release SRS.IND nvIPC buffers owned by a slot command.
     *
     * Used on the task-framework submit failure path after SRS parser state has
     * been registered in the slot command but before ownership can transfer to
     * cuphydriver.
     *
     * @param srs_params SRS params from the slot command.
     * @param slot       Slot indication for diagnostics.
     */
    void release_srs_indication_buffers(slot_command_api::srs_params& srs_params,
                                        const slot_command_api::slot_indication& slot);

    /**
     * @brief Handle a slot with no channel tasks (n_dl_tasks + n_ul_tasks == 0 at EOM).
     * Drops the sentinel, resets PHY instances without submitting a driver command.
     * @param slot_u32 Packed SFN/slot.
     */
    void submit_slot_command_empty(uint32_t slot_u32);

    /**
     * @brief EOM scan — build active channel masks for DL and UL, then enqueue all
     * channel aggr tasks under a single combined sentinel drop.
     * Called after global EOM (all cells reported SLOT.response).
     * Adds n_dl_tasks + n_ul_tasks to @c task_pool_.tasks_in_flight[ring_idx]
     * then drops the sentinel.
     * @param ss_curr Current slot at EOM.
     */
    void enqueue_channel_tasks(sfn_slot_t ss_curr);

    /**
     * @brief Drop a direct C-plane slot whose FAPI input arrived too late to launch.
     *
     * Startup warm-up slots and slots whose @p slot_interval exceeds
     * @c kMaxDirectCplaneStaleSlotInterval are dropped and the ring is fully reset.
     * After @c kMaxDirectCplaneConsecutiveStaleDrops consecutive non-startup drops the
     * ring is left untouched and a fatal logged. Must be called before any C-plane batch
     * is fired for the slot so zeroing the counters cannot race a task completion.
     * @param ss_curr Packed SFN/slot reaching EOM.
     * @param ring_idx Ring index for @p ss_curr.
     * @param slot_interval FAPI latency (in slots) snapshotted under tick_lock.
     * @param ss_tick_snapshot SFN/slot tick snapshot, diagnostics only.
     * @param curr_tick_snapshot current_tick_ snapshot, diagnostics only.
     * @return @c true if the slot was dropped and the caller must return; @c false to proceed.
     */
    [[nodiscard]] bool try_drop_stale_direct_cplane_slot(sfn_slot_t               ss_curr,
                                                         uint32_t                 ring_idx,
                                                         uint32_t                 slot_interval,
                                                         sfn_slot_t               ss_tick_snapshot,
                                                         std::chrono::nanoseconds curr_tick_snapshot);

    /**
     * @brief Slot-boundary backstop — force-complete a prior slot whose active-cell EOM
     * never fired (mirrors the SLOT.ind branch of the legacy recv_msg loop).
     *
     * Invoked on every SCF_FAPI_SLOT_INDICATION for the slot whose window just closed. If
     * that slot still holds a pure @c SLOT_TASK_SENTINEL (armed but no EOM), whatever was
     * accumulated is enqueued (or replayed/released when the store was dropped). A slot the
     * active-cell EOM already fired has had its sentinel replaced by the real task count and
     * is skipped, so this never double-fires. This guarantees every armed slot is released
     * within one slot boundary, independent of the active-cell snapshot being complete.
     * @param prior_slot_u32 Packed SFN/slot of the slot whose window just closed.
     */
    void finalize_slot_on_boundary(uint32_t prior_slot_u32);

    // =========================================================================
    // DL task framework
    // =========================================================================

    /**
     * @brief Accumulate a per-cell DL C-plane entry for EOM batch scheduling.
     *
     * Replaces the former one-task-per-cell enqueue_dlc_task_for_cell().
     * Fills dlc_task_args[cell_id] and appends cell_id to the pending batch for
     * this ring slot.  EOM planning in enqueue_channel_tasks() partitions the
     * accumulated cells into one or more DL batches using
     * config_options_.cplane_processing_dl_batch_size.
     *
     * tasks_in_flight[ring_idx] is incremented by 1 per batch (not per cell).
     *
     * @param cell_id   Cell that sent SLOT.response.
     * @param ss_curr   Current slot.
     */
    void accumulate_dlc_for_cell(uint32_t cell_id, sfn_slot_t ss_curr);

    /**
     * @brief Framework iteration layer for PDSCH across all cells.
     *
     * Retrieves the slot's DL_TTI lane from slot_message_storage_[ctx.ring_idx],
     * iterates all stored DL_TTI.req messages, and filters for
     * DL_TTI_PDU_TYPE_PDSCH PDUs. Invoked from a worker thread via DL_DISPATCH_TABLE.
     *
     * @todo For each PDSCH PDU, call the per-cell PDSCH update function and
     *       write into slot_command_array[ctx.slot_cmd_idx] directly.
     * @param ctx  Read-only shared context captured at EOM.
     */
    void process_aggr_pdsch_channel(const DlAggrTaskArg& ctx);

    /**
     * @brief Parallel DL aggregation handler for CSI-RS across all cells.
     *
     * Walks stored DL_TTI messages, precomputes PDSCH ordinals via
     * FapiSlotMessageStorage::dl_tti_has_pdsch_pdu for co-scheduling,
     * and calls scf_5g_fapi::apply_csirs_dl_aggr_slot_command_for_pdu
     * for every CSI-RS PDU.
     *
     * @note PDSCH co-scheduling uses dl_tti_has_pdsch_pdu() to determine
     *       whether the CSI-RS mirrors into PDSCH RrcDynPrms.
     * @param ctx  Read-only shared context captured at EOM.
     */
    void process_aggr_csirs_channel(const DlAggrTaskArg& ctx);

    /**
     * @brief Framework iteration layer for PDCCH across all cells.
     * @todo For each DL_TTI PDCCH PDU and each UL_DCI message, call
     *       the per-cell PDCCH update function.
     * @param ctx  Read-only shared context captured at EOM.
     */
    void process_aggr_pdcch_channel(const DlAggrTaskArg& ctx);

    /**
     * @brief Runs the DL_TTI PDCCH aggregation pass for one slot.
     *
     * Extracted from process_aggr_pdcch_channel so the caller's null/empty
     * guards stay flat; uses an early return for the "no PDCCH registration"
     * case instead of an extra nesting level.
     *
     * @param ctx   Read-only shared context captured at EOM.
     * @param msgs  Non-null DL_TTI message array (caller guarantees msgs != nullptr).
     * @param n     Number of DL_TTI messages (caller guarantees n != 0).
     */
    void process_aggr_pdcch_dl_tti(const DlAggrTaskArg& ctx,
                                   const phy_mac_msg_desc* msgs, uint16_t n);

    /**
     * @brief Framework iteration layer for SSB across all cells.
     * @todo For each DL_TTI_PDU_TYPE_SSB PDU, populate pbch_group_params
     *       in cell_group_command.
     * @param ctx  Read-only shared context captured at EOM.
     */
    void process_aggr_ssb_channel(const DlAggrTaskArg& ctx);

    /**
     * @brief Framework iteration layer for DL beamforming weights across all cells.
     * @todo For each stored DL_BFW_CVI_REQUEST, call
     *       PHY_instances()[cell_id].get().on_dl_bfw_request().
     * @param ctx  Read-only shared context captured at EOM.
     */
    void process_aggr_dlbfw_channel(const DlAggrTaskArg& ctx);

    /**
     * @brief Phase 1 — stage one cell's TX_DATA.req descriptor for batched H2D copy.
     *
     * CPU-only. Called on the msg thread per TX_DATA.req arrival (before EOM).
     * Accumulates copy descriptors into the per-slot batched memcpy helper.
     * Phase 2 (launch_tx_data_h2d) is fired at EOM from enqueue_channel_tasks()
     * after all TX_DATA.req messages have been staged — this eliminates any
     * TX_DATA.req / DL_TTI.req arrival-order dependency.
     *
     * @param cell_id   Physical cell ID of the arriving TX_DATA.req.
     * @param smsg      Message descriptor; msg_buf and data_buf fields are valid.
     * @param ring_idx  Ring slot index (slot % SLOT_STORAGE_DEPTH).
     * @param ss_msg    Packed SFN/slot of the message (already computed at call site).
     */
    void stage_tx_data_h2d(uint32_t cell_id, const phy_mac_msg_desc& smsg, uint32_t ring_idx, sfn_slot_t ss_msg);

    /**
     * @brief Phase 2 — validate staged TX_DATA count against DL_TTI PDSCH cells
     *        and launch the batched H2D copy for this slot (non-blocking CUDA).
     *
     * Called from enqueue_channel_tasks() at EOM. Validates that the number of
     * staged TX_DATA cells matches the PDSCH cell count from DL_TTI, then calls
     * l1_launch_tb_h2d() to fire the batched DMA. On mismatch or launch failure,
     * clears split-phase state so workers skip PDSCH (pTbInput null guard).
     *
     * @param ss_curr   Current slot SFN/slot.
     * @param ring_idx  Ring slot index for per-slot event selection.
     */
    void launch_tx_data_h2d(sfn_slot_t ss_curr, uint32_t ring_idx);

    /**
     * @brief Clear per-ring TX_DATA staging when the slot store is released.
     *
     * @param[in] ring_idx  Ring slot index in [0, SLOT_STORAGE_DEPTH). No-op when
     *                      @p ring_idx is out of range.
     */
    void reset_txdata_h2d_state_for_ring(std::size_t ring_idx);

    /**
     * Reset the driver's shared batched-memcpy accumulator, unless a sibling
     * ring slot holds staged-but-unlaunched TX_DATA — a global reset would
     * cancel the sibling's pending copies and its PDSCH would transmit stale
     * GPU TB data. When skipped, this ring's orphaned entries are flushed
     * harmlessly by the sibling's launch (copies land in per-slot buffers).
     *
     * @param[in] ring_idx  Ring slot being dropped/cleared; excluded from the
     *                      sibling staged-count check.
     */
    void reset_batched_memcpy_unless_sibling_staged(uint32_t ring_idx);

    /**
     * @brief Clear the per-cell DLC in-flight guard for a given cell.
     *
     * Called from task_work_fn_dlc via the s_clear shim before
     * on_slot_channel_task_complete() — allows the next slot's SLOT.response to
     * re-enqueue task_dlc for this cell immediately.
     *
     * @param cell_id Cell whose guard flag to clear.
     */
    // -----------------------------------------------------------------------
    // DL static shims: bridge instance methods to raw function-pointer signatures
    // required by DlChannelDispatch / DlcTaskArg. Assigned into DL_DISPATCH_TABLE.
    // -----------------------------------------------------------------------
    static void s_pdsch(PHY_module* s, const DlAggrTaskArg& c) { s->process_aggr_pdsch_channel(c); }
    static void s_csirs(PHY_module* s, const DlAggrTaskArg& c) { s->process_aggr_csirs_channel(c); }
    static void s_pdcch(PHY_module* s, const DlAggrTaskArg& c) { s->process_aggr_pdcch_channel(c); }
    static void s_ssb(PHY_module* s, const DlAggrTaskArg& c) { s->process_aggr_ssb_channel(c); }
    static void s_dlbfw(PHY_module* s, const DlAggrTaskArg& c) { s->process_aggr_dlbfw_channel(c); }
    static void s_done(PHY_module* s, uint32_t r, uint32_t sl) { s->on_slot_channel_task_complete(r, sl); }
    // Channel-aggr completion trampoline (dispatch-table 'done'): decrements BOTH the
    // channel-only and all-tasks counters. C-plane batches must NOT use this (they keep s_done).
    static void s_aggr_done(PHY_module* s, uint32_t r, uint32_t sl) { s->on_slot_channel_aggr_complete(r, sl); }

    /// Batch DLC shim: build DL C-plane for one cell.
    /// Storage is ring-indexed (ring_idx * MAX_CELLS_PER_SLOT + cell_id) so no
    /// per-cell in-flight guard is needed — cross-slot aliasing is impossible.
    /// Returns the int from build_and_send_dl_cplane so the dispatcher can
    /// accumulate failing cells for batch-end L2 fan-out.
    [[nodiscard]] static int s_process_dlc_cell(PHY_module* s,
                                               uint32_t ring_idx,
                                               uint32_t cid,
                                               std::size_t transaction_id)
    {
        auto& cell = s->task_pool_.dlc_task_args[ring_idx * MAX_CELLS_PER_SLOT + cid];
        return cell.phy_inst->build_and_send_dl_cplane(
            cell.dl_tti, cell.ul_dci, transaction_id, cell.slot_map_dl);
    }

    // -----------------------------------------------------------------------
    // Batched C-plane policy structs (private, used by fire_cplane_batch)
    // -----------------------------------------------------------------------
    struct DlcBatchPolicy
    {
        static nv::cplane_process_cell_fn_t process_cell_fn() { return s_process_dlc_cell; }
        static nv::post_batch_fn_t post_batch_fn() { return &l1_signal_dl_cplane_batch_done; }
        static nv::cplane_on_batch_error_fn_t on_batch_error_fn() { return &l1_handle_dl_cplane_batch_error; }
        static constexpr auto on_complete_fn() { return &PHY_module::s_done; }
        static void* get_slot_map(PHY_module* m, uint32_t ring_idx, uint8_t cell_id)
        {
            return m->task_pool_.dlc_task_args[ring_idx * MAX_CELLS_PER_SLOT + cell_id].slot_map_dl;
        }
        static void on_drop(PHY_module* m, uint32_t ring_idx, uint32_t)
        {
            m->task_pool_.tasks_in_flight[ring_idx].fetch_sub(1, std::memory_order_acq_rel);
        }
        [[nodiscard]] static bool push_task(uint64_t ts, l1_task_work_fn_t fn, void* arg);
    };

    struct UlcBatchPolicy
    {
        static nv::cplane_process_cell_fn_t process_cell_fn() { return su_process_ulc_cell; }
        static nv::post_batch_fn_t post_batch_fn() { return &l1_signal_ul_cplane_batch_done; }
        static nv::cplane_on_batch_error_fn_t on_batch_error_fn() { return &l1_handle_ul_cplane_batch_error; }
        static constexpr auto on_complete_fn() { return &PHY_module::s_done; }
        static void* get_slot_map(PHY_module* m, uint32_t ring_idx, uint8_t cell_id)
        {
            return m->task_pool_.ulc_task_args[ring_idx * MAX_CELLS_PER_SLOT + cell_id].slot_map_ul;
        }
        static void on_drop(PHY_module* m, uint32_t ring_idx, uint32_t /*slot_u32*/)
        {
            m->task_pool_.tasks_in_flight[ring_idx].fetch_sub(1, std::memory_order_acq_rel);
        }
        [[nodiscard]] static bool push_task(uint64_t ts, l1_task_work_fn_t fn, void* arg);
    };

    template <CplaneBatchPolicy Policy>
    void fire_cplane_batch(
        uint32_t                          ring_idx,
        sfn_slot_t                        ss_curr,
        std::span<nv::CplaneBatchAccum>   accums,
        std::span<nv::CplaneBatchTaskArg> batch_pool,
        uint8_t local_batch_idx,
        uint8_t n_cells_in_batch,
        uint8_t batch_id,
        uint8_t total_batches,
        uint64_t task_ts_ns);

    /**
     * @brief Singleton DL dispatch table passed to DlAggrTaskArg at EOM enqueue.
     *
     * Worker threads call through this table to invoke private PHY_module methods
     * without requiring public access. Zero runtime cost — static const data.
     */
    static constexpr DlChannelDispatch DL_DISPATCH_TABLE = {
        s_pdsch, s_csirs, s_pdcch, s_ssb, s_dlbfw, s_aggr_done};

    // =========================================================================
    // UL task framework
    // =========================================================================

    /**
     * @brief Accumulate a per-cell UL C-plane entry for EOM batch scheduling.
     *
     * Replaces the former one-task-per-cell enqueue_ulc_task_for_cell().
     * Mirrors accumulate_dlc_for_cell() for the UL path: fills ulc_task_args[cell_id]
     * and appends cell_id to the pending ULC batch. EOM planning in
     * enqueue_channel_tasks() partitions accumulated cells into one or more
     * UL batches using config_options_.cplane_processing_ul_batch_size.
     *
     * tasks_in_flight[ring_idx] is incremented by 1 per batch (not per cell).
     *
     * @param cell_id   Cell that sent UL_TTI.req.
     * @param ss_curr   Current slot.
     */
    void accumulate_ulc_for_cell(uint32_t cell_id, sfn_slot_t ss_curr);

    /**
     * @brief Framework iteration layer for PUSCH across all cells.
     *
     * Retrieves the slot's UL_TTI lane from slot_message_storage_[ctx.ring_idx],
     * iterates all stored UL_TTI.req messages, and filters for
     * UL_TTI_PDU_TYPE_PUSCH PDUs. Invoked from a worker thread via UL_DISPATCH_TABLE.
     *
     * @todo For each PUSCH PDU, update slot_command_array[ctx.slot_cmd_idx] directly.
     * @param ctx  Read-only shared context captured at EOM.
     */
    void process_aggr_pusch_channel(const UlAggrTaskArg& ctx);

    /**
     * @brief Framework iteration layer for PRACH across all cells.
     *
     * @todo For each UL_TTI_PDU_TYPE_PRACH PDU, call the per-cell PRACH
     *       update function and write into slot_command_array[ctx.slot_cmd_idx].
     * @param ctx  Read-only shared context captured at EOM.
     */
    void process_aggr_prach_channel(const UlAggrTaskArg& ctx);

    /**
     * @brief Framework iteration layer for PUCCH across all cells.
     *
     * Handles all PUCCH PDU types: format 0/1 (UL_TTI_NPDUS_IDX_PUCCH_F01) and
     * format 2/3/4 (UL_TTI_NPDUS_IDX_PUCCH_F234). A cell is skipped only when
     * both counts are zero.
     *
     * @todo For each PUCCH PDU, call the per-cell PUCCH update function.
     * @param ctx  Read-only shared context captured at EOM.
     */
    void process_aggr_pucch_channel(const UlAggrTaskArg& ctx);

    /**
     * @brief Framework iteration layer for SRS across all cells.
     *
     * @todo For each UL_TTI_PDU_TYPE_SRS PDU, call the per-cell SRS update function.
     * @param ctx  Read-only shared context captured at EOM.
     */
    void process_aggr_srs_channel(const UlAggrTaskArg& ctx);

    /**
     * @brief Framework iteration layer for UL beamforming weights across all cells.
     *
     * Retrieves the slot's UL_BFW lane from slot_message_storage_[ctx.ring_idx]
     * and iterates all stored UL_BFW_CVI_REQUEST messages.
     * Invoked from a worker thread via UL_DISPATCH_TABLE.
     *
     * For each UL_BFW_CVI_REQUEST, applies the stored PDUs through
     * PHY_instance::apply_ul_bfw_pdu().
     * @param ctx  Read-only shared context captured at EOM.
     */
    void process_aggr_ulbfw_channel(const UlAggrTaskArg& ctx);

    /**
     * @brief Clear the per-cell ULC in-flight guard for a given cell.
     *
     * Called from task_work_fn_ulc via the su_clear shim before
     * on_slot_channel_task_complete() — allows the next slot's UL_TTI.req to
     * re-enqueue task_ulc for this cell immediately.
     *
     * @param cell_id Cell whose guard flag to clear.
     */
    // -----------------------------------------------------------------------
    // UL static shims: bridge instance methods to raw function-pointer signatures
    // required by UlChannelDispatch. Assigned into UL_DISPATCH_TABLE.
    // -----------------------------------------------------------------------
    static void su_pusch(PHY_module* s, const UlAggrTaskArg& c) { s->process_aggr_pusch_channel(c); }
    static void su_prach(PHY_module* s, const UlAggrTaskArg& c) { s->process_aggr_prach_channel(c); }
    static void su_pucch(PHY_module* s, const UlAggrTaskArg& c) { s->process_aggr_pucch_channel(c); }
    static void su_srs(PHY_module* s, const UlAggrTaskArg& c) { s->process_aggr_srs_channel(c); }
    static void su_ulbfw(PHY_module* s, const UlAggrTaskArg& c) { s->process_aggr_ulbfw_channel(c); }
    static void su_done(PHY_module* s, uint32_t r, uint32_t sl) { s->on_slot_channel_task_complete(r, sl); }
    // UL channel-aggr completion trampoline: same two-counter semantics as s_aggr_done.
    static void su_aggr_done(PHY_module* s, uint32_t r, uint32_t sl) { s->on_slot_channel_aggr_complete(r, sl); }

    /// Batch ULC shim: build UL C-plane for one cell.
    /// Storage is ring-indexed (ring_idx * MAX_CELLS_PER_SLOT + cell_id).
    /// Returns the int from build_and_send_ul_cplane so the dispatcher can
    /// accumulate failing cells for batch-end L2 fan-out.
    [[nodiscard]] static int su_process_ulc_cell(PHY_module* s,
                                                uint32_t ring_idx,
                                                uint32_t cid,
                                                std::size_t transaction_id)
    {
        auto& cell = s->task_pool_.ulc_task_args[ring_idx * MAX_CELLS_PER_SLOT + cid];
        return cell.phy_inst->build_and_send_ul_cplane(
            cell.ul_tti, transaction_id, cell.slot_map_ul);
    }

    /**
     * @brief Singleton UL dispatch table passed to UlAggrTaskArg at EOM enqueue.
     *
     * Mirrors DL_DISPATCH_TABLE for the UL path. Zero runtime cost — static const data.
     */
    static constexpr UlChannelDispatch UL_DISPATCH_TABLE = {
        su_pusch, su_prach, su_pucch, su_srs, su_ulbfw, su_aggr_done};

public:
    /**
     * @brief True when the FAPI store-replay path is active (ENABLE_FAPI_STORE_REPLAY).
     *
     * TX_DATA nvipc buffer lifetime is managed by FapiSlotMessageStorage and the
     * tx_data_release_fn callback, not by dl_tbs_queue_.
     *
     * @return  true when ENABLE_FAPI_STORE_REPLAY is defined (split-phase path active);
     *          false otherwise (legacy single-phase path).
     *          Return value must be checked.
     */
    [[nodiscard]] bool use_split_phase_txdata_h2d() noexcept;

    /**
     * @brief Callback shim for cuphydriver's dl_slot_callbacks::tx_data_release_fn.
     *
     * Casts the opaque void* context back to @c PHY_module* and delegates to
     * @c release_deferred_tx_data().
     *
     * @param[in] ctx   Opaque context registered with cuphydriver — must be a
     *                  valid @c PHY_module*.
     * @param[in] sfn   System frame number of the completed DL slot.
     * @param[in] slot  Slot number of the completed DL slot.
     */
    static void s_tx_data_release(void* ctx, uint16_t sfn, uint8_t slot);

    /**
     * @brief Release deferred TX_DATA lane for a given SFN/slot.
     *
     * Called from cuphydriver's task_work_function_dl_aggr_1_pdsch via the
     * tx_data_release_fn callback after PhyPdschAggr::setup() has waited on the
     * H2D copy event.  Idempotent: no-op when the flag is already false.
     *
     * @param[in] sfn   System frame number.
     * @param[in] slot  Slot number.
     */
    void release_deferred_tx_data(uint16_t sfn, uint8_t slot);

    /**
     * @brief Process PHY commands for current slot
     * @param force_process Force processing even if conditions not met
     */
    void process_phy_commands(bool);
    /**
     * @brief Release all buffered slot messages for the ring entry mapped by slot.
     * @param slot_u32 Absolute slot encoded as SFN/slot u32.
     * @note Slot-to-ring mapping uses modulo depth; this releases the mapped entry.
     */
    void release_stored_slot_messages(uint32_t slot_u32);
    /**
     * @brief Release all buffered non-slot messages for one cell.
     * @param cell_id Cell identifier whose non-slot buffers are released.
     * @note Logs and returns when the cell id is out of range.
     */
    /**
     * @brief Callback type invoked when a slot becomes ready for processing.
     *
     * The callable receives the absolute slot index (SFN * slots-per-frame +
     * slot-within-frame) that is now ready.  The callback is fired on the
     * msg-processing thread; implementations must not block or perform
     * long-running work inside the callback.
     *
     * @param slot_index Absolute slot index of the slot that became ready.
     */
    using SlotReadyCallback = std::function<void(uint32_t)>;
    /**
     * @brief Register a callback to be invoked when a slot becomes ready.
     *
     * The provided @p callback is move-constructed into the internal
     * `slot_ready_callback_` member.  Any previously registered callback is
     * replaced.  Pass an empty `SlotReadyCallback{}` to clear the callback.
     *
     * @param callback Callable to register; ownership is transferred via move.
     *
     * @note Must be called before the msg-processing thread starts (i.e. before
     *       PHY_module::start()) to avoid a data race on `slot_ready_callback_`.
     */
    void set_slot_ready_callback(SlotReadyCallback callback) { slot_ready_callback_ = std::move(callback); }

    /**
     * @brief Callback when DL transport block has been processed
     */
    void on_dl_tb_processed();

    /**
     * @brief Callback when DL transport block has been processed
     * @param params Pointer to PDSCH parameters
     */
    void on_dl_tb_processed(const slot_command_api::pdsch_params* params);

    /**
     * @brief Callback when DL TTI has been processed (UNUSED)
     */
    void on_dl_tti_processed();

    /**
     * @brief Callback when DL TTI has been processed (UNUSED)
     * @param num_dl_tti Number of DL TTI messages processed
     */
    void on_dl_tti_processed(int num_dl_tti);

    /**
     * @brief Handle received timing tick
     * @param tick_time Tick timestamp in nanoseconds
     */
    void tick_received(std::chrono::nanoseconds&);

    /**
     * @brief Set the TTI flag
     * @param flag TTI flag value
     */
    void set_tti_flag(bool flag);

    /**
     * @brief Stop the tick generator
     */
    void stop_tick_generator();

    /**
     * @brief Set whether all cells are configured
     * @param value true if all cells configured, false otherwise
     */
    void set_all_cells_configured(bool value) { all_cells_configured = value; }

    /**
     * @brief Send callbacks to registered handlers
     */
    void send_call_backs();

    /**
     * @brief Get the highest numerology (mu) configured
     * @return Highest mu value
     */
    uint32_t get_mu_highest() { return tick_updater_.mu_highest_; }

    /**
     * @brief Get the slot advance value
     * @return Number of slots to advance
     */
    uint32_t get_slot_advance() { return tick_updater_.slot_advance_; }

    /**
     * @brief Get previous tick slot indication
     * @return Reference to previous slot indication
     */
    slot_command_api::slot_indication& get_prev_tick() { return tick_updater_.prev_slot_info_; }

    /**
     * @brief Get downlink transport block location type
     * @return DL TB location (inline, host buffer, or GPU buffer)
     */
    dl_tb_loc dl_tb_location() { return dl_tb_location_; }

    /**
     * @brief Check if dynamic SFN/slot tick is enabled
     * @return 1 if enabled, 0 otherwise
     */
    int tickDynamicSfnSlotIsEnabled() { return config_options_.enableTickDynamicSfnSlot; }

    /**
     * @brief Get static PUSCH slot number for testing
     * @return PUSCH slot number
     */
    int staticPuschSlotNum() { return config_options_.staticPuschSlotNum; }

    /**
     * @brief Get static PDSCH slot number for testing
     * @return PDSCH slot number
     */
    int staticPdschSlotNum() { return config_options_.staticPdschSlotNum; }

    /**
     * @brief Get static PDCCH slot number for testing
     * @return PDCCH slot number
     */
    int staticPdcchSlotNum() { return config_options_.staticPdcchSlotNum; }

    /**
     * @brief Get static CSI-RS slot number for testing
     * @return CSI-RS slot number
     */
    int staticCsiRsSlotNum() { return config_options_.staticCsiRsSlotNum; }

    /**
     * @brief Get static SSB physical cell ID for testing
     * @return SSB PCID
     */
    int staticSsbPcid() { return config_options_.staticSsbPcid; }

    /**
     * @brief Get static SSB SFN for testing
     * @return SSB SFN
     */
    int staticSsbSFN() { return config_options_.staticSsbSFN; }

    /**
     * @brief Get static SSB slot number for testing
     * @return SSB slot number
     */
    int staticSsbSlotNum() { return config_options_.staticSsbSlotNum; }

    /**
     * @brief Get static PUCCH slot number for testing
     * @return PUCCH slot number
     */
    int staticPucchSlotNum() { return config_options_.staticPucchSlotNum; }

    /**
     * @brief Check if beamforming is enabled
     * @return true if beamforming enabled, false otherwise
     */
    bool bf_enabled() { return config_options_.bf_enabled; }

    /**
     * @brief Check if precoding/precoding matrix is enabled
     * @return true if precoding enabled, false otherwise
     */
    bool pm_enabled() { return config_options_.precoding_enabled; }

    /**
     * @brief Get Limited Buffer Rate Matching (LBRM) value
     * @return LBRM configuration value
     */
    uint8_t lbrm() { return lbrm_; }

    /**
     * @brief Get GPS alpha value for timing synchronization
     * @return GPS alpha value
     */
    uint64_t gps_alpha() { return gps_alpha_; }

    /**
     * @brief Get GPS beta value for timing synchronization
     * @return GPS beta value
     */
    int64_t gps_beta() { return gps_beta_; }

    /**
     * @brief Get uplink CQI value
     * @return UL CQI value (0xff if not set)
     */
    uint8_t ul_cqi() { return ul_cqi_; }

    /**
     * @brief Get RSSI (Received Signal Strength Indicator) value
     * @return RSSI value (0xffff if not set)
     */
    uint16_t rssi() { return rssi_; }

    /**
     * @brief Get RSRP (Reference Signal Received Power) value
     * @return RSRP value (0xffff if not set)
     */
    uint16_t rsrp() { return rsrp_; }
    // static cell_group_command* group_command() {
    //     return &group_command_;
    // }
    static std::unordered_map<uint32_t, pm_weights_t>& pm_map()
    {
        return pm_weight_map_;
    }
    static std::unordered_map<uint16_t, digBeam_t>& static_digBeam_map()
    {
        return static_digBeam_weight_map_;
    }
    bool                    cell_group() { return cell_group_; }
    const pucch_dtx_t_list& dtx_thresholds() const { return dtx_thresholds_; }
    const float&            dtx_thresholds_pusch() const { return dtx_thresholds_pusch_; }

    std::chrono::nanoseconds l2a_start_tick() { return l2a_start_tick_; }
    std::chrono::nanoseconds l2a_end_tick() { return l2a_end_tick_; }
    void                     l2a_start_tick(std::chrono::nanoseconds time) { l2a_start_tick_ = time; }
    void                     l2a_end_tick(std::chrono::nanoseconds time) { l2a_end_tick_ = time; }
    void                     last_fapi_msg_tick(std::chrono::nanoseconds time) { last_fapi_msg_tick_ = time; }
    bool                     new_slot() { return new_slot_; }
    void                     new_slot(bool new_slot) { new_slot_ = new_slot; }
    bool                     is_ul_slot() { return is_ul_slot_; }
    bool                     is_dl_slot() { return is_dl_slot_; }
    void                     is_ul_slot(bool is_ul_slot) { is_ul_slot_ = is_ul_slot; }
    void                     is_dl_slot(bool is_dl_slot) { is_dl_slot_ = is_dl_slot; }
    bool                     is_csirs_slot() { return is_csirs_slot_; }
    void                     is_csirs_slot(bool is_csirs_slot) { is_csirs_slot_ = is_csirs_slot; }
    sfn_slot_t&              get_curr_sfn_slot() { return ss_curr; }
    sfn_slot_t               get_next_sfn_slot(sfn_slot_t& ss);
    uint32_t                 get_fapi_latency(sfn_slot_t ss_msg);
    uint32_t                 get_slot_interval(sfn_slot_t ss_old, sfn_slot_t ss_new);
#ifdef ENABLE_L2_SLT_RSP
    /**
     * @brief Set current SFN/slot state (ss_curr).
     * @param[in] ss SFN/slot to adopt as the slot in progress.
     */
    void set_curr_sfn_slot(const sfn_slot_t ss);
    [[nodiscard]] uint64_t get_active_cell_bitmap() const noexcept;
    [[nodiscard]] uint32_t get_active_cell_count() const noexcept;
    void set_active_cell_bitmap(uint16_t cell_id);
    void unset_active_cell_bitmap(uint16_t cell_id);
    // Defer-commit: record that a staged cell has produced its first slot message. The cell is
    // committed (enters the EOM expectation) at the NEXT SLOT.IND, not here -- so its first
    // produced slot is not awaited and a late join cannot reopen an already-finalized slot.
    void mark_cell_produced(uint16_t cell_id) noexcept;
    void update_eom_rcvd_bitmap(uint16_t cell_id) {  fapi_eom_rcvd_bitmap |= 1ULL << cell_id; }
#else
    uint32_t get_allowed_fapi_latency()
    {
        return allowed_fapi_latency;
    }
    sfn_slot_t&                           get_last_sfn_slot() { return ss_last; }
#endif
    void set_first_tick(bool ft)
    {
        first_tick = ft;
    }
    slot_command_api::cell_sub_command& cell_sub_command(uint32_t cell_index) { return slot_command_array.at(current_slot_cmd_index).cells.at(cell_index); }
    slot_command_api::slot_command&     slot_command() { return slot_command_array.at(current_slot_cmd_index); }
    cell_group_command*                 group_command()
    {
        return &(slot_command_array.at(current_slot_cmd_index).cell_groups);
    }
    /**
     * Resolve the per-ring, per-channel, per-cell UL ordering sym_prb_info scratch slot.
     *
     * Parallel UL channel tasks (PRACH, PUSCH, PUCCH, SRS) write their order
     * metadata here instead of the shared slot-command sym_prb_info;
     * merge_ul_order_scratch() consolidates the scratch at channel-drain.
     * Entries are pre-allocated at construction; no allocation on this path.
     * Called from worker-thread channel tasks via the order-scratch resolver.
     *
     * @param[in] ring_idx    Ring slot index in [0, SLOT_STORAGE_DEPTH).
     * @param[in] channel_idx UL ordering-scratch channel in [0, kUlOrderScratchChannels).
     * @param[in] cell_idx    Cell index in [0, MAX_CELLS_PER_SLOT).
     * @return Scratch slot_info for the (ring, channel, cell), or nullptr when any
     *         index is out of range (or in non-store-replay builds).
     */
    [[nodiscard]] slot_command_api::slot_info_t*
    ul_order_scratch_sym_prb_info(uint32_t ring_idx, uint32_t channel_idx, uint32_t cell_idx);

    /**
     * Allocate every UL ordering-scratch entry once, at construction.
     *
     * Runs on the constructor thread so the per-slot reset_ul_order_scratch and
     * worker-side ul_order_scratch_sym_prb_info paths never allocate on live
     * traffic. An entry that fails to allocate is left null and treated as empty
     * downstream.
     */
    void prealloc_ul_order_scratch();

    /**
     * Reset every UL ordering-scratch entry of @p ring_idx for the next occupant.
     *
     * Called from reset_for_slot() on the msg-processing thread. noexcept:
     * out-of-range @p ring_idx is a no-op. This path only clears pre-allocated
     * storage (see prealloc_ul_order_scratch()); it never allocates.
     *
     * @param[in] ring_idx  Ring slot index in [0, SLOT_STORAGE_DEPTH).
     */
    void reset_ul_order_scratch(uint32_t ring_idx) noexcept;

    /**
     * Clear per-cell direct C-plane task/batch args for @p ring_idx.
     *
     * The accumulate bitmaps are duplicate-delivery guards and can be set before
     * an empty-PDU early return. Leftover task args from a prior ring occupant
     * must be zeroed on every ring reclaim path so early SlotMap creation does
     * not invent phantom peer-ready cells.
     *
     * @param[in] ring_idx  Ring slot index in [0, SLOT_STORAGE_DEPTH).
     */
    void clear_direct_cplane_task_args_for_ring(uint32_t ring_idx) noexcept;

    /**
     * Merge all UL ordering-scratch entries of @p ring_idx into the slot
     * command's per-cell sym_prb_info.
     *
     * Runs exactly once per slot when the channel-task counter drains (from
     * on_slot_channel_aggr_complete on the last completing worker, or inline on
     * the msg thread), immediately before publish_slot_command(). Appends PRB
     * entries and per-symbol index lists with an overflow guard against
     * MAX_PRB_INFO. noexcept: out-of-range indexes degrade to a logged skip.
     *
     * @param[in] ring_idx  Ring slot index in [0, SLOT_STORAGE_DEPTH).
     */
    void merge_ul_order_scratch(uint32_t ring_idx) noexcept;
    void update_slot_cmds_indexes()
    {
        current_slot_cmd_index = (current_slot_cmd_index + 1) % slot_command_array.size();
    }

    bfw_coeff_mem_info_t* get_bfw_coeff_buff_info(uint32_t cell_index, uint8_t slot_index) { return &(bfwCoeff_mem_info[cell_index][slot_index]); }

    /// @brief Acquire a beamforming coefficient buffer guaranteed to be in FREE state.
    ///
    /// Tell-don't-ask wrapper around get_bfw_coeff_buff_info(): if the buffer's
    /// header was left in a non-FREE state by a previous slot, it is reset to
    /// BFW_COFF_MEM_FREE before returning. Callers no longer need to inspect or
    /// mutate the header field directly.
    ///
    /// @param cell_index        Logical cell index.
    /// @param slot_store_index  Slot rotation index (0 .. MAX_BFW_COFF_STORE_INDEX-1).
    /// @return Non-null pointer to the buffer info; *header is guaranteed FREE.
    slot_command_api::bfw_coeff_mem_info_t* acquire_free_bfw_coeff_buff(
        uint32_t cell_index, uint8_t slot_store_index)
    {
        auto* info = get_bfw_coeff_buff_info(cell_index, slot_store_index);
        if (*info->header != slot_command_api::BFW_COFF_MEM_FREE)
        {
            NVLOGW_FMT(NVLOG_TAG_BASE_L2_ADAPTER + 6,
                       "BFW coeff buffer header not marked as FREE; forcing FREE before reuse "
                       "cell_index={} slot_store_index={} header={} prev_sfn={} prev_slot={}",
                       cell_index,
                       slot_store_index,
                       static_cast<uint32_t>(*info->header),
                       info->sfn,
                       info->slot);
            *info->header = slot_command_api::BFW_COFF_MEM_FREE;
        }
        return info;
    }

    void set_bfw_coeff_buff_info(uint32_t cell_index, bfw_buffer_info* buff);

    void incr_active_cells();

    void decr_active_cells();

#ifdef ENABLE_FAPI_STORE_REPLAY
    /**
     * @brief Stage a cell activation for the next slot-boundary commit.
     * @param cell_id Carrier index to mark active in staged state.
     */
    void stage_cell_started(uint16_t cell_id) noexcept;

    /**
     * @brief Stage a cell deactivation for the next slot-boundary commit.
     * @param cell_id Carrier index to clear in staged state.
     */
    void stage_cell_stopped(uint16_t cell_id) noexcept;
#endif

    void oam_cell_eaxcids_update(uint16_t mplane_id, std::unordered_map<int, std::vector<uint16_t>>& eaxcids_ch_map);

    void oam_cell_multi_attri_update(uint16_t mplane_id, std::unordered_map<std::string, double>& attrs, std::unordered_map<std::string, int>& res);

    void create_cell_update_call_back();

    ::CellUpdateCallBackFn& cell_update_cb()
    {
        return cell_update_cb_fn;
    }
    uint8_t prepone_h2d_copy() { return prepone_h2d_copy_; };

    inline phy_config_option& config_options() { return config_options_; }
    uint16_t                  get_stat_prm_idx_to_cell_id_map_size() { return stat_prm_idx_to_cell_id_map.size(); };
    void                      insert_cell_id_in_stat_prm_map(uint16_t cell_id, uint16_t stat_idx) { stat_prm_idx_to_cell_id_map.insert({stat_idx, cell_id}); };
    uint16_t                  get_cell_id_from_stat_prm_idx(uint16_t stat_idx) { return stat_prm_idx_to_cell_id_map[stat_idx]; };
    uint64_t                  fapi_config_check_mask() { return fapi_config_check_mask_; };

#ifdef SCF_FAPI_10_04
    PHY_config& get_phy_config()
    {
        return phy_config;
    };
#endif
    int               send_sfn_slot_sync_grpc_command();
    bool              get_sfn_slot_sync_cmd_sent() { return sfn_slot_sync_cmd_sent; };
    void              set_sfn_slot_sync_cmd_sent(bool val) { sfn_slot_sync_cmd_sent = val; };
    uint8_t           get_enable_se_sync_cmd() { return enable_se_sync_cmd; };
    uint8_t           get_target_node() { return target_node; };
    bool              check_sync_rcvd_from_ue() { return (sync_rcvd_from_ue == true); };
    bool              check_sync_rcvd_from_du() { return (sync_rcvd_from_du == true); };
    ch_indexes&       get_ch_proc_indexes(uint type) { return ch_proc_seg_indexes[type]; }
    ch_seg_timelines& get_ch_timeline(uint type) { return ch_proc_seg_timelines[type]; }

    // Accessor methods for slot limit errors
    slot_limit_cell_error_t& get_cell_limit_errors(uint16_t cell_id)
    {
        return cell_limit_errors_[cell_id];
    }

    slot_limit_group_error_t& get_group_limit_errors()
    {
        return group_limit_errors_;
    }

    void reset_l1_limit_errors()
    {
        // Reset all cell errors
        for(auto& cell_error : cell_limit_errors_)
        {
            std::fill_n(reinterpret_cast<std::uint8_t*>(&cell_error),
                        sizeof(cell_error),
                        0);
        }

        // Reset all group errors
        std::fill_n(reinterpret_cast<std::uint8_t*>(&group_limit_errors_),
                    sizeof(group_limit_errors_),
                    0);
    }

private:
    typedef std::unique_ptr<PHY_instance> PHY_instance_ptr;
    //------------------------------------------------------------------
    // Data
    phy_mac_transport_wrapper      transport_wrapper_;
    std::thread                    thread_; // module thread
    std::unique_ptr<thread_config> thread_cfg_;

    std::vector<PHY_instance_ptr> phy_instances_;
    std::vector<PHY_instance_ref> phy_refs_;

    std::unique_ptr<member_event_callback<PHY_module>> msg_mcb_p;
    std::unique_ptr<phy_epoll_context>                 epoll_ctx_p;
    ///// Queue holding the DL TB to released in DL callback
    nv_preallocated_queue<phy_mac_msg_desc> dl_tbs_queue_;
    std::mutex                              dl_tbs_lock;

    ///// Queue holding the DL TTI msg to be released in FH prepare callback
    std::queue<phy_mac_msg_desc> dl_tti_queue_;
    std::mutex                   dl_tti_lock;

    // Tick member variables
    // Atomic variables can be used in both tick_generator and msg_processing threads
    nv::TickUpdater         tick_updater_;
    bool                    first_tick = true;
    std::atomic<sfn_slot_t> ss_tick;
    std::mutex              tick_lock;
    nanoseconds             current_tick_;
    nv::tti_gen             tti_module_;

    // TODO: Remove the limitation that all cells need to be configured at initial
    bool all_cells_configured = false;

    bool sfn_slot_sync_cmd_sent = false;

    std::once_flag              cb_flag;
    slot_command_api::callbacks callbacks_;
    dl_tb_loc                   dl_tb_location_;
    uint64_t                    timer_thread_wakeup_threshold_ = 15000;  //15 us
    uint32_t                    l2a_allowed_latency_           = 100000; //100 us
    uint8_t                     lbrm_                          = 0;
    uint64_t                    gps_alpha_                     = 0;
    int64_t                     gps_beta_                      = 0;
    uint8_t                     ul_cqi_                        = 0xff;
    uint16_t                    rssi_                          = 0xffff;
    uint16_t                    rsrp_                          = 0xffff;

    // test_type: 0 - normal; 1 - l2adapter standalone; 2 - tick unit test
    int32_t test_type = 0;

    bool cell_group_ = false;

    uint32_t total_cell_num = 0;

    // Tick interval deviation statistic logger
    int32_t  allowed_tick_error;
    uint8_t  prepone_h2d_copy_;
    uint64_t fapi_config_check_mask_ = 0x0UL;

    // -----------------------------------------------------------------------
    // TX_DATA split-phase H2D copy state (msg thread only — no concurrency concern)
    // -----------------------------------------------------------------------
    stat_log_t* tick_logger;
    stat_log_t* slot_latency;
#ifdef ENABLE_FAPI_STORE_REPLAY
    /// Per-ring telemetry snapshot — store-replay path only.
    /// Type comes from "nv_slot_telemetry.hpp" which is itself include-guarded
    /// by ENABLE_FAPI_STORE_REPLAY.
    std::array<SlotTelemetrySnapshot, SLOT_STORAGE_DEPTH> slot_telemetry_{};
    /// Serializes slot_latency->add() calls from cuphydriver worker threads
    /// in publish_slot_command(). stat_log_add() in
    /// cuPHY/nvlog/src/stat_log.c performs unlocked read-modify-write on
    /// min/max/sum/carry/counter, so without this mutex DlPhyDriverNN /
    /// UlPhyDriverNN workers landing here concurrently for different ring
    /// slots would race. The direct path (process_phy_commands(), msg
    /// thread only) does not take this lock — it cannot run concurrently
    /// with publish_slot_command() because the two paths are selected by
    /// the runtime fapi_to_cplane_direct flag and are mutually exclusive.
    std::mutex slot_latency_mu_;
#endif
    pucch_dtx_t_list dtx_thresholds_;
    float            dtx_thresholds_pusch_;
    /// Consolidated ring-buffer replacing five scattered per-slot arrays
    /// (tx_data_deferred_, expected_tx_data_count_, tx_batch_state_,
    /// txdata_staged_gpu_ptr_, txdata_staged_msg_buf_). See TxDataRingBuffer.
    nv::TxDataRingBuffer tx_data_ring_;
    //SE sync params
    std::string       server_addr;
    uint8_t           target_node;
    uint8_t           enable_se_sync_cmd;
    std::atomic<bool> sync_rcvd_from_ue;
    std::atomic<bool> sync_rcvd_from_du;

    bool new_slot_ = true;
    // One-shot boot latch: set the first time a slot at or past the startup window
    // (SFN >= kStartupDirectCplaneWarmupSfns) is armed. Gates the SFN-based startup
    // warmup drop and ring-reuse tolerance so they fire only at boot, not on every
    // ~10.24 s SFN rollover (SFN 0-3 recur every cycle). Msg-thread-only.
    bool startup_boot_complete_ = false;
    std::array<std::chrono::nanoseconds,10> current_tick_list_;//Array for storing last 10 ticks (indexed based on slot%10)
    std::array<std::chrono::nanoseconds,10> l1_slot_ind_tick_;//Array for storing last 10 slot indicator times (indexed based on slot%10)
    std::chrono::nanoseconds l2a_start_tick_;
    std::chrono::nanoseconds l2a_end_tick_;
    std::chrono::nanoseconds last_fapi_msg_tick_;
    bool is_ul_slot_ = false;
    bool is_dl_slot_ = false;
    bool is_csirs_slot_ = false;
    std::array<FapiSlotMessageStorage, SLOT_STORAGE_DEPTH> slot_message_storage_{};
#ifdef ENABLE_FAPI_STORE_REPLAY
    NonSlotDispatchState nonslot_dispatch_state_{};
    static constexpr uint32_t kUlOrderScratchChannels = 4U; // PRACH, PUSCH, PUCCH, SRS
    std::array<
        std::array<
            std::array<std::unique_ptr<slot_command_api::slot_info_t>, MAX_CELLS_PER_SLOT>,
            kUlOrderScratchChannels>,
        SLOT_STORAGE_DEPTH> ul_order_scratch_{};
#endif
    SlotReadyCallback slot_ready_callback_{};

    // Below variables should only use in msg_processing thread
    sfn_slot_t                                                ss_curr;
    // Slot value from the previous SLOT.ind; finalize_slot_on_boundary is called with
    // this so slot N is finalized at SLOT.ind for N+2, matching the store_slot_message
    // lag > 2 acceptance window.
    uint32_t                                                  deferred_prior_slot_u32_{SFN_SLOT_INVALID};
    int32_t                                                   tti_event_count;
    uint32_t                                                  current_slot_cmd_index;
    uint                                                      num_cells_active;
    uint16_t                                                  curr_slot_fapi_num{};
    uint16_t                                                  next_slot_fapi_num{};
    std::array<nv::phy_mac_msg_desc, MAX_CELLS_PER_SLOT * 12> next_slot_fapi_cache;
#ifdef ENABLE_L2_SLT_RSP
    // Store-replay builds keep canonical active-cell state in NonSlotDispatchState.
    // These legacy fields remain mirrored for existing PHY_module call sites.
    uint64_t active_cell_bitmap{};
    uint64_t fapi_eom_rcvd_bitmap{};
    uint64_t cached_cell_bitmap{};

    /// Slot task scheduling state — in-flight counters, task arg pools, DLC/ULC guards.
    nv::SlotTaskPool task_pool_;
#else
    uint32_t                              allowed_fapi_latency;
    std::array<nv::phy_mac_msg_desc, 128> slot_msgs_received;
    uint16_t                              num_msgs;
    sfn_slot_t                            ss_last;
#endif
    ::CellUpdateCallBackFn cell_update_cb_fn;
#ifdef SCF_FAPI_10_04
    PHY_config phy_config;
#endif
public:
    // static slot_command_api::cell_group_command group_command_;
    static std::unordered_map<uint32_t, pm_weights_t> pm_weight_map_;
    static std::unordered_map<uint16_t, digBeam_t>    static_digBeam_weight_map_;
    std::vector<slot_command_api::slot_command>       slot_command_array;
    std::once_flag                                    cell_update_flag;
    /// 0 - Sync per cell, 1 - sync per slot
    sync_mode_t ipc_sync_mode;
    bool        first_dl_slot_ = true;
    bool        first_ul_slot_ = true;

    phy_config_option            config_options_;
    std::map<uint16_t, uint16_t> stat_prm_idx_to_cell_id_map;

    bfw_coeff_mem_info_t                  bfwCoeff_mem_info[MAX_CELLS_PER_SLOT][MAX_BFW_COFF_STORE_INDEX];
    bfw_coeff_mem_info_t                  static_bfwCoeff_mem_info[MAX_CELLS_PER_SLOT][MAX_STATIC_BFW_COFF_STORE_INDEX];
    unordered_map<uint, ch_indexes>       ch_proc_seg_indexes;
    unordered_map<uint, ch_seg_timelines> ch_proc_seg_timelines;

#ifdef ENABLE_L2_SLT_RSP
    std::array<slot_limit_cell_error_t, MAX_CELLS_PER_SLOT> cell_limit_errors_;
    slot_limit_group_error_t                                group_limit_errors_;
#endif
};

} // namespace nv

#endif // !defined(NV_PHY_MODULE_HPP_INCLUDED_)

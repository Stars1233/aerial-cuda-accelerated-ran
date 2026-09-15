/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NV_FAPI_MESSAGE_STORAGE_HPP
#define NV_FAPI_MESSAGE_STORAGE_HPP

#include "enum_utils.hpp"
#include "nv_fapi_tti_pdu_counts.hpp"
#include "nv_phy_mac_transport.hpp"
#include "scf_5g_fapi.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace nv
{

/// Max per-lane entries; must be >= @c MAX_CELLS_PER_SLOT from cuphy.h (20 or 40).
inline constexpr std::size_t k_fapi_lane_capacity_max = 40;

/**
 * Result of FapiSlotMessageStorage::store_message().
 *
 * Three semantically distinct outcomes that a plain bool cannot express:
 *
 *   Stored      — payload message written to the lane buffer; replayed on EOM.
 *   ControlSkip — control message (SLOT_IND, ERROR_IND, SLOT_RESP) handled on
 *                 ingress; not stored, but not an error.
 *   Rejected    — unknown msg_id; not a payload.
 */
enum class StoreResult : uint8_t
{
    Stored,      ///< message queued for EOM-gated replay
    ControlSkip, ///< control message processed on ingress — not stored
    Rejected,    ///< unknown msg_id — not a payload
};

/**
 * Models a FAPI IPC transport capable of releasing a receive buffer.
 *
 * Used to constrain `release_all` templates on both storage classes.
 * Any type providing `rx_release(phy_mac_msg_desc&)` satisfies this concept.
 */
template <typename T>
concept FapiTransport = requires(T& t, phy_mac_msg_desc& m)
{
    t.rx_release(m);
};

/**
 * Slot-scoped storage for FAPI messages grouped by type.
 *
 * Uses a single flat buffer divided into fixed-size per-type lanes.
 * All message descriptors are contiguous in memory; per-type counts
 * are kept in a small array on the stack. This eliminates the 6 separate
 * vector header cache lines and the per-push_back size+capacity reads.
 *
 * Layout of msgs_ (each lane has capacity_per_type_ slots):
 *   [DL_TTI ... | UL_TTI ... | UL_DCI ... | TX_DATA ... | DL_BFW ... | UL_BFW ...]
 *
 * store_message() : switch → lane base pointer + count write + count increment.
 * clear()         : std::fill on counts_[] + 2 scalar resets (no pointer/capacity updates).
 */
class FapiSlotMessageStorage final {
public:
    // Lane identifiers — order must match lane_offset() and the replay sequence
    // in process_ready_slot_messages().
    enum class MsgType : uint8_t
    {
        DL_TTI = 0,
        UL_TTI,
        UL_DCI,
        TX_DATA,
        DL_BFW,
        UL_BFW,
        NUM_TYPES
    };
    static constexpr std::size_t NUM_TYPES = to_underlying(MsgType::NUM_TYPES);

    FapiSlotMessageStorage() = default;
    /**
     * Construct a storage entry and pre-allocate the flat buffer.
     * @param[in] capacity_per_type  Maximum number of messages to reserve per
     *                               message-type lane.  Equivalent to calling
     *                               the default constructor followed by
     *                               reserve_per_type(capacity_per_type).
     */
    explicit FapiSlotMessageStorage(std::size_t capacity_per_type);

    // The std::atomic<uint16_t> counts_ member makes the implicit copy/move
    // ineligible. The class is used move-only (PHY_module's move ctor and the
    // per-slot re-init assignment in the PHY_module ctor), so define move
    // operations that transfer the atomic counts by value. Copy stays deleted.
    //
    // Inlined here (rather than in nv_fapi_message_storage.cpp) so PHY_module —
    // which is compiled unconditionally and holds a FapiSlotMessageStorage
    // member — can move-construct/move-assign the member even in configurations
    // where nv_fapi_message_storage.cpp is excluded from libnvphy
    // (ENABLE_FAPI_STORE_REPLAY=OFF; the .cpp contains FAPI-10.04-specific
    // helpers gated behind that flag, but the move ops themselves have no such
    // dependency).
    FapiSlotMessageStorage(FapiSlotMessageStorage&& other) noexcept
        : msgs_(std::move(other.msgs_)),
          capacity_per_type_(other.capacity_per_type_),
          tracked_slot_(other.tracked_slot_),
          ready_(other.ready_)
    {
        dl_tti_pdu_counts_ = other.dl_tti_pdu_counts_;
        ul_tti_pdu_counts_ = other.ul_tti_pdu_counts_;
        for (std::size_t i = 0; i < NUM_TYPES; ++i)
        {
            counts_[i].store(other.counts_[i].load(std::memory_order_relaxed),
                             std::memory_order_relaxed);
        }
    }

    FapiSlotMessageStorage& operator=(FapiSlotMessageStorage&& other) noexcept
    {
        if (this != &other)
        {
            msgs_              = std::move(other.msgs_);
            dl_tti_pdu_counts_ = other.dl_tti_pdu_counts_;
            ul_tti_pdu_counts_ = other.ul_tti_pdu_counts_;
            capacity_per_type_ = other.capacity_per_type_;
            tracked_slot_      = other.tracked_slot_;
            ready_             = other.ready_;
            for (std::size_t i = 0; i < NUM_TYPES; ++i)
            {
                counts_[i].store(other.counts_[i].load(std::memory_order_relaxed),
                                 std::memory_order_relaxed);
            }
        }
        return *this;
    }

    FapiSlotMessageStorage(const FapiSlotMessageStorage&)            = delete;
    FapiSlotMessageStorage& operator=(const FapiSlotMessageStorage&) = delete;
    ~FapiSlotMessageStorage()                                        = default;

    /**
     * Allocate or reallocate the flat message buffer.
     * @details The flat buffer holds @p capacity_per_type × NUM_TYPES message
     *          descriptors contiguously.  If the buffer was previously
     *          allocated with a different capacity, it is replaced.  Not
     *          called on the hot path; expected to be called once during ring
     *          construction.
     * @param[in] capacity_per_type  Maximum number of messages to reserve per
     *                               message-type lane.
     */
    void reserve_per_type(std::size_t capacity_per_type);

    /**
     * Store a slot-scoped message into the appropriate lane.
     * @param[in] msg  FAPI message descriptor to classify and store.
     * @return StoreResult::Stored      if the message was written to the lane buffer.
     *         StoreResult::ControlSkip if it is a control message (SLOT_IND, ERROR_IND,
     *                                  SLOT_RESP) handled on ingress.
     *         StoreResult::Rejected    if the msg_id is unknown.
     */
    [[nodiscard]] StoreResult store_message(const phy_mac_msg_desc& msg);

    /**
     * Release all stored messages via any FapiTransport implementation.
     * @tparam Transport  Type satisfying the FapiTransport concept.
     * @param[in] transport  Transport used to release each stored message buffer.
     */
    template <FapiTransport Transport>
    void release_all(Transport& transport)
    {
        phy_mac_msg_desc* base = msgs_.data();
        for(std::size_t t = 0; t < NUM_TYPES; ++t)
        {
            phy_mac_msg_desc* lane = base + t * capacity_per_type_;
            const uint16_t    n    = counts_[t].exchange(0, std::memory_order_acq_rel);
            for(uint16_t i = 0; i < n; ++i)
            {
                transport.rx_release(lane[i]);
            }
        }
        tracked_slot_.reset();
        ready_ = false;
    }

    /**
     * Release a single lane's messages via any FapiTransport implementation.
     * @tparam Transport  Type satisfying the FapiTransport concept.
     * @param[in] type       Lane to release.
     * @param[in] transport  Transport used to release each stored message buffer.
     */
    template <FapiTransport Transport>
    void release_lane(MsgType type, Transport& transport)
    {
        const auto        idx       = to_underlying(type);
        phy_mac_msg_desc* lane_base = msgs_.data() + idx * capacity_per_type_;
        // Atomically claim the lane count so two threads racing to release the
        // same lane cannot double-free. This matters for the TX_DATA lane, which
        // the H2D-completion callback and the inline message thread may both try
        // to release: whichever performs the exchange first frees the buffers,
        // the other observes 0 and frees nothing.
        const uint16_t n = counts_[idx].exchange(0, std::memory_order_acq_rel);
        for(uint16_t i = 0; i < n; ++i)
        {
            transport.rx_release(lane_base[i]);
        }
    }

    /**
     * Discard all stored messages without releasing transport buffers.
     * @details Resets every per-lane message count to zero, clears the
     *          tracked-slot binding, and clears the ready flag.  The flat
     *          buffer allocation is preserved.  Typically called via
     *          reset_for_slot(), which immediately re-binds the entry to a
     *          new slot after clearing.
     */
    void clear();

    /**
     * Bind this ring entry to a new slot and clear accumulated state.
     * @details Calls clear() and then records @p slot_u32 as the tracked slot,
     *          transitioning the entry from unbound (or previously bound) to
     *          bound-and-empty.  Subsequent store_message() calls are accepted
     *          only for this slot until the entry is cleared or reset again.
     * @param[in] slot_u32  Absolute slot encoded as a packed uint32_t
     *                      (upper 16 bits: SFN, lower 16 bits: slot).
     */
    void reset_for_slot(uint32_t slot_u32);

    /// Result of bind_to_slot(): whether the entry was already bound to the
    /// requested slot, freshly bound from unbound, or rebound over a different slot.
    /// Callers on the store path collapse FreshBind/Rebound to "!= AlreadyBound"
    /// (both clear state), but the two are kept distinct as a tested contract:
    /// FreshBind clears from cold, Rebound clears a drained ring's stale state -
    /// unit tests assert each transition, and the split leaves room for a future
    /// per-outcome diagnostic (cold bind vs mid-stream ring reuse).
    enum class BindOutcome { AlreadyBound, FreshBind, Rebound };

    /**
     * Tell this entry to bind to @p slot_u32, clearing accumulated state on a
     * fresh bind or a rebind. Replaces the has_slot()/tracked_slot() ask-then-reset
     * idiom so a caller can pair it with its coupled per-ring reset (the telemetry
     * snapshot) in a single owner, and the two cannot desync.
     * @param[in] slot_u32  Absolute packed slot (upper 16 bits SFN, lower 16 slot).
     * @return AlreadyBound (no change), FreshBind (was unbound), or Rebound (was
     *         bound to a different slot; the old accumulated state is cleared).
     */
    [[nodiscard]] BindOutcome bind_to_slot(uint32_t slot_u32)
    {
        if (tracked_slot_.has_value())
        {
            if (tracked_slot_.value() == slot_u32)
            {
                return BindOutcome::AlreadyBound;
            }
            reset_for_slot(slot_u32);
            return BindOutcome::Rebound;
        }
        reset_for_slot(slot_u32);
        return BindOutcome::FreshBind;
    }

    /**
     * Clear the slot binding and ready flag without touching lane counts.
     *
     * Called after lane buffers have been individually released via release_lane()
     * so that has_slot() returns false.  Deferred TX_DATA lane counts are
     * intentionally preserved for the cuphydriver callback.
     */
    void unbind()
    {
        tracked_slot_.reset();
        ready_ = false;
    }

    /**
     * Check whether this entry is bound to a slot.
     * @return @c true if reset_for_slot() has been called and no subsequent
     *         clear() or unbind() has discarded the binding; @c false otherwise.
     */
    [[nodiscard]] bool has_slot() const { return tracked_slot_.has_value(); }

    /**
     * Return the absolute slot this entry is currently bound to.
     * @details Only valid when has_slot() is @c true.  Calling this on an
     *          unbound entry (has_slot() == false) is undefined behaviour
     *          (std::optional::value() throws std::bad_optional_access).
     * @return The packed uint32_t slot value passed to the last
     *         reset_for_slot() call.
     */
    [[nodiscard]] uint32_t tracked_slot() const { return tracked_slot_.value(); }

    /**
     * Query whether this slot entry has been marked ready for replay.
     * @details An entry becomes ready when the EOM condition is satisfied for
     *          its slot (i.e., all expected SLOT.RESPONSE messages have been
     *          received).  The flag is cleared by clear().
     * @return @c true if set_ready(true) has been called since the last
     *         clear(); @c false otherwise.
     */
    [[nodiscard]] bool ready() const { return ready_; }

    /**
     * Set or clear the ready flag for this slot entry.
     * @details Called by the EOM handler to signal that message replay may
     *          begin, or by the reset path to withdraw that signal.
     * @param[in] ready  @c true to mark the entry ready for replay;
     *                   @c false to withdraw the ready state.
     */
    void set_ready(bool ready) { ready_ = ready; }

    /// @name Named lane accessors — public API for the replay path and tests.
    /// Each pair returns a pointer to the base of the message array for its
    /// lane (MsgType).  The pointer remains valid until the next store_message()
    /// or clear() call.  Pair the pointer with the corresponding count accessor
    /// to iterate over stored messages.
    /// @see lane(), MsgType
    ///@{

    /**
     * Access the DL_TTI message lane.
     * @return Pointer to the first stored @c DL_TTI message descriptor, or a
     *         valid base pointer into an empty lane when dl_tti_count() == 0.
     *         Valid until the next store_message() or clear().
     * @see MsgType::DL_TTI, dl_tti_count()
     */
    [[nodiscard]] phy_mac_msg_desc* dl_tti_messages() { return lane(MsgType::DL_TTI); }
    /** @copydoc dl_tti_messages() */
    [[nodiscard]] const phy_mac_msg_desc* dl_tti_messages() const { return lane(MsgType::DL_TTI); }

    /**
     * Access the UL_TTI message lane.
     * @return Pointer to the first stored @c UL_TTI message descriptor.
     *         Valid until the next store_message() or clear().
     * @see MsgType::UL_TTI, ul_tti_count()
     */
    [[nodiscard]] phy_mac_msg_desc* ul_tti_messages() { return lane(MsgType::UL_TTI); }
    /** @copydoc ul_tti_messages() */
    [[nodiscard]] const phy_mac_msg_desc* ul_tti_messages() const { return lane(MsgType::UL_TTI); }

    /**
     * Access the UL_DCI message lane.
     * @return Pointer to the first stored @c UL_DCI message descriptor.
     *         Valid until the next store_message() or clear().
     * @see MsgType::UL_DCI, ul_dci_count()
     */
    [[nodiscard]] phy_mac_msg_desc* ul_dci_messages() { return lane(MsgType::UL_DCI); }
    /** @copydoc ul_dci_messages() */
    [[nodiscard]] const phy_mac_msg_desc* ul_dci_messages() const { return lane(MsgType::UL_DCI); }

    /**
     * Access the TX_DATA message lane.
     * @return Pointer to the first stored @c TX_DATA message descriptor.
     *         Valid until the next store_message() or clear().
     * @see MsgType::TX_DATA, tx_data_count()
     */
    [[nodiscard]] phy_mac_msg_desc* tx_data_messages() { return lane(MsgType::TX_DATA); }
    /** @copydoc tx_data_messages() */
    [[nodiscard]] const phy_mac_msg_desc* tx_data_messages() const { return lane(MsgType::TX_DATA); }

    /**
     * Access the DL_BFW message lane.
     * @return Pointer to the first stored @c DL_BFW message descriptor.
     *         Valid until the next store_message() or clear().
     * @see MsgType::DL_BFW, dl_bfw_count()
     */
    [[nodiscard]] phy_mac_msg_desc* dl_bfw_messages() { return lane(MsgType::DL_BFW); }
    /** @copydoc dl_bfw_messages() */
    [[nodiscard]] const phy_mac_msg_desc* dl_bfw_messages() const { return lane(MsgType::DL_BFW); }

    /**
     * Access the UL_BFW message lane.
     * @return Pointer to the first stored @c UL_BFW message descriptor.
     *         Valid until the next store_message() or clear().
     * @see MsgType::UL_BFW, ul_bfw_count()
     */
    [[nodiscard]] phy_mac_msg_desc* ul_bfw_messages() { return lane(MsgType::UL_BFW); }
    /** @copydoc ul_bfw_messages() */
    [[nodiscard]] const phy_mac_msg_desc* ul_bfw_messages() const { return lane(MsgType::UL_BFW); }

    ///@}

    /// @name Lane count accessors — number of messages stored in each lane.
    /// @see count(), MsgType
    ///@{

    /** Number of DL_TTI messages currently stored. @return Message count; 0 if none stored. @see MsgType::DL_TTI, dl_tti_messages() */
    [[nodiscard]] uint16_t dl_tti_count() const { return count(MsgType::DL_TTI); }

    /**
     * @brief Whether the DL_TTI message at @p index carries at least one PDSCH PDU.
     *
     * Cached at @c store_message() time so that CSI-RS aggregation can decide
     * PDSCH co-scheduling behaviour without re-scanning the FAPI payload.
     * @param[in] index  @c 0 .. dl_tti_count()-1, parallel to @c dl_tti_messages()[index].
     * @return @c true if the DL_TTI at @p index contains at least one PDSCH PDU; @c false
     *         otherwise (including when @p index is out of range, in which case the
     *         function returns @c false rather than reading past the lane).
     */
    [[nodiscard]] bool dl_tti_has_pdsch_pdu(uint16_t index) const;

    /// Cached per-type PDU counts for DL_TTI at @p index (empty if out of range).
    [[nodiscard]] NvDlTtiPduCounts dl_tti_pdu_counts(uint16_t index) const noexcept;

    /// Cached per-type PDU counts for UL_TTI at @p index (empty if out of range).
    [[nodiscard]] NvUlTtiPduCounts ul_tti_pdu_counts(uint16_t index) const noexcept;

    /// Base of the UL_TTI PDU-count sidecars, parallel to @c ul_tti_messages();
    /// valid over 0 .. ul_tti_count()-1. Lets the SRS parser reuse the store-time
    /// count instead of re-walking the payload.
    [[nodiscard]] const NvUlTtiPduCounts* ul_tti_pdu_counts_data() const noexcept
    {
        return ul_tti_pdu_counts_.data();
    }

    /**
     * Cell ID of the stored DL_TTI.req at @p index.
     *
     * Delegates to @c dl_tti_messages()[index].cell_id cast to uint16_t.
     * Required by the FapiSlotStorage concept in nv_pdsch_tb_merge.hpp.
     *
     * @param[in] index  @c 0 .. dl_tti_count()-1, parallel to @c dl_tti_messages()[index].
     * @return Cell ID of the DL_TTI.req at @p index.
     *         Return value must be checked.
     */
    [[nodiscard]] uint16_t dl_tti_cell_id(uint16_t index) const noexcept
    {
        return static_cast<uint16_t>(dl_tti_messages()[index].cell_id);
    }
    /** Number of UL_TTI messages currently stored. @return Message count; 0 if none stored. @see MsgType::UL_TTI, ul_tti_messages() */
    [[nodiscard]] uint16_t ul_tti_count() const { return count(MsgType::UL_TTI); }
    /** Number of UL_DCI messages currently stored. @return Message count; 0 if none stored. @see MsgType::UL_DCI, ul_dci_messages() */
    [[nodiscard]] uint16_t ul_dci_count() const { return count(MsgType::UL_DCI); }
    /** Number of TX_DATA messages currently stored. @return Message count; 0 if none stored. @see MsgType::TX_DATA, tx_data_messages() */
    [[nodiscard]] uint16_t tx_data_count() const { return count(MsgType::TX_DATA); }
    /** Number of DL_BFW messages currently stored. @return Message count; 0 if none stored. @see MsgType::DL_BFW, dl_bfw_messages() */
    [[nodiscard]] uint16_t dl_bfw_count() const { return count(MsgType::DL_BFW); }
    /** Number of UL_BFW messages currently stored. @return Message count; 0 if none stored. @see MsgType::UL_BFW, ul_bfw_messages() */
    [[nodiscard]] uint16_t ul_bfw_count() const { return count(MsgType::UL_BFW); }

    ///@}

private:
    // lane() and count() are implementation helpers for the named accessors above.
    // Callers must use the named accessors; lane(MsgType) is not part of the public API.
    [[nodiscard]] phy_mac_msg_desc*       lane(MsgType t) { return msgs_.data() + to_underlying(t) * capacity_per_type_; }
    [[nodiscard]] const phy_mac_msg_desc* lane(MsgType t) const { return msgs_.data() + to_underlying(t) * capacity_per_type_; }
    [[nodiscard]] uint16_t                count(MsgType t) const { return counts_[to_underlying(t)].load(std::memory_order_relaxed); }

    // Single flat allocation: NUM_TYPES lanes × capacity_per_type_ descriptors each.
    std::vector<phy_mac_msg_desc> msgs_{};

    /// Parallel to the DL/UL TTI lanes; fixed capacity (@c k_fapi_lane_capacity_max),
    /// indexed by lane slot. Filled in @c store_message; no heap on the hot path.
    /// Replaces the old @c dl_tti_has_pdsch_ vector (PDSCH presence is
    /// @c by_type[DL_TTI_NPDUS_IDX_PDSCH] > 0).
    std::array<NvDlTtiPduCounts, k_fapi_lane_capacity_max> dl_tti_pdu_counts_{};
    std::array<NvUlTtiPduCounts, k_fapi_lane_capacity_max> ul_tti_pdu_counts_{};
    /// Per-lane occupancy; zeroed on clear(). Atomic so the TX_DATA lane can be
    /// released concurrently by the H2D-completion callback and the inline
    /// message thread without a torn read or a double-free (see release_lane).
    std::atomic<uint16_t>   counts_[NUM_TYPES]{};
    std::size_t             capacity_per_type_ = 0;
    std::optional<uint32_t> tracked_slot_{}; // empty == "no slot assigned"
    bool                    ready_ = false;
};

/**
 * No-op transport — satisfies FapiTransport concept for callers with no
 * buffer to release. Pass NullFapiTransport{} instead of a null pointer.
 */
struct NullFapiTransport
{
    void rx_release(phy_mac_msg_desc&) const noexcept {}
};

/**
 * Per-cell storage for non-slot FAPI messages.
 *
 * Stores a single CONFIG/START/STOP message per cell. New arrivals replace
 * previous ones, releasing the old buffer via the supplied transport.
 */
class FapiNonSlotMessageStorage final {
public:
    /**
     * Store a non-slot message (CONFIG/START/STOP), releasing any previously
     * stored message of the same type via `transport`.
     * @tparam Transport  Type satisfying the FapiTransport concept.
     * @param[in] msg        FAPI message descriptor to store.
     * @param[in] transport  Transport used to release a previously stored message of the same type.
     * @return @c true if stored; @c false if @p msg carries an unsupported msg_id.
     */
    template <FapiTransport Transport>
    [[nodiscard]] bool store_message(const phy_mac_msg_desc& msg, Transport&& transport)
    {
        auto* slot = find_slot(static_cast<uint8_t>(msg.msg_id));
        if(!slot) return false;
        if(*slot) transport.rx_release(**slot);
        *slot = msg;
        return true;
    }

    /**
     * Store a non-slot message with no transport (nothing to release).
     * @param[in] msg  FAPI message descriptor to store.
     * @return @c true if stored; @c false if @p msg carries an unsupported msg_id.
     */
    [[nodiscard]] bool store_message(const phy_mac_msg_desc& msg)
    {
        NullFapiTransport null{};
        return store_message(msg, null);
    }

    /**
     * Release all stored messages via any FapiTransport implementation.
     * @tparam Transport  Type satisfying the FapiTransport concept.
     * @param[in] transport  Transport used to release each stored message buffer.
     */
    template <FapiTransport Transport>
    void release_all(Transport& transport)
    {
        auto release_one = [&transport](auto& stored) {
            if(stored)
            {
                transport.rx_release(*stored);
                stored.reset();
            }
        };
        release_one(config_request_);
        release_one(start_request_);
        release_one(stop_request_);
    }

    /**
     * Clear stored messages without releasing transport buffers.
     */
    void clear();

    [[nodiscard]] const phy_mac_msg_desc* config_request() const { return config_request_ ? &*config_request_ : nullptr; }
    [[nodiscard]] const phy_mac_msg_desc* start_request() const { return start_request_ ? &*start_request_ : nullptr; }
    [[nodiscard]] const phy_mac_msg_desc* stop_request() const { return stop_request_ ? &*stop_request_ : nullptr; }

    /** Mutable overloads — for callers that pass descriptors to non-const APIs (e.g. on_msg). */
    [[nodiscard]] phy_mac_msg_desc* config_request_mut() { return config_request_ ? &*config_request_ : nullptr; }
    [[nodiscard]] phy_mac_msg_desc* start_request_mut() { return start_request_ ? &*start_request_ : nullptr; }
    [[nodiscard]] phy_mac_msg_desc* stop_request_mut() { return stop_request_ ? &*stop_request_ : nullptr; }

private:
    /**
     * Map msg_id to the corresponding optional slot pointer.
     * @param[in] msg_id  FAPI message identifier to look up.
     * @return Pointer to the matching optional member, or @c nullptr if @p msg_id is unsupported.
     */
    std::optional<phy_mac_msg_desc>* find_slot(uint8_t msg_id);

    std::optional<phy_mac_msg_desc> config_request_{};
    std::optional<phy_mac_msg_desc> start_request_{};
    std::optional<phy_mac_msg_desc> stop_request_{};
};

} // namespace nv

#endif // NV_FAPI_MESSAGE_STORAGE_HPP

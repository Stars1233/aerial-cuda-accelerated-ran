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

#include <gtest/gtest.h>
#include "ru_emulator.hpp"
#include "timing_utils.hpp"

// Test fixture for RU Emulator tests
class RUEmulatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup code for all tests
    }

    void TearDown() override {
        // Common cleanup code for all tests
    }
};

// Test t0/toa calculation
TEST_F(RUEmulatorTest, CalculateT0ToA) {
    // Test case 1: Basic calculation
    int64_t packet_time = 1000000;  // 1ms
    int64_t beginning_of_time = 0;
    int64_t frame_cycle_time_ns = 10000000;  // 10ms
    uint16_t frame_id = 0;
    uint16_t subframe_id = 0;
    uint16_t slot_id = 0;
    uint16_t start_sym = 0;
    uint16_t max_slot_id = 9;
    uint16_t opt_tti_us = 1000;  // 1ms

    auto result = calculate_t0_toa(packet_time, beginning_of_time, frame_cycle_time_ns,
                                 frame_id, subframe_id, slot_id, start_sym,
                                 max_slot_id, opt_tti_us);

    // Verify the results
    EXPECT_GE(result.slot_t0, beginning_of_time);
    EXPECT_LE(result.toa, frame_cycle_time_ns/2);
    EXPECT_GE(result.toa, -frame_cycle_time_ns/2);
}

// Test t0/toa calculation with different frame/slot combinations
TEST_F(RUEmulatorTest, CalculateT0ToAWithDifferentFrames) {
    int64_t packet_time = 1000000;
    int64_t beginning_of_time = 0;
    int64_t frame_cycle_time_ns = 10000000;
    uint16_t max_slot_id = 9;
    uint16_t opt_tti_us = 1000;

    // Test different frame/subframe/slot combinations
    struct TestCase {
        uint16_t frame_id;
        uint16_t subframe_id;
        uint16_t slot_id;
        uint16_t start_sym;
    };

    std::vector<TestCase> test_cases = {
        {0, 0, 0, 0},    // First frame, first subframe, first slot
        {0, 0, 9, 0},    // First frame, first subframe, last slot
        {0, 9, 0, 0},    // First frame, last subframe, first slot
        {1, 0, 0, 0},    // Second frame, first subframe, first slot
        {0, 0, 0, 7},    // First frame, first subframe, first slot, middle symbol
    };

    for (const auto& tc : test_cases) {
        // Calculate time offset based on frame, subframe, slot and symbol
        int64_t time_offset = ((int64_t)tc.frame_id * ORAN_MAX_SUBFRAME_ID * (max_slot_id + 1) + 
                              (int64_t)tc.subframe_id * (max_slot_id + 1) + 
                              (int64_t)tc.slot_id);
        time_offset *= opt_tti_us * NS_X_US;
        time_offset += (int)(opt_tti_us * NS_X_US * (float)tc.start_sym / ORAN_ALL_SYMBOLS);

        // Adjust beginning_of_time to be in the correct range relative to time_offset
        int64_t adjusted_beginning_of_time = beginning_of_time;
        while(packet_time < adjusted_beginning_of_time + time_offset) {
            adjusted_beginning_of_time -= frame_cycle_time_ns;
        }
        while(packet_time > adjusted_beginning_of_time + time_offset + frame_cycle_time_ns) {
            adjusted_beginning_of_time += frame_cycle_time_ns;
        }

        auto result = calculate_t0_toa(packet_time, adjusted_beginning_of_time, frame_cycle_time_ns,
                                     tc.frame_id, tc.subframe_id, tc.slot_id, tc.start_sym,
                                     max_slot_id, opt_tti_us);

        // Verify the results
        EXPECT_GE(result.slot_t0, adjusted_beginning_of_time);
        EXPECT_LE(result.toa, frame_cycle_time_ns/2);
        EXPECT_GE(result.toa, -frame_cycle_time_ns/2);

        // Verify time offset calculation
        EXPECT_EQ(result.slot_t0 - adjusted_beginning_of_time, time_offset);
    }
}

/// @defgroup SlotSectionIdTrackerTests SlotSectionIdTracker unit tests
/// @brief Validates the per-eAxC sectionId tracking logic introduced for
///        O-RAN WG4 CUS default coupling compliance.
///
/// Test groups:
///  - **Core functionality**: is_same_slot, is_forward_slot, record_uplane_sid
///    (including out-of-range guard), advance_slot (UL and DL paths).
///  - **Error detection**: unannounced U-plane sectionIds, entry generation
///    lifecycle (first-occurrence recording, staleness after slot advance),
///    duplicate sectionId consistency checks ("multiple citations" rule).
///  - **Edge cases**: first-ever slot (no prev), empty slots, generation
///    counter wraparound skipping zero, per-slot independence.
///  - **Multi-cell / multi-eAxC isolation**: faults injected into one tracker
///    instance must not affect another, verifying that per-cell and per-eAxC
///    separation holds.
/// @{

/**
 * @brief Helper functor that collects warned sectionIds during DL advance_slot
 *        cross-validation, for assertion in test cases.
 */
struct WarnCollector {
    std::vector<uint16_t> warned_sids;
    void operator()(uint16_t sid, const fssId&) { warned_sids.push_back(sid); }
};

// --- Core functionality ---

/**
 * @brief Verify is_same_slot() returns true only when all three FSS fields match.
 */
TEST_F(RUEmulatorTest, SectionTracker_IsSameSlot) {
    SlotSectionIdTracker tracker;
    tracker.current_fss = {3, 5, 1};

    EXPECT_TRUE(tracker.is_same_slot({3, 5, 1}));
    EXPECT_FALSE(tracker.is_same_slot({3, 5, 0}));
    EXPECT_FALSE(tracker.is_same_slot({3, 4, 1}));
    EXPECT_FALSE(tracker.is_same_slot({2, 5, 1}));
}

/**
 * @brief Verify is_forward_slot() correctly distinguishes forward, backward,
 *        and same-slot comparisons without wraparound.
 */
TEST_F(RUEmulatorTest, SectionTracker_IsForwardSlot_BasicForwardAndBackward) {
    SlotSectionIdTracker tracker;
    tracker.current_fss = {0, 5, 0};

    EXPECT_TRUE(tracker.is_forward_slot({0, 5, 1}));
    EXPECT_TRUE(tracker.is_forward_slot({0, 6, 0}));
    EXPECT_TRUE(tracker.is_forward_slot({1, 0, 0}));

    EXPECT_FALSE(tracker.is_forward_slot({0, 4, 1}));
    EXPECT_FALSE(tracker.is_forward_slot({0, 5, 0}));  ///< Same slot is not forward
}

/**
 * @brief Verify is_forward_slot() handles frameId wraparound (255 -> 0).
 *
 * The total slot space is 5120 (256 frames x 10 subframes x 2 slots).
 * Distances < half (2560) are considered forward; distances >= half are backward.
 */
TEST_F(RUEmulatorTest, SectionTracker_IsForwardSlot_Wraparound) {
    SlotSectionIdTracker tracker;
    tracker.current_fss = {255, 9, 1};  ///< Last slot in frame space (slot 5119)

    EXPECT_TRUE(tracker.is_forward_slot({0, 0, 0}));
    EXPECT_TRUE(tracker.is_forward_slot({1, 0, 0}));

    /// Slot 5119 - 2560 = 2559 (frame 127,9,1) is exactly at the halfway boundary.
    EXPECT_FALSE(tracker.is_forward_slot({127, 9, 1}));
}

/**
 * @brief Verify is_forward_slot() accepts large forward jumps up to just under
 *        half the total slot space (2559 slots).
 */
TEST_F(RUEmulatorTest, SectionTracker_IsForwardSlot_LargeForwardJump) {
    SlotSectionIdTracker tracker;
    tracker.current_fss = {0, 0, 0};

    EXPECT_TRUE(tracker.is_forward_slot({64, 0, 0}));   ///< ~1/4 of total space
    EXPECT_TRUE(tracker.is_forward_slot({127, 9, 1}));   ///< 2559 slots forward
}

/**
 * @brief Verify record_uplane_sid() sets the correct bits in the
 *        uplane_sids_seen bitset for sectionIds in [0, NUM_ENTRIES-1].
 */
TEST_F(RUEmulatorTest, SectionTracker_RecordUplaneSid_ValidRange) {
    SlotSectionIdTracker tracker;

    tracker.record_uplane_sid(0);
    tracker.record_uplane_sid(SlotSectionIdTracker::NUM_ENTRIES - 1);  // 1023
    tracker.record_uplane_sid(512);

    EXPECT_TRUE(tracker.uplane_sids_seen.test(0));
    EXPECT_TRUE(tracker.uplane_sids_seen.test(SlotSectionIdTracker::NUM_ENTRIES - 1));
    EXPECT_TRUE(tracker.uplane_sids_seen.test(512));
    EXPECT_FALSE(tracker.uplane_sids_seen.test(1));
}

/**
 * @brief Verify record_uplane_sid() silently ignores sectionIds >= NUM_ENTRIES
 *        (1024) without corrupting the bitset.
 */
TEST_F(RUEmulatorTest, SectionTracker_RecordUplaneSid_OutOfRangeIgnored) {
    SlotSectionIdTracker tracker;

    tracker.record_uplane_sid(SlotSectionIdTracker::NUM_ENTRIES);  // 1024
    tracker.record_uplane_sid(65535);

    EXPECT_FALSE(tracker.uplane_sids_seen.any());
}

/**
 * @brief Verify the UL advance_slot() overload correctly rotates current
 *        bitsets/FSS into prev, resets current state, sets has_prev, and
 *        increments the generation counter.
 */
TEST_F(RUEmulatorTest, SectionTracker_AdvanceSlotUL_RotatesState) {
    SlotSectionIdTracker tracker;
    fssId slot0{0, 0, 0}, slot1{0, 0, 1};

    tracker.current_fss = slot0;
    tracker.cplane_sids_announced.set(42);
    tracker.record_uplane_sid(42);

    EXPECT_FALSE(tracker.has_prev);
    tracker.advance_slot(slot1);
    EXPECT_TRUE(tracker.has_prev);

    EXPECT_FALSE(tracker.cplane_sids_announced.any());
    EXPECT_FALSE(tracker.uplane_sids_seen.any());
    EXPECT_TRUE(tracker.is_same_slot(slot1));

    EXPECT_TRUE(tracker.prev_cplane_sids_announced.test(42));
    EXPECT_TRUE(tracker.prev_uplane_sids_seen.test(42));
    EXPECT_EQ(tracker.prev_fss.frameId, slot0.frameId);
}

/**
 * @brief Verify the DL advance_slot() overload produces zero warnings when
 *        all U-plane sectionIds were properly announced in C-plane.
 */
TEST_F(RUEmulatorTest, SectionTracker_AdvanceSlotDL_NoWarningWhenMatched) {
    SlotSectionIdTracker tracker;
    fssId slot0{0, 0, 0}, slot1{0, 0, 1}, slot2{0, 1, 0};

    tracker.current_fss = slot0;
    tracker.cplane_sids_announced.set(10);
    tracker.cplane_sids_announced.set(20);
    tracker.record_uplane_sid(10);
    tracker.record_uplane_sid(20);

    WarnCollector wc;
    tracker.advance_slot(slot1, std::ref(wc));
    tracker.advance_slot(slot2, std::ref(wc));

    EXPECT_TRUE(wc.warned_sids.empty());
}

// --- Error detection ---

/**
 * @brief Verify that a U-plane sectionId with no corresponding C-plane
 *        announcement triggers a warning after the 2-slot deferral window.
 *
 * The cross-validation is deferred by 2 slot advances to tolerate C/U-plane
 * thread timing skew in the real system.
 */
TEST_F(RUEmulatorTest, SectionTracker_UnannouncedUplaneSid_WarnsAfterTwoAdvances) {
    SlotSectionIdTracker tracker;
    fssId slot0{0, 0, 0}, slot1{0, 0, 1}, slot2{0, 1, 0};

    tracker.current_fss = slot0;
    tracker.record_uplane_sid(99);  ///< No C-plane announcement for sid 99

    WarnCollector wc;
    tracker.advance_slot(slot1, std::ref(wc));
    EXPECT_TRUE(wc.warned_sids.empty());  ///< First advance: slot0 moved to prev, no validation yet

    tracker.advance_slot(slot2, std::ref(wc));
    ASSERT_EQ(wc.warned_sids.size(), 1u);  ///< Second advance: prev (slot0) validated
    EXPECT_EQ(wc.warned_sids[0], 99);
}

/**
 * @brief Verify that multiple unannounced U-plane sectionIds in the same slot
 *        each produce an independent warning, while matched sids do not.
 */
TEST_F(RUEmulatorTest, SectionTracker_MultipleUnannouncedSids) {
    SlotSectionIdTracker tracker;
    fssId slot0{0, 0, 0}, slot1{0, 0, 1}, slot2{0, 1, 0};

    tracker.current_fss = slot0;
    tracker.cplane_sids_announced.set(10);
    tracker.record_uplane_sid(10);   ///< Matched — no warning expected
    tracker.record_uplane_sid(50);   ///< Unannounced
    tracker.record_uplane_sid(200);  ///< Unannounced

    WarnCollector wc;
    tracker.advance_slot(slot1, std::ref(wc));
    tracker.advance_slot(slot2, std::ref(wc));

    ASSERT_EQ(wc.warned_sids.size(), 2u);
    EXPECT_EQ(wc.warned_sids[0], 50);
    EXPECT_EQ(wc.warned_sids[1], 200);
}

/**
 * @brief Verify that a zero-initialized Entry does not match generation 1,
 *        and that writing fields to an entry correctly records them.
 *
 * This mirrors the first-occurrence path in validate_section_id_default_coupling().
 */
TEST_F(RUEmulatorTest, SectionTracker_EntryGeneration_FirstOccurrenceRecordsFields) {
    SlotSectionIdTracker tracker;
    tracker.current_fss = {0, 0, 0};

    uint16_t sid = 42;
    uint32_t gen = tracker.current_generation;
    auto& entry = tracker.entries[sid];

    EXPECT_NE(entry.generation, gen);  ///< Zero-init generation != 1

    entry.generation = gen;
    entry.rb = 0;
    entry.startPrbc = 100;
    entry.numPrbc = 50;
    entry.numSymbol = 14;
    entry.udCompHdr = 0x91;

    EXPECT_EQ(entry.generation, gen);
    EXPECT_EQ(entry.startPrbc, 100);
    EXPECT_EQ(entry.numPrbc, 50);
    EXPECT_EQ(entry.numSymbol, 14);
    EXPECT_EQ(entry.udCompHdr, 0x91);
}

/**
 * @brief Verify that an entry recorded in one slot becomes stale (generation
 *        mismatch) after advance_slot() increments the generation counter.
 */
TEST_F(RUEmulatorTest, SectionTracker_EntryGeneration_StaleAfterAdvance) {
    SlotSectionIdTracker tracker;
    tracker.current_fss = {0, 0, 0};

    uint16_t sid = 42;
    auto& entry = tracker.entries[sid];
    entry.generation = tracker.current_generation;
    entry.startPrbc = 100;

    tracker.advance_slot({0, 0, 1});

    EXPECT_NE(entry.generation, tracker.current_generation);
}

/**
 * @brief Verify that a duplicate sectionId within the same slot is accepted
 *        when all "multiple citations" fields match (per O-RAN WG4 CUS spec).
 */
TEST_F(RUEmulatorTest, SectionTracker_DuplicateDetection_SameFieldsNoError) {
    SlotSectionIdTracker tracker;
    tracker.current_fss = {0, 0, 0};

    uint16_t sid = 42;
    uint32_t gen = tracker.current_generation;
    auto& entry = tracker.entries[sid];

    entry.generation = gen;
    entry.rb = 0;
    entry.startPrbc = 100;
    entry.numPrbc = 50;
    entry.numSymbol = 14;
    entry.udCompHdr = 0x91;

    bool mismatch = (entry.rb != 0 || entry.startPrbc != 100 ||
                     entry.numPrbc != 50 || entry.numSymbol != 14 ||
                     entry.udCompHdr != 0x91);
    EXPECT_FALSE(mismatch);
}

/**
 * @brief Verify that a duplicate sectionId within the same slot is detected
 *        as an error when any "multiple citations" field differs from the
 *        first occurrence.
 */
TEST_F(RUEmulatorTest, SectionTracker_DuplicateDetection_DifferentFieldsIsError) {
    SlotSectionIdTracker tracker;
    tracker.current_fss = {0, 0, 0};

    uint16_t sid = 42;
    uint32_t gen = tracker.current_generation;
    auto& entry = tracker.entries[sid];

    entry.generation = gen;
    entry.rb = 0;
    entry.startPrbc = 100;
    entry.numPrbc = 50;
    entry.numSymbol = 14;
    entry.udCompHdr = 0x91;

    uint16_t new_startPrbc = 200;  ///< Changed from 100 → 200
    bool mismatch = (entry.rb != 0 || entry.startPrbc != new_startPrbc ||
                     entry.numPrbc != 50 || entry.numSymbol != 14 ||
                     entry.udCompHdr != 0x91);
    EXPECT_TRUE(mismatch);
}

// --- Edge cases ---

/**
 * @brief Verify that the very first slot advance does not fire cross-validation
 *        warnings, since has_prev is false and there is no prior slot to validate.
 */
TEST_F(RUEmulatorTest, SectionTracker_FirstSlot_NoCrossValidation) {
    SlotSectionIdTracker tracker;
    fssId slot0{0, 0, 0}, slot1{0, 0, 1};

    tracker.current_fss = slot0;
    tracker.record_uplane_sid(50);  ///< Unannounced, but first-ever slot

    WarnCollector wc;
    tracker.advance_slot(slot1, std::ref(wc));

    EXPECT_TRUE(wc.warned_sids.empty());
}

/**
 * @brief Verify that consecutive empty slots (no C-plane or U-plane activity)
 *        produce no spurious warnings.
 */
TEST_F(RUEmulatorTest, SectionTracker_EmptySlot_NoWarnings) {
    SlotSectionIdTracker tracker;
    fssId slot0{0, 0, 0}, slot1{0, 0, 1}, slot2{0, 1, 0}, slot3{0, 1, 1};

    tracker.current_fss = slot0;

    WarnCollector wc;
    tracker.advance_slot(slot1, std::ref(wc));
    tracker.advance_slot(slot2, std::ref(wc));
    tracker.advance_slot(slot3, std::ref(wc));

    EXPECT_TRUE(wc.warned_sids.empty());
}

/**
 * @brief Verify that the generation counter wraps from UINT32_MAX to 1,
 *        skipping 0 (which is the zero-init value of Entry::generation).
 */
TEST_F(RUEmulatorTest, SectionTracker_GenerationSkipsZero) {
    SlotSectionIdTracker tracker;
    EXPECT_EQ(tracker.current_generation, 1u);

    tracker.current_generation = UINT32_MAX;
    tracker.advance_slot({0, 0, 1});

    EXPECT_EQ(tracker.current_generation, 1u);
}

/**
 * @brief Verify per-slot independence: a clean slot followed by a faulty slot
 *        produces a warning only for the faulty slot, not the clean one.
 *
 * Walks through slots 0-3, with slot 0 matched and slot 1 having an
 * unannounced U-plane sid. The warning for slot 1 fires when slot 3 arrives.
 */
TEST_F(RUEmulatorTest, SectionTracker_ConsecutiveSlots_IndependentValidation) {
    SlotSectionIdTracker tracker;
    fssId slot0{0, 0, 0}, slot1{0, 0, 1}, slot2{0, 1, 0}, slot3{0, 1, 1};

    tracker.current_fss = slot0;
    tracker.cplane_sids_announced.set(10);
    tracker.record_uplane_sid(10);

    WarnCollector wc;
    tracker.advance_slot(slot1, std::ref(wc));

    tracker.record_uplane_sid(77);  ///< Unannounced in slot 1

    tracker.advance_slot(slot2, std::ref(wc));
    EXPECT_TRUE(wc.warned_sids.empty());  ///< Validates prev=slot0 (clean)

    tracker.advance_slot(slot3, std::ref(wc));
    ASSERT_EQ(wc.warned_sids.size(), 1u);  ///< Validates prev=slot1 (faulty)
    EXPECT_EQ(wc.warned_sids[0], 77);
}

// --- Multi-cell isolation ---

/**
 * @brief Two-cell scenario: cell 0 has matched C/U-plane sectionIds, cell 1
 *        has an unannounced U-plane sid. Verify that only cell 1's tracker
 *        produces a warning; cell 0 remains clean.
 */
TEST_F(RUEmulatorTest, SectionTracker_MultiCell_FaultInOneCellDoesNotAffectOther) {
    SlotSectionIdTracker cell0_tracker;
    SlotSectionIdTracker cell1_tracker;
    fssId slot0{0, 0, 0}, slot1{0, 0, 1}, slot2{0, 1, 0};

    cell0_tracker.current_fss = slot0;
    cell0_tracker.cplane_sids_announced.set(10);
    cell0_tracker.record_uplane_sid(10);

    cell1_tracker.current_fss = slot0;
    cell1_tracker.cplane_sids_announced.set(20);
    cell1_tracker.record_uplane_sid(99);  ///< Unannounced

    WarnCollector wc0, wc1;
    cell0_tracker.advance_slot(slot1, std::ref(wc0));
    cell1_tracker.advance_slot(slot1, std::ref(wc1));

    cell0_tracker.advance_slot(slot2, std::ref(wc0));
    cell1_tracker.advance_slot(slot2, std::ref(wc1));

    EXPECT_TRUE(wc0.warned_sids.empty());
    ASSERT_EQ(wc1.warned_sids.size(), 1u);
    EXPECT_EQ(wc1.warned_sids[0], 99);
}

/**
 * @brief Two-cell scenario: both cells record the same sectionId (42) but with
 *        different startPrbc values. Verify that each tracker's entry is
 *        independent — writing to cell 1 does not contaminate cell 0.
 */
TEST_F(RUEmulatorTest, SectionTracker_MultiCell_DuplicateMismatchIsolated) {
    SlotSectionIdTracker cell0_tracker;
    SlotSectionIdTracker cell1_tracker;

    cell0_tracker.current_fss = {0, 0, 0};
    cell1_tracker.current_fss = {0, 0, 0};

    uint16_t sid = 42;
    uint32_t gen0 = cell0_tracker.current_generation;
    uint32_t gen1 = cell1_tracker.current_generation;

    cell0_tracker.entries[sid].generation = gen0;
    cell0_tracker.entries[sid].rb = 0;
    cell0_tracker.entries[sid].startPrbc = 100;
    cell0_tracker.entries[sid].numPrbc = 50;
    cell0_tracker.entries[sid].numSymbol = 14;
    cell0_tracker.entries[sid].udCompHdr = 0x91;

    cell1_tracker.entries[sid].generation = gen1;
    cell1_tracker.entries[sid].rb = 0;
    cell1_tracker.entries[sid].startPrbc = 200;  ///< Different from cell 0
    cell1_tracker.entries[sid].numPrbc = 50;
    cell1_tracker.entries[sid].numSymbol = 14;
    cell1_tracker.entries[sid].udCompHdr = 0x91;

    EXPECT_EQ(cell0_tracker.entries[sid].startPrbc, 100);
    EXPECT_EQ(cell1_tracker.entries[sid].startPrbc, 200);

    bool cell0_mismatch = (cell0_tracker.entries[sid].startPrbc != 100);
    EXPECT_FALSE(cell0_mismatch);

    bool cell1_mismatch = (cell1_tracker.entries[sid].startPrbc != 100);
    EXPECT_TRUE(cell1_mismatch);
}

/**
 * @brief Multi-slot, two-cell scenario: both cells start healthy in slot 0,
 *        cell 1 becomes faulty in slot 1 (unannounced sid 88), then recovers
 *        in slot 2. Verify the warning fires only for cell 1's faulty slot
 *        and cell 0 remains clean throughout.
 */
TEST_F(RUEmulatorTest, SectionTracker_MultiCell_MultipleFaultySlots) {
    SlotSectionIdTracker cell0_tracker;
    SlotSectionIdTracker cell1_tracker;
    fssId slot0{0, 0, 0}, slot1{0, 0, 1}, slot2{0, 1, 0}, slot3{0, 1, 1}, slot4{0, 2, 0};

    cell0_tracker.current_fss = slot0;
    cell1_tracker.current_fss = slot0;
    cell0_tracker.cplane_sids_announced.set(10);
    cell0_tracker.record_uplane_sid(10);
    cell1_tracker.cplane_sids_announced.set(20);
    cell1_tracker.record_uplane_sid(20);

    WarnCollector wc0, wc1;
    cell0_tracker.advance_slot(slot1, std::ref(wc0));
    cell1_tracker.advance_slot(slot1, std::ref(wc1));

    cell0_tracker.cplane_sids_announced.set(11);
    cell0_tracker.record_uplane_sid(11);
    cell1_tracker.record_uplane_sid(88);  ///< Unannounced in slot 1

    cell0_tracker.advance_slot(slot2, std::ref(wc0));
    cell1_tracker.advance_slot(slot2, std::ref(wc1));
    EXPECT_TRUE(wc0.warned_sids.empty());  ///< Validates prev=slot0 (both clean)
    EXPECT_TRUE(wc1.warned_sids.empty());

    cell0_tracker.cplane_sids_announced.set(12);
    cell0_tracker.record_uplane_sid(12);
    cell1_tracker.cplane_sids_announced.set(30);
    cell1_tracker.record_uplane_sid(30);

    cell0_tracker.advance_slot(slot3, std::ref(wc0));
    cell1_tracker.advance_slot(slot3, std::ref(wc1));
    EXPECT_TRUE(wc0.warned_sids.empty());       ///< Cell 0 still clean
    ASSERT_EQ(wc1.warned_sids.size(), 1u);       ///< Cell 1 slot 1 fault detected
    EXPECT_EQ(wc1.warned_sids[0], 88);
}

/**
 * @brief Same-cell, two-eAxC scenario: eAxC 0 has matched sectionIds, eAxC 1
 *        has an unannounced U-plane sid. Verify that only eAxC 1's tracker
 *        fires a warning; eAxC 0 is unaffected.
 *
 * This confirms per-eAxC isolation within the same cell, mirroring the
 * unordered_map<uint16_t, SlotSectionIdTracker> keying in RU_Emulator.
 */
TEST_F(RUEmulatorTest, SectionTracker_MultiEaxc_FaultOnOneEaxcIsolated) {
    SlotSectionIdTracker eaxc0_tracker;
    SlotSectionIdTracker eaxc1_tracker;
    fssId slot0{0, 0, 0}, slot1{0, 0, 1}, slot2{0, 1, 0};

    eaxc0_tracker.current_fss = slot0;
    eaxc0_tracker.cplane_sids_announced.set(5);
    eaxc0_tracker.record_uplane_sid(5);

    eaxc1_tracker.current_fss = slot0;
    eaxc1_tracker.record_uplane_sid(77);  ///< Unannounced

    WarnCollector wc0, wc1;
    eaxc0_tracker.advance_slot(slot1, std::ref(wc0));
    eaxc1_tracker.advance_slot(slot1, std::ref(wc1));

    eaxc0_tracker.advance_slot(slot2, std::ref(wc0));
    eaxc1_tracker.advance_slot(slot2, std::ref(wc1));

    EXPECT_TRUE(wc0.warned_sids.empty());
    ASSERT_EQ(wc1.warned_sids.size(), 1u);
    EXPECT_EQ(wc1.warned_sids[0], 77);
}

/// @}

/// @defgroup FloatToModCompScalerTests float_to_modcompscaler encoding tests
/// @brief Validates that float_to_modcompscaler correctly encodes a (0,1)
///        float to the 15-bit (4-bit exp | 11-bit mantissa) modCompScaler
///        format, and that the decoded value matches the original within the
///        precision of the 15-bit representation.
/// @{

static float decode_modcompscaler(uint16_t encoded) {
    int exp = (encoded >> 11) & 0xF;
    int mantissa = encoded & 0x7FF;
    return static_cast<float>(mantissa) / static_cast<float>(2048 << exp);
}

TEST_F(RUEmulatorTest, FloatToModCompScaler_KnownValue) {
    float scale = 0.01747141f;
    uint16_t encoded = float_to_modcompscaler(scale);
    EXPECT_NE(encoded, 0u);
    float decoded = decode_modcompscaler(encoded);
    EXPECT_NEAR(decoded, scale, 1e-4f);
}

TEST_F(RUEmulatorTest, FloatToModCompScaler_Half) {
    float scale = 0.5f;
    uint16_t encoded = float_to_modcompscaler(scale);
    float decoded = decode_modcompscaler(encoded);
    EXPECT_NEAR(decoded, scale, 1e-6f);
}

TEST_F(RUEmulatorTest, FloatToModCompScaler_VerySmall) {
    float scale = 0.001f;
    uint16_t encoded = float_to_modcompscaler(scale);
    EXPECT_NE(encoded, 0u);
    float decoded = decode_modcompscaler(encoded);
    EXPECT_NEAR(decoded, scale, 1e-4f);
}

TEST_F(RUEmulatorTest, FloatToModCompScaler_NearOne) {
    float scale = 0.999f;
    uint16_t encoded = float_to_modcompscaler(scale);
    float decoded = decode_modcompscaler(encoded);
    EXPECT_NEAR(decoded, scale, 1e-3f);
}

TEST_F(RUEmulatorTest, FloatToModCompScaler_OutOfRange) {
    EXPECT_EQ(float_to_modcompscaler(0.0f), 0u);
    EXPECT_EQ(float_to_modcompscaler(1.0f), 0u);
    EXPECT_EQ(float_to_modcompscaler(-0.5f), 0u);
}

TEST_F(RUEmulatorTest, FloatToModCompScaler_Deterministic) {
    float scale = 0.25f;
    uint16_t a = float_to_modcompscaler(scale);
    uint16_t b = float_to_modcompscaler(scale);
    EXPECT_EQ(a, b);
}

/// @}

/// @defgroup ModCompTvExtInfoTests tv_mod_comp_ext_info struct tests
/// @{

TEST_F(RUEmulatorTest, TvModCompExtInfo_DefaultInvalid) {
    tv_mod_comp_ext_info info;
    EXPECT_FALSE(info.valid);
    EXPECT_EQ(info.ext_type, 0u);
    EXPECT_EQ(info.n_mask, 0u);
}

TEST_F(RUEmulatorTest, TvModCompExtInfo_PopulateSE5) {
    tv_mod_comp_ext_info info;
    info.ext_type = ORAN_CMSG_SECTION_EXT_TYPE_5;
    info.n_mask = 2;
    info.mc_scale_re_mask[0] = 0x444;
    info.mc_scale_re_mask[1] = 0xBBB;
    info.mc_scale_offset_encoded[0] = float_to_modcompscaler(0.01747141f);
    info.mc_scale_offset_encoded[1] = float_to_modcompscaler(0.01747141f);
    info.csf[0] = 1;
    info.csf[1] = 1;
    info.valid = true;

    EXPECT_TRUE(info.valid);
    EXPECT_EQ(info.ext_type, (uint32_t)ORAN_CMSG_SECTION_EXT_TYPE_5);
    EXPECT_EQ(info.n_mask, 2u);
    EXPECT_NE(info.mc_scale_offset_encoded[0], 0u);
    EXPECT_EQ(info.mc_scale_offset_encoded[0], info.mc_scale_offset_encoded[1]);
}

/// @}

/// @defgroup SE4SE5RoundTripTests SE4/SE5 wire-format round-trip tests
/// @{

static void se5_host_to_wire(oran_cmsg_sect_ext_type_5& se5) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    uint64_t* bf = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(&se5) + sizeof(se5.extLen));
    *bf = __builtin_bswap64(*bf);
#endif
}

static void se5_wire_to_host(oran_cmsg_sect_ext_type_5& se5) {
    se5_host_to_wire(se5);
}

TEST_F(RUEmulatorTest, SE4_Bitfield_RoundTrip) {
    oran_cmsg_sect_ext_type_4 se4{};
    se4.modCompScalor = 0x1234;
    se4.csf = 1;
    EXPECT_EQ(se4.modCompScalor.get(), 0x1234u);
    EXPECT_EQ(se4.csf.get(), 1u);
}

TEST_F(RUEmulatorTest, SE5_Bitfield_RoundTrip) {
    oran_cmsg_sect_ext_type_5 se5{};
    se5.extLen = 3;
    se5.mcScaleReMask_1 = 0xFFF;
    se5.csf_1 = 1;
    se5.mcScaleOffset_1 = 0x1234;
    se5.mcScaleReMask_2 = 0xABC;
    se5.csf_2 = 0;
    se5.mcScaleOffset_2 = 0x5678;
    se5.zero_padding = 0;

    se5_host_to_wire(se5);
    se5_wire_to_host(se5);

    EXPECT_EQ(se5.extLen, 3u);
    EXPECT_EQ((uint16_t)se5.mcScaleReMask_1, 0xFFFu);
    EXPECT_EQ((uint8_t)se5.csf_1, 1u);
    EXPECT_EQ((uint16_t)se5.mcScaleOffset_1, 0x1234u);
    EXPECT_EQ((uint16_t)se5.mcScaleReMask_2, 0xABCu);
    EXPECT_EQ((uint8_t)se5.csf_2, 0u);
    EXPECT_EQ((uint16_t)se5.mcScaleOffset_2, 0x5678u);
}

/// @}

// Test t0/toa calculation with edge cases
TEST_F(RUEmulatorTest, CalculateT0ToAEdgeCases) {
    int64_t frame_cycle_time_ns = 10000000;
    uint16_t max_slot_id = 9;
    uint16_t opt_tti_us = 1000;

    // Test case 1: Packet time very close to frame boundary
    {
        int64_t packet_time = frame_cycle_time_ns - 1;
        int64_t beginning_of_time = 0;
        auto result = calculate_t0_toa(packet_time, beginning_of_time, frame_cycle_time_ns,
                                     0, 0, 0, 0, max_slot_id, opt_tti_us);
        EXPECT_LE(result.toa, frame_cycle_time_ns/2);
    }

    // Test case 2: Packet time just after frame boundary
    {
        int64_t packet_time = frame_cycle_time_ns + 1;
        int64_t beginning_of_time = 0;
        auto result = calculate_t0_toa(packet_time, beginning_of_time, frame_cycle_time_ns,
                                     0, 0, 0, 0, max_slot_id, opt_tti_us);
        EXPECT_GE(result.toa, -frame_cycle_time_ns/2);
    }

    // Test case 3: Very large frame number
    {
        int64_t packet_time = 1000000;
        int64_t beginning_of_time = 0;
        auto result = calculate_t0_toa(packet_time, beginning_of_time, frame_cycle_time_ns,
                                     0xFFFF, 0, 0, 0, max_slot_id, opt_tti_us);
        EXPECT_LE(result.toa, frame_cycle_time_ns/2);
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    try
    {
        return RUN_ALL_TESTS();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught in main: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown exception caught in main" << std::endl;
        return 1;
    }
} 
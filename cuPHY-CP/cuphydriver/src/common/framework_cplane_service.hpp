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

#ifndef FRAMEWORK_CPLANE_SERVICE_HPP
#define FRAMEWORK_CPLANE_SERVICE_HPP

#include <concepts>
#include <memory>
#include <optional>
#include <cstdint>
#include <cstddef>
#include <vector>

class PhyDriverCtx;
class Cell;
class SlotMapDl;
class SlotMapUl;
struct UplaneConversionParams;
struct PartialUplaneSlotInfo;
struct direct_bfw_cvi_record;
using PartialUplaneSlotInfo_t = struct PartialUplaneSlotInfo;

namespace nv { struct phy_mac_msg_desc; }

class FrameworkCPlaneService {
public:
    struct Config { bool enabled{false}; };

    /**
     * Construct the service with the given configuration.
     *
     * No framework resources are created until init() is called.
     *
     * @param[in] cfg Service configuration; cfg.enabled=false makes init() a no-op.
     */
    explicit FrameworkCPlaneService(Config cfg);

    /**
     * Destroy the service and release CPlaneGenerator and CplaneSender resources.
     */
    ~FrameworkCPlaneService();

    FrameworkCPlaneService(const FrameworkCPlaneService&) = delete;
    FrameworkCPlaneService& operator=(const FrameworkCPlaneService&) = delete;

    /**
     * Construct the framework CPlaneGenerator and per-NIC per-cell CplaneSenders.
     *
     * Populates the phy_id -> cell-index map and, on success, marks the service
     * active. If Config::enabled is false this is a no-op that returns success.
     *
     * @param[in] pdctx             PHY driver context used for NIC/FhProxy access.
     * @param[in] cells             Cells to configure; must be non-empty when enabled.
     * @param[in] bf_enabled        Whether beamforming is enabled for PortMaskConfig.
     * @param[in] precoding_enabled Whether precoding is enabled for PortMaskConfig.
     * @return 0 on success, -1 on failure (empty cell list, invalid SSB, or sender creation failure).
     */
    [[nodiscard]] int init(PhyDriverCtx* pdctx, const std::vector<Cell*>& cells,
                           bool bf_enabled, bool precoding_enabled);

    /**
     * Report whether init() succeeded and the CPlaneGenerator is live.
     *
     * @return true if the service is initialised and active; false otherwise.
     */
    [[nodiscard]] bool is_active() const;

    /**
     * Generate and transmit DL C-plane packets for the cell referenced by dl_tti / ul_dci.
     *
     * Drives ran::oran::CPlaneGenerator::prepare_dl + generate_dl_packets over the
     * cell's CplaneSender. Also updates SlotMapDl timing fields and, when a DL
     * buffer is present, primes PartialUplaneSlotInfo via convert_dl_cplane_to_uplane().
     *
     * @param[in]     dl_tti      Stored DL_TTI.request (may be null if ul_dci provides cell_id).
     * @param[in]     ul_dci      Stored UL_DCI.request (may be null).
     * @param[in]     transaction_id C-plane generator transaction id.
     * @param[in,out] slot_map_dl DL slot map; timing and partial-uplane state are updated in place.
     * @return 0 on success, 1 if there is no DL body to send (no-op), -1 on error.
     */
    [[nodiscard]] int send_dl_cplane(const nv::phy_mac_msg_desc* dl_tti,
                                     const nv::phy_mac_msg_desc* ul_dci,
                                     std::size_t transaction_id,
                                     SlotMapDl* slot_map_dl);

    /**
     * Generate and transmit UL C-plane packets for the cell referenced by ul_tti.
     *
     * @param[in]     ul_tti      Stored UL_TTI.request; must be non-null.
     * @param[in]     transaction_id C-plane generator transaction id.
     * @param[in,out] slot_map_ul UL slot map; timing fields are updated in place.
     * @return 0 on success, 1 if the UL_TTI body is absent (no-op), -1 on error.
     */
    [[nodiscard]] int send_ul_cplane(const nv::phy_mac_msg_desc* ul_tti,
                                     std::size_t transaction_id,
                                     SlotMapUl* slot_map_ul);

    /**
     * Convert a generated DL C-plane section layout into PartialUplaneSlotInfo.
     *
     * @param[in]  transaction_id C-plane generator transaction id.
     * @param[in]  p   Conversion parameters prepared by setupUPlaneGpuComm.
     * @param[out] out Partial U-plane slot info to populate.
     * @return 0 on success, -1 if the generator is absent or conversion fails.
     */
    [[nodiscard]] int convert_dl_cplane_to_uplane(std::size_t transaction_id,
                                                   const UplaneConversionParams& p,
                                                   PartialUplaneSlotInfo_t& out);

    /**
     * Record one prior-slot direct BFW CVI completion into the per-cell store.
     *
     * Producer entry point (slot N-1). The matching records are consumed by
     * send_dl_cplane / send_ul_cplane in slot N. The service owns the store; this
     * is the only mutation path (DOP / Tell-Don't-Ask).
     *
     * @param[in] cell_id SDK cell index the record belongs to.
     * @param[in] record  Producer-built BFW completion record.
     * @return 0 on success, -1 on validation failure or unknown cell.
     */
    [[nodiscard]] int record_bfw_cvi(uint16_t cell_id, const direct_bfw_cvi_record& record);

    /**
     * Look up the CPlaneGenerator cell index corresponding to a PHY cell id.
     *
     * @param[in] phy_id PHY cell id (Cell::getIdx()).
     * @return The cell index registered in init(), or std::nullopt if unknown.
     */
    [[nodiscard]] std::optional<std::size_t> resolve_cell_index(uint32_t phy_id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * Compile-time interface contract for FrameworkCPlaneService implementations.
 *
 * Both the real and stub translation units static_assert against this concept so
 * signature drift between the two is caught at compile time. See the requires
 * clause below for the exact required expressions and return types.
 *
 * @tparam T Candidate service type to check for conformance.
 */
template <typename T>
concept FrameworkCPlaneServiceContract = requires(
    T svc,
    const T csvc,
    PhyDriverCtx* pdctx,
    const std::vector<Cell*>& cells,
    bool flag,
    const nv::phy_mac_msg_desc* msg,
    SlotMapDl* smdl,
    SlotMapUl* smul,
    const UplaneConversionParams& ucp,
    PartialUplaneSlotInfo_t& pusi,
    uint32_t phy_id,
    const direct_bfw_cvi_record& bfw_rec)
{
    { svc.init(pdctx, cells, flag, flag) } -> std::same_as<int>;
    { csvc.is_active() } -> std::same_as<bool>;
    { svc.send_dl_cplane(msg, msg, std::size_t{0}, smdl) } -> std::same_as<int>;
    { svc.send_ul_cplane(msg, std::size_t{0}, smul) } -> std::same_as<int>;
    { svc.convert_dl_cplane_to_uplane(std::size_t{0}, ucp, pusi) } -> std::same_as<int>;
    { csvc.resolve_cell_index(phy_id) } -> std::same_as<std::optional<std::size_t>>;
    { svc.record_bfw_cvi(uint16_t{0}, bfw_rec) } -> std::same_as<int>;
};

static_assert(FrameworkCPlaneServiceContract<FrameworkCPlaneService>);

#endif // FRAMEWORK_CPLANE_SERVICE_HPP

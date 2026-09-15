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

#include "framework_cplane_service.hpp"

struct FrameworkCPlaneService::Impl {};

FrameworkCPlaneService::FrameworkCPlaneService(Config) : impl_(nullptr) {}
FrameworkCPlaneService::~FrameworkCPlaneService() = default;

int  FrameworkCPlaneService::init(PhyDriverCtx*, const std::vector<Cell*>&, bool, bool) { return 0; }
bool FrameworkCPlaneService::is_active() const { return false; }

int  FrameworkCPlaneService::send_dl_cplane(const nv::phy_mac_msg_desc*, const nv::phy_mac_msg_desc*, std::size_t, SlotMapDl*) { return 0; }
int  FrameworkCPlaneService::send_ul_cplane(const nv::phy_mac_msg_desc*, std::size_t, SlotMapUl*) { return 0; }
int  FrameworkCPlaneService::convert_dl_cplane_to_uplane(std::size_t, const UplaneConversionParams&, PartialUplaneSlotInfo_t&) { return 0; }

std::optional<std::size_t> FrameworkCPlaneService::resolve_cell_index(uint32_t) const { return std::nullopt; }
int  FrameworkCPlaneService::record_bfw_cvi(uint16_t, const direct_bfw_cvi_record&) { return 0; }

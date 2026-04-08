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

#if !defined(NV_PHY_GROUP_HPP_INCLUDED_)
#define NV_PHY_GROUP_HPP_INCLUDED_

#include "nv_phy_module.hpp"
#include <vector>
#include <stdio.h>
#include "yaml.hpp"

namespace nv
{

////////////////////////////////////////////////////////////////////////
// PHY_group
class PHY_group
{
public:
    //------------------------------------------------------------------
    // PHY_group()
    // Constructor
    PHY_group(const char* configFile)
    {
        yaml::file_parser fp(configFile);
        // Each document in the configuration file corresponds to an
        // L1 module
        yaml::document doc = fp.next_document();
        while(doc.is_valid())
        {
            // The document root is used to initialize the L1 module
            modules_.push_back(std::move(PHY_module(doc.root())));
            doc = fp.next_document();
        }
    }
    //------------------------------------------------------------------
    // modules()
    // Vector of PHY modules
    std::vector<PHY_module>& modules() { return modules_; }
    
    //------------------------------------------------------------------
    // join()
    // Block until all modules run to completion
    void join()
    {
        for(auto& m : modules_) { m.join(); }
    }
    //------------------------------------------------------------------
    // start()
    // Call the start() function on all contained modules
    void start()
    {
        for(auto& m : modules_) { m.start(); }
    }
    //------------------------------------------------------------------
    // stop()
    // Call the stop() function on all contained modules
    void stop()
    {
        for(auto& m : modules_) { m.stop(); }
    }
private:
    //------------------------------------------------------------------
    // Data
    std::vector<PHY_module> modules_;
};

}

#endif // !defined(NV_PHY_GROUP_HPP_INCLUDED_)

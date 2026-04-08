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

#include <string_view>
#include "ru_emulator.hpp"

// Forward declaration
void usage(std::string prog);

/**
 * Main execution function with exception handling
 *
 * This function orchestrates the RU Emulator lifecycle:
 * 1. Initialization and configuration parsing
 * 2. Test vector loading and flow setup
 * 3. Worker thread creation and execution
 * 4. Finalization and cleanup
 *
 * @param[in,out] re RU Emulator instance
 * @param[in] argc Command line argument count
 * @param[in] argv Command line argument vector
 * @return EXIT_SUCCESS on successful execution, EXIT_FAILURE on error
 */
int main_throw(RU_Emulator& re, int argc, char ** argv)
{
    pthread_t bg_thread_id;
    try {
        // Initialize RU Emulator: parse config, setup logging, initialize FH driver
        bg_thread_id = re.init(argc, argv);
        if(bg_thread_id == -1)
        {
            return EXIT_FAILURE;
        }
    } catch (std::exception& e) {
        re_err(AERIAL_RU_EMULATOR_EVENT, "Exception caught at initialization phase: {}", e.what());
        set_force_quit();
        return EXIT_FAILURE;
    } catch (...) {
        re_err(AERIAL_RU_EMULATOR_EVENT, "Uncaught exception");
        set_force_quit();
        return EXIT_FAILURE;
    }

    // Verify and apply configuration parameters
    re.verify_and_apply_configs();

    // Setup slot data structures for IQ samples
    re.setup_slots();

    // Load test vectors from HDF5 files
    re.load_tvs();

    // Add flow mappings for ORAN packet routing
    re.add_flows();
    // re.start_fh_driver(); //-> eal_init() // FH driver started in init()

    // Pre-compute UL TX cache (all dependencies ready after add_flows)
    re.precompute_ul_tx_cache();

    // Setup ring buffers for inter-thread communication
    re.setup_rings();

    // Print final configuration summary
    re.print_configs();

    // Initialize OAM (Operation, Administration, and Maintenance)
    re.oam_init();

    // Start worker threads and main processing loop
    re.start();

    // Allow time for graceful shutdown (1 second)
    usleep(1'000'000);

    // Finalize: stop threads, print statistics, cleanup resources
    re.finalize();

    // Allow time for log flushing
    usleep(1'000'000);

    // Close logging subsystem
    nvlog_fmtlog_close(bg_thread_id);

    return EXIT_SUCCESS;
}

//! Global RU Emulator instance
RU_Emulator re;

/**
 * Application entry point
 *
 * Wraps main_throw() with additional exception handling to ensure
 * graceful failure in case of unexpected exceptions during execution.
 *
 * @param[in] argc Command line argument count
 * @param[in] argv Command line argument vector
 * @return EXIT_SUCCESS on successful execution, EXIT_FAILURE on error
 */
int main(int argc, char **argv)
{
    // Check for help flag first
    for(int i = 1; i < argc; i++)
    {
        const std::string_view arg(argv[i]);
        if(arg == "--help" || arg == "-h")
        {
            usage(std::string(argv[0]));
            return EXIT_SUCCESS;
        }
    }

    try {
        return main_throw(re, argc, argv);
    } catch (std::exception& e) {
        re_err(AERIAL_RU_EMULATOR_EVENT, "main() Exception caught: {}", e.what());
        set_force_quit();
        // re.finalize(); // Cannot call finalize if init failed
    } catch (...) {
        re_err(AERIAL_RU_EMULATOR_EVENT, "Uncaught exception");
        set_force_quit();
        // re.finalize(); // Cannot call finalize if init failed
    }
    return EXIT_FAILURE;
}

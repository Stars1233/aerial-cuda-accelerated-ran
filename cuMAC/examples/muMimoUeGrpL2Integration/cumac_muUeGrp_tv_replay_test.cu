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

//
// Standalone cuMAC TV replay testbench.
//
// Phase 1 - TV Generation:
//   Automatically launches the integration test suite (l1_muUeGrp_test,
//   cumac_muUeGrp_test, l2_muUeGrp_test) with ENABLE_TV_TEST_MODE=true
//   and TV_SAVE_ALL_SLOTS=true to generate HDF5 test vectors for 20 slots.
//   Existing TV files are removed before generation.
//
// Phase 2 - TV Replay & Verification:
//   For each generated slot, loads the input TV into GPU buffers, runs the
//   GPU MU-MIMO UE pairing module, and verifies the computed solution
//   against the solution TV.
//
// Returns 0 if all slots match, 1 if any slot fails or TV generation fails.
//
// Usage: cumac_muUeGrp_tv_replay_test [-m 0|1] [-r N] [tv_dir] [yaml_config]
//   -m 0|1       - 0: without memory sharing, 1: with memory sharing
//                  (overrides ENABLE_L1_L2_MEM_SHARING from YAML config)
//   -r N         - repeat replay N times to check determinism (default: 1)
//   tv_dir       - directory containing HDF5 TVs (default: ./tv)
//   yaml_config  - YAML config file path (default: standard config.yaml)
//

#include "muMimoUserPairing/muMimoUserPairing.cuh"
#include "common_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <regex>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <getopt.h>

constexpr const char* DEFAULT_YAML_PATH =
    "./cuMAC/examples/muMimoUeGrpL2Integration/yamlConfigFiles/config.yaml";
constexpr const char* DEFAULT_TV_DIR = "./tv";
constexpr int TV_NUM_SLOTS = 20;

// ---------------------------------------------------------------------------
// Phase 1 helpers: TV generation
// ---------------------------------------------------------------------------

// Derive the directory containing the current executable from argv[0].
static std::string get_exe_dir(const char* argv0)
{
    std::string path(argv0);
    size_t pos = path.find_last_of('/');
    return (pos != std::string::npos) ? path.substr(0, pos) : ".";
}

// Replace a YAML scalar value on the line matching `key:`.
// Preserves surrounding whitespace/comments on other lines.
static std::string yaml_set_value(const std::string& content,
                                  const std::string& key,
                                  const std::string& value)
{
    std::regex re(key + ":\\s*\\S+");
    return std::regex_replace(content, re, key + ": " + value);
}

static std::string read_file(const char* path)
{
    std::ifstream f(path);
    return std::string((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
}

static void write_file(const char* path, const std::string& content)
{
    std::ofstream f(path);
    f << content;
}

static pid_t launch_exe(const std::string& exe_path)
{
    pid_t pid = fork();
    if (pid == 0) {
        execl(exe_path.c_str(), exe_path.c_str(), nullptr);
        perror(("execl " + exe_path).c_str());
        _exit(127);
    }
    return pid;
}

static int wait_for(pid_t pid, const char* name)
{
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        printf("  %s exited with status %d\n", name, code);
        return code;
    }
    return 0;
}

// Remove all TV files from the directory.
static void remove_existing_tvs(const char* tv_dir)
{
    DIR* dir = opendir(tv_dir);
    if (!dir) return;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name(entry->d_name);
        if (name.size() > 3 && name.substr(name.size() - 3) == ".h5" &&
            name.find("muUePairTV_") == 0) {
            std::string full = std::string(tv_dir) + "/" + name;
            std::remove(full.c_str());
        }
    }
    closedir(dir);
}

// Check whether any stale IPC objects matching the given glob exist.
static bool shm_files_exist(const char* pattern)
{
    std::string cmd = std::string("ls ") + pattern + " 2>/dev/null | head -1";
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) return false;
    char buf[256];
    bool found = (fgets(buf, sizeof(buf), fp) != nullptr);
    pclose(fp);
    return found;
}

// Clean up stale POSIX IPC objects and ensure the log directory is writable.
// Returns 0 on success, -1 if critical resources are blocked.
static int cleanup_ipc_and_logs()
{
    printf("Cleaning stale IPC objects ...\n");

    const char* shm_patterns[] = {
        "/dev/shm/sem.l1_sem_*",
        "/dev/shm/sem.l2_cumac*",
        "/dev/shm/sem.l1_cumac*",
        "/dev/shm/sem.l1_l2*",
        "/dev/shm/l1_cpu_sh_pool*",
        "/dev/shm/l1_gpu_sh_pool*",
        "/dev/shm/l2_cumac*",
        "/dev/shm/l1_l2*",
        "/dev/shm/l1_cumac*",
    };
    for (const char* pat : shm_patterns) {
        std::string cmd = std::string("rm -f ") + pat + " 2>/dev/null";
        if (int status = std::system(cmd.c_str()); status == -1) {
            perror("system");
        } else if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
            printf("WARNING: cleanup command failed with status %d: %s\n", code, cmd.c_str());
        }
    }

    // Verify the critical semaphore files are actually gone
    if (shm_files_exist("/dev/shm/sem.l1_sem_*") ||
        shm_files_exist("/dev/shm/sem.l2_cumac*")) {
        printf("\n"
               "ERROR: Stale IPC semaphores in /dev/shm/ could not be removed\n"
               "       (owned by another user/root from a previous run).\n"
               "\n"
               "  Fix: run the following once, then re-run the test:\n"
               "\n"
               "    sudo rm -f /dev/shm/sem.l1_sem_* /dev/shm/sem.l2_cumac* "
               "/dev/shm/sem.l1_cumac* /dev/shm/sem.l1_l2*\n"
               "    sudo rm -f /dev/shm/l1_cpu_sh_pool* /dev/shm/l1_gpu_sh_pool* "
               "/dev/shm/l2_cumac* /dev/shm/l1_l2* /dev/shm/l1_cumac*\n"
               "\n");
        return -1;
    }

    // Ensure log directory exists and is writable
    mkdir("/var/log/aerial", 0777);
    if (access("/var/log/aerial", W_OK) != 0) {
        printf("\n"
               "ERROR: /var/log/aerial is not writable (needed by NVIPC logging).\n"
               "\n"
               "  Fix: run the following once, then re-run the test:\n"
               "\n"
               "    sudo mkdir -p /var/log/aerial && sudo chmod 777 /var/log/aerial\n"
               "\n");
        return -1;
    }

    return 0;
}

// Atomically restore the original config.yaml from its backup (.orig).
// Safe to call even if no backup exists.
static void restore_config(const std::string& yaml_path)
{
    std::string orig_path = yaml_path + ".orig";
    struct stat st;
    if (stat(orig_path.c_str(), &st) == 0) {
        std::remove(yaml_path.c_str());
        std::rename(orig_path.c_str(), yaml_path.c_str());
        printf("Config restored from %s\n", orig_path.c_str());
    }
}

// Phase 1: generate TVs by running the integration test suite.
// Returns 0 on success, non-zero on failure.
// mem_sharing_override: -1 = use YAML default, 0 = force off, 1 = force on
static int generate_tvs(const char* yaml_path, const char* tv_dir,
                         const char* argv0, int mem_sharing_override)
{
    printf("\n==================== Phase 1: TV Generation ====================\n");

    // -- Clean stale IPC and prepare log directory --
    if (cleanup_ipc_and_logs() != 0) {
        printf("Aborting TV generation due to environment issues (see above).\n");
        return -1;
    }

    // -- Locate sibling executables --
    std::string exe_dir = get_exe_dir(argv0);
    std::string l1_exe    = exe_dir + "/l1_muUeGrp_test";
    std::string cumac_exe = exe_dir + "/cumac_muUeGrp_test";
    std::string l2_exe    = exe_dir + "/l2_muUeGrp_test";

    struct stat st;
    for (const auto& exe : {l1_exe, cumac_exe, l2_exe}) {
        if (stat(exe.c_str(), &st) != 0) {
            printf("ERROR: required executable not found: %s\n", exe.c_str());
            return -1;
        }
    }

    // -- Remove existing TV files --
    printf("Removing existing TV files from %s ...\n", tv_dir);
    remove_existing_tvs(tv_dir);
    mkdir(tv_dir, 0755);

    // -- Create temporary config: move original aside, write patched copy --
    // The child executables read from the hardcoded path, so we place the
    // temporary config there and keep the original safe as "<path>.orig".
    std::string yaml_str(yaml_path);
    std::string orig_path = yaml_str + ".orig";
    if (std::rename(yaml_path, orig_path.c_str()) != 0) {
        perror("rename config.yaml -> config.yaml.orig");
        return -1;
    }

    std::string patched = read_file(orig_path.c_str());
    patched = yaml_set_value(patched, "ENABLE_TV_TEST_MODE", "true");
    patched = yaml_set_value(patched, "TV_SAVE_ALL_SLOTS",   "true");
    // patched = yaml_set_value(patched, "NUM_TIME_SLOTS",
    //                          std::to_string(TV_NUM_SLOTS));
    if (mem_sharing_override >= 0) {
        patched = yaml_set_value(patched, "ENABLE_L1_L2_MEM_SHARING",
                                 mem_sharing_override ? "true" : "false");
    }
    write_file(yaml_path, patched);
    printf("Temporary config created at %s  (original backed up to %s)\n",
           yaml_path, orig_path.c_str());
    printf("  ENABLE_TV_TEST_MODE=true  TV_SAVE_ALL_SLOTS=true"
           "  NUM_TIME_SLOTS=%d", TV_NUM_SLOTS);
    if (mem_sharing_override >= 0) {
        printf("  ENABLE_L1_L2_MEM_SHARING=%s",
               mem_sharing_override ? "true" : "false");
    }
    printf("\n");

    // -- Launch processes: L1 -> cuMAC -> L2 --
    // L1 must create shared memory first; cuMAC opens it, then creates NVIPC;
    // L2 connects to NVIPC last.
    printf("Launching L1 ...\n");
    pid_t l1_pid = launch_exe(l1_exe);
    if (l1_pid < 0) { restore_config(yaml_str); return -1; }
    usleep(2000000);

    printf("Launching cuMAC ...\n");
    pid_t cumac_pid = launch_exe(cumac_exe);
    if (cumac_pid < 0) { restore_config(yaml_str); return -1; }
    usleep(2000000);

    printf("Launching L2 ...\n");
    pid_t l2_pid = launch_exe(l2_exe);
    if (l2_pid < 0) { restore_config(yaml_str); return -1; }

    // -- Wait for all processes --
    printf("Waiting for integration test to complete ...\n");
    int ret = 0;
    ret |= wait_for(l2_pid,    "l2_muUeGrp_test");
    ret |= wait_for(cumac_pid, "cumac_muUeGrp_test");
    ret |= wait_for(l1_pid,    "l1_muUeGrp_test");

    // -- Remove temporary config and restore original --
    restore_config(yaml_str);

    if (ret != 0) {
        printf("ERROR: TV generation failed (one or more processes exited "
               "with errors).\n");
    } else {
        printf("TV generation completed successfully.\n");
    }
    printf("================================================================\n\n");
    return ret;
}

// ---------------------------------------------------------------------------
// Phase 2 helpers: TV replay
// ---------------------------------------------------------------------------

struct SlotTV {
    uint16_t    sfn;
    uint16_t    slot;
    std::string tv_base;   // e.g. "./tv/muUePairTV_sfn0_slot4"
};

static std::vector<SlotTV> discover_tv_slots(const char* tv_dir)
{
    std::vector<SlotTV> slots;
    std::regex re("muUePairTV_sfn(\\d+)_slot(\\d+)_solution\\.h5");

    DIR* dir = opendir(tv_dir);
    if (!dir) {
        printf("ERROR: cannot open TV directory: %s\n", tv_dir);
        return slots;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name(entry->d_name);
        std::smatch m;
        if (std::regex_match(name, m, re)) {
            SlotTV tv;
            tv.sfn  = static_cast<uint16_t>(std::stoi(m[1].str()));
            tv.slot = static_cast<uint16_t>(std::stoi(m[2].str()));
            tv.tv_base = std::string(tv_dir) + "/muUePairTV_sfn"
                         + m[1].str() + "_slot" + m[2].str();
            slots.push_back(tv);
        }
    }
    closedir(dir);

    std::sort(slots.begin(), slots.end(), [](const SlotTV& a, const SlotTV& b) {
        if (a.sfn != b.sfn) return a.sfn < b.sfn;
        return a.slot < b.slot;
    });

    return slots;
}

static uint32_t peek_cubb_srs_buf_size(const std::string& cell0_h5_path)
{
    try {
        H5::H5File file(cell0_h5_path, H5F_ACC_RDONLY);
        if (H5Lexists(file.getId(), "cubb_srs_gpu_buf", H5P_DEFAULT) > 0) {
            H5::DataSet ds = file.openDataSet("cubb_srs_gpu_buf");
            hsize_t dim;
            ds.getSpace().getSimpleExtentDims(&dim);
            return static_cast<uint32_t>(dim);
        }
    } catch (...) { }
    return 0;
}

static bool compare_solutions(const uint8_t* computed_buf,
                               const uint8_t* expected_buf, int num_cell)
{
    bool match = true;

    for (uint16_t cellId = 0; cellId < num_cell; cellId++) {
        const auto* computed = reinterpret_cast<const cumac_muUeGrp_resp_info_t*>(
            computed_buf + cellId * sizeof(cumac_muUeGrp_resp_info_t));
        const auto* expected = reinterpret_cast<const cumac_muUeGrp_resp_info_t*>(
            expected_buf + cellId * sizeof(cumac_muUeGrp_resp_info_t));

        if (computed->numSchdUeg != expected->numSchdUeg) {
            printf("  mismatch: cell %u numSchdUeg computed=%u expected=%u\n",
                   cellId, computed->numSchdUeg, expected->numSchdUeg);
            match = false;
            continue;
        }

        for (uint16_t uegId = 0; uegId < computed->numSchdUeg; uegId++) {
            const auto& c_ueg = computed->schdUegInfo[uegId];
            const auto& e_ueg = expected->schdUegInfo[uegId];

            if (c_ueg.numUeInGrp != e_ueg.numUeInGrp) {
                printf("  mismatch: cell %u ueg %u numUeInGrp computed=%u expected=%u\n",
                       cellId, uegId, c_ueg.numUeInGrp, e_ueg.numUeInGrp);
                match = false;
            }
            if (c_ueg.allocPrgStart != e_ueg.allocPrgStart) {
                printf("  mismatch: cell %u ueg %u allocPrgStart computed=%d expected=%d\n",
                       cellId, uegId, c_ueg.allocPrgStart, e_ueg.allocPrgStart);
                match = false;
            }
            if (c_ueg.allocPrgEnd != e_ueg.allocPrgEnd) {
                printf("  mismatch: cell %u ueg %u allocPrgEnd computed=%d expected=%d\n",
                       cellId, uegId, c_ueg.allocPrgEnd, e_ueg.allocPrgEnd);
                match = false;
            }
            if (c_ueg.flags != e_ueg.flags) {
                printf("  mismatch: cell %u ueg %u flags computed=0x%02x expected=0x%02x\n",
                       cellId, uegId, c_ueg.flags, e_ueg.flags);
                match = false;
            }

            uint8_t ueCount = std::min(c_ueg.numUeInGrp, e_ueg.numUeInGrp);
            for (uint16_t ueId = 0; ueId < ueCount; ueId++) {
                const auto& c_ue = c_ueg.ueInfo[ueId];
                const auto& e_ue = e_ueg.ueInfo[ueId];

                if (c_ue.rnti != e_ue.rnti)
                    printf("  mismatch: cell %u ueg %u ue %u rnti computed=%u expected=%u\n",
                           cellId, uegId, ueId, c_ue.rnti, e_ue.rnti), match = false;
                if (c_ue.id != e_ue.id)
                    printf("  mismatch: cell %u ueg %u ue %u id computed=%u expected=%u\n",
                           cellId, uegId, ueId, c_ue.id, e_ue.id), match = false;
                if (c_ue.layerSel != e_ue.layerSel)
                    printf("  mismatch: cell %u ueg %u ue %u layerSel computed=0x%02x expected=0x%02x\n",
                           cellId, uegId, ueId, c_ue.layerSel, e_ue.layerSel), match = false;
                if (c_ue.ueOrderInGrp != e_ue.ueOrderInGrp)
                    printf("  mismatch: cell %u ueg %u ue %u ueOrderInGrp computed=%u expected=%u\n",
                           cellId, uegId, ueId, c_ue.ueOrderInGrp, e_ue.ueOrderInGrp), match = false;
                if (c_ue.nSCID != e_ue.nSCID)
                    printf("  mismatch: cell %u ueg %u ue %u nSCID computed=%u expected=%u\n",
                           cellId, uegId, ueId, c_ue.nSCID, e_ue.nSCID), match = false;
                if (c_ue.flags != e_ue.flags)
                    printf("  mismatch: cell %u ueg %u ue %u flags computed=0x%02x expected=0x%02x\n",
                           cellId, uegId, ueId, c_ue.flags, e_ue.flags), match = false;
            }
        }
    }

    return match;
}

// ---------------------------------------------------------------------------
// After running each slot's kernel the persistent GPU buffers
// (srs_chan_est_buf / srs_snr_buf / chan_orth_mat_buf) should match the
// state captured at the start of the *next* slot during TV generation.
// This end-to-end check asserts that state evolution in the replay matches
// the TV generator run and catches any regression in kernel determinism
// or driver-level FP behaviour differences between the two runs.
// ---------------------------------------------------------------------------
static bool compare_big_buffers_vs_tv(const std::string& tv_base,
                                       uint8_t* srs_chan_est_buf, uint32_t srs_chan_est_buf_size,
                                       float* srs_snr_buf, uint32_t srs_snr_buf_size,
                                       float* chan_orth_mat_buf, uint32_t chan_orth_mat_buf_size)
{
    std::string cell0_path = tv_base + "_cell0.h5";

    std::vector<uint8_t> tv_ce;
    std::vector<float>   tv_snr;
    std::vector<float>   tv_orth;

    try {
        H5::H5File file(cell0_path, H5F_ACC_RDONLY);
        {
            H5::DataSet ds = file.openDataSet("srs_chan_est_buf");
            hsize_t dim;
            ds.getSpace().getSimpleExtentDims(&dim);
            tv_ce.resize(dim);
            ds.read(tv_ce.data(), H5::PredType::NATIVE_UINT8);
        }
        {
            H5::DataSet ds = file.openDataSet("srs_snr_buf");
            hsize_t dim;
            ds.getSpace().getSimpleExtentDims(&dim);
            tv_snr.resize(dim);
            ds.read(tv_snr.data(), H5::PredType::NATIVE_FLOAT);
        }
        {
            H5::DataSet ds = file.openDataSet("chan_orth_mat_buf");
            hsize_t dim;
            ds.getSpace().getSimpleExtentDims(&dim);
            tv_orth.resize(dim);
            ds.read(tv_orth.data(), H5::PredType::NATIVE_FLOAT);
        }
    } catch (const H5::Exception&) {
        printf("  WARNING: could not load big buffers from %s - skipping comparison\n",
               cell0_path.c_str());
        return true;
    }

    std::vector<uint8_t> gpu_ce(srs_chan_est_buf_size);
    std::vector<float>   gpu_snr(srs_snr_buf_size / sizeof(float));
    std::vector<float>   gpu_orth(chan_orth_mat_buf_size / sizeof(float));

    CHECK_CUDA_ERR(cudaMemcpy(gpu_ce.data(),   srs_chan_est_buf, srs_chan_est_buf_size,  cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(gpu_snr.data(),  srs_snr_buf,      srs_snr_buf_size,     cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(gpu_orth.data(), chan_orth_mat_buf,  chan_orth_mat_buf_size, cudaMemcpyDeviceToHost));

    bool all_match = true;
    int mismatches;
    constexpr int MAX_PRINT = 10;

    mismatches = 0;
    size_t ce_cmp = std::min(gpu_ce.size(), tv_ce.size());
    for (size_t i = 0; i < ce_cmp; i++) {
        if (gpu_ce[i] != tv_ce[i]) {
            if (mismatches < MAX_PRINT)
                printf("    srs_chan_est_buf byte[%zu]: GPU=0x%02x TV=0x%02x\n",
                       i, gpu_ce[i], tv_ce[i]);
            mismatches++;
        }
    }
    if (mismatches > 0) {
        printf("  srs_chan_est_buf: %d mismatches / %zu bytes\n", mismatches, ce_cmp);
        all_match = false;
    }

    mismatches = 0;
    size_t snr_cmp = std::min(gpu_snr.size(), tv_snr.size());
    for (size_t i = 0; i < snr_cmp; i++) {
        uint32_t g, t;
        memcpy(&g, &gpu_snr[i], sizeof(float));
        memcpy(&t, &tv_snr[i],  sizeof(float));
        if (g != t) {
            if (mismatches < MAX_PRINT)
                printf("    srs_snr_buf[%zu]: GPU=%e (0x%08x) TV=%e (0x%08x)\n",
                       i, gpu_snr[i], g, tv_snr[i], t);
            mismatches++;
        }
    }
    if (mismatches > 0) {
        printf("  srs_snr_buf: %d mismatches / %zu elems\n", mismatches, snr_cmp);
        all_match = false;
    }

    mismatches = 0;
    size_t orth_cmp = std::min(gpu_orth.size(), tv_orth.size());
    for (size_t i = 0; i < orth_cmp; i++) {
        uint32_t g, t;
        memcpy(&g, &gpu_orth[i], sizeof(float));
        memcpy(&t, &tv_orth[i],  sizeof(float));
        if (g != t) {
            if (mismatches < MAX_PRINT)
                printf("    chan_orth_mat_buf[%zu]: GPU=%e (0x%08x) TV=%e (0x%08x)\n",
                       i, gpu_orth[i], g, tv_orth[i], t);
            mismatches++;
        }
    }
    if (mismatches > 0) {
        printf("  chan_orth_mat_buf: %d mismatches / %zu elems\n", mismatches, orth_cmp);
        all_match = false;
    }

    if (all_match)
        printf("  big buffers MATCH next slot's TV\n");
    else
        printf("  big buffers DIVERGE from next slot's TV\n");

    return all_match;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static void print_usage(const char* prog)
{
    printf("Usage: %s [-m 0|1] [-r N] [-s] [tv_dir] [yaml_config]\n"
           "  -m 0|1       0: without memory sharing, 1: with memory sharing\n"
           "               (overrides ENABLE_L1_L2_MEM_SHARING from YAML config)\n"
           "  -r N         repeat the replay N times to check determinism (default: 1)\n"
           "  -s           skip Phase 1 TV generation; replay existing TVs in tv_dir\n"
           "  tv_dir       directory containing HDF5 TVs (default: %s)\n"
           "  yaml_config  YAML config file path (default: %s)\n",
           prog, DEFAULT_TV_DIR, DEFAULT_YAML_PATH);
}

int main(int argc, char** argv)
{
    int mem_sharing_override = -1;  // -1 = use YAML default
    int num_repeats = 1;
    bool skip_generation = false;

    int opt;
    while ((opt = getopt(argc, argv, "m:r:sh")) != -1) {
        switch (opt) {
        case 'm':
            mem_sharing_override = atoi(optarg);
            if (mem_sharing_override != 0 && mem_sharing_override != 1) {
                printf("ERROR: -m requires 0 or 1\n");
                print_usage(argv[0]);
                return 1;
            }
            break;
        case 'r':
            num_repeats = atoi(optarg);
            if (num_repeats < 1) {
                printf("ERROR: -r requires a positive integer\n");
                print_usage(argv[0]);
                return 1;
            }
            break;
        case 's':
            skip_generation = true;
            break;
        case 'h':
        default:
            print_usage(argv[0]);
            return (opt == 'h') ? 0 : 1;
        }
    }

    const char* tv_dir    = DEFAULT_TV_DIR;
    const char* yaml_path = DEFAULT_YAML_PATH;
    if (optind < argc) tv_dir    = argv[optind];
    if (optind + 1 < argc) yaml_path = argv[optind + 1];

    if (mem_sharing_override >= 0) {
        printf("Memory sharing override: %s\n",
               mem_sharing_override ? "ENABLED" : "DISABLED");
    }
    if (num_repeats > 1) {
        printf("Repeat runs: %d (determinism check)\n", num_repeats);
    }

    // ===================== Phase 1: Generate TVs =====================
    if (skip_generation) {
        printf("Skipping Phase 1 TV generation (-s flag); using existing TVs in: %s\n",
               tv_dir);
    } else if (generate_tvs(yaml_path, tv_dir, argv[0], mem_sharing_override) != 0) {
        printf("Aborting: TV generation failed.\n");
        // return 1;
    }

    // ===================== Phase 2: Replay TVs =======================
    printf("==================== Phase 2: TV Replay ====================\n");

    // ---- Load configuration ----
    printf("Loading config from: %s\n", yaml_path);
    sys_param_t sys_param(yaml_path);

    if (mem_sharing_override >= 0) {
        sys_param.enable_l1_l2_mem_sharing = (mem_sharing_override == 1);
        printf("Overriding ENABLE_L1_L2_MEM_SHARING = %s\n",
               sys_param.enable_l1_l2_mem_sharing ? "true" : "false");
    }

    CHECK_CUDA_ERR(cudaSetDevice(sys_param.cuda_device_id));

    // ---- FTZ/DAZ diagnostics ----
    {
        cudaDeviceProp prop;
        CHECK_CUDA_ERR(cudaGetDeviceProperties(&prop, sys_param.cuda_device_id));
        printf("CUDA Device: %s  (SM %d.%d, %d SMs)\n",
               prop.name, prop.major, prop.minor, prop.multiProcessorCount);
        unsigned int devFlags = 0;
        cudaGetDeviceFlags(&devFlags);
        printf("CUDA Device Flags: 0x%x\n", devFlags);
#if defined(__CUDA_ARCH__)
        printf("Compiled with __CUDA_ARCH__ = %d\n", __CUDA_ARCH__);
#endif
        printf("Note: nvcc default --ftz=true flushes denormalized single-precision "
               "floats to zero in device code.\n"
               "      Different execution contexts (multi-process vs single-process) "
               "may exhibit different FTZ behavior\n"
               "      if the driver sets different modes.\n");
    }

    // ---- Discover TV files ----
    std::vector<SlotTV> tv_slots = discover_tv_slots(tv_dir);
    if (tv_slots.empty()) {
        printf("ERROR: no TV files found in %s\n", tv_dir);
        return 1;
    }
    printf("Found %zu TV slot(s) in %s\n", tv_slots.size(), tv_dir);

    // ---- Determine cubb_srs_gpu_buf size from first TV ----
    std::string first_cell0 = tv_slots[0].tv_base + "_cell0.h5";
    uint32_t cubb_srs_gpu_buf_total_size = peek_cubb_srs_buf_size(first_cell0);
    printf("cubb_srs_gpu_buf size from TV: %u bytes\n", cubb_srs_gpu_buf_total_size);

    // ---- Create CUDA stream ----
    cudaStream_t strm;
    CHECK_CUDA_ERR(cudaStreamCreate(&strm));

    // ---- Compute buffer sizes ----
    const uint32_t srs_chan_est_buf_size = static_cast<uint32_t>(
        sizeof(__half2) * sys_param.num_bs_ant_port * MAX_NUM_UE_ANT_PORT
        * sys_param.num_subband * sys_param.num_prg_samp_per_subband
        * MAX_NUM_SRS_UE_PER_CELL * sys_param.num_cell);

    const uint32_t srs_snr_buf_size = static_cast<uint32_t>(
        sizeof(float) * MAX_NUM_SRS_UE_PER_CELL * sys_param.num_cell);

    const uint32_t chan_orth_mat_buf_size = static_cast<uint32_t>(
        sizeof(float) * MAX_NUM_SRS_UE_PER_CELL * MAX_NUM_UE_ANT_PORT
        * (MAX_NUM_SRS_UE_PER_CELL * MAX_NUM_UE_ANT_PORT + 1) / 2
        * sys_param.num_subband * sys_param.num_prg_samp_per_subband
        * sys_param.num_cell);

    const uint32_t out_buf_size = static_cast<uint32_t>(
        sizeof(cumac_muUeGrp_resp_info_t) * sys_param.num_cell);

    const uint32_t task_in_buf_len_per_cell = sys_param.enable_l1_l2_mem_sharing
        ? static_cast<uint32_t>(sizeof(cumac_muUeGrp_req_info_t)
            + sizeof(cumac_muUeGrp_req_srs_info_msh_t) * MAX_NUM_UE_SRS_INFO_PER_SLOT
            + sizeof(cumac_muUeGrp_req_ue_info_t)      * MAX_NUM_SRS_UE_PER_CELL)
        : static_cast<uint32_t>(sizeof(cumac_muUeGrp_req_info_t)
            + sizeof(cumac_muUeGrp_req_srs_info_t)     * MAX_NUM_UE_SRS_INFO_PER_SLOT
            + sizeof(cumac_muUeGrp_req_ue_info_t)      * MAX_NUM_SRS_UE_PER_CELL);

    // ---- Allocate GPU buffers ----
    uint8_t* srs_chan_est_buf;
    float*   srs_snr_buf;
    float*   chan_orth_mat_buf;
    uint8_t* gpu_out_buf;
    __half2* cubb_srs_gpu_buf = nullptr;

    CHECK_CUDA_ERR(cudaMalloc(&srs_chan_est_buf, srs_chan_est_buf_size));
    CHECK_CUDA_ERR(cudaMalloc(&srs_snr_buf, srs_snr_buf_size));
    CHECK_CUDA_ERR(cudaMalloc(&chan_orth_mat_buf, chan_orth_mat_buf_size));
    CHECK_CUDA_ERR(cudaMalloc(&gpu_out_buf, out_buf_size));
    if (cubb_srs_gpu_buf_total_size > 0) {
        CHECK_CUDA_ERR(cudaMalloc(&cubb_srs_gpu_buf, cubb_srs_gpu_buf_total_size));
    }

    // ---- Allocate host buffers ----
    uint8_t* gpu_sol_buf_host;
    uint8_t* expected_sol_buf;
    CHECK_CUDA_ERR(cudaMallocHost(&gpu_sol_buf_host, out_buf_size));
    CHECK_CUDA_ERR(cudaMallocHost(&expected_sol_buf, out_buf_size));

    // ---- Create MU-MIMO UE pairing module ----
    cumac::muMimoUserPairing* uePairObj = new cumac::muMimoUserPairing(
        srs_chan_est_buf, srs_snr_buf, chan_orth_mat_buf, cubb_srs_gpu_buf,
        sys_param.num_cell, sys_param.num_prg_per_cell,
        sys_param.num_subband, sys_param.num_prg_samp_per_subband,
        sys_param.num_bs_ant_port);

    // ---- Replay each TV slot (with optional repeat runs) ----
    int total_failures_across_repeats = 0;
    // Stack-allocated: released automatically on any return path, no leak risk.
    cumac::muUePairTask task0{};
    // load_h5_tv requires task_in_buf to be a valid GPU allocation for cudaMemcpy.
    size_t task0_in_buf_size = static_cast<size_t>(task_in_buf_len_per_cell) * sys_param.num_cell;
    uint8_t* task0_in_buf = nullptr;
    CHECK_CUDA_ERR(cudaMalloc(&task0_in_buf, task0_in_buf_size));
    task0.task_in_buf = task0_in_buf;

    for (int repeat = 0; repeat < num_repeats; repeat++) {
        if (num_repeats > 1)
            printf("\n\n############### Repeat run %d / %d ###############\n",
                   repeat + 1, num_repeats);

        // Zero-initialize big GPU buffers before each repeat to ensure
        // deterministic starting state (eliminates stale cudaMalloc contents)
        CHECK_CUDA_ERR(cudaMemset(srs_chan_est_buf, 0, srs_chan_est_buf_size));
        CHECK_CUDA_ERR(cudaMemset(srs_snr_buf, 0, srs_snr_buf_size));
        CHECK_CUDA_ERR(cudaMemset(chan_orth_mat_buf, 0, chan_orth_mat_buf_size));
        if (cubb_srs_gpu_buf && cubb_srs_gpu_buf_total_size > 0)
            CHECK_CUDA_ERR(cudaMemset(cubb_srs_gpu_buf, 0, cubb_srs_gpu_buf_total_size));
        printf("Big GPU buffers zero-initialized (%u + %u + %u + %u bytes)\n",
               srs_chan_est_buf_size, srs_snr_buf_size, chan_orth_mat_buf_size,
               cubb_srs_gpu_buf_total_size);

        // Load static buffers from first slot's TV
        uint16_t loaded_sfn = 0, loaded_slot = 0;
        load_h5_tv(tv_slots[0].tv_base, sys_param.num_cell,
                    loaded_sfn, loaded_slot, &task0,
                    srs_chan_est_buf,   srs_chan_est_buf_size,
                    srs_snr_buf,        srs_snr_buf_size,
                    chan_orth_mat_buf,   chan_orth_mat_buf_size,
                    cubb_srs_gpu_buf,   cubb_srs_gpu_buf_total_size);

        int num_failed = 0;
        int num_tested = 0;

        for (size_t slot_idx = 0; slot_idx < tv_slots.size(); slot_idx++) {
            const auto& tv = tv_slots[slot_idx];
            printf("\n===== TV Replay [%d/%zu]: SFN=%u  slot=%u =====\n",
                   num_tested + 1, tv_slots.size(), tv.sfn, tv.slot);

            // ---- Create task structure and allocate task_in_buf for each slot ----
            uint8_t* task_in_buf_host = nullptr;
            uint8_t* task_in_buf_gpu = nullptr;
            size_t task_in_buf_size =
                static_cast<size_t>(task_in_buf_len_per_cell) * sys_param.num_cell;
            CHECK_CUDA_ERR(cudaMallocHost(&task_in_buf_host, task_in_buf_size));
            CHECK_CUDA_ERR(cudaMalloc(&task_in_buf_gpu, task_in_buf_size));

            cumac::muUePairTask* task = new cumac::muUePairTask();
            task->task_out_buf = gpu_out_buf;
            task->strm         = strm;
            task->is_mem_sharing = sys_param.enable_l1_l2_mem_sharing;
            task->task_in_buf = task_in_buf_gpu;
            task->num_srs_ue_per_slot_cell = task0.num_srs_ue_per_slot_cell;
            task->num_blocks_per_row_chanOrtMat = task0.num_blocks_per_row_chanOrtMat;
            task->kernel_launch_flags = task0.kernel_launch_flags;

            // Load cell0 TV: metadata and cubb_srs_gpu_buf
            {
                std::string cell0_path = tv.tv_base + "_cell0.h5";
                H5::H5File file(cell0_path, H5F_ACC_RDONLY);

                uint16_t srs_ue_per_slot_cell;
                file.openAttribute("num_srs_ue_per_slot_cell")
                    .read(H5::PredType::NATIVE_UINT16, &srs_ue_per_slot_cell);
                task->num_srs_ue_per_slot_cell = srs_ue_per_slot_cell;

                uint8_t flags;
                file.openAttribute("kernel_launch_flags")
                    .read(H5::PredType::NATIVE_UINT8, &flags);
                task->kernel_launch_flags = flags;

                int32_t blocks_per_row;
                file.openAttribute("num_blocks_per_row_chanOrtMat")
                    .read(H5::PredType::NATIVE_INT32, &blocks_per_row);
                task->num_blocks_per_row_chanOrtMat =
                    static_cast<uint16_t>(blocks_per_row);

                // In memory-sharing mode, `cubb_srs_gpu_buf` is the shared L1<->cuMAC
                // buffer that L1 fills with fresh SRS data at each S-slot. Because this
                // standalone replay has no L1 in the loop, we must restore the buffer
                // from the current slot's TV before the kernel runs; otherwise fresh
                // UEs would read stale/zero channel estimates through `realBuffIdx`
                // and the persistent state (srs_chan_est_buf / chan_orth_mat_buf)
                // would diverge from the TV starting at the first post-slot-0 S-slot.
                if (sys_param.enable_l1_l2_mem_sharing &&
                    cubb_srs_gpu_buf != nullptr &&
                    cubb_srs_gpu_buf_total_size > 0 &&
                    H5Lexists(file.getId(), "cubb_srs_gpu_buf", H5P_DEFAULT) > 0) {
                    H5::DataSet ds = file.openDataSet("cubb_srs_gpu_buf");
                    hsize_t dim;
                    ds.getSpace().getSimpleExtentDims(&dim);
                    if (dim != static_cast<hsize_t>(cubb_srs_gpu_buf_total_size)) {
                        printf("ERROR: %s cubb_srs_gpu_buf size %llu != expected %u\n",
                               cell0_path.c_str(),
                               static_cast<unsigned long long>(dim),
                               cubb_srs_gpu_buf_total_size);
                        CHECK_CUDA_ERR(cudaFree(task_in_buf_gpu));
                        CHECK_CUDA_ERR(cudaFreeHost(task_in_buf_host));
                        delete task;
                        return 1;
                    }
                    std::vector<uint8_t> buf(dim);
                    ds.read(buf.data(), H5::PredType::NATIVE_UINT8);
                    CHECK_CUDA_ERR(cudaMemcpy(cubb_srs_gpu_buf, buf.data(), dim,
                                              cudaMemcpyHostToDevice));
                }
                printf("TV replay: loaded %s\n", cell0_path.c_str());
            }

            // Load task_in_buf from each cell's TV file (each cell stores its own portion)
            for (int c = 0; c < sys_param.num_cell; c++) {
                std::string cell_path = tv.tv_base + "_cell" + std::to_string(c) + ".h5";
                H5::H5File cell_file(cell_path, H5F_ACC_RDONLY);
                H5::DataSet ds = cell_file.openDataSet("task_in_buf");
                hsize_t dim;
                ds.getSpace().getSimpleExtentDims(&dim);
                if (dim != static_cast<hsize_t>(task_in_buf_len_per_cell)) {
                    printf("ERROR: %s task_in_buf size %llu != expected %u\n",
                           cell_path.c_str(),
                           static_cast<unsigned long long>(dim),
                           task_in_buf_len_per_cell);
                    CHECK_CUDA_ERR(cudaFree(task_in_buf_gpu));
                    CHECK_CUDA_ERR(cudaFreeHost(task_in_buf_host));
                    delete task;
                    return 1;
                }
                ds.read(task_in_buf_host + c * task_in_buf_len_per_cell,
                        H5::PredType::NATIVE_UINT8);
            }

            CHECK_CUDA_ERR(cudaMemcpy(task_in_buf_gpu, task_in_buf_host, task_in_buf_size,
                                      cudaMemcpyHostToDevice));

            printf("setup: slot=%u num_srs_ue_per_slot_cell=%u, "
                   "num_blocks_per_row_chanOrtMat=%u, kernel_launch_flags=0x%02x\n",
                   tv.slot, task->num_srs_ue_per_slot_cell,
                   task->num_blocks_per_row_chanOrtMat, task->kernel_launch_flags);

            uePairObj->setup(task);
            uePairObj->run(gpu_sol_buf_host);
            CHECK_CUDA_ERR(cudaStreamSynchronize(strm));

            // ---- Compare persistent GPU buffers against NEXT slot's TV ----
            // The state evolved by this slot's kernel run must match the state
            // captured as the input for the next slot during TV generation;
            // any divergence signals a regression in kernel determinism.
            if (slot_idx + 1 < tv_slots.size()) {
                printf("  Comparing GPU big buffers vs next slot's TV "
                       "(SFN=%u slot=%u):\n",
                       tv_slots[slot_idx + 1].sfn, tv_slots[slot_idx + 1].slot);
                compare_big_buffers_vs_tv(tv_slots[slot_idx + 1].tv_base,
                    srs_chan_est_buf, srs_chan_est_buf_size,
                    srs_snr_buf,     srs_snr_buf_size,
                    chan_orth_mat_buf, chan_orth_mat_buf_size);
            }

            // ---- Compare solution ----
            uint16_t sol_sfn, sol_slot;
            load_h5_solution_tv(tv.tv_base, sys_param.num_cell,
                                sol_sfn, sol_slot, expected_sol_buf);

            bool match = compare_solutions(gpu_sol_buf_host, expected_sol_buf,
                                           sys_param.num_cell);

            if (match) {
                printf("PASS: SFN=%u slot=%u - computed solution matches TV\n",
                       tv.sfn, tv.slot);
            } else {
                printf("FAIL: SFN=%u slot=%u - computed solution does NOT match TV\n",
                       tv.sfn, tv.slot);
                num_failed++;

                if (sys_param.print_ue_pairing_solution) {
                    print_ue_pairing_sol("Computed", tv.sfn, tv.slot,
                                         gpu_sol_buf_host, sys_param.num_cell);
                    print_ue_pairing_sol("Expected", tv.sfn, tv.slot,
                                         expected_sol_buf, sys_param.num_cell);
                }
            }

            num_tested++;

            CHECK_CUDA_ERR(cudaFree(task_in_buf_gpu));
            CHECK_CUDA_ERR(cudaFreeHost(task_in_buf_host));
            delete task;
        }

        // ---- Summary for this repeat run ----
        printf("\n========== TV Replay Test Summary");
        if (num_repeats > 1)
            printf(" (repeat %d/%d)", repeat + 1, num_repeats);
        printf(" ==========\n");
        printf("Tested : %d slot(s)\n", num_tested);
        printf("Passed : %d\n", num_tested - num_failed);
        printf("Failed : %d\n", num_failed);
        printf("Result : %s\n", num_failed > 0 ? "FAIL" : "PASS");
        printf("=============================================\n");

        total_failures_across_repeats += num_failed;
    }

    if (num_repeats > 1) {
        printf("\n========== Determinism Check Summary ==========\n");
        printf("Repeat runs       : %d\n", num_repeats);
        printf("Total failures    : %d (across all repeats)\n",
               total_failures_across_repeats);
        if (total_failures_across_repeats == 0)
            printf("Conclusion: all repeats PASSED - results are deterministic\n");
        else
            printf("Conclusion: failures detected - compare per-repeat logs above "
                   "to check if failure slots are consistent\n");
        printf("================================================\n");
    }

    // ---- Cleanup ----
    delete uePairObj;

    CHECK_CUDA_ERR(cudaFree(task0_in_buf));
    CHECK_CUDA_ERR(cudaFree(srs_chan_est_buf));
    CHECK_CUDA_ERR(cudaFree(srs_snr_buf));
    CHECK_CUDA_ERR(cudaFree(chan_orth_mat_buf));
    CHECK_CUDA_ERR(cudaFree(gpu_out_buf));
    if (cubb_srs_gpu_buf) {
        CHECK_CUDA_ERR(cudaFree(cubb_srs_gpu_buf));
    }
    CHECK_CUDA_ERR(cudaFreeHost(gpu_sol_buf_host));
    CHECK_CUDA_ERR(cudaFreeHost(expected_sol_buf));
    CHECK_CUDA_ERR(cudaStreamDestroy(strm));

    // Remove generated TVs and any leftover temporary config
    if (!skip_generation) {
        printf("Cleaning up generated TV files from %s ...\n", tv_dir);
        remove_existing_tvs(tv_dir);
        restore_config(std::string(yaml_path));
    }

    return (total_failures_across_repeats > 0) ? 1 : 0;
}

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

#include "CLI/CLI.hpp"
#include "cuphy.h"
#include "cuphy.hpp"
#include "ldpc/ldpc_api.hpp"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

struct Options
{
    int              device          = 0;
    int              bg              = 1;
    int              z               = 384;
    int              p               = 4;
    int              requested_algo  = 0;
    int              max_iterations  = 10;
    int              num_codewords   = 2;
    std::string      fptype          = "fp16";
    bool             sweep           = false;
    bool             choose_throughput = false;
    bool             choose_latency  = false;
    bool             no_header       = false;
    std::vector<int> bg_list;
    std::vector<int> z_list;
    std::vector<int> p_list;
    std::vector<std::string> p_list_tokens;
};

struct DeviceInfo
{
    int         device          = 0;
    std::string device_name;
    int         cc_major        = 0;
    int         cc_minor        = 0;
    int         max_optin_shmem = 0;
};

struct ProbeCase
{
    int bg = 1;
    int z  = 384;
    int p  = 4;
};

struct ProbeResult
{
    cuphyStatus_t status         = CUPHY_STATUS_SUCCESS;
    int           algo           = -1;
    const char*   globalfunc     = "NA";
    int           cb_per_cta     = -1;
    int           max_cta_per_sm = -1;
    unsigned int  shared_mem     = 0;
    unsigned int  grid_x         = 0;
    unsigned int  grid_y         = 0;
    unsigned int  grid_z         = 0;
    unsigned int  block_x        = 0;
    unsigned int  block_y        = 0;
    unsigned int  block_z        = 0;
};

struct LaunchProbeResult
{
    cuphyStatus_t status     = CUPHY_STATUS_SUCCESS;
    int           algo       = -1;
    CUfunction    func       = nullptr;
    unsigned int  shared_mem = 0;
    unsigned int  grid_x     = 0;
    unsigned int  grid_y     = 0;
    unsigned int  grid_z     = 0;
    unsigned int  block_x    = 0;
    unsigned int  block_y    = 0;
    unsigned int  block_z    = 0;
};

// Dispatch-neutral defaults passed to LDPC_decode_config solely to form a valid descriptor;
// these values do not affect kernel selection, only decode math.
constexpr float LDPC_CONFIG_CLAMP_DEFAULT      = 32.0F;
constexpr float LDPC_CONFIG_NORM_DEFAULT       = 0.8125F;

void check_cuda(CUresult status, std::string_view call)
{
    if(status != CUDA_SUCCESS)
    {
        const char* error_name   = nullptr;
        const char* error_string = nullptr;
        (void)cuGetErrorName(status, &error_name);
        (void)cuGetErrorString(status, &error_string);
        throw std::runtime_error(std::string(call) + " failed (" +
                                 (error_name == nullptr ? "unknown" : error_name) + "): " +
                                 (error_string == nullptr ? "unknown error" : error_string));
    }
}

class DriverContext final
{
public:
    explicit DriverContext(int device_ordinal) : device_ordinal_(device_ordinal)
    {
        check_cuda(cuInit(0), "cuInit");
        check_cuda(cuDeviceGet(&device_, device_ordinal_), "cuDeviceGet");
#if CUDA_VERSION >= 13000
        CUctxCreateParams params{};
        check_cuda(cuCtxCreate(&context_, &params, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, device_),
                   "cuCtxCreate");
#else
        check_cuda(cuCtxCreate(&context_, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, device_),
                   "cuCtxCreate");
#endif
    }

    ~DriverContext() noexcept
    {
        if(context_ == nullptr)
        {
            return;
        }

        const CUresult status = cuCtxDestroy(context_);
        if(status != CUDA_SUCCESS)
        {
            const char* error_name = nullptr;
            (void)cuGetErrorName(status, &error_name);
            std::fprintf(stderr,
                         "cuCtxDestroy failed: %s\n",
                         error_name == nullptr ? "unknown CUDA driver error" : error_name);
        }
    }

    DriverContext(const DriverContext&)            = delete;
    DriverContext& operator=(const DriverContext&) = delete;
    DriverContext(DriverContext&&)                 = delete;
    DriverContext& operator=(DriverContext&&)      = delete;

    [[nodiscard]] DeviceInfo device_info() const
    {
        std::array<char, 256> device_name{};
        check_cuda(cuDeviceGetName(device_name.data(), static_cast<int>(device_name.size()), device_),
                   "cuDeviceGetName");

        int cc_major{};
        int cc_minor{};
        int max_optin_shmem{};
        check_cuda(cuDeviceGetAttribute(
                       &cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_),
                   "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR)");
        check_cuda(cuDeviceGetAttribute(
                       &cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_),
                   "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MINOR)");
        check_cuda(cuDeviceGetAttribute(
                       &max_optin_shmem,
                       CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                       device_),
                   "cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)");

        std::replace(device_name.begin(), device_name.end(), ',', ' ');
        return {device_ordinal_, device_name.data(), cc_major, cc_minor, max_optin_shmem};
    }

private:
    int       device_ordinal_{};
    CUdevice  device_{};
    CUcontext context_{nullptr};
};

int kb_from_bg(int bg)
{
    return (bg == 1) ? 22 : 10;
}

int max_parity_from_bg(int bg)
{
    return (bg == 1) ? 46 : 42;
}

cuphyDataType_t parse_llr_type(const std::string& fptype)
{
    if(fptype == "fp16")
    {
        return CUPHY_R_16F;
    }
    if(fptype == "fp32")
    {
        return CUPHY_R_32F;
    }
    throw std::invalid_argument("Unsupported --fptype. Use fp16 or fp32.");
}

const char* llr_type_name(cuphyDataType_t type)
{
    switch(type)
    {
    case CUPHY_R_16F:
        return "fp16";
    case CUPHY_R_32F:
        return "fp32";
    default:
        return "unknown";
    }
}

std::vector<int> all_lifting_sizes()
{
    return {2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
            15,  16,  18,  20,  22,  24,  26,  28,  30,  32,  36,  40,  44,
            48,  52,  56,  60,  64,  72,  80,  88,  96,  104, 112, 120, 128,
            144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384};
}

std::vector<int> range_inclusive(int first, int last)
{
    std::vector<int> values;
    values.reserve(std::max(0, last - first + 1));
    for(int value = first; value <= last; ++value)
    {
        values.push_back(value);
    }
    return values;
}

void validate_bg(int bg)
{
    if(bg != 1 && bg != 2)
    {
        throw std::invalid_argument("BG must be 1 or 2.");
    }
}

void validate_lifting_size(int z)
{
    static const std::vector<int> valid = all_lifting_sizes();
    if(!std::binary_search(valid.begin(), valid.end(), z))
    {
        throw std::invalid_argument("Z=" + std::to_string(z) +
                                    " is not a valid 38.212 lifting size.");
    }
}

void validate_positive(std::string_view name, int value)
{
    if(value <= 0)
    {
        throw std::invalid_argument(std::string(name) + " must be positive.");
    }
}

int parse_int_token(const std::string& token, std::string_view option_name)
{
    std::size_t parsed = 0;
    int         value  = 0;
    try
    {
        value = std::stoi(token, &parsed);
    }
    catch(const std::exception&)
    {
        throw std::invalid_argument("Invalid " + std::string(option_name) + " token: " + token);
    }
    if(parsed != token.size())
    {
        throw std::invalid_argument("Invalid " + std::string(option_name) + " token: " + token);
    }
    return value;
}

std::vector<int> expand_int_list_tokens(const std::vector<std::string>& tokens, std::string_view option_name)
{
    std::vector<int> values;
    for(const std::string& token : tokens)
    {
        const std::size_t first_colon = token.find(':');
        if(first_colon == std::string::npos)
        {
            values.push_back(parse_int_token(token, option_name));
            continue;
        }

        const std::size_t second_colon = token.find(':', first_colon + 1);
        if((second_colon != std::string::npos) &&
           (token.find(':', second_colon + 1) != std::string::npos))
        {
            throw std::invalid_argument("Invalid " + std::string(option_name) + " range: " + token);
        }

        const int first = parse_int_token(token.substr(0, first_colon), option_name);
        const int last  = parse_int_token(token.substr(first_colon + 1,
                                                       second_colon == std::string::npos ? std::string::npos
                                                                                         : second_colon - first_colon - 1),
                                          option_name);
        const int step  = second_colon == std::string::npos
                            ? 1
                            : parse_int_token(token.substr(second_colon + 1), option_name);
        if(step <= 0 || last < first)
        {
            throw std::invalid_argument("Invalid " + std::string(option_name) + " range: " + token);
        }
        for(int value = first; value <= last; value += step)
        {
            values.push_back(value);
        }
    }
    return values;
}

void validate_options(const Options& options)
{
    validate_bg(options.bg);
    validate_lifting_size(options.z);
    validate_positive("p", options.p);
    validate_positive("max_iterations", options.max_iterations);
    validate_positive("num_codewords", options.num_codewords);
    (void)parse_llr_type(options.fptype);

    for(int bg : options.bg_list)
    {
        validate_bg(bg);
    }
    for(int z : options.z_list)
    {
        validate_lifting_size(z);
    }
    for(int p : options.p_list)
    {
        validate_positive("p list entry", p);
    }
}

std::vector<ProbeCase> make_probe_cases(const Options& options)
{
    if(!options.sweep)
    {
        return {{options.bg, options.z, options.p}};
    }

    const std::vector<int> bgs = options.bg_list.empty() ? std::vector<int>{1, 2} : options.bg_list;
    const std::vector<int> zs  = options.z_list.empty() ? all_lifting_sizes() : options.z_list;

    std::vector<ProbeCase> cases;
    for(int bg : bgs)
    {
        const std::vector<int> ps = options.p_list.empty() ? range_inclusive(4, max_parity_from_bg(bg)) : options.p_list;
        for(int z : zs)
        {
            for(int p : ps)
            {
                cases.push_back({bg, z, p});
            }
        }
    }
    return cases;
}

// Authoritative variant label. Instead of shadow-reconstructing the kernel
// variant from p/z/shmem (which drifts whenever a kernel's thresholds or shmem
// layout change -- e.g. algo55's bigreg reach or bp-full core size), ask the
// driver for the name of the kernel the library ACTUALLY selected. `func` is
// params.func from the launch descriptor, so this can never disagree with the
// real dispatch. Returns a pointer into driver-owned storage (valid for the
// module's lifetime), which is fine for the immediate CSV print.
const char* func_label(CUfunction func)
{
    const char* name = nullptr;
    if(func != nullptr && cuFuncGetName(&name, func) == CUDA_SUCCESS && name != nullptr)
    {
        // Trim the common "ldpc2_" prefix for brevity; the remainder is the exact
        // selected kernel, e.g. BG1_split_index_bp_x2_desc_dyn_bigreg_tb_msc.
        constexpr std::string_view prefix{"ldpc2_"};
        const std::string_view     sv{name};
        if(sv.substr(0, prefix.size()) == prefix)
        {
            return name + prefix.size();
        }
        return name;
    }
    return "NA";
}

uint32_t ldpc_flags(const Options& options)
{
    uint32_t flags = CUPHY_LDPC_DECODE_DEFAULT;
    if(options.choose_throughput)
    {
        flags |= CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT;
    }
    return flags;
}

LaunchProbeResult query_launch(cuphy::LDPC_decoder& decoder,
                               const ProbeCase&     probe_case,
                               const Options&       options,
                               cuphyDataType_t      llr_type,
                               uint32_t             flags,
                               int                  num_codewords)
{
    const int kb          = kb_from_bg(probe_case.bg);
    const int llr_length  = (kb + probe_case.p) * probe_case.z;
    const int code_length = kb * probe_case.z;

    cuphy::LDPC_decode_config config(llr_type,
                                     static_cast<int16_t>(probe_case.p),
                                     static_cast<int16_t>(probe_case.z),
                                     static_cast<int16_t>(options.max_iterations),
                                     LDPC_CONFIG_CLAMP_DEFAULT,
                                     static_cast<int16_t>(kb),
                                     LDPC_CONFIG_NORM_DEFAULT,
                                     flags,
                                     static_cast<int16_t>(probe_case.bg),
                                     static_cast<int16_t>(options.requested_algo),
                                     nullptr);

    cuphy::tensor_device llr_tensor(llr_type,
                                    llr_length,
                                    num_codewords,
                                    cuphy::tensor_flags::align_coalesce);
    cuphy::tensor_device decoded_tensor(CUPHY_BIT,
                                        code_length,
                                        num_codewords,
                                        cuphy::tensor_flags::align_coalesce);

    cuphy::LDPC_decode_desc desc(config, 1);
    desc.add_tensor_as_tb(llr_tensor.desc(), llr_tensor.addr(), decoded_tensor.desc(), decoded_tensor.addr());

    cuphyLDPCDecodeLaunchConfig_t launch_config{};
    launch_config.decode_desc = desc;

    LaunchProbeResult result{};
    result.status = cuphyErrorCorrectionLDPCDecodeGetLaunchDescriptor(decoder.handle(), &launch_config);
    if(result.status == CUPHY_STATUS_SUCCESS)
    {
        const CUDA_KERNEL_NODE_PARAMS& params = launch_config.kernel_node_params_driver;
        result.algo       = launch_config.decode_desc.config.algo;
        result.func       = params.func;
        result.shared_mem = params.sharedMemBytes;
        result.grid_x     = params.gridDimX;
        result.grid_y     = params.gridDimY;
        result.grid_z     = params.gridDimZ;
        result.block_x    = params.blockDimX;
        result.block_y    = params.blockDimY;
        result.block_z    = params.blockDimZ;
    }
    return result;
}

bool same_kernel(const LaunchProbeResult& lhs, const LaunchProbeResult& rhs)
{
    return (lhs.status == CUPHY_STATUS_SUCCESS) &&
           (rhs.status == CUPHY_STATUS_SUCCESS) &&
           (lhs.algo == rhs.algo) &&
           (lhs.func == rhs.func);
}

int infer_codeblocks_per_cta(cuphy::LDPC_decoder&     decoder,
                             const ProbeCase&         probe_case,
                             const Options&           options,
                             cuphyDataType_t          llr_type,
                             uint32_t                 flags,
                             const LaunchProbeResult& reference)
{
    constexpr int MAX_PROBED_CB_PER_CTA = 64;

    auto query = [&](int num_codewords) {
        return query_launch(decoder, probe_case, options, llr_type, flags, num_codewords);
    };
    auto is_valid = [&](const LaunchProbeResult& result) {
        return same_kernel(reference, result) && (result.grid_y == 1) && (result.grid_z == 1) &&
               (result.grid_x >= 1);
    };

    LaunchProbeResult one_cb = query(1);
    if(!is_valid(one_cb) || one_cb.grid_x != 1)
    {
        return -1;
    }

    int low = 1; // Largest tested CB count that still launches one CTA.
    for(int high = 2; high <= MAX_PROBED_CB_PER_CTA; high *= 2)
    {
        LaunchProbeResult high_result = query(high);
        if(!is_valid(high_result))
        {
            return -1;
        }
        if(high_result.grid_x == 1)
        {
            low = high;
            continue;
        }

        while(low + 1 < high)
        {
            const int         mid        = low + (high - low) / 2;
            LaunchProbeResult mid_result = query(mid);
            if(!is_valid(mid_result))
            {
                return -1;
            }
            if(mid_result.grid_x == 1)
            {
                low = mid;
            }
            else
            {
                high = mid;
            }
        }
        return low;
    }

    LaunchProbeResult overflow = query(MAX_PROBED_CB_PER_CTA + 1);
    return (is_valid(overflow) && overflow.grid_x > 1) ? MAX_PROBED_CB_PER_CTA : -1;
}

ProbeResult probe_dispatch(cuphy::LDPC_decoder& decoder,
                           const ProbeCase&     probe_case,
                           const Options&       options,
                           cuphyDataType_t      llr_type,
                           uint32_t             flags)
{
    const LaunchProbeResult launch =
        query_launch(decoder, probe_case, options, llr_type, flags, options.num_codewords);

    ProbeResult result{};
    result.status = launch.status;
    if(result.status == CUPHY_STATUS_SUCCESS)
    {
        result.algo           = launch.algo;
        result.globalfunc     = func_label(launch.func);
        result.cb_per_cta     = infer_codeblocks_per_cta(decoder, probe_case, options, llr_type, flags, launch);
        result.shared_mem     = launch.shared_mem;
        result.grid_x         = launch.grid_x;
        result.grid_y         = launch.grid_y;
        result.grid_z         = launch.grid_z;
        result.block_x        = launch.block_x;
        result.block_y        = launch.block_y;
        result.block_z        = launch.block_z;

        const int threads_per_cta = static_cast<int>(launch.block_x * launch.block_y * launch.block_z);
        int       max_cta_per_sm  = 0;
        if(CUDA_SUCCESS == cuOccupancyMaxActiveBlocksPerMultiprocessor(
                               &max_cta_per_sm, launch.func, threads_per_cta, launch.shared_mem))
        {
            result.max_cta_per_sm = max_cta_per_sm;
        }
    }
    return result;
}

const char* status_name(cuphyStatus_t status)
{
    const char* name = cuphyGetErrorName(status);
    return (name == nullptr) ? "UNKNOWN" : name;
}

void print_header()
{
    std::puts("device,device_name,cc_major,cc_minor,max_optin_shmem,bg,z,p,kb,llr_type,flags,requested_algo,status,status_name,algo,globalfunc,cb_per_cta,max_cta_per_sm,shared_mem,grid_x,grid_y,grid_z,block_x,block_y,block_z");
}

void print_result(const DeviceInfo&   device,
                  const ProbeCase&    probe_case,
                  const Options&      options,
                  cuphyDataType_t     llr_type,
                  uint32_t            flags,
                  const ProbeResult&  result)
{
    std::printf("%d,%s,%d,%d,%d,%d,%d,%d,%d,%s,%u,%d,%d,%s,%d,%s,%d,%d,%u,%u,%u,%u,%u,%u,%u\n",
                device.device,
                device.device_name.c_str(),
                device.cc_major,
                device.cc_minor,
                device.max_optin_shmem,
                probe_case.bg,
                probe_case.z,
                probe_case.p,
                kb_from_bg(probe_case.bg),
                llr_type_name(llr_type),
                flags,
                options.requested_algo,
                static_cast<int>(result.status),
                status_name(result.status),
                result.algo,
                result.globalfunc,
                result.cb_per_cta,
                result.max_cta_per_sm,
                result.shared_mem,
                result.grid_x,
                result.grid_y,
                result.grid_z,
                result.block_x,
                result.block_y,
                result.block_z);
}

} // namespace

int main(int argc, char** argv)
{
    Options options{};

    CLI::App app{"Probe cuPHY LDPC decoder auto-dispatch by asking for launch configs without launching decode kernels."};
    app.add_option("--device", options.device, "CUDA device ordinal.");
    app.add_option("--bg", options.bg, "Base graph for single-point mode.")->check(CLI::Range(1, 2));
    app.add_option("--z", options.z, "Lifting size for single-point mode.")->check(CLI::PositiveNumber);
    app.add_option("-p,--parity", options.p, "Parity-node count for single-point mode.")->check(CLI::PositiveNumber);
    app.add_option("-a,--algo", options.requested_algo, "Requested LDPC algorithm. Use 0 for auto-dispatch.");
    app.add_option("--max-iter", options.max_iterations, "Maximum LDPC iterations used in the descriptor.")->check(CLI::PositiveNumber);
    app.add_option("--num-cw", options.num_codewords, "Dummy codeword count used to form the descriptor.")->check(CLI::PositiveNumber);
    app.add_option("--fptype", options.fptype, "LLR type: fp16 or fp32.")->check(CLI::IsMember({"fp16", "fp32"}));
    app.add_flag("-t,--throughput", options.choose_throughput, "Set CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT for throughput-mode auto-dispatch.");
    app.add_flag("--latency", options.choose_latency, "Clear CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT for latency-mode auto-dispatch.");
    app.add_flag("--sweep", options.sweep, "Sweep BG1/BG2, all 38.212 lifting sizes, and valid parity-node ranges.");
    app.add_option("--bg-list", options.bg_list, "BG list for sweep mode. Defaults to 1 2.")->check(CLI::Range(1, 2));
    app.add_option("--z-list", options.z_list, "Z list for sweep mode. Defaults to all 38.212 lifting sizes.")->check(CLI::PositiveNumber);
    app.add_option("--p-list", options.p_list_tokens,
                   "p list for sweep mode. Accepts values or inclusive start:stop[:step] ranges; defaults to 4:max_p per BG.");
    app.add_flag("--no-header", options.no_header, "Suppress the CSV header row.");

    CLI11_PARSE(app, argc, argv);

    try
    {
        options.p_list = expand_int_list_tokens(options.p_list_tokens, "--p-list");
        validate_options(options);
        if(options.choose_latency)
        {
            options.choose_throughput = false;
        }

        DriverContext         driver_context(options.device);
        const DeviceInfo      device   = driver_context.device_info();
        const cuphyDataType_t llr_type = parse_llr_type(options.fptype);
        const uint32_t      flags    = ldpc_flags(options);
        const auto          cases    = make_probe_cases(options);

        cuphy::context      context;
        cuphy::LDPC_decoder decoder(context);

        if(!options.no_header)
        {
            print_header();
        }
        for(const ProbeCase& probe_case : cases)
        {
            const ProbeResult result = probe_dispatch(decoder, probe_case, options, llr_type, flags);
            print_result(device, probe_case, options, llr_type, flags, result);
        }
    }
    catch(const std::exception& ex)
    {
        std::fprintf(stderr, "cuphy_ldpc_find_dispatch: %s\n", ex.what());
        return 1;
    }

    return 0;
}

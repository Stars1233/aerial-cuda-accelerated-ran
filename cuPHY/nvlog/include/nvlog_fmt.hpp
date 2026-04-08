/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once
#ifndef NVLOG_FMT_HPP

//#define FMTLOG_HEADER_ONLY
#ifdef NVIPC_FMTLOG_ENABLE
#include "fmtlog.h"
#endif

#ifdef NVIPC_FMTLOG_ENABLE
#include <string_view>
#include <type_traits>
#include "enum_utils.hpp"

// =============================================================================
// Generic formatter for ALL enum types to work with fmt library
// =============================================================================
// This allows using enums directly in NVLOG_FMT without explicit static_cast:
//   NVLOGC_FMT(TAG, "value: {}", my_enum);  // Works automatically!
//
// Technical approach:
// - Uses SFINAE (std::enable_if_t) instead of C++20 concepts for NVCC compatibility
// - SFINAE condition: only applies when T is an enum (std::is_enum_v<T> == true)
// - Inherits from formatter for the enum's underlying integer type
// - Uses to_underlying() for clean enum-to-int conversion (C++23 compatible)
// - Zero runtime cost: enum-to-int cast is optimized away (same bit representation)
// =============================================================================
template <typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_enum_v<T>, char>> : fmt::formatter<std::underlying_type_t<T>>
{
    using underlying_type = std::underlying_type_t<T>;  //!< Compile-time query: what int type backs this enum?
    using base = fmt::formatter<underlying_type>;       //!< Inherit formatting logic from underlying type

    /**
     * Parse format specifications at compile time
     *
     * Called by fmt during compile-time format string validation.
     * Delegates to the base integer formatter to handle specs like {:x}, {:d}, etc.
     *
     * @param ctx Format parse context containing format spec string
     * @return Iterator to end of parsed format specification
     */
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin())
    {
        return base::parse(ctx);
    }

    /**
     * Format the enum value at runtime
     * 
     * Converts enum to its underlying integer type using to_underlying() and 
     * delegates to the base formatter. The conversion is zero-cost since enum 
     * and underlying type share identical memory layout.
     * 
     * @param value The enum value to format
     * @param ctx Format context for output
     * @return Iterator to end of formatted output
     */
    template <typename FormatContext>
    auto format(const T& value, FormatContext& ctx) const -> decltype(ctx.out())
    {
        // Convert enum to underlying type and delegate to base formatter
        return base::format(to_underlying(value), ctx);
    }
};
#endif
#include <unistd.h>
#include "memtrace.h"
#include "exit_handler.hpp"
    
    
extern exit_handler& pExitHandler;

inline void EXIT_APP(int val)
{
    if(val == EXIT_FAILURE)
    {
        pExitHandler.test_trigger_exit(__FILE__, __LINE__, "NULL");
    }
    else
    {
        exit(EXIT_SUCCESS);
    }
}

#ifdef NVIPC_FMTLOG_ENABLE
#define NVLOG_PRINTF(log_level, component_id, format_printf, ...) do { \
    static_assert(nvlog_component_is_valid(component_id),"Component ID (TAG) doesn't exist"); \
    if (log_level >= g_nvlog_component_levels[nvlog_get_component_id(component_id)]) { \
        try { \
            MemtraceDisableScope md; \
            FMTLOG_ONCE(log_level, "[%s] " format_printf, nvlog_get_component_name(component_id), ##__VA_ARGS__); \
        } \
        catch (...) { \
            printf("Caught exception in NVLOG_PRINTF with log_level %d format string %s at %s:%d\n", log_level, format_printf, __FILE__,__LINE__); \
            ::EXIT_APP(EXIT_FAILURE); \
        } \
    } \
} while (0)

#define NVLOGE_PRINTF(log_level, event, component_id, format_printf, ...) do { \
    static_assert(nvlog_component_is_valid(component_id),"Component ID (TAG) doesn't exist"); \
    static_assert(std::is_same<decltype(event),aerial_event_code_t>::value,"Event is not of type aerial_event_code_t"); \
    if (log_level >= g_nvlog_component_levels[nvlog_get_component_id(component_id)]) { \
        try { \
            MemtraceDisableScope md; \
            FMTLOG_ONCE(log_level, "[%s][%s] " format_printf, #event, nvlog_get_component_name(component_id), ##__VA_ARGS__); \
        } \
        catch (...) { \
            printf("Caught exception in NVLOG_PRINTF with log_level %d format string %s at %s:%d\n", log_level, format_printf, __FILE__, __LINE__); \
            ::EXIT_APP(EXIT_FAILURE); \
        } \
    } \
} while (0)

#define NVLOGV_FMT(component_id, format_fmt, ...) NVLOG_FMT(fmtlog::VEB,component_id, format_fmt, ##__VA_ARGS__)
#define NVLOGD_FMT(component_id, format_fmt, ...) NVLOG_FMT(fmtlog::DBG,component_id, format_fmt, ##__VA_ARGS__)

#define NVLOGI_FMT(component_id, format_fmt, ...) \
    _Pragma("vcast_dont_instrument_start") \
    NVLOG_FMT(fmtlog::INF,component_id, format_fmt, ##__VA_ARGS__) \
    _Pragma("vcast_dont_instrument_end") \

#define NVLOGW_FMT(component_id, format_fmt, ...) NVLOG_FMT(fmtlog::WRN,component_id, format_fmt, ##__VA_ARGS__)
#define NVLOGC_FMT(component_id, format_fmt, ...) NVLOG_FMT(fmtlog::CON,component_id, format_fmt, ##__VA_ARGS__)
#define NVLOGE_NO_FMT(component_id, event_level, format_fmt, ...) NVLOG_FMT_EVT(fmtlog::ERR,component_id, event_level, format_fmt, ##__VA_ARGS__)

#define NVLOGI_FMT_EVT(component_id, event_level, format_fmt, ...) NVLOG_FMT_EVT(fmtlog::INF,component_id, event_level, format_fmt, ##__VA_ARGS__)

#define NVLOGE_FMT(component_id, event_level, format_fmt, ...) \
    _Pragma("vcast_dont_instrument_start") \
    NVLOG_FMT_EVT(fmtlog::ERR,component_id, event_level, format_fmt, ##__VA_ARGS__) \
    _Pragma("vcast_dont_instrument_end") \

#define NVLOGF_FMT(component_id, event_level, format_fmt, ...) \
    _Pragma("vcast_dont_instrument_start") \
    do { \
        MemtraceDisableScope md; \
        NVLOG_FMT_EVT(fmtlog::FAT, component_id, event_level, format_fmt, ##__VA_ARGS__); \
        ::pExitHandler.test_trigger_exit(__FILE__, __LINE__, format_fmt); \
    } while(0)
    _Pragma("vcast_dont_instrument_end") \

#define EXIT_L1(val) \
    do { \
        MemtraceDisableScope md; \
        if(val == EXIT_SUCCESS) \
        { \
            printf("EXIT successfully at: %s line %d \n", __FILE__, __LINE__); \
            exit(EXIT_SUCCESS); \
        } \
        else \
        { \
            ::pExitHandler.test_trigger_exit(__FILE__, __LINE__, "NULL"); \
        } \
    } while(0)

#define ENTER_L1_RECOVERY() {::pExitHandler.set_exit_handler_flag(exit_handler::l1_state::L1_RECOVERY);}

#define NVLOG_FMT(log_level, component_id, format_fmt, ...) do { \
    static_assert(nvlog_component_is_valid(component_id),"Component ID (TAG) doesn't exist"); \
    if (log_level >= g_nvlog_component_levels[nvlog_get_component_id(component_id)]) { \
        try { \
            if (1) {/* Optimized method, string formatting in background thread */ \
                FMTLOG(log_level, "[{}] " format_fmt, nvlog_get_component_name(component_id), ##__VA_ARGS__); \
            } else { /* Slow method, formatting in foreground thread, useful for catching exceptions */ \
                std::string s = fmt::format(format_fmt, ##__VA_ARGS__); \
                FMTLOG(log_level, "[{}] {}", nvlog_get_component_name(component_id), s); \
            } \
        } \
        catch (...) { \
            printf("Caught exception in NVLOG_FMT with log_level %d format string %s at %s:%d\n", log_level, format_fmt, __FILE__, __LINE__); \
        } \
    } \
} while (0)

#define NVLOG_FMT_EVT(log_level, component_id, event_level, format_fmt, ...) do { \
    static_assert(nvlog_component_is_valid(component_id),"Component ID (TAG) doesn't exist"); \
	static_assert(std::is_same<decltype(event_level),aerial_event_code_t>::value,"Event is not of type aerial_event_code_t"); \
	try { \
		FMTLOG(log_level, "[{}] [{}] " format_fmt, #event_level, nvlog_get_component_name(component_id), ##__VA_ARGS__); \
	} \
	catch (...) { \
		printf("Caught exception in NVLOG_FMT with log_level %d format string %s at %s:%d\n", log_level, format_fmt, __FILE__, __LINE__); \
    } \
} while (0)

#define NVLOG_FMT_ARRAY(level, tag, info, array, num)                                                                                                                                                                                                                                                                                    \
    do {                                                                                                                                                                                                                                                                                                                                 \
        switch(num)                                                                                                                                                                                                                                                                                                                      \
        {                                                                                                                                                                                                                                                                                                                                \
        case 0: NVLOG_FMT(level, tag, "{}[{}]: empty array", info, num); break;                                                                                                                                                                                                                                                          \
        case 1: NVLOG_FMT(level, tag, "{}[{}]: {}", info, num, array[0]); break;                                                                                                                                                                                                                                                         \
        case 2: NVLOG_FMT(level, tag, "{}[{}]: {} {}", info, num, array[0], array[1]); break;                                                                                                                                                                                                                                            \
        case 3: NVLOG_FMT(level, tag, "{}[{}]: {} {} {}", info, num, array[0], array[1], array[2]); break;                                                                                                                                                                                                                               \
        case 4: NVLOG_FMT(level, tag, "{}[{}]: {} {} {} {}", info, num, array[0], array[1], array[2], array[3]); break;                                                                                                                                                                                                                  \
        case 5: NVLOG_FMT(level, tag, "{}[{}]: {} {} {} {} {}", info, num, array[0], array[1], array[2], array[3], array[4]); break;                                                                                                                                                                                                     \
        case 6: NVLOG_FMT(level, tag, "{}[{}]: {} {} {} {} {} {}", info, num, array[0], array[1], array[2], array[3], array[4], array[5]); break;                                                                                                                                                                                        \
        case 7: NVLOG_FMT(level, tag, "{}[{}]: {} {} {} {} {} {} {}", info, num, array[0], array[1], array[2], array[3], array[4], array[5], array[6]); break;                                                                                                                                                                           \
        case 8: NVLOG_FMT(level, tag, "{}[{}]: {} {} {} {} {} {} {} {}", info, num, array[0], array[1], array[2], array[3], array[4], array[5], array[6], array[7]); break;                                                                                                                                                              \
        default: NVLOG_FMT(level, tag, "{}[{}]: {} {} {} {} {} {} {} {} ... {} {} {} {} {} {} {} {}", info, num, array[0], array[1], array[2], array[3], array[4], array[5], array[6], array[7], array[num - 8], array[num - 7], array[num - 6], array[num - 5], array[num - 4], array[num - 3], array[num - 2], array[num - 1]); break; \
        }                                                                                                                                                                                                                                                                                                                                \
    } while(0)

#define NVLOGE_FMT_ARRAY(tag, info, array, num) NVLOG_FMT_ARRAY(fmtlog::ERR, tag, info, array, num)
#define NVLOGW_FMT_ARRAY(tag, info, array, num) NVLOG_FMT_ARRAY(fmtlog::WRN, tag, info, array, num)
#define NVLOGC_FMT_ARRAY(tag, info, array, num) NVLOG_FMT_ARRAY(fmtlog::CON, tag, info, array, num)
#define NVLOGI_FMT_ARRAY(tag, info, array, num) NVLOG_FMT_ARRAY(fmtlog::INF, tag, info, array, num)
#define NVLOGD_FMT_ARRAY(tag, info, array, num) NVLOG_FMT_ARRAY(fmtlog::DBG, tag, info, array, num)
#define NVLOGV_FMT_ARRAY(tag, info, array, num) NVLOG_FMT_ARRAY(fmtlog::VEB, tag, info, array, num)

struct nvlog_component_ids
{
    int id;
    const char* name;
};

constexpr int nvlog_max_id = 1024;

inline constexpr nvlog_component_ids g_nvlog_component_ids[] {
    //Reserve number 0 for no tag print
    {0, ""},

    // nvlog
    {10, "NVLOG"},
    {11, "NVLOG.TEST"},
    {12, "NVLOG.ITAG"},
    {13, "NVLOG.STAG"},
    {14, "NVLOG.STAT"},
    {15, "NVLOG.OBSERVER"},
    {16, "NVLOG.CPP"},
    {17, "NVLOG.SHM"},
    {18, "NVLOG.UTILS"},
    {19, "NVLOG.C"},
    {20, "NVLOG.EXIT_HANDLER"},

    // nvipc
    {30, "NVIPC"},
    {31, "NVIPC:YAML"},
    {32, "NVIPC.SHM_UTILS"},
    {33, "NVIPC.CUDAPOOL"},
    {34, "NVIPC.CUDAUTILS"},
    {35, "NVIPC.TESTCUDA"},
    {36, "NVIPC.SHM"},
    {37, "NVIPC.QUEUE"},
    {38, "NVIPC.IPC"},
    {39, "NVIPC.FD_SHARE"},
    {40, "NVIPC.DEBUG"},
    {41, "NVIPC.EFD"},
    {42, "NVIPC.EPOLL"},
    {43, "NVIPC.MEMPOOL"},
    {44, "NVIPC.RING"},
    {45, "NVIPC.SEM"},
    {46, "NVIPC.SHMLOG"},
    {47, "NVIPC.CONF"},
    {48, "NVIPC.DOCA"},
    {49, "NVIPC.DOCA_UTILS"},
    {50, "NVIPC.DPDK"},
    {51, "NVIPC.DPDK_UTILS"},
    {52, "NVIPC.GPUDATAUTILS"},
    {53, "NVIPC.GPUDATAPOOL"},
    {54, "NVIPC.TIMING"},
    {55, "NVIPC.DUMP"},
    {56, "NVIPC.UDP"},
    {57, "INIT"},
    {58, "TEST"},
    {59, "PHY"},
    {60, "MAC"},
    {61, "NVIPC.PCAP"},

    // aerial_utils
    {80, "UTIL"},
    {81, "UTIL.MEMFOOT"},

    // cuPHYController
    {100, "CTL"},
    {101, "CTL.SCF"},
    {103, "CTL.DRV"},
    {104, "CTL.YAML"},
    {105, "CTL.STARTUP_TIMES"},
    {106, "CTL.DATA_LAKE"},
    {107, "CTL.E3"},

    // cuPHYDriver
    {200, "DRV"},
    {201, "DRV.SA"},
    {202, "DRV.TIME"},
    {203, "DRV.CTX"},
    {204, "DRV.API"},
    {205, "DRV.FH"},
    {206, "DRV.GEN_CUDA"},
    {207, "DRV.GPUDEV"},
    {208, "DRV.PHYCH"},
    {209, "DRV.TASK"},
    {210, "DRV.WORKER"},
    {211, "DRV.DLBUF"},
    {212, "DRV.CSIRS"},
    {213, "DRV.PBCH"},
    {214, "DRV.PDCCH_DL"},
    {215, "DRV.PDSCH"},
    {216, "DRV.MAP_DL"},
    {217, "DRV.FUNC_DL"},
    {218, "DRV.HARQ_POOL"},
    {219, "DRV.ORDER_CUDA"},
    {220, "DRV.ORDER_ENTITY"},
    {221, "DRV.PRACH"},
    {222, "DRV.PUCCH"},
    {223, "DRV.PUSCH"},
    {224, "DRV.MAP_UL"},
    {225, "DRV.FUNC_UL"},
    {226, "DRV.ULBUF"},
    {227, "DRV.MPS"},
    {228, "DRV.METRICS"},
    {229, "DRV.MEMFOOT"},
    {230, "DRV.CELL"},
    {231, "DRV.EXCP"},
    {232, "DRV.CV_MEM_BNK"},
    {233, "DRV.DLBFW"},
    {234, "DRV.ULBFW"},
    {235, "DRV.CUPHY_PTI"},
    {236, "DRV.SYMBOL_TIMINGS"},
    {237, "DRV.PACKET_TIMINGS"},
    {238, "DRV.UL_PACKET_SUMMARY"},
    {239, "DRV.SRS"},
    {240, "DRV.MAP_DL_VERBOSE"},
    {241, "DRV.MAP_UL_VERBOSE"},
    {242, "DRV.PMU"},
    {243, "DRV.SRS_PACKET_SUMMARY"},
    {244, "DRV.SRS_FAPI_PACKET_SUMMARY"},
    {245, "DRV.SYMBOL_TIMINGS_SRS"},
    {246, "DRV.PACKET_TIMINGS_SRS"},
    {247, "DRV.WAVGCFO_POOL"},
    {248, "DRV.PERF_METRICS"},

    // cuphyl2adapter
    {300, "L2A"},
    {301, "L2A.MAC"},
    {302, "L2A.MACFACT"},
    {303, "L2A.PROXY"},
    {304, "L2A.EPOLL"},
    {305, "L2A.TRANSPORT"},
    {306, "L2A.MODULE"},
    {307, "L2A.TICK"},
    {308, "L2A.UEMD"},
    {309, "L2A.PARAM"},
    {310, "L2A.SIM"},
    {311, "L2A.PROCESSING_TIMES"},
    {312, "L2A.TICK_TIMES"},

    // scfl2adapter
    {330, "SCF"},
    {331, "SCF.MAC"},
    {332, "SCF.DISPATCH"},
    {333, "SCF.PHY"},
    {334, "SCF.SLOTCMD"},
    {335, "SCF.L2SA"},
    {336, "SCF.DUMMYMAC"},
    {337, "SCF.CALLBACK"},
    {338, "SCF.TICK_TEST"},
    {339, "SCF.UL_FAPI_VALIDATE"},
    {340, "SCF.DL_FAPI_VALIDATE"},

    // testMAC
    {400, "MAC"},
    {401, "MAC.LP"},
    {402, "MAC.FAPI"},
    {403, "MAC.UTILS"},
    {404, "MAC.SCF"},
    {406, "MAC.CFG"},
    {407, "MAC.PROC"},
    {408, "MAC.VALD"},
    {409, "MAC.PROCESSING_TIMES"},

    // testMAC - cuMAC
    {420, "CUMAC"},
    {421, "CUMAC.CFG"},
    {422, "CUMAC.HANDLER"},
    {423, "CUMAC.PATTERN"},
    {424, "CUMAC.VALD"},

    // cuMAC-CP
    {450, "CUMCP"},
    {451, "CUMCP.MAIN"},
    {452, "CUMCP.CFG"},
    {453, "CUMCP.RECV"},
    {454, "CUMCP.HANDLER"},
    {455, "CUMCP.TASK"},

    // ru-emulator
    {500, "RU"},
    {501, "RU.EMULATOR"},
    {502, "RU.PARSER"},
    {503, "RU.LATE_PACKETS"},
    {504, "RU.SYMBOL_TIMINGS"},
    {505, "RU.TX_TIMINGS"},
    {506, "RU.TX_TIMINGS_SUM"},
    {507, "RU.TV_CONFIGS"},
    {508, "RU.UL_LATE_TX"},
    {509, "RU.CP_WORKER_TRACING"},

    // aerial-fh-driver
    {600, "FH"},
    {601, "FH.FLOW"},
    {602, "FH.FH"},
    {603, "FH.GPU_MP"},
    {604, "FH.LIB"},
    {605, "FH.MEMREG"},
    {606, "FH.METRICS"},
    {608, "FH.PDUMP"},
    {609, "FH.PEER"},
    {610, "FH.QUEUE"},
    {611, "FH.RING"},
    {612, "FH.TIME"},
    {613, "FH.GPU_COMM"},
    {614, "FH.STREAMRX"},
    {615, "FH.GPU"},
    {616, "FH.RMAX"},
    {617, "FH.GPU_COMM_CUDA"},
    {618, "FH.DOCA"},
    {619, "FH.NIC"},
    {620, "FH.STATS"},
    {621, "FH.LATE_PACKETS"},
    {622, "FH.SYMBOL_TIMINGS"},
    {623, "FH.TX_TIMINGS"},
    {624, "FH.PACKET_SUMMARY"},
    {625, "FH.DEBUG"},
    // fh_generator
    {650, "FHGEN"},
    {651, "FHGEN.GEN"},
    {652, "FHGEN.WORKER"},
    {653, "FHGEN.YAML"},
    {654, "FHGEN.ORAN_SLOT_ITER"},

    // compression_decompression
    {700, "COMP"},

    // cuphyoam
    {800, "OAM"},
    {801, "OAM.YMMGR"},
    {802, "OAM.YMSVC"},
    {803, "OAM.CUSConnMgr"},
    {804, "OAM.COMMSVC"},

    // cuphy
    // cuPHY channels
    {900, "CUPHY"},
    {901, "CUPHY.SSB_TX"},
    {902, "CUPHY.PDCCH_TX"},
    {904, "CUPHY.PDSCH_TX"},
    {905, "CUPHY.CSIRS_TX"},
    {906, "CUPHY.PRACH_RX"},
    {907, "CUPHY.PUCCH_RX"},
    {908, "CUPHY.PUSCH_RX"},
    {909, "CUPHY.BFW"},
    {910, "CUPHY.SRS_RX"},
    {911, "CUPHY.SRS_TX"},
    {912, "CUPHY.CSIRS_RX"},

    // cuPHY components and common utilities
    {931, "CUPHY.UTILS"}, // do not change
    {932, "CUPHY.MEMFOOT"}, // do not change
    {933, "CUPHY.PTI"},
    {934, "CUPHY.CUPTI"},
    {935, "CUPHY.PYAERIAL"},

    // test bench (currently phase-3)
    {1000, "TESTBENCH"},
    {1001, "TESTBENCH.PHY"},
    {1002, "TESTBENCH.MAC"},
    
    // Unit testbench
    {1010, "UNIT_TB"},
    {1011, "UNIT_TB.COMMON"},
    {1012, "UNIT_TB.DLC"},

    // Order kernel test bench
    {1050, "ORDER_TB"},
    {1051, "ORDER_TB.INIT"},
    {1052, "ORDER_TB.RUN"},

    // App Configs and utilities
    {1100, "APP"},
    {1101, "APP.CONFIG"},
    {1102, "APP.UTILS"},

    // cuMAC
    {1200, "CUMAC"},
    {1210, "CUMAC.UTILS"},
};

#define NVLOG_FMTLOG_NUM_TAGS (sizeof(g_nvlog_component_ids) / sizeof(nvlog_component_ids))
inline int g_nvlog_component_levels[NVLOG_FMTLOG_NUM_TAGS];

constexpr bool nvlog_component_is_valid(int id)
{
    for (auto &c : g_nvlog_component_ids)
    {
        if (id == c.id)
        {
            return true;
        }
    }
    return false;
};

constexpr bool nvlog_component_is_valid(std::string_view name)
{
    for (auto &c : g_nvlog_component_ids)
    {
        if (name == c.name)
        {
            return true;
        }
    }
    return false;
};


constexpr const char * nvlog_get_component_name(int id)
{
    for (auto &c : g_nvlog_component_ids)
    {
        if (id == c.id)
        {
            return c.name;
        }
    }
    return "UNKNOWN";
};

constexpr const char * nvlog_get_component_name(std::string_view name)
{
    for (auto &c : g_nvlog_component_ids)
    {
        if (name == c.name)
        {
            return c.name;
        }
    }
    return "UNKNOWN";
};

constexpr int nvlog_get_component_id(int id)
{
    for(int i = 0; i < NVLOG_FMTLOG_NUM_TAGS; ++i)
    {
        auto &c = g_nvlog_component_ids[i];
        if (id == c.id)
        {
            return i;
        }
    }
    return 0;
};

constexpr int nvlog_get_component_id(std::string_view name)
{
    for(int i = 0; i < NVLOG_FMTLOG_NUM_TAGS; ++i)
    {
        auto &c = g_nvlog_component_ids[i];
        if (name == c.name)
        {
            return i;
        }
    }
    return 0;
};

#else

#define NVLOG_FMT_EVT(log_level, component_id, event_level, format_fmt, ...)

#define NVLOGV_FMT(component_id, format_fmt, ...) 
#define NVLOGD_FMT(component_id, format_fmt, ...) 
#define NVLOGI_FMT(component_id, format_fmt, ...) 
#define NVLOGW_FMT(component_id, format_fmt, ...) 
#define NVLOGC_FMT(component_id, format_fmt, ...) 
#define NVLOGE_NO_FMT(component_id, event_level, format_fmt, ...) 

#define NVLOGI_FMT_EVT(component_id, event_level, format_fmt, ...) 
#define NVLOGE_FMT(component_id, event_level, format_fmt, ...) 

#define NVLOGF_FMT(component_id, event_level, format_fmt, ...) do { \
    usleep(100000); \
    ::pExitHandler.test_trigger_exit(__FILE__, __LINE__, "NULL"); \
} while (0)

#endif

#endif // NVLOG_FMT_HPP

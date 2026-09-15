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

#ifndef FH_GENERATOR_UTILS_HPP__
#define FH_GENERATOR_UTILS_HPP__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <time.h>
#include <unistd.h>

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <vector>

#include "constant.hpp"
#include "nvlog.hpp"
#include "aerial-fh-driver/api.hpp"
#include "aerial-fh-driver/oran.hpp"
#include "aerial-fh-driver/packet_stats.hpp"

#ifndef likely
#define likely(x) __builtin_expect((x), 1)
#endif
#ifndef unlikely
#define unlikely(x) __builtin_expect((x), 0)
#endif

#define STR_(x) #x
#define STR(x) STR_(x)

#define TAG 650 //"FHGEN"

#ifndef ACCESS_ONCE
    #define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))
#endif

#define FH_GEN_CATCH_EXCEPTIONS()                  \
    catch(std::exception const& e)                 \
    {                                              \
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Exception! {}", e.what()); \
        if(gen)                                    \
        {                                          \
            delete gen;                            \
        }                                          \
                                                   \
        return -1;                                 \
    }                                              \
    catch(...)                                     \
    {                                              \
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Uncaught Exception!");     \
        if(gen)                                    \
        {                                          \
            delete gen;                            \
        }                                          \
                                                   \
        return -1;                                 \
    }

#define SET_THREAD_NAME(name) \
    { \
        char *str = strdup(name); \
        if (strlen(str) > 15) \
            str[15] = '\0'; \
        pthread_setname_np(pthread_self(), str); \
        free(str); \
    }

using namespace fh_gen;

enum FhGenType
{
    DU,
    RU,
};

inline aerial_fh::Ns now_ns()
{
    timespec t;
    if(unlikely(clock_gettime(CLOCK_REALTIME, &t) != 0))
    {
        printf("Could not access timer\n");
        throw "Could not access timer";
    }

    auto ns = static_cast<aerial_fh::Ns>(t.tv_nsec);
    ns += static_cast<aerial_fh::Ns>(t.tv_sec * 1000 * 1000 * 1000);

    return ns;
}

inline void wait_ns(aerial_fh::Ns end_time)
{
    while(now_ns() < end_time)
    {
        usleep(1);
    }
}

inline bool file_exists(const std::string& name)
{
    std::ifstream f(name.c_str());
    return f.good();
}

class StringBuilder {
    std::stringstream ss_;

public:
    template <class T>
    StringBuilder& operator<<(T const& x)
    {
        ss_ << x;
        return *this;
    }
    operator std::string()
    {
        return ss_.str();
    }
};

class FhGenExc : public std::runtime_error {
    const char* file_;
    const char* func_;
    int         lineno_;

public:
    FhGenExc(const std::string& what, const char* file, const char* func, int lineno) :
        std::runtime_error{what},
        file_{file},
        func_{func},
        lineno_{lineno} {}
    const char* file() const { return file_; }
    const char* func() const { return func_; }
    int         lineno() const { return lineno_; }
};

#define FILE_BNAME (__builtin_strrchr(__FILE__, '/') ? __builtin_strrchr(__FILE__, '/') + 1 : __FILE__)

// FH exception throw macro
#define THROW(what) throw FhGenExc(what, FILE_BNAME, __FUNCTION__, __LINE__);

inline size_t get_prb_size(size_t iq_sample_size, aerial_fh::UserDataCompressionMethod compression_method)
{
    auto                  method             = static_cast<size_t>(compression_method);
    std::array<size_t, 7> ud_comp_param_size = {0, 1, 1, 1, 0, 2, 2};

    if(unlikely((iq_sample_size > UD_IQ_WIDH_MAX) || (iq_sample_size == 0)))
        THROW(StringBuilder() << "Invalid user data IQ sample size: " << iq_sample_size);

    if(unlikely(method >= ud_comp_param_size.size()))
        THROW("Invalid user data IQ compression method");

    return PRB_SIZE(iq_sample_size) + ud_comp_param_size[method];
}

inline void read_binary_file(std::string filepath, void* buffer, size_t buffer_size)
{
    std::ifstream infile;
    infile.open(filepath, std::ios::binary | std::ios::in);

    if(!infile)
    {
        THROW(StringBuilder() << "Failed to open file " << filepath);
    }

    infile.read(static_cast<char*>(buffer), buffer_size);

    if(!infile.good())
    {
        THROW(StringBuilder() << "Failed to load " << buffer_size << " bytes from file " << filepath);
    }

    infile.close();
}

#endif //ifndef FH_GENERATOR_UTILS_HPP__

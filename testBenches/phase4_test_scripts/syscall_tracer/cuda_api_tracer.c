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

/**
 * @file cuda_api_tracer.c
 * @brief CUDA API tracer - LD_PRELOAD shared library.
 *
 * Intercepts CUDA Driver and Runtime API calls, increments per-API counters,
 * and dumps totals to a log file on process exit.
 *
 * APIs traced:
 *   Driver (libcuda): cuEventQuery, cuEventRecord, cuGraphExecKernelNodeSetParams_v2,
 *   cuGraphLaunch, cuGraphUpload, cuKernelSetAttribute, cuLaunchKernel,
 *   cuMemcpyHtoDAsync_v2, cuMemsetD8Async, cuStreamWaitEvent
 *   Runtime (libcudart): cudaEventQuery, cudaEventRecord, cudaStreamWaitEvent
 *   (cuMemcpyBatchAsync_v2 excluded - causes segfault on aarch64 with LD_PRELOAD)
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include <unistd.h>

/* Minimal CUDA types for ABI compatibility (avoids cuda.h at build time) */
typedef int CUresult;
typedef void* CUevent;
typedef void* CUstream;
typedef void* CUfunction;
typedef void* CUgraphExec;
typedef void* CUgraphNode;
typedef unsigned long long CUdeviceptr;

/* cudaError_t for Runtime API */
typedef int cudaError_t;
typedef void* cudaEvent_t;
typedef void* cudaStream_t;

#define NUM_APIS 13

typedef struct {
    const char* name;
    _Atomic uint64_t count;
} api_counter_t;

static api_counter_t g_counters[NUM_APIS] = {
    /* Driver API (libcuda) */
    {"cuEventQuery", 0},
    {"cuEventRecord", 0},
    {"cuGraphExecKernelNodeSetParams_v2", 0},
    {"cuGraphLaunch", 0},
    {"cuGraphUpload", 0},
    {"cuKernelSetAttribute", 0},
    {"cuLaunchKernel", 0},
    /* cuMemcpyBatchAsync_v2 EXCLUDED: causes segfault on aarch64 */
    {"cuMemcpyHtoDAsync_v2", 0},
    {"cuMemsetD8Async", 0},
    {"cuStreamWaitEvent", 0},
    /* Runtime API (libcudart) - workers often use these instead of Driver API */
    {"cudaEventQuery", 0},
    {"cudaEventRecord", 0},
    {"cudaStreamWaitEvent", 0},
};

static _Atomic int g_atexit_registered = 0;

/**
 * @brief Increment the call counter for the named API.
 * @param name API symbol name (e.g. "cuLaunchKernel"). Must match g_counters[].name.
 */
static void count_inc(const char* name) {
    for (int i = 0; i < NUM_APIS; i++) {
        if (strcmp(g_counters[i].name, name) == 0) {
            atomic_fetch_add_explicit(&g_counters[i].count, 1, memory_order_relaxed);
            return;
        }
    }
}

/**
 * @brief Write all API call counts to the output log file and close it.
 *
 * Output path is taken from CUDA_API_TRACER_OUTPUT; if unset, uses
 * /tmp/cuda_api_tracer_<pid>.log. Called from atexit() on process exit.
 */
static void dump_counts(void) {
    char buf[128];
    const char* out_path = getenv("CUDA_API_TRACER_OUTPUT");
    if (!out_path || out_path[0] == '\0') {
        snprintf(buf, sizeof(buf), "/tmp/cuda_api_tracer_%d.log", (int)getpid());
        out_path = buf;
    }

    FILE* f = fopen(out_path, "w");
    if (!f) {
        fprintf(stderr, "cuda_api_tracer: failed to open %s for write\n", out_path);
        return;
    }

    fprintf(f, "# CUDA API call counts (libcuda_api_tracer.so)\n");
    fprintf(f, "# Format: API_NAME COUNT\n");
    fprintf(f, "# ---\n");

    uint64_t total = 0;
    for (int i = 0; i < NUM_APIS; i++) {
        uint64_t c = atomic_load_explicit(&g_counters[i].count, memory_order_relaxed);
        fprintf(f, "%s %lu\n", g_counters[i].name, (unsigned long)c);
        total += c;
    }

    fprintf(f, "# ---\n");
    fprintf(f, "# TOTAL %lu\n", (unsigned long)total);

    fclose(f);
    fprintf(stderr, "cuda_api_tracer: dumped counts to %s (total %lu calls)\n", out_path, (unsigned long)total);
}

/**
 * @brief Register dump_counts with atexit() once (thread-safe).
 *
 * Ensures counts are dumped on process exit even if no traced API is ever called.
 */
static void maybe_register_atexit(void) {
    int prev = atomic_exchange(&g_atexit_registered, 1);
    if (prev == 0) {
        atexit(dump_counts);
    }
}

/**
 * @brief Library constructor: register atexit so we dump on process exit.
 */
__attribute__((constructor))
static void tracer_init(void) {
    maybe_register_atexit();
}

#define WRAP0(api)                                                              \
    static CUresult api##_impl(void) {                                          \
        typedef CUresult (*fn_t)(void);                                          \
        static _Atomic uintptr_t real_ptr;                                       \
        fn_t r = (fn_t)atomic_load_explicit(&real_ptr, memory_order_acquire);    \
        if (!r) {                                                               \
            r = (fn_t)dlsym(RTLD_NEXT, #api);                                   \
            atomic_store_explicit(&real_ptr, (uintptr_t)r, memory_order_release); \
        }                                                                       \
        if (!r) return -1;                                                      \
        count_inc(#api);                                                        \
        maybe_register_atexit();                                                 \
        return r();                                                             \
    }

/* Single-param wrappers */
#define WRAP1(api, t1, a1)                                                      \
    static CUresult api##_impl(t1 a1) {                                         \
        typedef CUresult (*fn_t)(t1);                                            \
        static _Atomic uintptr_t real_ptr;                                       \
        fn_t r = (fn_t)atomic_load_explicit(&real_ptr, memory_order_acquire);   \
        if (!r) {                                                               \
            r = (fn_t)dlsym(RTLD_NEXT, #api);                                   \
            atomic_store_explicit(&real_ptr, (uintptr_t)r, memory_order_release); \
        }                                                                       \
        if (!r) return -1;                                                      \
        count_inc(#api);                                                        \
        maybe_register_atexit();                                                 \
        return r(a1);                                                           \
    }

#define WRAP2(api, t1, a1, t2, a2)                                               \
    static CUresult api##_impl(t1 a1, t2 a2) {                                   \
        typedef CUresult (*fn_t)(t1, t2);                                        \
        static _Atomic uintptr_t real_ptr;                                       \
        fn_t r = (fn_t)atomic_load_explicit(&real_ptr, memory_order_acquire);    \
        if (!r) {                                                               \
            r = (fn_t)dlsym(RTLD_NEXT, #api);                                   \
            atomic_store_explicit(&real_ptr, (uintptr_t)r, memory_order_release); \
        }                                                                       \
        if (!r) return -1;                                                      \
        count_inc(#api);                                                        \
        maybe_register_atexit();                                                 \
        return r(a1, a2);                                                       \
    }

#define WRAP3(api, t1, a1, t2, a2, t3, a3)                                       \
    static CUresult api##_impl(t1 a1, t2 a2, t3 a3) {                            \
        typedef CUresult (*fn_t)(t1, t2, t3);                                    \
        static _Atomic uintptr_t real_ptr;                                       \
        fn_t r = (fn_t)atomic_load_explicit(&real_ptr, memory_order_acquire);    \
        if (!r) {                                                               \
            r = (fn_t)dlsym(RTLD_NEXT, #api);                                   \
            atomic_store_explicit(&real_ptr, (uintptr_t)r, memory_order_release); \
        }                                                                       \
        if (!r) return -1;                                                      \
        count_inc(#api);                                                        \
        maybe_register_atexit();                                                 \
        return r(a1, a2, a3);                                                   \
    }

#define WRAP4(api, t1, a1, t2, a2, t3, a3, t4, a4)                              \
    static CUresult api##_impl(t1 a1, t2 a2, t3 a3, t4 a4) {                     \
        typedef CUresult (*fn_t)(t1, t2, t3, t4);                                \
        static _Atomic uintptr_t real_ptr;                                       \
        fn_t r = (fn_t)atomic_load_explicit(&real_ptr, memory_order_acquire);    \
        if (!r) {                                                               \
            r = (fn_t)dlsym(RTLD_NEXT, #api);                                   \
            atomic_store_explicit(&real_ptr, (uintptr_t)r, memory_order_release); \
        }                                                                       \
        if (!r) return -1;                                                      \
        count_inc(#api);                                                        \
        maybe_register_atexit();                                                 \
        return r(a1, a2, a3, a4);                                               \
    }

#define WRAP5(api, t1, a1, t2, a2, t3, a3, t4, a4, t5, a5)                       \
    static CUresult api##_impl(t1 a1, t2 a2, t3 a3, t4 a4, t5 a5) {               \
        typedef CUresult (*fn_t)(t1, t2, t3, t4, t5);                            \
        static _Atomic uintptr_t real_ptr;                                       \
        fn_t r = (fn_t)atomic_load_explicit(&real_ptr, memory_order_acquire);    \
        if (!r) {                                                               \
            r = (fn_t)dlsym(RTLD_NEXT, #api);                                   \
            atomic_store_explicit(&real_ptr, (uintptr_t)r, memory_order_release); \
        }                                                                       \
        if (!r) return -1;                                                      \
        count_inc(#api);                                                        \
        maybe_register_atexit();                                                 \
        return r(a1, a2, a3, a4, a5);                                            \
    }

/* cuLaunchKernel has 11 params */
#define WRAP_CULAUNCHKERNEL()                                                    \
    static CUresult cuLaunchKernel_impl(CUfunction f, unsigned int gx, unsigned int gy, \
        unsigned int gz, unsigned int bx, unsigned int by, unsigned int bz,     \
        unsigned int sharedMemBytes, CUstream hStream, void** kernelParams,     \
        void** extra) {                                                          \
        typedef CUresult (*fn_t)(CUfunction, unsigned int, unsigned int,        \
            unsigned int, unsigned int, unsigned int, unsigned int,              \
            unsigned int, CUstream, void**, void**);                             \
        static _Atomic uintptr_t real_ptr;                                       \
        fn_t r = (fn_t)atomic_load_explicit(&real_ptr, memory_order_acquire);    \
        if (!r) {                                                               \
            r = (fn_t)dlsym(RTLD_NEXT, "cuLaunchKernel");                        \
            atomic_store_explicit(&real_ptr, (uintptr_t)r, memory_order_release); \
        }                                                                       \
        if (!r) return -1;                                                      \
        count_inc("cuLaunchKernel");                                             \
        maybe_register_atexit();                                                 \
        return r(f, gx, gy, gz, bx, by, bz, sharedMemBytes, hStream, kernelParams, extra); \
    }

/* cuKernelSetAttribute: CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev) */
typedef int CUkernel_attribute;
typedef void* CUkernel;
typedef void* CUdevice;
#define WRAP_CUKERNELSETATTR()                                                   \
    static CUresult cuKernelSetAttribute_impl(CUkernel_attribute attrib, int val, CUkernel kernel, CUdevice dev) { \
        typedef CUresult (*fn_t)(CUkernel_attribute, int, CUkernel, CUdevice);   \
        static _Atomic uintptr_t real_ptr;                                       \
        fn_t r = (fn_t)atomic_load_explicit(&real_ptr, memory_order_acquire);    \
        if (!r) {                                                               \
            r = (fn_t)dlsym(RTLD_NEXT, "cuKernelSetAttribute");                  \
            atomic_store_explicit(&real_ptr, (uintptr_t)r, memory_order_release); \
        }                                                                       \
        if (!r) return -1;                                                      \
        count_inc("cuKernelSetAttribute");                                       \
        maybe_register_atexit();                                                 \
        return r(attrib, val, kernel, dev);                                     \
    }

/* cuGraphExecKernelNodeSetParams_v2: (CUgraphExec, CUgraphNode, const void*) */
#define WRAP_CUGRAPHEXECKERNELNODESETPARAMS()                                    \
    static CUresult cuGraphExecKernelNodeSetParams_v2_impl(CUgraphExec hGraphExec, \
        CUgraphNode node, const void* nodeParams) {                              \
        typedef CUresult (*fn_t)(CUgraphExec, CUgraphNode, const void*);         \
        static _Atomic uintptr_t real_ptr;                                       \
        fn_t r = (fn_t)atomic_load_explicit(&real_ptr, memory_order_acquire);    \
        if (!r) {                                                               \
            r = (fn_t)dlsym(RTLD_NEXT, "cuGraphExecKernelNodeSetParams_v2");     \
            atomic_store_explicit(&real_ptr, (uintptr_t)r, memory_order_release); \
        }                                                                       \
        if (!r) return -1;                                                      \
        count_inc("cuGraphExecKernelNodeSetParams_v2");                           \
        maybe_register_atexit();                                                 \
        return r(hGraphExec, node, nodeParams);                                 \
    }

/* cuMemcpyBatchAsync_v2: EXCLUDED - causes segfault on aarch64 (see comment above) */

/* cuMemcpyHtoDAsync_v2: (CUdeviceptr, const void*, size_t, CUstream) */
#define WRAP_CUMEMCPYHTODASYNC()                                                 \
    static CUresult cuMemcpyHtoDAsync_v2_impl(CUdeviceptr dstDevice, const void* srcHost, \
        size_t ByteCount, CUstream hStream) {                                    \
        typedef CUresult (*fn_t)(CUdeviceptr, const void*, size_t, CUstream);   \
        static _Atomic uintptr_t real_ptr;                                       \
        fn_t r = (fn_t)atomic_load_explicit(&real_ptr, memory_order_acquire);    \
        if (!r) {                                                               \
            r = (fn_t)dlsym(RTLD_NEXT, "cuMemcpyHtoDAsync_v2");                  \
            atomic_store_explicit(&real_ptr, (uintptr_t)r, memory_order_release); \
        }                                                                       \
        if (!r) return -1;                                                      \
        count_inc("cuMemcpyHtoDAsync_v2");                                       \
        maybe_register_atexit();                                                 \
        return r(dstDevice, srcHost, ByteCount, hStream);                        \
    }

/* cuMemsetD8Async: (CUdeviceptr, unsigned char, size_t, CUstream) */
#define WRAP_CUMEMSETD8ASYNC()                                                   \
    static CUresult cuMemsetD8Async_impl(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) { \
        typedef CUresult (*fn_t)(CUdeviceptr, unsigned char, size_t, CUstream); \
        static _Atomic uintptr_t real_ptr;                                        \
        fn_t r = (fn_t)atomic_load_explicit(&real_ptr, memory_order_acquire);    \
        if (!r) {                                                               \
            r = (fn_t)dlsym(RTLD_NEXT, "cuMemsetD8Async");                       \
            atomic_store_explicit(&real_ptr, (uintptr_t)r, memory_order_release); \
        }                                                                       \
        if (!r) return -1;                                                      \
        count_inc("cuMemsetD8Async");                                            \
        maybe_register_atexit();                                                 \
        return r(dstDevice, uc, N, hStream);                                     \
    }

/* cuStreamWaitEvent: (CUstream, CUevent, unsigned int) */
#define WRAP_CUSTREAMWAITEVENT()                                                 \
    static CUresult cuStreamWaitEvent_impl(CUstream hStream, CUevent hEvent, unsigned int Flags) { \
        typedef CUresult (*fn_t)(CUstream, CUevent, unsigned int);              \
        static _Atomic uintptr_t real_ptr;                                       \
        fn_t r = (fn_t)atomic_load_explicit(&real_ptr, memory_order_acquire);    \
        if (!r) {                                                               \
            r = (fn_t)dlsym(RTLD_NEXT, "cuStreamWaitEvent");                     \
            atomic_store_explicit(&real_ptr, (uintptr_t)r, memory_order_release); \
        }                                                                       \
        if (!r) return -1;                                                      \
        count_inc("cuStreamWaitEvent");                                          \
        maybe_register_atexit();                                                 \
        return r(hStream, hEvent, Flags);                                        \
    }

/* Runtime API wrappers (libcudart) */
#define WRAP_CUDA_EVENTQUERY()                                                   \
    static cudaError_t cudaEventQuery_impl(cudaEvent_t event) {                  \
        typedef cudaError_t (*fn_t)(cudaEvent_t);                                \
        static _Atomic uintptr_t real_ptr;                                       \
        fn_t r = (fn_t)atomic_load_explicit(&real_ptr, memory_order_acquire);    \
        if (!r) {                                                               \
            r = (fn_t)dlsym(RTLD_NEXT, "cudaEventQuery");                        \
            atomic_store_explicit(&real_ptr, (uintptr_t)r, memory_order_release); \
        }                                                                       \
        if (!r) return -1;                                                      \
        count_inc("cudaEventQuery");                                             \
        maybe_register_atexit();                                                 \
        return r(event);                                                        \
    }
#define WRAP_CUDA_EVENTRECORD()                                                  \
    static cudaError_t cudaEventRecord_impl(cudaEvent_t event, cudaStream_t stream) { \
        typedef cudaError_t (*fn_t)(cudaEvent_t, cudaStream_t);                   \
        static _Atomic uintptr_t real_ptr;                                       \
        fn_t r = (fn_t)atomic_load_explicit(&real_ptr, memory_order_acquire);   \
        if (!r) {                                                               \
            r = (fn_t)dlsym(RTLD_NEXT, "cudaEventRecord");                        \
            atomic_store_explicit(&real_ptr, (uintptr_t)r, memory_order_release); \
        }                                                                       \
        if (!r) return -1;                                                      \
        count_inc("cudaEventRecord");                                            \
        maybe_register_atexit();                                                 \
        return r(event, stream);                                                 \
    }
#define WRAP_CUDA_STREAMWAITEVENT()                                              \
    static cudaError_t cudaStreamWaitEvent_impl(cudaStream_t stream, cudaEvent_t event, unsigned int flags) { \
        typedef cudaError_t (*fn_t)(cudaStream_t, cudaEvent_t, unsigned int);     \
        static _Atomic uintptr_t real_ptr;                                        \
        fn_t r = (fn_t)atomic_load_explicit(&real_ptr, memory_order_acquire);    \
        if (!r) {                                                               \
            r = (fn_t)dlsym(RTLD_NEXT, "cudaStreamWaitEvent");                     \
            atomic_store_explicit(&real_ptr, (uintptr_t)r, memory_order_release); \
        }                                                                       \
        if (!r) return -1;                                                      \
        count_inc("cudaStreamWaitEvent");                                         \
        maybe_register_atexit();                                                 \
        return r(stream, event, flags);                                          \
    }

/* Generate impls */
WRAP1(cuEventQuery, CUevent, hEvent)
WRAP2(cuEventRecord, CUevent, hEvent, CUstream, hStream)
WRAP2(cuGraphLaunch, CUgraphExec, hGraphExec, CUstream, hStream)
WRAP2(cuGraphUpload, CUgraphExec, hGraphExec, CUstream, hStream)
WRAP_CULAUNCHKERNEL()
WRAP_CUKERNELSETATTR()
WRAP_CUGRAPHEXECKERNELNODESETPARAMS()
WRAP_CUMEMCPYHTODASYNC()
WRAP_CUMEMSETD8ASYNC()
WRAP_CUSTREAMWAITEVENT()
WRAP_CUDA_EVENTQUERY()
WRAP_CUDA_EVENTRECORD()
WRAP_CUDA_STREAMWAITEVENT()

/* Public symbols - override libcuda.so */

/** @brief Intercept cuEventQuery; count call and forward to real implementation. */
CUresult cuEventQuery(CUevent hEvent) {
    return cuEventQuery_impl(hEvent);
}
/** @brief Intercept cuEventRecord; count call and forward to real implementation. */
CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    return cuEventRecord_impl(hEvent, hStream);
}
/** @brief Intercept cuGraphExecKernelNodeSetParams_v2; count call and forward to real implementation. */
CUresult cuGraphExecKernelNodeSetParams_v2(CUgraphExec hGraphExec, CUgraphNode node, const void* nodeParams) {
    return cuGraphExecKernelNodeSetParams_v2_impl(hGraphExec, node, nodeParams);
}
/** @brief Intercept cuGraphLaunch; count call and forward to real implementation. */
CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
    return cuGraphLaunch_impl(hGraphExec, hStream);
}
/** @brief Intercept cuGraphUpload; count call and forward to real implementation. */
CUresult cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) {
    return cuGraphUpload_impl(hGraphExec, hStream);
}
/** @brief Intercept cuKernelSetAttribute; count call and forward to real implementation. */
CUresult cuKernelSetAttribute(CUkernel_attribute attrib, int val, CUkernel kernel, CUdevice dev) {
    return cuKernelSetAttribute_impl(attrib, val, kernel, dev);
}
/** @brief Intercept cuLaunchKernel; count call and forward to real implementation. */
CUresult cuLaunchKernel(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz,
    unsigned int bx, unsigned int by, unsigned int bz, unsigned int sharedMemBytes,
    CUstream hStream, void** kernelParams, void** extra) {
    return cuLaunchKernel_impl(f, gx, gy, gz, bx, by, bz, sharedMemBytes, hStream, kernelParams, extra);
}
/* cuMemcpyBatchAsync_v2: not intercepted (causes segfault) */
/** @brief Intercept cuMemcpyHtoDAsync_v2; count call and forward to real implementation. */
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream) {
    return cuMemcpyHtoDAsync_v2_impl(dstDevice, srcHost, ByteCount, hStream);
}
/** @brief Intercept cuMemsetD8Async; count call and forward to real implementation. */
CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
    return cuMemsetD8Async_impl(dstDevice, uc, N, hStream);
}
/** @brief Intercept cuStreamWaitEvent; count call and forward to real implementation. */
CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
    return cuStreamWaitEvent_impl(hStream, hEvent, Flags);
}

/* Public symbols - override libcudart.so (Runtime API) */

/** @brief Intercept cudaEventQuery; count call and forward to real implementation. */
cudaError_t cudaEventQuery(cudaEvent_t event) {
    return cudaEventQuery_impl(event);
}
/** @brief Intercept cudaEventRecord; count call and forward to real implementation. */
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    return cudaEventRecord_impl(event, stream);
}
/** @brief Intercept cudaStreamWaitEvent; count call and forward to real implementation. */
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    return cudaStreamWaitEvent_impl(stream, event, flags);
}

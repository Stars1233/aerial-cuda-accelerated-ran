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

#if !defined CUPHY_INTERNAL_H_INCLUDED_
#define CUPHY_INTERNAL_H_INCLUDED_

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <memory>
#include <math.h>
#include <algorithm>
#include "nvlog.hpp"
#include "common_utils.hpp"
#include "memfoot_global.h"

#if USE_NVTX
#include "nvtx3/nvToolsExt.h"

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif


#ifdef __CUDACC__
#define CUDA_BOTH __host__ __device__
#define CUDA_BOTH_INLINE __forceinline__ __host__ __device__
#define CUDA_INLINE __forceinline__ __device__
#else
#define CUDA_BOTH
#define CUDA_INLINE
#ifdef WINDOWS
#define CUDA_BOTH_INLINE __inline
#else
#define CUDA_BOTH_INLINE __inline__
#endif
#endif


#define CU_CHECK(expr_to_check) do {            \
    CUresult result  = expr_to_check;           \
    if(result != CUDA_SUCCESS)                  \
    {                                           \
        const char* pErrStr;                    \
        CUresult e2 = cuGetErrorString(result,&pErrStr);      \
        if (e2 != CUDA_SUCCESS)                 \
        {                                       \
        fprintf(stderr,                         \
                "CUDA Error: %s:%i:%d; cuGetErrorString() failed with %d\n",       \
                __FILE__,                       \
                __LINE__,                       \
                result, e2);                    \
        } else {                                \
        fprintf(stderr,                         \
                "CUDA Error: %s:%i:%s\n",       \
                __FILE__,                       \
                __LINE__,                       \
                pErrStr);                       \
        }                                       \
    }                                           \
} while (0)

#if CUPHY_DEBUG
  #define DEBUG_PRINTF(...) do { printf(__VA_ARGS__); } while(0)
  #define DEBUG_PRINT_FUNC_ATTRIBUTES(func) do                                              \
          {                                                                                 \
              cudaFuncAttributes fAttr;                                                     \
              CUDA_CHECK(cudaFuncGetAttributes(&fAttr, func));                              \
              printf(#func ":\n");                                                          \
              printf("\tbinaryVersion:             %i\n",  fAttr.binaryVersion);            \
              printf("\tcacheModeCA:               %i\n",  fAttr.cacheModeCA);              \
              printf("\tconstSizeBytes:            %lu\n", fAttr.constSizeBytes);           \
              printf("\tlocalSizeBytes:            %lu\n", fAttr.localSizeBytes);           \
              printf("\tmaxDynamicSharedSizeBytes: %i\n", fAttr.maxDynamicSharedSizeBytes); \
              printf("\tmaxThreadsPerBlock:        %i\n", fAttr.maxThreadsPerBlock);        \
              printf("\tnumRegs:                   %i\n", fAttr.numRegs);                   \
              printf("\tpreferredShmemCarveout:    %i\n", fAttr.preferredShmemCarveout);    \
              printf("\tptxVersion:                %i\n", fAttr.ptxVersion);                \
              printf("\tsharedSizeBytes:           %lu\n", fAttr.sharedSizeBytes);          \
          } while(0)
  #define DEBUG_PRINT_FUNC_MAX_BLOCKS(func, blkDim, dynShMem) do                            \
          {                                                                                 \
              int maxBlocks = 0;                                                            \
              CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks,          \
                         func,                                                              \
                         blkDim.x * blkDim.y * blkDim.z,                                    \
                         dynShMem));                                                        \
              printf(#func " max blocks per SM: %i\n", maxBlocks);                          \
          } while(0)
#else
  #define DEBUG_PRINTF(...)
  #define DEBUG_PRINT_FUNC_ATTRIBUTES(func)
  #define DEBUG_PRINT_FUNC_MAX_BLOCKS(func, blkSize, dynShMem)
#endif

#define TIME_KERNEL(kernel_call, ITER_COUNT, strm)                              \
    do                                                                          \
    {                                                                           \
        cudaEvent_t eStart, eFinish;                                            \
        CUDA_CHECK(cudaEventCreate(&eStart));                                   \
        CUDA_CHECK(cudaEventCreate(&eFinish));                                  \
        cudaEventRecord(eStart, strm);                                          \
        for(size_t i = 0; i < ITER_COUNT; ++i)                                  \
        {                                                                       \
            kernel_call;                                                        \
        }                                                                       \
        cudaEventRecord(eFinish, strm);                                         \
        cudaEventSynchronize(eFinish);                                          \
        cudaError_t e = cudaGetLastError();                                     \
        if(cudaSuccess != e)                                                    \
        {                                                                       \
            fprintf(stderr, "CUDA ERROR: (%s:%i) %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(e));                                     \
        }                                                                       \
        float elapsed_ms = 0.0f;                                                \
        cudaEventElapsedTime(&elapsed_ms, eStart, eFinish);                     \
        printf("Average (%i iterations) elapsed time in usec = %.0f\n",         \
               ITER_COUNT, elapsed_ms * 1000 / ITER_COUNT);                     \
    cudaEventDestroy(eStart);                                                   \
    cudaEventDestroy(eFinish);                                                  \
  } while (0)


CUDA_BOTH_INLINE bool is_set(unsigned int flag, unsigned int mask)
{
    return (0 != (flag & mask));
}
CUDA_BOTH_INLINE unsigned int bit(unsigned int pos) { return (1UL << pos); }
CUDA_BOTH_INLINE unsigned int bit_set(unsigned int pos, unsigned int mask)
{
    return (mask |= bit(pos));
}

template <typename T>
CUDA_BOTH_INLINE T round_up_to_next(T val, T increment)
{
    return ((val + (increment - 1)) / increment) * increment;
}

template <typename T>
constexpr CUDA_BOTH_INLINE T round_up_to_next_cexp(T val, T increment)
{
    return ((val + (increment - 1)) / increment) * increment;
}

template <typename T>
CUDA_BOTH_INLINE T div_round_up(T val, T divide_by)
{
    return ((val + (divide_by - 1)) / divide_by);
}

////////////////////////////////////////////////////////////////////////
// round_up_t
template< int M, int N >
struct round_up_t
{
    enum { value = (M + N-1) / N * N };
};

////////////////////////////////////////////////////////////////////////
// div_round_up_t
template< int M, int N >
struct div_round_up_t
{
    enum { value = (M + N-1) / N };
};

////////////////////////////////////////////////////////////////////////
// div_round_down_t
template< int M, int N >
struct div_round_down_t
{
    enum { value = (M / N) };
};

////////////////////////////////////////////////////////////////////////
// Return the maximum of two integers
template <int A, int B>
struct max_t
{
    static const int value = (A < B) ? B : A;
};

/////////////////////////////////////////////////////////////////////////
template<typename Tscalar>
inline bool compare_approx(const Tscalar &a, const Tscalar &b, const Tscalar tolerance) {
    Tscalar diff = fabs(a - b);
    Tscalar m = std::max(fabs(a), fabs(b));
    Tscalar ratio = (diff >= tolerance) ? (Tscalar)(diff / m) : diff;

    return (ratio <= tolerance);
}

//specialization for __half
template<>
inline bool compare_approx(const __half &a, const __half &b, const __half tolerance) {
#if CUDART_VERSION < 12020
    float af = __half2float(a);
    float bf = __half2float(b);
    float tolf = __half2float(tolerance);
    float diff = fabs(af - bf);
    float m = std::max(fabs(af), fabs(bf));
    float ratio = (diff >= tolf) ? (diff / m) : diff;
    return (ratio <= tolf);
#else
    __half diff = __habs(a - b);
    __half m = __hmax(__habs(a), __habs(b));
    __half ratio = (diff >= tolerance) ? (diff / m) : diff;
    return (ratio <= tolerance);
#endif
}

template<typename Tcomplex, typename Tscalar>
inline bool complex_approx_equal(Tcomplex & a, Tcomplex & b, const Tscalar tol) {
    return (compare_approx<Tscalar>(a.x, b.x, tol) && compare_approx<Tscalar>(a.y, b.y, tol));
}

////////////////////////////////////////////////////////////////////////


namespace cuphy_i // cuphy internal
{

union word_t
{
    float       f32;
    uint32_t    u32;
    int32_t     i32;
    __half_raw  f16;
    __half2_raw f16x2;
    ushort2     u16x2;
};

union u128_half2x4 {
    __uint128_t u128;
    __half2 h2[4];
};

union uint4_half2x4 {
    uint4   u4;
    __half2 h2[4];
};


////////////////////////////////////////////////////////////////////////
// data types for 64-bit LD/ST

struct alignas(8) __hf2x2
{
    __half2 x;
    __half2 y;
};

struct alignas(8) half2x2_u64
{
    union
    {
        __hf2x2   hf2;
        uint64_t  u64;
    };
};

struct alignas(8) half4 {
    __half x[4];
};

struct alignas(16) half8 {
    __half x[8];
};


struct alignas(8) uint8x8
{
    union
    {
        uint8_t u8[8];
        uint64_t u64;
    };
};

struct alignas(8) bool8
{
    union
    {
        bool    b8[8];
        uint64_t  u64;
    };
};

////////////////////////////////////////////////////////////////////////
// tex_result_v4
// Struct for results of 4-component texture fetches, templated on the
// scalar output type (float or __half).
template <typename T> struct tex_result_v4;
template <> struct tex_result_v4<float>
{
    float x;
    float y;
    float z;
    float w;
};
template <> struct tex_result_v4<__half>
{
    word_t a;
    word_t b;
};

////////////////////////////////////////////////////////////////////////
// cuda_exception
// Exception class for errors from CUDA
class cuda_exception : public std::exception //
{
public:
    cuda_exception(cudaError_t s) :
        status_(s) {}
    virtual ~cuda_exception() = default;
    virtual const char* what() const noexcept
    {
        return cudaGetErrorString(status_);
    }
    cudaError_t status() const { return status_; }

private:
    cudaError_t status_;
};

////////////////////////////////////////////////////////////////////////
// cu_exception
// Exception class for errors from the CUDA driver API
class cu_exception : public std::exception //
{
public:
    cu_exception(CUresult s) : res_(s) {}
    virtual ~cu_exception() = default;
    virtual const char* what() const noexcept
    {
        const char* desc = nullptr;
        CUresult e2 = cuGetErrorString(res_, &desc);
        if (e2 != CUDA_SUCCESS) 
        {
                fprintf(stderr, "CUDA Error: %s:%i:%d; cuGetErrorString() failed with %d\n",
                __FILE__,
                __LINE__,
                res_, e2);
        }
        return desc;
    }
    CUresult result() const { return res_; }
private:
    CUresult res_;
};

template <class T>
struct device_deleter
{
    // typedef typename std::remove_all_extents<T>::type ptr_t;
    typedef T ptr_t;
    void      operator()(ptr_t* p) const
    {
        //printf("Freeing device bytes at 0x%p\n", p);
        CUDA_CHECK(cudaFree(p));
    }
};
template <class T>
struct pinned_deleter
{
    // typedef typename std::remove_all_extents<T>::type ptr_t;
    typedef T ptr_t;
    void      operator()(ptr_t* p) const { CUDA_CHECK(cudaFreeHost(p)); }
};

struct array_deleter
{
    void operator()(cudaArray_t a) const { CUDA_CHECK(cudaFreeArray(a)); }
};

struct mipmapped_array_deleter
{
    void operator()(cudaMipmappedArray_t ma) const { CUDA_CHECK(cudaFreeMipmappedArray(ma)); }
};

template <typename T>
using unique_device_ptr = std::unique_ptr<T, device_deleter<T>>;

template <typename T>
unique_device_ptr<T> make_unique_device(size_t count = 1)
{
    typedef typename unique_device_ptr<T>::pointer pointer_t;
    pointer_t                                      p;
    cudaError_t                                    res = cudaMalloc(&p, count * sizeof(T));
    if(cudaSuccess != res)
    {
        throw cuda_exception(res);
    }
    //printf("Allocated %lu device bytes at 0x%p\n", count * sizeof(T), p);
    return unique_device_ptr<T>(p);
}

typedef std::unique_ptr<struct cudaArray,          array_deleter>           unique_array_ptr;
typedef std::unique_ptr<struct cudaMipmappedArray, mipmapped_array_deleter> unique_mipmapped_array_ptr;

////////////////////////////////////////////////////////////////////////
// make_unique_array()
// Allocate a CUDA array with unique_ptr semantics
// Note: cudaArray_t is a typedef for "struct cudaArray*", so we use
// "struct cudaArray" for the unique_ptr element_type.
inline
unique_array_ptr make_unique_array(const cudaChannelFormatDesc& fdesc,
                                   size_t                       w,
                                   size_t                       h = 0,
                                   unsigned int                 flags = 0)
{
    cudaArray_t a;
    cudaError_t e = cudaMallocArray(&a, &fdesc, w, h, flags);
    if(cudaSuccess != e)
    {
        throw cuda_exception(e);
    }
    return unique_array_ptr(a);
}

////////////////////////////////////////////////////////////////////////
// make_unique_mipmapped_array()
// Allocate a CUDA mipmapped array with unique_ptr semantics
// Note: cudaMipmappedArray_t is a typedef for "struct
// cudaMipmappedArray*", so we use "struct cudaMipmappedArray" for the
// unique_ptr element_type.
inline
unique_mipmapped_array_ptr make_unique_mipmapped_array(const cudaChannelFormatDesc& fdesc,
                                                       const cudaExtent&            ext,
                                                       unsigned int                 numLevels,
                                                       unsigned int                 flags = 0)
{
    cudaMipmappedArray_t a;
    cudaError_t e = cudaMallocMipmappedArray(&a, &fdesc, ext, numLevels, flags);
    if(cudaSuccess != e)
    {
        throw cuda_exception(e);
    }
    return unique_mipmapped_array_ptr(a);
}

////////////////////////////////////////////////////////////////////////
// cu_module
// CUmodule wrapper (CUDA driver API)
class cu_module
{
public:
    cu_module(CUmodule m = nullptr) : module_(m)    {}
    cu_module(cu_module&& cm) : module_(cm.module_) { cm.module_ = nullptr; }
    ~cu_module() { if(module_) CU_CHECK(cuModuleUnload(module_)); }

    void load_from_file(const char* fname)
    {
        if(module_) CU_CHECK(cuModuleUnload(module_));
        CUresult res = cuModuleLoad(&module_, fname);
        if(CUDA_SUCCESS != res)
        {
            throw cu_exception(res);
        }
    }
    void load(const void* img)
    {
        CUresult res = cuModuleLoadData(&module_, img);
        if(CUDA_SUCCESS != res)
        {
            throw cu_exception(res);
        }
    }
    CUfunction get_function(const char* funcName)
    {
        CUfunction f;
        CUresult res = cuModuleGetFunction(&f, module_, funcName);
        if(CUDA_SUCCESS != res)
        {
            throw cu_exception(res);
        }
        return f;
    }
    CUfunction get_function(const char* funcName, std::nothrow_t)
    {
        CUfunction f   = nullptr;
        CUresult   res = cuModuleGetFunction(&f, module_, funcName);
        return (res == CUDA_SUCCESS) ? f : nullptr;
    }
    cu_module& operator=(const cu_module&) = delete;
    cu_module(const cu_module&)            = delete;
private:
    CUmodule module_;
};


////////////////////////////////////////////////////////////////////////
// channel_format_traits
// Provides texture channel format traits for different C/C++ types
template <typename T> struct channel_format_traits;
template<> struct channel_format_traits<float>
{
    static constexpr cudaChannelFormatKind kind = cudaChannelFormatKindFloat;
    static constexpr int                   bits = sizeof(float) * CHAR_BIT;
};
template<> struct channel_format_traits<__half>
{
    static constexpr cudaChannelFormatKind kind = cudaChannelFormatKindFloat;
    static constexpr int                   bits = sizeof(__half) * CHAR_BIT;
};

////////////////////////////////////////////////////////////////////////
// channel_format
// Provides a (texture) channel format descriptor for a given type and
// number of components.
template <typename T, int N> struct channel_format
{
    static cudaChannelFormatDesc desc()
    {
        cudaChannelFormatDesc d =
        {
            channel_format_traits<T>::bits,
            ((N > 1) ? channel_format_traits<T>::bits : 0),
            ((N > 2) ? channel_format_traits<T>::bits : 0),
            ((N > 3) ? channel_format_traits<T>::bits : 0),
            channel_format_traits<T>::kind
        };
        return d;
    };
};

////////////////////////////////////////////////////////////////////////
// texture_object
// Class wrapper for the CUDA texture object API
class texture_object
{
public:
    //------------------------------------------------------------------
    // Constructor
    // Initialize with explicit resource and texture descriptor
    texture_object(const cudaResourceDesc& rdesc,
                   const cudaTextureDesc&  tdesc)
    {
        texObject_    = 0;
        cudaError_t e = cudaCreateTextureObject(&texObject_,
                                                &rdesc,
                                                &tdesc,
                                                nullptr);
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
    }
    //------------------------------------------------------------------
    // Constructor
    // Initialize with a CUDA array and a texture descriptor
    // (cudaResourceDesc is initialized with the array object)
    texture_object(cudaArray_t             a,
                   const cudaTextureDesc&  tdesc)
    {
        cudaResourceDesc rdesc = {}; // zero initialize
        rdesc.resType          = cudaResourceTypeArray;
        rdesc.res.array.array  = a;
        texObject_             = 0;
        cudaError_t e = cudaCreateTextureObject(&texObject_,
                                                &rdesc,
                                                &tdesc,
                                                nullptr);
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
    }
    //------------------------------------------------------------------
    // Constructor
    // Initialize with a CUDA mipmapped array and a texture descriptor
    // (cudaResourceDesc is initialized with the mipmapped array object)
    texture_object(cudaMipmappedArray_t    a,
                   const cudaTextureDesc&  tdesc)
    {
        cudaResourceDesc rdesc  = {}; // zero initialize
        rdesc.resType           = cudaResourceTypeMipmappedArray;
        rdesc.res.mipmap.mipmap = a;
        texObject_              = 0;
        cudaError_t e = cudaCreateTextureObject(&texObject_,
                                                &rdesc,
                                                &tdesc,
                                                nullptr);
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
    }
    //------------------------------------------------------------------
    // Destructor
    ~texture_object() { CUDA_CHECK_NO_THROW(cudaDestroyTextureObject(texObject_)); }
    //------------------------------------------------------------------
    // handle()
    cudaTextureObject_t handle() const { return texObject_; }

    texture_object& operator=(const texture_object&) = delete;
    texture_object(const texture_object&)            = delete;
private:
    cudaTextureObject_t texObject_;
};

////////////////////////////////////////////////////////////////////////
// texture
// Class that manages the lifetime of a CUDA array and an associated
// CUDA texture object.
class texture
{
public:
    //------------------------------------------------------------------
    // Constructor
    texture(const cudaChannelFormatDesc& formatDesc,
            const cudaTextureDesc&       texDesc,
            size_t                       w,
            size_t                       h = 0) :
      array_(make_unique_array(formatDesc, w)),
      texObj_(array_.get(), texDesc)
    {
    }
    //------------------------------------------------------------------
    // width()
    size_t width() const
    {
        cudaExtent  ext = {}; // zero initialize
        cudaError_t e   = cudaArrayGetInfo(nullptr,
                                           &ext,
                                           nullptr,
                                           array_.get());
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
        return ext.width;
    }
    //------------------------------------------------------------------
    // width_bytes()
    size_t width_bytes() const
    {
        cudaChannelFormatDesc desc = {}; // zero initialize
        cudaExtent            ext  = {}; // zero initialize
        cudaError_t e              = cudaArrayGetInfo(&desc,
                                                      &ext,
                                                      nullptr,
                                                      array_.get());
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
        return ext.width * (desc.x + desc.y + desc.z + desc.w) / CHAR_BIT;
    }
    //------------------------------------------------------------------
    // height()
    size_t height() const
    {
        cudaExtent  ext = {}; // zero initialize
        cudaError_t e   = cudaArrayGetInfo(nullptr,
                                           &ext,
                                           nullptr,
                                           array_.get());
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
        return ext.height;
    }
    //------------------------------------------------------------------
    // copy_to()
    // (Full copy of data, with width and height equal to the array
    // size at creation.)
    void copy_to(const void*    src,
                 size_t         srcPitch,
                 cudaMemcpyKind copyKind,
                 cudaStream_t   strm)
    {
        size_t      h = std::max(height(), 1UL);
        cudaError_t e = cudaMemcpy2DToArrayAsync(array_.get(),  // array
                                                 0,             // destination X
                                                 0,             // destination Y
                                                 src,           // source data
                                                 srcPitch,      // pitch of source
                                                 width_bytes(), // width (bytes)
                                                 h,             // height
                                                 copyKind,      // kind
                                                 strm);         // stream
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
    }
    //------------------------------------------------------------------
    // tex_obj()
    const texture_object& tex_obj() const { return texObj_; }
    //------------------------------------------------------------------
    // array()
    cudaArray_t array() const { return array_.get(); }    
private:
    unique_array_ptr array_;
    texture_object   texObj_;
};

////////////////////////////////////////////////////////////////////////
// mipmapped_texture
// Class that manages the lifetime of a CUDA mipmapped array and an
// associated CUDA texture object.
class mipmapped_texture
{
public:
    //------------------------------------------------------------------
    // Constructor
    // Note: 1-D mipmapped array requires h = 0 according to CUDA docs
    mipmapped_texture(const cudaChannelFormatDesc& formatDesc,
                      const cudaTextureDesc&       texDesc,
                      unsigned int                 numLevels,
                      size_t                       w,
                      size_t                       h = 0) :
      mipArray_(make_unique_mipmapped_array(formatDesc, make_cudaExtent(w, h, 0), numLevels)),
      texObj_(mipArray_.get(), texDesc),
      numLevels_(numLevels)
    {
        // The dimension w and h should not be 0 together. If one dimension is 0, size equals to the other dimension.
        gpu_size_alloc = (w == 0 ? 1 : w) * (h == 0 ? 1 : h);
    }
    //------------------------------------------------------------------
    // tex_obj()
    // Returns a handle to the CUDA texture wrapper instance
    const texture_object& tex_obj() const { return texObj_; }
    //------------------------------------------------------------------
    // mipmapped_array()
    // Returns a handle to the underlying mimapped array allocation
    cudaMipmappedArray_t mipmapped_array() const { return mipArray_.get(); }
    //------------------------------------------------------------------
    // num_levels()
    unsigned int num_levels() const { return numLevels_; }
    //------------------------------------------------------------------
    // level_array()
    // Returns the CUDA array for a given level index
    cudaArray_t level_array(unsigned int lvl) const
    {
        cudaArray_t a = nullptr;
        cudaError_t e = cudaGetMipmappedArrayLevel(&a,
                                                   mipArray_.get(),
                                                   lvl);
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
        return a;
    }
    //------------------------------------------------------------------
    // copy_to_level()
    // (Full copy of data, with width and height equal to the array
    // size at creation.)
    void copy_to_level(const void*    src,
                       size_t         srcPitch,
                       unsigned int   lvl,
                       cudaMemcpyKind copyKind,
                       cudaStream_t   strm)
    {
        cudaArray_t           a    = level_array(lvl);
        cudaChannelFormatDesc desc = {}; // zero initialize
        cudaExtent            ext  = {}; // zero initialize
        cudaError_t           e    = cudaArrayGetInfo(&desc,
                                                      &ext,
                                                      nullptr,
                                                      a);
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
        size_t                width_bytes = ext.width * (desc.x + desc.y + desc.z + desc.w) / CHAR_BIT;
        size_t                h           = std::max(ext.height, 1UL);
        e                                 = cudaMemcpy2DToArrayAsync(a,           // array
                                                                     0,           // destination X
                                                                     0,           // destination Y
                                                                     src,         // source data
                                                                     srcPitch,    // pitch of source
                                                                     width_bytes, // width (bytes)
                                                                     h,           // height
                                                                     copyKind,    // kind
                                                                     strm);       // stream
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
    }

    // Returns the GPU size allocated for this object
    size_t get_gpu_size_alloc() const { return gpu_size_alloc; }

private:
    //------------------------------------------------------------------
    // Data
    unique_mipmapped_array_ptr mipArray_;
    texture_object             texObj_;
    unsigned int               numLevels_;
    size_t                     gpu_size_alloc;
};

} // namespace cuphy_i

////////////////////////////////////////////////////////////////////////
/**
 * Calculate the actual layer ID from PDSCH DMRS port configuration
 * 
 * This function implements the layer ID calculation formula used across PDSCH
 * processing kernels (rate matching, DMRS generation, modulation mapping) and
 * host-side code.
 * 
 * The formula accounts for:
 * - Base port ID (0-11 for Type 1, 0-5 for Type 2)
 * - Scrambling ID offset (n_scid: 0 or 1, adds 0 or 8)
 * - Extended layer support (nlAbove16: 0 or 1, adds 0 or 16 for 32-layer support)
 * 
 * @param[in] port_id DMRS port ID value
 * @param[in] n_scid Scrambling ID (0 or 1)
 * @param[in] nlAbove16 Indicator for layers above 16 (0 or 1)
 * @return Calculated layer ID for tensor indexing
 */
CUDA_BOTH_INLINE int calculate_layer_id(
    const int port_id,
    const uint8_t n_scid,
    const uint8_t nlAbove16)
{
    return port_id + 8 * n_scid + 16 * nlAbove16;
}

#endif // !defined(CUPHY_INTERNAL_H_INCLUDED_)

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

//#define CUPHY_DEBUG 1
#include "cuphy_internal.h"
#include "cuphy_kernel_util.cuh"
#include "rng.hpp"
#include "variant.hpp"
#include "type_convert.hpp"

namespace
{
const int CUPHY_RNG_NUM_THREADS = 1024;

////////////////////////////////////////////////////////////////////////
// rng_value_gen_normal
// Type-specific random number generator, using curand device functions
template <typename T> struct rng_value_gen_normal;

template <> struct rng_value_gen_normal<double>
{
    __device__ static double generate(curandState& randState) { return curand_normal_double(&randState); }
};
template <> struct rng_value_gen_normal<float>
{
    __device__ static float generate(curandState& randState) { return curand_normal(&randState); }
};
template <> struct rng_value_gen_normal<cuComplex>
{
    __device__ static cuComplex generate(curandState& randState) { return curand_normal2(&randState); }
};
template <> struct rng_value_gen_normal<cuDoubleComplex>
{
    __device__ static cuDoubleComplex generate(curandState& randState) { return curand_normal2_double(&randState); }
};

////////////////////////////////////////////////////////////////////////
// rng_value_gen_uniform
// Type-specific random number generator, using curand device functions
template <typename T> struct rng_value_gen_uniform;

template <> struct rng_value_gen_uniform<double>
{
    __device__ static double generate(curandState& randState) { return curand_uniform_double(&randState); }
};
template <> struct rng_value_gen_uniform<float>
{
    __device__ static float generate(curandState& randState) { return curand_uniform(&randState); }
};
template <> struct rng_value_gen_uniform<cuComplex>
{
    __device__ static cuComplex generate(curandState& randState)
    {
        return make_cuComplex(curand_uniform(&randState),
                              curand_uniform(&randState));
    }
};
template <> struct rng_value_gen_uniform<cuDoubleComplex>
{
    __device__ static cuDoubleComplex generate(curandState& randState)
    {
        return make_cuDoubleComplex(curand_uniform_double(&randState),
                                    curand_uniform_double(&randState));
    }
};

////////////////////////////////////////////////////////////////////////
// rng_value_gen_bits
// Random bit generator, using curand device functions
template <typename T> struct rng_value_gen_bits;

template <> struct rng_value_gen_bits<uint32_t>
{
    __device__ static uint32_t generate(curandState& randState) { return curand(&randState); }
};

////////////////////////////////////////////////////////////////////////
// rng_value_adjust
// Type-specific scale and offset application to a random value. Used
// for both normal and uniform distributions.
template <typename T> struct rng_value_adjust;

template <> struct rng_value_adjust<double>
{
    __device__ static float apply(double val, double scale, double offset)
    {
        return (val * scale) + offset;
    }
};
template <> struct rng_value_adjust<float>
{
    __device__ static float apply(float val, float scale, float offset)
    {
        return (val * scale) + offset;
    }
};
template <> struct rng_value_adjust<cuComplex>
{
    __device__ static cuComplex apply(cuComplex val, cuComplex scale, cuComplex offset)
    {
        return make_cuFloatComplex((val.x * scale.x) + offset.x,
                                   (val.y * scale.y) + offset.y);
    }
};
template <> struct rng_value_adjust<cuDoubleComplex>
{
    __device__ static cuDoubleComplex apply(cuDoubleComplex val, cuDoubleComplex scale, cuDoubleComplex offset)
    {
        return make_cuDoubleComplex((val.x * scale.x) + offset.x,
                                    (val.y * scale.y) + offset.y);
    }
};

template <typename T>
__device__
float rng_clamp_float(T min_val, T max_val, float f)
{
    f = max(f, static_cast<float>(min_val));
    f = min(f, static_cast<float>(max_val));
    return f;
}

template <typename T> struct rng_value_cast;

template <> struct rng_value_cast<float>
{
    __device__ static float cast(float f) { return f; }
};
template <> struct rng_value_cast<double>
{
    __device__ static double cast(double f) { return f; }
};
template <> struct rng_value_cast<__half>
{
    __device__ static __half cast(float f) { return __float2half(f); }
};
template <> struct rng_value_cast<__nv_fp8_e4m3>
{
    __device__ static __nv_fp8_e4m3 cast(float f) { return __nv_fp8_e4m3(f); }
};
template <> struct rng_value_cast<__nv_fp8_e5m2>
{
    __device__ static __nv_fp8_e5m2 cast(float f) { return __nv_fp8_e5m2(f); }
};
template <> struct rng_value_cast<uint32_t>
{
    __device__ static uint32_t cast(float f)
    {
        return __float2uint_rn(rng_clamp_float(0U, UINT_MAX, f));
    }
};
template <> struct rng_value_cast<int32_t>
{
    __device__ static int32_t cast(float f)
    {
        return lroundf(rng_clamp_float(INT_MIN, INT_MAX, f));
    }
};
template <> struct rng_value_cast<uint16_t>
{
    __device__ static uint16_t cast(float f)
    {
        return static_cast<uint16_t>(lroundf(rng_clamp_float(0, USHRT_MAX, f)));
    }
};
template <> struct rng_value_cast<int16_t>
{
    __device__ static int16_t cast(float f)
    {
        return static_cast<int16_t>(lroundf(rng_clamp_float(SHRT_MIN, SHRT_MAX, f)));
    }
};
template <> struct rng_value_cast<uint8_t>
{
    __device__ static uint8_t cast(float f)
    {
        return static_cast<uint8_t>(rintf(rng_clamp_float(0, UCHAR_MAX, f)));
    }
};
template <> struct rng_value_cast<int8_t>
{
    __device__ static int8_t cast(float f)
    {
        return static_cast<int8_t>(lroundf(rng_clamp_float(SCHAR_MIN, SCHAR_MAX, f)));
    }
};
template <> struct rng_value_cast<cuComplex>
{
    __device__ static cuComplex cast(cuComplex f) { return f; }
};
template <> struct rng_value_cast<cuDoubleComplex>
{
    __device__ static cuDoubleComplex cast(cuDoubleComplex f) { return f; }
};
template <> struct rng_value_cast<__half2>
{
    __device__ static __half2 cast(cuComplex f) { return type_convert<__half2>(f); }
};
template <> struct rng_value_cast<int2>
{
    __device__ static int2 cast(cuComplex f)
    {
        int2 v = {rng_value_cast<int>::cast(f.x), rng_value_cast<int>::cast(f.y)};
        return v;
    }
};
template <> struct rng_value_cast<short2>
{
    __device__ static short2 cast(cuComplex f)
    {
        short2 v = {rng_value_cast<short>::cast(f.x),
                    rng_value_cast<short>::cast(f.y)};
        return v;
    }
};
template <> struct rng_value_cast<uint2>
{
    __device__ static uint2 cast(cuComplex f)
    {
        uint2 v = {rng_value_cast<unsigned int>::cast(f.x),
                   rng_value_cast<unsigned int>::cast(f.y)};
        return v;
    }
};
template <> struct rng_value_cast<ushort2>
{
    __device__ static ushort2 cast(cuComplex f)
    {
        ushort2 v = {rng_value_cast<uint16_t>::cast(f.x),
                     rng_value_cast<uint16_t>::cast(f.y)};
        return v;
    }
};
template <> struct rng_value_cast<char2>
{
    __device__ static char2 cast(cuComplex f)
    {
        char2 v = {rng_value_cast<signed char>::cast(f.x),
                   rng_value_cast<signed char>::cast(f.y)};
        return v;
    }
};
template <> struct rng_value_cast<uchar2>
{
    __device__ static uchar2 cast(cuComplex f)
    {
        uchar2 v = {rng_value_cast<unsigned char>::cast(f.x),
                    rng_value_cast<unsigned char>::cast(f.y)};
        return v;
    }
};

////////////////////////////////////////////////////////////////////////
// cuphy_rng_init()
// Kernel to initialize RNG state in global memory
__global__ void cuphy_rng_init(unsigned long long seed,
                               curandState*       s)
{
    // Each thread gets the same seed, a different sequence number, and
    // no offset.
    curand_init(seed,             // seed
                threadIdx.x,      // sequence number
                0,                // offset
                s + threadIdx.x); // state address
}

////////////////////////////////////////////////////////////////////////
// tensor_rng_kernel()
// Kernel to generate random values with a normal distribution for tensors
template <typename TOut, typename TRand, template <typename> class TGenerator>
__global__ void tensor_rng_kernel(curandState*      s,
                                  tensor_layout_any layoutDst,
                                  TOut*             dst,
                                  TRand             scale,
                                  TRand             offset)
{
    static_assert(CUPHY_DIM_MAX == 5, "cuphy_rng_normal only defined for 5D tensors");
    //------------------------------------------------------------------
    // Retrieve random generator state from global memory
    unsigned int     idx       = threadIdx.x + (blockDim.x * blockIdx.x);
    curandState      randState = s[idx];
    typedef TGenerator<TRand>       generator_t;
    typedef rng_value_adjust<TRand> adjustor_t;
    typedef rng_value_cast<TOut>    cast_t;
    //------------------------------------------------------------------
    // Loop over destination tensor (up to 5-D)
    for(int i4 = 0; i4 < layoutDst.dimensions[4]; ++i4)
    {    
        for(int i3 = 0; i3 < layoutDst.dimensions[3]; ++i3)
        {
            for(grid_stride_index<2> it2; it2 < layoutDst.dimensions[2]; it2.next())
            {
                for(grid_stride_index<1> it1; it1 < layoutDst.dimensions[1]; it1.next())
                {
                    for(grid_stride_index<0> it0; it0 < layoutDst.dimensions[0]; it0.next())
                    {
                        int    n[5]    = {static_cast<int>(it0.value),
                                        static_cast<int>(it1.value),
                                        static_cast<int>(it2.value),
                                        i3,
                                        i4};
                        size_t out_idx = layoutDst.offset(n);
                        //printf("(i0, i1, i2, i3) = (%u, %u, %u %u), offset = %lu\n",
                        //       it0.value,
                        //       it1.value,
                        //       it2.value,
                        //       i3,
                        //       out_idx);
                        TRand val          = generator_t::generate(randState);
                        // Scale and shift to get desired mean and stddev
                        TRand val_adjusted = adjustor_t::apply(val, scale, offset);
                        dst[out_idx] = cast_t::cast(val_adjusted);
                    } // it0
                }     // it1
            }         // it2
        }             // it3
    }
    //------------------------------------------------------------------
    // Write random generator state to global memory
    s[idx] = randState;
}

////////////////////////////////////////////////////////////////////////
// tensor_rng_kernel_bits()
// Kernel to generate random values with a normal distribution for
// tensors with type CUPHY_bit. The input layout is assumed to be a
// CUPHY_BIT layout converted to the equivalent 32-bit word layout.
// The dim0Bits argument is used to zero bits in the last element of
// the first dimension. This would only be necessary if dim0 is not a
// multiple of 32.
__global__ void tensor_rng_kernel_bits(curandState*      s,
                                       tensor_layout_any layoutDst,
                                       uint32_t*         dst,
                                       int               dim0Bits)
{
    static_assert(CUPHY_DIM_MAX == 5, "cuphy_rng_normal only defined for 5D tensors");
    //------------------------------------------------------------------
    // Retrieve random generator state from global memory
    unsigned int     idx       = threadIdx.x + (blockDim.x * blockIdx.x);
    curandState      randState = s[idx];
    //------------------------------------------------------------------
    // Loop over destination tensor (up to 5-D)
    for(int i4 = 0; i4 < layoutDst.dimensions[4]; ++i4)
    {    
        for(int i3 = 0; i3 < layoutDst.dimensions[3]; ++i3)
        {
            for(grid_stride_index<2> it2; it2 < layoutDst.dimensions[2]; it2.next())
            {
                for(grid_stride_index<1> it1; it1 < layoutDst.dimensions[1]; it1.next())
                {
                    for(grid_stride_index<0> it0; it0 < layoutDst.dimensions[0]; it0.next())
                    {
                        int    n[5]    = {static_cast<int>(it0.value),
                                        static_cast<int>(it1.value),
                                        static_cast<int>(it2.value),
                                        i3,
                                        i4};
                        size_t out_idx = layoutDst.offset(n);
                        //printf("(i0, i1, i2, i3) = (%u, %u, %u %u), offset = %lu\n",
                        //       it0.value,
                        //       it1.value,
                        //       it2.value,
                        //       i3,
                        //       out_idx);
                        uint32_t val = rng_value_gen_bits<uint32_t>::generate(randState);
                        // See if we need to zero any high order bits for
                        // the last element of the first column.
                        if(((it0.value + 1) * 32) > dim0Bits)
                        {
                            int validBits = dim0Bits - (it0.value * 32);
                            uint32_t mask = (1 << validBits) - 1;
                            val           = val & mask;
                        }
                        dst[out_idx] = val;
                    } // it0
                }     // it1
            }         // it2
        }             // it3
    }                 // it4
    //------------------------------------------------------------------
    // Write random generator state to global memory
    s[idx] = randState;
}

////////////////////////////////////////////////////////////////////////
// launch_rng_normal()
// Template parameters:
// TType: cuPHY data type for output value
// TRand: data type for cuRAND functions (double for double precision
//        cuPHY types and float for everything else)
template <cuphyDataType_t TType,
          typename        TRand>
cuphyStatus_t launch_rng_normal(const tensor_layout_any& tLayout,
                                void*                    dst,
                                curandState*             randState,
                                const cuphyVariant_t&    scale,
                                const cuphyVariant_t&    offset,
                                cudaStream_t             strm)
{
    dim3 gridDim(1);
    dim3 blockDim(CUPHY_RNG_NUM_THREADS);

    typedef typename data_type_traits<TType>::type value_t;
    //------------------------------------------------------------------
    // Convert caller-provided variant values to the random type used in
    // the kernel (typically float, double, cuComplex or cuDoubleComplex)
    cuphyVariant_t           scale0    = scale;
    cuphyVariant_t           offset0   = offset;
    const cuphyDataType_t    rand_type = type_to_cuphy_type<TRand>::value;
    cuphyStatus_t            s         = cuphy_i::convert_variant(scale0, rand_type);
    if(CUPHY_STATUS_SUCCESS != s) { return s; }
    s                                  = cuphy_i::convert_variant(offset0, rand_type);
    if(CUPHY_STATUS_SUCCESS != s) { return s; }
    //------------------------------------------------------------------
    // Kernel launch
    tensor_rng_kernel<value_t, TRand, rng_value_gen_normal><<<gridDim, blockDim, 0, strm>>>(randState,                              // stored state
                                                                                            tLayout,                                // tensor dims
                                                                                            static_cast<value_t*>(dst),             // output
                                                                                            cuphy_i::variant_as_t<TRand>(scale0),   // scale
                                                                                            cuphy_i::variant_as_t<TRand>(offset0)); // offset
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

template <typename T>
T get_uniform_scale(const T& a, const T& b)
{
    return (b - a);
}
template <>
cuComplex get_uniform_scale(const cuComplex& a, const cuComplex& b)
{
    return make_cuComplex(b.x - a.x, b.y - a.y);
}
template <>
cuDoubleComplex get_uniform_scale(const cuDoubleComplex& a, const cuDoubleComplex& b)
{
    return make_cuDoubleComplex(b.x - a.x, b.y - a.y);
}

////////////////////////////////////////////////////////////////////////
// launch_rng_uniform()
// Template parameters:
// TType: cuPHY data type for output value
// TRand: data type for cuRAND functions (double for double precision
//        cuPHY types and float for everything else)
template <cuphyDataType_t TType,
          typename        TRand>
cuphyStatus_t launch_rng_uniform(const tensor_layout_any& tLayout,
                                 void*                    dst,
                                 curandState*             randState,
                                 const cuphyVariant_t&    minVal,
                                 const cuphyVariant_t&    maxVal,
                                 cudaStream_t             strm)
{
    dim3 gridDim(1);
    dim3 blockDim(CUPHY_RNG_NUM_THREADS);

    typedef typename data_type_traits<TType>::type value_t;
    //------------------------------------------------------------------
    // Convert caller provided variant values to the random type used in
    // the kernel (typically float or double)
    cuphyVariant_t           minVal0   = minVal;
    cuphyVariant_t           maxVal0   = maxVal;
    const cuphyDataType_t    rand_type = type_to_cuphy_type<TRand>::value;
    cuphyStatus_t            s         = cuphy_i::convert_variant(minVal0, rand_type);
    if(CUPHY_STATUS_SUCCESS != s) { return s; }
    s                                  = cuphy_i::convert_variant(maxVal0, rand_type);
    if(CUPHY_STATUS_SUCCESS != s) { return s; }
    TRand                    scale     = get_uniform_scale(cuphy_i::variant_as_t<TRand>(minVal0),
                                                           cuphy_i::variant_as_t<TRand>(maxVal0));
    //------------------------------------------------------------------
    // Kernel launch
    tensor_rng_kernel<value_t, TRand, rng_value_gen_uniform><<<gridDim, blockDim, 0, strm>>>(randState,                              // stored state
                                                                                             tLayout,                                // tensor dims
                                                                                             static_cast<value_t*>(dst),             // output
                                                                                             scale,                                  // scale
                                                                                             cuphy_i::variant_as_t<TRand>(minVal0)); // offset
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

////////////////////////////////////////////////////////////////////////
// launch_rng_bits()
cuphyStatus_t launch_rng_bits(const tensor_layout_any& tLayout,
                              void*                    dst,
                              curandState*             randState,
                              int                      dim0Bits,
                              cudaStream_t             strm)
{
    dim3 gridDim(1);
    dim3 blockDim(CUPHY_RNG_NUM_THREADS);

    typedef typename data_type_traits<CUPHY_BIT>::type value_t;
 
    tensor_rng_kernel_bits<<<gridDim, blockDim, 0, strm>>>(randState,                  // stored state
                                                           tLayout,                    // tensor dims
                                                           static_cast<value_t*>(dst), // output
                                                           dim0Bits);
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

} // namespace

namespace cuphy_i
{

////////////////////////////////////////////////////////////////////////
// rng::rng()
rng::rng(unsigned long long seed, cudaStream_t s) :
    randStates_(cuphy_i::make_unique_device<curandState>(CUPHY_RNG_NUM_THREADS))
{
    // Launch a kernel to populate random number generator state data in
    // global memory
    cuphy_rng_init<<<1, CUPHY_RNG_NUM_THREADS, 0, s>>>(seed, randStates_.get());

    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    if(e != cudaSuccess)
    {
        throw cuphy_i::cuda_exception(e);
    }
}

////////////////////////////////////////////////////////////////////////
// rng::normal()
cuphyStatus_t rng::normal(const tensor_desc&    t,
                          void*                 p,
                          const cuphyVariant_t& mean,
                          const cuphyVariant_t& stddev,
                          cudaStream_t          strm)
{
    //------------------------------------------------------------------
    // Create kernel structures representing the layout
    const tensor_layout_any& tLayout      = t.layout();
    cuphyDataType_t          tType        = t.type();
    //------------------------------------------------------------------
    cuphyStatus_t s  = CUPHY_STATUS_UNSUPPORTED_TYPE;
    switch(tType)
    {
    case CUPHY_R_16F:     s = launch_rng_normal<CUPHY_R_16F,     float>          (tLayout, p, randStates_.get(), stddev, mean, strm); break;
    case CUPHY_R_32F:     s = launch_rng_normal<CUPHY_R_32F,     float>          (tLayout, p, randStates_.get(), stddev, mean, strm); break;
    case CUPHY_R_64F:     s = launch_rng_normal<CUPHY_R_64F,     double>         (tLayout, p, randStates_.get(), stddev, mean, strm); break;
    case CUPHY_R_8F_E4M3: s = launch_rng_normal<CUPHY_R_8F_E4M3, float>          (tLayout, p, randStates_.get(), stddev, mean, strm); break;
    case CUPHY_R_8F_E5M2: s = launch_rng_normal<CUPHY_R_8F_E5M2, float>          (tLayout, p, randStates_.get(), stddev, mean, strm); break;
    case CUPHY_C_16F:     s = launch_rng_normal<CUPHY_C_16F,     cuComplex>      (tLayout, p, randStates_.get(), stddev, mean, strm); break;
    case CUPHY_C_32F:     s = launch_rng_normal<CUPHY_C_32F,     cuComplex>      (tLayout, p, randStates_.get(), stddev, mean, strm); break;
    case CUPHY_C_64F:     s = launch_rng_normal<CUPHY_C_64F,     cuDoubleComplex>(tLayout, p, randStates_.get(), stddev, mean, strm); break;
    default:                                                                                                                  break;
    } // switch 

    return s;
}

////////////////////////////////////////////////////////////////////////
// rng::uniform()
cuphyStatus_t rng::uniform(const tensor_desc&    t,
                           void*                 p,
                           const cuphyVariant_t& min_v,
                           const cuphyVariant_t& max_v,
                           cudaStream_t          strm)
{
    //------------------------------------------------------------------
    // Create kernel structures representing the layout
    const tensor_layout_any& tLayout      = t.layout();
    cuphyDataType_t          tType        = t.type();
    //------------------------------------------------------------------
    cuphyStatus_t            s            = CUPHY_STATUS_UNSUPPORTED_TYPE;
    switch(tType)
    {
    case CUPHY_R_8I:      s = launch_rng_uniform<CUPHY_R_8I,  float>    (tLayout, p, randStates_.get(), min_v, max_v, strm); break;
    case CUPHY_C_8I:      s = launch_rng_uniform<CUPHY_C_8I,  float2>   (tLayout, p, randStates_.get(), min_v, max_v, strm); break;
    case CUPHY_R_8U:      s = launch_rng_uniform<CUPHY_R_8U,  float>    (tLayout, p, randStates_.get(), min_v, max_v, strm); break;
    case CUPHY_C_8U:      s = launch_rng_uniform<CUPHY_C_8U,  float2>   (tLayout, p, randStates_.get(), min_v, max_v, strm); break;
    case CUPHY_R_16I:     s = launch_rng_uniform<CUPHY_R_16I, float>    (tLayout, p, randStates_.get(), min_v, max_v, strm); break;
    case CUPHY_C_16I:     s = launch_rng_uniform<CUPHY_C_16I, float2>   (tLayout, p, randStates_.get(), min_v, max_v, strm); break;
    case CUPHY_R_16U:     s = launch_rng_uniform<CUPHY_R_16U, float>    (tLayout, p, randStates_.get(), min_v, max_v, strm); break;
    case CUPHY_C_16U:     s = launch_rng_uniform<CUPHY_C_16U, float2>   (tLayout, p, randStates_.get(), min_v, max_v, strm); break;
    case CUPHY_R_32I:     s = launch_rng_uniform<CUPHY_R_32I, float>    (tLayout, p, randStates_.get(), min_v, max_v, strm); break;
    case CUPHY_C_32I:     s = launch_rng_uniform<CUPHY_C_32I, float2>   (tLayout, p, randStates_.get(), min_v, max_v, strm); break;
    case CUPHY_R_32U:     s = launch_rng_uniform<CUPHY_R_32U, float>    (tLayout, p, randStates_.get(), min_v, max_v, strm); break;
    case CUPHY_C_32U:     s = launch_rng_uniform<CUPHY_C_32U, float2>   (tLayout, p, randStates_.get(), min_v, max_v, strm); break;
    case CUPHY_R_16F:     s = launch_rng_uniform<CUPHY_R_16F, float>    (tLayout, p, randStates_.get(), min_v, max_v, strm); break;
    case CUPHY_C_16F:     s = launch_rng_uniform<CUPHY_C_16F, float2>   (tLayout, p, randStates_.get(), min_v, max_v, strm); break;
    case CUPHY_R_32F:     s = launch_rng_uniform<CUPHY_R_32F, float>    (tLayout, p, randStates_.get(), min_v, max_v, strm); break;
    case CUPHY_C_32F:     s = launch_rng_uniform<CUPHY_C_32F, float2>   (tLayout, p, randStates_.get(), min_v, max_v, strm); break;
    case CUPHY_R_64F:     s = launch_rng_uniform<CUPHY_R_64F, double>   (tLayout, p, randStates_.get(), min_v, max_v, strm); break;
    case CUPHY_C_64F:     s = launch_rng_uniform<CUPHY_C_64F, double2>  (tLayout, p, randStates_.get(), min_v, max_v, strm); break;
    case CUPHY_R_8F_E4M3: s = launch_rng_uniform<CUPHY_R_8F_E4M3, float>(tLayout, p, randStates_.get(), min_v, max_v, strm); break;
    case CUPHY_R_8F_E5M2: s = launch_rng_uniform<CUPHY_R_8F_E5M2, float>(tLayout, p, randStates_.get(), min_v, max_v, strm); break;
    case CUPHY_BIT:
        {
            tensor_layout_any tBits = word_layout_from_bit_layout(tLayout);
            s = launch_rng_bits(tBits, p, randStates_.get(), tLayout.dimensions[0], strm); break;
        }
        break;
    default:                                                                                                                  break;
    } // switch 

    return s;
}

} // namespace cuphy_i

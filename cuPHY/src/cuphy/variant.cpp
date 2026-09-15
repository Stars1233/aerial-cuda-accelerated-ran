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

#include <limits>
#include "variant.hpp"
#include "tensor_desc.hpp"
#include "type_convert.hpp"

namespace
{

template <typename T, typename U>
struct have_same_signedness
{
    constexpr static bool value = (std::is_signed<T>::value == std::is_signed<U>::value);
};

template <typename T, typename U> T cuphy_narrow_cast(U u) { return static_cast<T>(u); };
//template <> __half cuphy_narrow_cast(signed char s)    { return static_cast<__half>(static_cast<float>(s)); };
//template <> __half cuphy_narrow_cast(unsigned char u)  { return static_cast<__half>(static_cast<float>(u)); };
//template <> __half cuphy_narrow_cast(short s)          { return static_cast<__half>(static_cast<float>(s)); };
//template <> __half cuphy_narrow_cast(unsigned short s) { return static_cast<__half>(static_cast<float>(s)); };
//template <> __half cuphy_narrow_cast(int i)            { return static_cast<__half>(static_cast<float>(i)); };
//template <> __half cuphy_narrow_cast(unsigned int i)   { return static_cast<__half>(static_cast<float>(i)); };

// Narrowing casts
// See also:
// https://stackoverflow.com/questions/52863643/understanding-gslnarrow-implementation
//
// Discussion about different types of "failure":
// https://github.com/isocpp/CppCoreGuidelines/issues/1498    
template <typename TSrc, typename TDst>
bool safe_narrow_int_cast(TDst& dst, TSrc src)
{
    dst = cuphy_narrow_cast<TDst>(src);
    if(cuphy_narrow_cast<TSrc>(dst) != src)
    {
        return false;
    }
    // If the types have different "signedness", and if the signs don't
    // match, we return failure.
    if(!have_same_signedness<TDst, TSrc>::value && ((src < TSrc{}) != (dst < TDst{})))
    {
        return false;
    }
    return true;
}

template <typename TSrc, typename TDst>
bool safe_narrow_complex_int_cast(TDst& dst, TSrc src)
{
    typedef typename scalar_from_complex<TDst>::type TDstScalar;
    typedef typename scalar_from_complex<TSrc>::type TSrcScalar;
    dst.x = cuphy_narrow_cast<TDstScalar>(src.x);
    dst.y = cuphy_narrow_cast<TDstScalar>(src.y);
    if((cuphy_narrow_cast<TSrcScalar>(dst.x) != src.x) ||
       (cuphy_narrow_cast<TSrcScalar>(dst.y) != src.y))
    {
        return false;
    }
    // If the types have different "signedness", and if the signs don't
    // match, we return failure.
    if(!have_same_signedness<TDstScalar, TSrcScalar>::value &&
       (((src.x < TSrcScalar{}) != (dst.x < TDstScalar{})) || ((src.y < TSrcScalar{}) != (dst.y < TDstScalar{}))))
    {
        return false;
    }
    return true;
}

template <typename TSrc, typename TDst>
bool safe_narrow_float_to_int_cast(TDst& dst, TSrc src)
{
    const double srcRound = round(static_cast<double>(src));
    if((srcRound >= static_cast<double>(std::numeric_limits<TDst>::min())) &&
       (srcRound <= static_cast<double>(std::numeric_limits<TDst>::max())))
    {
        dst = static_cast<TDst>(srcRound);
        return true;
    }
    else
    {
        return false;
    }
}

template <typename TSrc, typename TDst>
bool safe_narrow_complex_float_to_int_cast(TDst& dst, TSrc src)
{
    return safe_narrow_float_to_int_cast(dst.x, src.x) &&
           safe_narrow_float_to_int_cast(dst.y, src.y);
}

} // namespace

namespace cuphy_i
{
    
// clang-format off
////////////////////////////////////////////////////////////////////////
// convert_variant()
cuphyStatus_t convert_variant(cuphyVariant_t& var,
                              cuphyDataType_t convertToType)
{
    switch(var.type)
    {
    case CUPHY_BIT:
        {
            unsigned char b = (0 == var.value.b1) ? 0 : 1;
            // We use static_cast instead of more generic casts (that
            // might take signed/unsigned considerations into account)
            // because here we know that the value is 0 or 1.
            switch(convertToType)
            {
            case CUPHY_BIT:       /* src and dst types identical */                                      break;
            case CUPHY_R_8I:      var.value.r8i      = static_cast<signed char>(b);                      break;
            case CUPHY_R_8U:      var.value.r8u      = static_cast<unsigned char>(b);                    break;
            case CUPHY_R_16I:     var.value.r16i     = static_cast<short>(b);                            break;
            case CUPHY_R_16U:     var.value.r16u     = static_cast<unsigned short>(b);                   break;
            case CUPHY_R_32I:     var.value.r32i     = static_cast<int>(b);                              break;
            case CUPHY_R_32U:     var.value.r32u     = static_cast<unsigned int>(b);                     break;
            case CUPHY_R_16F:     var.value.r16f     = static_cast<__half_raw>(type_convert<__half>(b)); break;
            case CUPHY_R_32F:     var.value.r32f     = static_cast<float>(b);                            break;
            case CUPHY_R_64F:     var.value.r64f     = static_cast<double>(b);                           break;
            case CUPHY_R_8F_E4M3: var.value.r8f_e4m3 = static_cast<__nv_fp8_e4m3>(b);                    break;
            case CUPHY_R_8F_E5M2: var.value.r8f_e5m2 = static_cast<__nv_fp8_e5m2>(b);                    break;
            default:
                // Conversion to complex types not supported
                return CUPHY_STATUS_INVALID_CONVERSION;
            }
        }
        break;
    case CUPHY_R_8I:
        {
            bool bSuccess = true;
            switch(convertToType)
            {
            case CUPHY_BIT:       var.value.b1 = (var.value.r8i == 0) ? 0 : 1;                                   break;
            case CUPHY_R_8I:      /* src and dst types identical */                                              break;
            case CUPHY_R_8U:      bSuccess = safe_narrow_int_cast(var.value.r8u,  var.value.r8i);                break;
            case CUPHY_R_16I:     var.value.r16i = type_convert<short>(var.value.r8i);                           break;
            case CUPHY_R_16U:     bSuccess = safe_narrow_int_cast(var.value.r16u, var.value.r8i);                break;
            case CUPHY_R_32I:     var.value.r32i = type_convert<int>(var.value.r8i);                             break;
            case CUPHY_R_32U:     bSuccess = safe_narrow_int_cast(var.value.r32u, var.value.r8i);                break;
            case CUPHY_R_16F:     var.value.r16f = static_cast<__half_raw>(type_convert<__half>(var.value.r8i)); break;
            case CUPHY_R_32F:     var.value.r32f = type_convert<float>(var.value.r8i);                           break;
            case CUPHY_R_64F:     var.value.r64f = type_convert<double>(var.value.r8i);                          break;
            case CUPHY_R_8F_E4M3: var.value.r8f_e4m3 = type_convert<__nv_fp8_e4m3>(var.value.r8i);               break;
            case CUPHY_R_8F_E5M2: var.value.r8f_e5m2 = type_convert<__nv_fp8_e5m2>(var.value.r8i);               break;
            default:
                // Conversion to complex types not supported
                return CUPHY_STATUS_INVALID_CONVERSION;
            }
            if(!bSuccess)
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case CUPHY_C_8I:
        {
            bool bSuccess = true;
            switch(convertToType)
            {
            case CUPHY_C_8I:   /* src and dst types identical */                                                break;
            case CUPHY_C_8U:   bSuccess = safe_narrow_complex_int_cast(var.value.c8u,  var.value.c8i);          break;
            case CUPHY_C_16I:  var.value.c16i = type_convert<short2>(var.value.c8i);                            break;
            case CUPHY_C_16U:  bSuccess = safe_narrow_complex_int_cast(var.value.c16u, var.value.c8i);          break;
            case CUPHY_C_32I:  var.value.c32i = type_convert<int2>(var.value.c8i);                              break;
            case CUPHY_C_32U:  bSuccess = safe_narrow_complex_int_cast(var.value.c32u, var.value.c8i);          break;
            case CUPHY_C_16F:  var.value.c16f = static_cast<__half2_raw>(type_convert<__half2>(var.value.c8i)); break;
            case CUPHY_C_32F:  var.value.c32f = type_convert<cuComplex>(var.value.c8i);                         break;
            case CUPHY_C_64F:  var.value.c64f = type_convert<cuDoubleComplex>(var.value.c8i);                   break;
            default:
                // Conversion to bit and real types from complex inputs not supported
                return CUPHY_STATUS_INVALID_CONVERSION;
            }
            if(!bSuccess)
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case CUPHY_R_8U:
        {
            bool bSuccess = true;
            switch(convertToType)
            {
            case CUPHY_BIT:       var.value.b1   = (var.value.r8u == 0) ? 0 : 1;                                 break;
            case CUPHY_R_8I:      bSuccess       = safe_narrow_int_cast(var.value.r8i,  var.value.r8u);          break;
            case CUPHY_R_8U:      /* src and dst types identical */                                              break;
            case CUPHY_R_16I:     var.value.r16i = type_convert<short>(var.value.r8u);                           break;
            case CUPHY_R_16U:     bSuccess       = type_convert<unsigned short>(var.value.r8u);                  break;
            case CUPHY_R_32I:     var.value.r32i = type_convert<int>(var.value.r8u);                             break;
            case CUPHY_R_32U:     var.value.r32u = type_convert<unsigned int>(var.value.r8u);                    break;
            case CUPHY_R_16F:     var.value.r16f = static_cast<__half_raw>(type_convert<__half>(var.value.r8u)); break;
            case CUPHY_R_32F:     var.value.r32f = type_convert<float>(var.value.r8u);                           break;
            case CUPHY_R_64F:     var.value.r64f = type_convert<double>(var.value.r8u);                          break;
            case CUPHY_R_8F_E4M3: var.value.r8f_e4m3 = type_convert<__nv_fp8_e4m3>(var.value.r8u);               break;
            case CUPHY_R_8F_E5M2: var.value.r8f_e5m2 = type_convert<__nv_fp8_e5m2>(var.value.r8u);               break;
            default:
                // Conversion to complex types not supported
                return CUPHY_STATUS_INVALID_CONVERSION;
            }
            if(!bSuccess)
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case CUPHY_C_8U:
        {
            bool bSuccess = true;
            switch(convertToType)
            {
            case CUPHY_C_8I:   bSuccess = safe_narrow_complex_int_cast(var.value.c8i, var.value.c8u);           break;
            case CUPHY_C_8U:   /* src and dst types identical */                                                break;
            case CUPHY_C_16I:  var.value.c16i = type_convert<short2>(var.value.c8u);                            break;
            case CUPHY_C_16U:  var.value.c16u = type_convert<ushort2>(var.value.c8u);                           break;
            case CUPHY_C_32I:  var.value.c32i = type_convert<int2>(var.value.c8u);                              break;
            case CUPHY_C_32U:  var.value.c32u = type_convert<uint2>(var.value.c8u);                             break;
            case CUPHY_C_16F:  var.value.c16f = static_cast<__half2_raw>(type_convert<__half2>(var.value.c8u)); break;
            case CUPHY_C_32F:  var.value.c32f = type_convert<cuComplex>(var.value.c8u);                         break;
            case CUPHY_C_64F:  var.value.c64f = type_convert<cuDoubleComplex>(var.value.c8u);                   break;
            default:
                // Conversion to bit and real types from complex inputs not supported
                return CUPHY_STATUS_INVALID_CONVERSION;
            }
            if(!bSuccess)
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case CUPHY_R_16I:
        {
            bool bSuccess = true;
            switch(convertToType)
            {
            case CUPHY_BIT:       var.value.b1 = (var.value.r16i == 0) ? 0 : 1;                                                  break;
            case CUPHY_R_8I:      bSuccess = safe_narrow_int_cast(var.value.r8i, var.value.r16i);                                break;
            case CUPHY_R_8U:      bSuccess = safe_narrow_int_cast(var.value.r8u, var.value.r16i);                                break;
            case CUPHY_R_16I:     /* src and dst types identical */                                                              break;
            case CUPHY_R_16U:     bSuccess = safe_narrow_int_cast(var.value.r16u, var.value.r16i);                               break;
            case CUPHY_R_32I:     var.value.r32i = type_convert<int>(var.value.r16i);                                            break;
            case CUPHY_R_32U:     bSuccess = safe_narrow_int_cast(var.value.r32u, var.value.r16i);                               break;
            case CUPHY_R_16F:     { __half h(static_cast<float>(var.value.r16i)); var.value.r16f = static_cast<__half_raw>(h); } break;
            case CUPHY_R_32F:     var.value.r32f = type_convert<float>(var.value.r16i);                                          break;
            case CUPHY_R_64F:     var.value.r64f = type_convert<double>(var.value.r16i);                                         break;
            case CUPHY_R_8F_E4M3: var.value.r8f_e4m3 = type_convert<__nv_fp8_e4m3>(var.value.r16i);                              break;
            case CUPHY_R_8F_E5M2: var.value.r8f_e5m2 = type_convert<__nv_fp8_e5m2>(var.value.r16i);                              break;
            default:
                // Conversion to complex types not supported
                return CUPHY_STATUS_INVALID_CONVERSION;
            }
            if(!bSuccess)
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case CUPHY_C_16I:
        {
            bool bSuccess = true;
            switch(convertToType)
            {
            case CUPHY_C_8I:   bSuccess = safe_narrow_complex_int_cast(var.value.c8i,  var.value.c16i); break;
            case CUPHY_C_8U:   bSuccess = safe_narrow_complex_int_cast(var.value.c8u,  var.value.c16i); break;
            case CUPHY_C_16I:  /* src and dst types identical */                                        break;
            case CUPHY_C_16U:  bSuccess = safe_narrow_complex_int_cast(var.value.c16u, var.value.c16i); break;
            case CUPHY_C_32I:  var.value.c32i = type_convert<int2>(var.value.c8i);                      break;
            case CUPHY_C_32U:  bSuccess = safe_narrow_complex_int_cast(var.value.c32u, var.value.c16i); break;
            case CUPHY_C_16F:
                {
                    __half2 h2 = __floats2half2_rn(var.value.c16i.x, var.value.c16i.y);
                    var.value.c16f = static_cast<__half2_raw>(h2);
                }
                break;
            case CUPHY_C_32F:  var.value.c32f = type_convert<cuComplex>(var.value.c16i);                break;
            case CUPHY_C_64F:  var.value.c64f = type_convert<cuDoubleComplex>(var.value.c16i);          break;
            default:
                // Conversion to bit and real types from complex inputs not supported
                return CUPHY_STATUS_INVALID_CONVERSION;
            }
            if(!bSuccess)
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case CUPHY_R_16U:
        {
            bool bSuccess = true;
            switch(convertToType)
            {
            case CUPHY_BIT:    var.value.b1   = (var.value.r16u == 0) ? 0 : 1;                        break;
            case CUPHY_R_8I:   bSuccess       = safe_narrow_int_cast(var.value.r8i, var.value.r16u);  break;
            case CUPHY_R_8U:   bSuccess       = safe_narrow_int_cast(var.value.r8u, var.value.r16u);  break;
            case CUPHY_R_16I:  bSuccess       = safe_narrow_int_cast(var.value.r16i, var.value.r16u); break;
            case CUPHY_R_16U:  /* src and dst types identical */                                      break;
            case CUPHY_R_32I:  var.value.r32i = type_convert<int>(var.value.r16u);                    break;
            case CUPHY_R_32U:  var.value.r32u = type_convert<unsigned int>(var.value.r16u);           break;
            case CUPHY_R_16F:
                {
                    __half h(type_convert<float>(var.value.r16u)); // may be inf
                    var.value.r16f = static_cast<__half_raw>(h);
                }
                break;
            case CUPHY_R_32F:     var.value.r32f     = type_convert<float>(var.value.r16u);           break;
            case CUPHY_R_64F:     var.value.r64f     = type_convert<double>(var.value.r16u);          break;
            case CUPHY_R_8F_E4M3: var.value.r8f_e4m3 = type_convert<__nv_fp8_e4m3>(var.value.r16u);   break;
            case CUPHY_R_8F_E5M2: var.value.r8f_e5m2 = type_convert<__nv_fp8_e5m2>(var.value.r16u);   break;
            default:
                // Conversion to complex types not supported
                return CUPHY_STATUS_INVALID_CONVERSION;
            }
            if(!bSuccess)
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case CUPHY_C_16U:
        {
            bool bSuccess = true;
            switch(convertToType)
            {
            case CUPHY_C_8I:   bSuccess = safe_narrow_complex_int_cast(var.value.c8i, var.value.c16u);  break;
            case CUPHY_C_8U:   bSuccess = safe_narrow_complex_int_cast(var.value.c8i, var.value.c16u);  break;
            case CUPHY_C_16I:  bSuccess = safe_narrow_complex_int_cast(var.value.c16i, var.value.c16u); break;
            case CUPHY_C_16U:  /* src and dst types identical */                                        break;
            case CUPHY_C_32I:  var.value.c32i = type_convert<int2>(var.value.c16u);                     break;
            case CUPHY_C_32U:  var.value.c32u = type_convert<uint2>(var.value.c16u);                    break;
            case CUPHY_C_16F:
                {
                    __half2 h2 = __floats2half2_rn(var.value.c16u.x, var.value.c16u.y); // may get inf
                    var.value.c16f = static_cast<__half2_raw>(h2);
                }
                break;
            case CUPHY_C_32F:  var.value.c32f = type_convert<cuComplex>(var.value.c16u);                break;
            case CUPHY_C_64F:  var.value.c64f = type_convert<cuDoubleComplex>(var.value.c16u);          break;
            default:
                // Conversion to bit and real types from complex inputs not supported
                return CUPHY_STATUS_INVALID_CONVERSION;
            }
            if(!bSuccess)
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case CUPHY_R_32I:
        {
            bool bSuccess = true;
            switch(convertToType)
            {
            case CUPHY_BIT:    var.value.b1 = (var.value.r32i == 0) ? 0 : 1;                    break;
            case CUPHY_R_8I:   bSuccess = safe_narrow_int_cast(var.value.r8i, var.value.r32i);  break;
            case CUPHY_R_8U:   bSuccess = safe_narrow_int_cast(var.value.r8u, var.value.r32i);  break;
            case CUPHY_R_16I:  bSuccess = safe_narrow_int_cast(var.value.r16i, var.value.r32i); break;
            case CUPHY_R_16U:  bSuccess = safe_narrow_int_cast(var.value.r16u, var.value.r32i); break;
            case CUPHY_R_32I:  /* src and dst types identical */                                break;
            case CUPHY_R_32U:  bSuccess = safe_narrow_int_cast(var.value.r32u, var.value.r32i); break;
            case CUPHY_R_16F:
                {
                    __half h(static_cast<float>(var.value.r32i)); // may get inf
                    var.value.r16f = static_cast<__half_raw>(h);
                }
                break;
            case CUPHY_R_32F:     var.value.r32f = type_convert<float>(var.value.r32i);             break;
            case CUPHY_R_64F:     var.value.r64f = type_convert<double>(var.value.r32i);            break;
            case CUPHY_R_8F_E4M3: var.value.r8f_e4m3 = type_convert<__nv_fp8_e4m3>(var.value.r32i); break;
            case CUPHY_R_8F_E5M2: var.value.r8f_e5m2 = type_convert<__nv_fp8_e5m2>(var.value.r32i); break;
            default:
                // Conversion to complex types not supported
                return CUPHY_STATUS_INVALID_CONVERSION;
            }
            if(!bSuccess)
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case CUPHY_C_32I:
        {
            bool bSuccess = true;
            switch(convertToType)
            {
            case CUPHY_C_8I:   bSuccess = safe_narrow_complex_int_cast(var.value.c8i,  var.value.c32i); break;
            case CUPHY_C_8U:   bSuccess = safe_narrow_complex_int_cast(var.value.c8u,  var.value.c32i); break;
            case CUPHY_C_16I:  bSuccess = safe_narrow_complex_int_cast(var.value.c16i, var.value.c32i); break;
            case CUPHY_C_16U:  bSuccess = safe_narrow_complex_int_cast(var.value.c16u, var.value.c32i); break;
            case CUPHY_C_32I:  /* src and dst types identical */                                        break;
            case CUPHY_C_32U:  bSuccess = safe_narrow_complex_int_cast(var.value.c32u, var.value.c32i); break;
            case CUPHY_C_16F:
                {
                    __half2 h2     = __floats2half2_rn(var.value.c32i.x, var.value.c32i.y); // may get inf
                    var.value.c16f = static_cast<__half2_raw>(h2);
                }
                break;
            case CUPHY_C_32F:  var.value.c32f = type_convert<cuComplex>(var.value.c32i);                break;
            case CUPHY_C_64F:  var.value.c64f = type_convert<cuDoubleComplex>(var.value.c32i);          break;
            default:
                // Conversion to bit and real types from complex inputs not supported
                return CUPHY_STATUS_INVALID_CONVERSION;
            }
            if(!bSuccess)
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case CUPHY_R_32U:
        {
            bool bSuccess = true;
            switch(convertToType)
            {
            case CUPHY_BIT:    var.value.b1   = (var.value.r32u == 0) ? 0 : 1;                        break;
            case CUPHY_R_8I:   bSuccess       = safe_narrow_int_cast(var.value.r8i, var.value.r32u);  break;
            case CUPHY_R_8U:   bSuccess       = safe_narrow_int_cast(var.value.r8u, var.value.r32u);  break;
            case CUPHY_R_16I:  bSuccess       = safe_narrow_int_cast(var.value.r16i, var.value.r32u); break;
            case CUPHY_R_16U:  bSuccess       = safe_narrow_int_cast(var.value.r16u, var.value.r32u); break;
            case CUPHY_R_32I:  bSuccess       = safe_narrow_int_cast(var.value.r32i, var.value.r32u); break;
            case CUPHY_R_32U:  /* src and dst types identical */                                      break;
            case CUPHY_R_16F:
                {
                    __half h(static_cast<float>(var.value.r32u));
                    var.value.r16f = static_cast<__half_raw>(h);
                }
                break;
            case CUPHY_R_32F:     var.value.r32f     = type_convert<float>(var.value.r32u);           break;
            case CUPHY_R_64F:     var.value.r64f     = type_convert<double>(var.value.r32u);          break;
            case CUPHY_R_8F_E4M3: var.value.r8f_e4m3 = type_convert<__nv_fp8_e4m3>(var.value.r32u);   break;
            case CUPHY_R_8F_E5M2: var.value.r8f_e5m2 = type_convert<__nv_fp8_e5m2>(var.value.r32u);   break;
            default:
                // Conversion to complex types not supported
                return CUPHY_STATUS_INVALID_CONVERSION;
            }
            if(!bSuccess)
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case CUPHY_C_32U:
        {
            bool bSuccess = true;
            switch(convertToType)
            {
            case CUPHY_C_8I:   bSuccess = safe_narrow_complex_int_cast(var.value.c8i, var.value.c32u);  break;
            case CUPHY_C_8U:   bSuccess = safe_narrow_complex_int_cast(var.value.c8i, var.value.c32u);  break;
            case CUPHY_C_16I:  bSuccess = safe_narrow_complex_int_cast(var.value.c16i, var.value.c32u); break;
            case CUPHY_C_16U:  bSuccess = safe_narrow_complex_int_cast(var.value.c16i, var.value.c32u); break;
            case CUPHY_C_32I:  bSuccess = safe_narrow_complex_int_cast(var.value.c32i, var.value.c32u); break;
            case CUPHY_C_32U:  /* src and dst types identical */                                        break;
            case CUPHY_C_16F:
                {
                    __half2 h2     = __floats2half2_rn(var.value.c32u.x, var.value.c32u.y); // may get inf
                    var.value.c16f = static_cast<__half2_raw>(h2);
                }
                break;
            case CUPHY_C_32F:  var.value.c32f = type_convert<cuComplex>(var.value.c32u);                break;
            case CUPHY_C_64F:  var.value.c64f = type_convert<cuDoubleComplex>(var.value.c32u);          break;
            default:
                // Conversion to bit and real types from complex inputs not supported
                return CUPHY_STATUS_INVALID_CONVERSION;
            }
            if(!bSuccess)
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case CUPHY_R_16F:
        {
            // Max FP16 value is +/-65504
            bool bSuccess = true;
            switch(convertToType)
            {
            case CUPHY_BIT:       var.value.b1 = (__half2float(__half(var.value.r16f)) == 0.0f) ? 0 : 1;              break;
            case CUPHY_R_8I:      bSuccess = safe_narrow_float_to_int_cast(var.value.r8i,  __half(var.value.r16f)); break;
            case CUPHY_R_8U:      bSuccess = safe_narrow_float_to_int_cast(var.value.r8u,  __half(var.value.r16f)); break;
            case CUPHY_R_16I:     bSuccess = safe_narrow_float_to_int_cast(var.value.r16i, __half(var.value.r16f)); break;
            case CUPHY_R_16U:     bSuccess = safe_narrow_float_to_int_cast(var.value.r16u, __half(var.value.r16f)); break;
            case CUPHY_R_32I:     bSuccess = safe_narrow_float_to_int_cast(var.value.r32i, __half(var.value.r16f)); break;
            case CUPHY_R_32U:     bSuccess = safe_narrow_float_to_int_cast(var.value.r32u, __half(var.value.r16f)); break;
            case CUPHY_R_16F:     /* src and dst types identical */                                                 break;
            case CUPHY_R_32F:     var.value.r32f = type_convert<float>(var.value.r16f);                             break;
            case CUPHY_R_64F:     var.value.r64f = type_convert<double>(var.value.r16f);                            break;
            case CUPHY_R_8F_E4M3: var.value.r8f_e4m3 = type_convert<__nv_fp8_e4m3>(var.value.r16f);                 break;
            case CUPHY_R_8F_E5M2: var.value.r8f_e5m2 = type_convert<__nv_fp8_e5m2>(var.value.r16f);                 break;
            default:
                // Conversion to complex types not supported
                return CUPHY_STATUS_INVALID_CONVERSION;
            }
            if(!bSuccess)
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case CUPHY_C_16F:
        {
            bool bSuccess = true;
            switch(convertToType)
            {
            case CUPHY_C_8I:   bSuccess = safe_narrow_complex_float_to_int_cast(var.value.c8i,  static_cast<__half2>(var.value.c16f)); break;
            case CUPHY_C_8U:   bSuccess = safe_narrow_complex_float_to_int_cast(var.value.c8u,  static_cast<__half2>(var.value.c16f)); break;
            case CUPHY_C_16I:  bSuccess = safe_narrow_complex_float_to_int_cast(var.value.c16i, static_cast<__half2>(var.value.c16f)); break;
            case CUPHY_C_16U:  bSuccess = safe_narrow_complex_float_to_int_cast(var.value.c16u, static_cast<__half2>(var.value.c16f)); break;
            case CUPHY_C_32I:  bSuccess = safe_narrow_complex_float_to_int_cast(var.value.c32i, static_cast<__half2>(var.value.c16f)); break;
            case CUPHY_C_32U:  bSuccess = safe_narrow_complex_float_to_int_cast(var.value.c32u, static_cast<__half2>(var.value.c16f)); break;
            case CUPHY_C_16F:  /* src and dst types identical */                                                                       break;
            case CUPHY_C_32F:  var.value.c32f = type_convert<cuComplex>(var.value.c16f);                                               break;
            case CUPHY_C_64F:  var.value.c64f = type_convert<cuDoubleComplex>(var.value.c16f);                                         break;
            default:
                // Conversion to bit and real types from complex inputs not supported
                return CUPHY_STATUS_INVALID_CONVERSION;
            }
            if(!bSuccess)
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case CUPHY_R_32F:
        {
            bool bSuccess = true;
            switch(convertToType)
            {
            case CUPHY_BIT:       var.value.b1 = (var.value.r32f == 0) ? 0 : 1;                              break;
            case CUPHY_R_8I:      bSuccess = safe_narrow_float_to_int_cast(var.value.r8i,  var.value.r32f);  break;
            case CUPHY_R_8U:      bSuccess = safe_narrow_float_to_int_cast(var.value.r8u,  var.value.r32f);  break;
            case CUPHY_R_16I:     bSuccess = safe_narrow_float_to_int_cast(var.value.r16i, var.value.r32f);  break;
            case CUPHY_R_16U:     bSuccess = safe_narrow_float_to_int_cast(var.value.r16u, var.value.r32f);  break;
            case CUPHY_R_32I:     bSuccess = safe_narrow_float_to_int_cast(var.value.r32i, var.value.r32f);  break;
            case CUPHY_R_32U:     bSuccess = safe_narrow_float_to_int_cast(var.value.r32u, var.value.r32f);  break;
            case CUPHY_R_16F:     { __half h(var.value.r32f); var.value.r16f = static_cast<__half_raw>(h); } break;
            case CUPHY_R_32F:     /* src and dst types identical */                                          break;
            case CUPHY_R_64F:     var.value.r64f = type_convert<double>(var.value.r32f);                     break;
            case CUPHY_R_8F_E4M3: var.value.r8f_e4m3 = type_convert<__nv_fp8_e4m3>(var.value.r32f);          break;
            case CUPHY_R_8F_E5M2: var.value.r8f_e5m2 = type_convert<__nv_fp8_e5m2>(var.value.r32f);          break;
            default:
                // Conversion to complex types not supported
                return CUPHY_STATUS_INVALID_CONVERSION;
            }
            if(!bSuccess)
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case CUPHY_C_32F:
        {
            bool bSuccess = true;
            switch(convertToType)
            {
            case CUPHY_C_8I:   bSuccess = safe_narrow_complex_float_to_int_cast(var.value.c8i,  var.value.c32f); break;
            case CUPHY_C_8U:   bSuccess = safe_narrow_complex_float_to_int_cast(var.value.c8u,  var.value.c32f); break;
            case CUPHY_C_16I:  bSuccess = safe_narrow_complex_float_to_int_cast(var.value.c16i, var.value.c32f); break;
            case CUPHY_C_16U:  bSuccess = safe_narrow_complex_float_to_int_cast(var.value.c16u, var.value.c32f); break;
            case CUPHY_C_32I:  bSuccess = safe_narrow_complex_float_to_int_cast(var.value.c32i, var.value.c32f); break;
            case CUPHY_C_32U:  bSuccess = safe_narrow_complex_float_to_int_cast(var.value.c32u, var.value.c32f); break;
            case CUPHY_C_16F:
                {
                    __half2 h2 = type_convert<__half2>(var.value.c32f); // may get inf
                    var.value.c16f = static_cast<__half2_raw>(h2);
                }
                break;
            case CUPHY_C_32F:  /* src and dst types identical */                                    break;
            case CUPHY_C_64F:  var.value.c64f = type_convert<cuDoubleComplex>(var.value.c32f);      break;
            default:
                // Conversion to bit and real types from complex inputs not supported
                return CUPHY_STATUS_INVALID_CONVERSION;
            }
            if(!bSuccess)
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case CUPHY_R_64F:
        {
            bool bSuccess = true;
            switch(convertToType)
            {
            case CUPHY_BIT:       var.value.b1 = (var.value.r64f == 0) ? 0 : 1;                              break;
            case CUPHY_R_8I:      bSuccess = safe_narrow_float_to_int_cast(var.value.r8i,  var.value.r64f);  break;
            case CUPHY_R_8U:      bSuccess = safe_narrow_float_to_int_cast(var.value.r8u,  var.value.r64f);  break;
            case CUPHY_R_16I:     bSuccess = safe_narrow_float_to_int_cast(var.value.r16i, var.value.r64f);  break;
            case CUPHY_R_16U:     bSuccess = safe_narrow_float_to_int_cast(var.value.r16u, var.value.r64f);  break;
            case CUPHY_R_32I:     bSuccess = safe_narrow_float_to_int_cast(var.value.r32i, var.value.r64f);  break;
            case CUPHY_R_32U:     bSuccess = safe_narrow_float_to_int_cast(var.value.r32u, var.value.r64f);  break;
            case CUPHY_R_16F:     { __half h(var.value.r64f); var.value.r16f = static_cast<__half_raw>(h); } break;
            case CUPHY_R_32F:     var.value.r32f = type_convert<float>(var.value.r64f);                      break;
            case CUPHY_R_64F:     /* src and dst types identical */                                          break;
            case CUPHY_R_8F_E4M3: var.value.r8f_e4m3 = type_convert<__nv_fp8_e4m3>(var.value.r64f);          break;
            case CUPHY_R_8F_E5M2: var.value.r8f_e5m2 = type_convert<__nv_fp8_e5m2>(var.value.r64f);          break;
            default:
                // Conversion to complex types not supported
                return CUPHY_STATUS_INVALID_CONVERSION;
            }
            if(!bSuccess)
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case CUPHY_C_64F:
        {
            bool bSuccess = true;
            switch(convertToType)
            {
            case CUPHY_C_8I:   bSuccess = safe_narrow_complex_float_to_int_cast(var.value.c8i,  var.value.c64f); break;
            case CUPHY_C_8U:   bSuccess = safe_narrow_complex_float_to_int_cast(var.value.c8u,  var.value.c64f); break;
            case CUPHY_C_16I:  bSuccess = safe_narrow_complex_float_to_int_cast(var.value.c16i, var.value.c64f); break;
            case CUPHY_C_16U:  bSuccess = safe_narrow_complex_float_to_int_cast(var.value.c16u, var.value.c64f); break;
            case CUPHY_C_32I:  bSuccess = safe_narrow_complex_float_to_int_cast(var.value.c32i, var.value.c64f); break;
            case CUPHY_C_32U:  bSuccess = safe_narrow_complex_float_to_int_cast(var.value.c32u, var.value.c64f); break;
            case CUPHY_C_16F:
                {
                    __half2 h2(__half(var.value.c64f.x), __half(var.value.c64f.y)); // may get inf
                    var.value.c16f = static_cast<__half2_raw>(h2);
                }
                break;
            case CUPHY_C_32F:  var.value.c32f = type_convert<cuComplex>(var.value.c64f);            break;
            case CUPHY_C_64F:  /* src and dst types identical */                                    break;
            default:
                // Conversion to bit and real types from complex inputs not supported
                return CUPHY_STATUS_INVALID_CONVERSION;
            }
            if(!bSuccess)
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case CUPHY_R_8F_E4M3:
        {
            bool bSuccess = true;
            switch(convertToType)
            {
            case CUPHY_BIT:       var.value.b1 = (static_cast<float>(var.value.r8f_e4m3) == 0.0f) ? 0 : 1;               break;
            case CUPHY_R_8I:      bSuccess = safe_narrow_float_to_int_cast(var.value.r8i,  __half(var.value.r8f_e4m3)); break;
            case CUPHY_R_8U:      bSuccess = safe_narrow_float_to_int_cast(var.value.r8u,  __half(var.value.r8f_e4m3)); break;
            case CUPHY_R_16I:     bSuccess = safe_narrow_float_to_int_cast(var.value.r16i, __half(var.value.r8f_e4m3)); break;
            case CUPHY_R_16U:     bSuccess = safe_narrow_float_to_int_cast(var.value.r16u, __half(var.value.r8f_e4m3)); break;
            case CUPHY_R_32I:     bSuccess = safe_narrow_float_to_int_cast(var.value.r32i, __half(var.value.r8f_e4m3)); break;
            case CUPHY_R_32U:     bSuccess = safe_narrow_float_to_int_cast(var.value.r32u, __half(var.value.r8f_e4m3)); break;
            case CUPHY_R_16F:     var.value.r16f = type_convert<__half>(var.value.r8f_e4m3);                            break;
            case CUPHY_R_32F:     var.value.r32f = type_convert<float>(var.value.r8f_e4m3);                             break;
            case CUPHY_R_64F:     var.value.r64f = type_convert<double>(var.value.r8f_e4m3);                            break;
            case CUPHY_R_8F_E4M3: /* src and dst types identical */                                                     break;
            case CUPHY_R_8F_E5M2: var.value.r8f_e5m2 = type_convert<__nv_fp8_e5m2>(var.value.r8f_e4m3);                 break;
            default:
                // Conversion to complex types not supported
                return CUPHY_STATUS_INVALID_CONVERSION;
            }
            if(!bSuccess)
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case CUPHY_R_8F_E5M2:
        {
            bool bSuccess = true;
            switch(convertToType)
            {
            case CUPHY_BIT:       var.value.b1 = (static_cast<float>(var.value.r8f_e5m2) == 0.0f) ? 0 : 1;               break;
            case CUPHY_R_8I:      bSuccess = safe_narrow_float_to_int_cast(var.value.r8i,  __half(var.value.r8f_e5m2)); break;
            case CUPHY_R_8U:      bSuccess = safe_narrow_float_to_int_cast(var.value.r8u,  __half(var.value.r8f_e5m2)); break;
            case CUPHY_R_16I:     bSuccess = safe_narrow_float_to_int_cast(var.value.r16i, __half(var.value.r8f_e5m2)); break;
            case CUPHY_R_16U:     bSuccess = safe_narrow_float_to_int_cast(var.value.r16u, __half(var.value.r8f_e5m2)); break;
            case CUPHY_R_32I:     bSuccess = safe_narrow_float_to_int_cast(var.value.r32i, __half(var.value.r8f_e5m2)); break;
            case CUPHY_R_32U:     bSuccess = safe_narrow_float_to_int_cast(var.value.r32u, __half(var.value.r8f_e5m2)); break;
            case CUPHY_R_16F:     var.value.r16f = type_convert<__half>(var.value.r8f_e5m2);                            break;
            case CUPHY_R_32F:     var.value.r32f = type_convert<float>(var.value.r8f_e5m2);                             break;
            case CUPHY_R_64F:     var.value.r64f = type_convert<double>(var.value.r8f_e5m2);                            break;
            case CUPHY_R_8F_E4M3: var.value.r8f_e4m3 = type_convert<__nv_fp8_e4m3>(var.value.r8f_e5m2);                 break;
            case CUPHY_R_8F_E5M2: /* src and dst types identical */                                                     break;
            default:
                // Conversion to complex types not supported
                return CUPHY_STATUS_INVALID_CONVERSION;
            }
            if(!bSuccess)
            {
                return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    default:
        // Don't expect to be here...
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    // On success, set the output type
    var.type = convertToType;
    return CUPHY_STATUS_SUCCESS;
}
// clang-format on

} // namespace cuphy_i

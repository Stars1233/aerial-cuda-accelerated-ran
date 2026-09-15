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

#if !defined(CUPHY_VARIANT_HPP_INCLUDED_)
#define CUPHY_VARIANT_HPP_INCLUDED_

#include "cuphy.h"
#include "tensor_desc.hpp"

namespace cuphy_i
{

template <cuphyDataType_t TType> typename data_type_traits<TType>::type variant_as(const cuphyVariant_t& v);
template <> inline typename data_type_traits<CUPHY_R_8I>::type      variant_as<CUPHY_R_8I>     (const cuphyVariant_t& v) { return v.value.r8i;      }
template <> inline typename data_type_traits<CUPHY_C_8I>::type      variant_as<CUPHY_C_8I>     (const cuphyVariant_t& v) { return v.value.c8i;      }
template <> inline typename data_type_traits<CUPHY_R_8U>::type      variant_as<CUPHY_R_8U>     (const cuphyVariant_t& v) { return v.value.r8u;      }
template <> inline typename data_type_traits<CUPHY_C_8U>::type      variant_as<CUPHY_C_8U>     (const cuphyVariant_t& v) { return v.value.c8u;      }
template <> inline typename data_type_traits<CUPHY_R_16I>::type     variant_as<CUPHY_R_16I>    (const cuphyVariant_t& v) { return v.value.r16i;     }
template <> inline typename data_type_traits<CUPHY_C_16I>::type     variant_as<CUPHY_C_16I>    (const cuphyVariant_t& v) { return v.value.c16i;     }
template <> inline typename data_type_traits<CUPHY_R_16U>::type     variant_as<CUPHY_R_16U>    (const cuphyVariant_t& v) { return v.value.r16u;     }
template <> inline typename data_type_traits<CUPHY_C_16U>::type     variant_as<CUPHY_C_16U>    (const cuphyVariant_t& v) { return v.value.c16u;     }
template <> inline typename data_type_traits<CUPHY_R_32I>::type     variant_as<CUPHY_R_32I>    (const cuphyVariant_t& v) { return v.value.r32i;     }
template <> inline typename data_type_traits<CUPHY_C_32I>::type     variant_as<CUPHY_C_32I>    (const cuphyVariant_t& v) { return v.value.c32i;     }
template <> inline typename data_type_traits<CUPHY_R_32U>::type     variant_as<CUPHY_R_32U>    (const cuphyVariant_t& v) { return v.value.r32u;     }
template <> inline typename data_type_traits<CUPHY_C_32U>::type     variant_as<CUPHY_C_32U>    (const cuphyVariant_t& v) { return v.value.c32u;     }
template <> inline typename data_type_traits<CUPHY_R_16F>::type     variant_as<CUPHY_R_16F>    (const cuphyVariant_t& v) { return v.value.r16f;     }
template <> inline typename data_type_traits<CUPHY_C_16F>::type     variant_as<CUPHY_C_16F>    (const cuphyVariant_t& v) { return v.value.c16f;     }
template <> inline typename data_type_traits<CUPHY_R_32F>::type     variant_as<CUPHY_R_32F>    (const cuphyVariant_t& v) { return v.value.r32f;     }
template <> inline typename data_type_traits<CUPHY_C_32F>::type     variant_as<CUPHY_C_32F>    (const cuphyVariant_t& v) { return v.value.c32f;     }
template <> inline typename data_type_traits<CUPHY_R_64F>::type     variant_as<CUPHY_R_64F>    (const cuphyVariant_t& v) { return v.value.r64f;     }
template <> inline typename data_type_traits<CUPHY_C_64F>::type     variant_as<CUPHY_C_64F>    (const cuphyVariant_t& v) { return v.value.c64f;     }
template <> inline typename data_type_traits<CUPHY_R_8F_E4M3>::type variant_as<CUPHY_R_8F_E4M3>(const cuphyVariant_t& v) { return v.value.r8f_e4m3; }
template <> inline typename data_type_traits<CUPHY_R_8F_E5M2>::type variant_as<CUPHY_R_8F_E5M2>(const cuphyVariant_t& v) { return v.value.r8f_e5m2; }

template <typename T> T variant_as_t(const cuphyVariant_t& v);
template <> inline int8_t          variant_as_t<int8_t>         (const cuphyVariant_t& v) { return v.value.r8i;  }
template <> inline char2           variant_as_t<char2>          (const cuphyVariant_t& v) { return v.value.c8i;  }
template <> inline uint8_t         variant_as_t<uint8_t>        (const cuphyVariant_t& v) { return v.value.r8u;  }
template <> inline uchar2          variant_as_t<uchar2>         (const cuphyVariant_t& v) { return v.value.c8u;  }
template <> inline int16_t         variant_as_t<int16_t>        (const cuphyVariant_t& v) { return v.value.r16i; }
template <> inline short2          variant_as_t<short2>         (const cuphyVariant_t& v) { return v.value.c16i; }
template <> inline uint16_t        variant_as_t<uint16_t>       (const cuphyVariant_t& v) { return v.value.r16u; }
template <> inline ushort2         variant_as_t<ushort2>        (const cuphyVariant_t& v) { return v.value.c16u; }
template <> inline int32_t         variant_as_t<int32_t>        (const cuphyVariant_t& v) { return v.value.r32i; }
template <> inline int2            variant_as_t<int2>           (const cuphyVariant_t& v) { return v.value.c32i; }
template <> inline uint32_t        variant_as_t<uint32_t>       (const cuphyVariant_t& v) { return v.value.r32u; }
template <> inline uint2           variant_as_t<uint2>          (const cuphyVariant_t& v) { return v.value.c32u; }
template <> inline __half_raw      variant_as_t<__half_raw>     (const cuphyVariant_t& v) { return v.value.r16f; }
template <> inline __half2_raw     variant_as_t<__half2_raw>    (const cuphyVariant_t& v) { return v.value.c16f; }
template <> inline __half          variant_as_t<__half>         (const cuphyVariant_t& v) { return static_cast<__half>(v.value.r16f); }
template <> inline __half2         variant_as_t<__half2>        (const cuphyVariant_t& v) { return static_cast<__half2>(v.value.c16f); }
template <> inline float           variant_as_t<float>          (const cuphyVariant_t& v) { return v.value.r32f; }
template <> inline cuComplex       variant_as_t<cuComplex>      (const cuphyVariant_t& v) { return v.value.c32f; }
template <> inline double          variant_as_t<double>         (const cuphyVariant_t& v) { return v.value.r64f; }
template <> inline cuDoubleComplex variant_as_t<cuDoubleComplex>(const cuphyVariant_t& v) { return v.value.c64f; }
template <> inline __nv_fp8_e4m3   variant_as_t<__nv_fp8_e4m3>  (const cuphyVariant_t& v) { return v.value.r8f_e4m3; }
template <> inline __nv_fp8_e5m2   variant_as_t<__nv_fp8_e5m2>  (const cuphyVariant_t& v) { return v.value.r8f_e5m2; }

////////////////////////////////////////////////////////////////////////
// convert_variant()
// Attempts to convert the given variant, in-place, to the given type.
// Returns one of the following values:
// CUPHY_STATUS_SUCCESS,
// CUPHY_STATUS_INVALID_ARGUMENT
// CUPHY_STATUS_INVALID_CONVERSION
// CUPHY_STATUS_VALUE_OUT_OF_RANGE
cuphyStatus_t convert_variant(cuphyVariant_t& var, cuphyDataType_t convertToType);

////////////////////////////////////////////////////////////////////////
// cuphy_i::variant
class variant : public cuphyVariant_t
{
public:
    variant()
    {
        type = CUPHY_VOID;
    }
    template <typename T>
    variant(T t)
    {
        type = type_to_cuphy_type<T>::value;
        set(t);
    }
    void set(const signed char&     sc)  { type = CUPHY_R_8I;  value.r8i  = sc;  }
    void set(const char2&           c2)  { type = CUPHY_C_8I;  value.c8i  = c2;  }
    void set(const unsigned char&   uc)  { type = CUPHY_R_8U;  value.r8u  = uc;  }
    void set(const uchar2&          uc2) { type = CUPHY_C_8U;  value.c8u  = uc2; }
    void set(const short&           s)   { type = CUPHY_R_16I; value.r16i = s;   }
    void set(const short2&          s2)  { type = CUPHY_C_16I; value.c16i = s2;  }
    void set(const unsigned short&  us)  { type = CUPHY_R_16U; value.r16u = us;  }
    void set(const ushort2&         us2) { type = CUPHY_C_16U; value.c16u = us2; }
    void set(const int&             i)   { type = CUPHY_R_32I; value.r32i = i;   }
    void set(const int2&            i2)  { type = CUPHY_C_32I; value.c32i = i2;  }
    void set(const unsigned int&    u)   { type = CUPHY_R_32U; value.r32u = u;   }
    void set(const uint2&           u2)  { type = CUPHY_C_32U; value.c32u = u2;  }
    void set(const __half&          h)   { type = CUPHY_R_16F; value.r16f = h;   }
    void set(const __half2&         h2)  { type = CUPHY_C_16F; value.c16f = h2;  }
    void set(const float&           f)   { type = CUPHY_R_32F; value.r32f = f;   }
    void set(const cuComplex&       c)   { type = CUPHY_C_32F; value.c32f = c;   }
    void set(const double&          d)   { type = CUPHY_R_64F; value.r64f = d;   }
    void set(const cuDoubleComplex& dc)  { type = CUPHY_C_64F; value.c64f = dc;  }
    template <typename T> T& as();
};

} // namespace cuphy_i

#endif // !defined(CUPHY_VARIANT_HPP_INCLUDED_)

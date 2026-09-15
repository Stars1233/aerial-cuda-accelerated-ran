// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <aerial/casts/casts.hpp>

// aerial_common propagates gsl_CONFIG_CONTRACT_VIOLATION_THROWS via its INTERFACE compile
// definitions. If this fires, the CMakeLists.txt INTERFACE propagation is broken.
#ifndef gsl_CONFIG_CONTRACT_VIOLATION_THROWS
#error "aerial_common must propagate gsl_CONFIG_CONTRACT_VIOLATION_THROWS to all consumers"
#endif

#include <cstdint>
#include <tuple>

#include <gsl-lite/gsl-lite.hpp>
#include <gtest/gtest.h>

namespace {

struct TrivialPod
{
    std::uint32_t a;
    std::uint32_t b;
};
static_assert(std::is_trivial_v<TrivialPod>);
static_assert(std::is_standard_layout_v<TrivialPod>);

TEST(AssumeCast, MutableVoidPtrReturnsMutableTypedPtr)
{
    TrivialPod obj{42U, 7U};
    void *raw = &obj;
    TrivialPod *result = aerial::casts::assume_cast<TrivialPod>(raw);
    ASSERT_EQ(result, &obj);
    EXPECT_EQ(result->a, 42U);
    EXPECT_EQ(result->b, 7U);
}

TEST(AssumeCast, ConstVoidPtrReturnsConstTypedPtr)
{
    const TrivialPod obj{1U, 2U};
    const void *raw = &obj;
    const TrivialPod *result = aerial::casts::assume_cast<TrivialPod>(raw);
    ASSERT_EQ(result, &obj);
    EXPECT_EQ(result->a, 1U);
    EXPECT_EQ(result->b, 2U);
}

TEST(AssumeCast, MutableVoidPtrToUint8T)
{
    alignas(TrivialPod) std::uint8_t buf[sizeof(TrivialPod)] = {};
    buf[0] = 0xDE;
    void *raw = buf;
    std::uint8_t *result = aerial::casts::assume_cast<std::uint8_t>(raw);
    ASSERT_EQ(result, buf);
    EXPECT_EQ(*result, 0xDE);
}

TEST(AssumeCastRef, MutableReferenceReturnsMutableRef)
{
    alignas(TrivialPod) std::uint8_t buf[sizeof(TrivialPod)];
    auto *obj = new (buf) TrivialPod{10U, 20U};
    TrivialPod &result = aerial::casts::assume_cast_ref<TrivialPod>(*obj);
    EXPECT_EQ(result.a, 10U);
    EXPECT_EQ(result.b, 20U);
    result.a = 99U;
    EXPECT_EQ(obj->a, 99U);
    EXPECT_EQ(&result, obj);
}

TEST(AssumeCastRef, ConstReferenceReturnsConstRef)
{
    const TrivialPod obj{5U, 6U};
    const TrivialPod &result =
            aerial::casts::assume_cast_ref<TrivialPod>(obj);
    EXPECT_EQ(result.a, 5U);
    EXPECT_EQ(result.b, 6U);
    EXPECT_EQ(&result, &obj);
}

TEST(AssumeCast, NullptrMutableReturnsNullptr)
{
    // nullptr is documented as a valid input; gsl_Expects(0 % alignof(T) == 0) passes.
    EXPECT_EQ(aerial::casts::assume_cast<TrivialPod>(static_cast<void *>(nullptr)), nullptr);
}

TEST(AssumeCast, NullptrConstReturnsNullptr)
{
    EXPECT_EQ(aerial::casts::assume_cast<TrivialPod>(static_cast<const void *>(nullptr)), nullptr);
}

TEST(AssumeCast, MisalignedPointerThrowsFailFast)
{
    // TrivialPod has alignof == 4, so offset by 1 is guaranteed unaligned.
    alignas(TrivialPod) std::uint8_t buf[sizeof(TrivialPod) + 1] = {};
    void *misaligned = buf + 1;
    EXPECT_THROW(std::ignore = aerial::casts::assume_cast<TrivialPod>(misaligned), gsl_lite::fail_fast);
}

TEST(AssumeCast, MisalignedPointerExceptionIsCatchable)
{
    alignas(TrivialPod) std::uint8_t buf[sizeof(TrivialPod) + 1] = {};
    void *misaligned = buf + 1;
    bool caught = false;
    try
    {
        std::ignore = aerial::casts::assume_cast<TrivialPod>(misaligned);
    }
    catch(const gsl_lite::fail_fast &)
    {
        caught = true;
    }
    EXPECT_TRUE(caught);
}

} // namespace

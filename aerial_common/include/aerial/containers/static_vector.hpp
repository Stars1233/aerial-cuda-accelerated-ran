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

#ifndef AERIAL_CONTAINERS_STATIC_VECTOR_HPP
#define AERIAL_CONTAINERS_STATIC_VECTOR_HPP

#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

#include <gsl-lite/gsl-lite.hpp>

namespace aerial {

/**
 * @brief Fixed-capacity contiguous vector with storage embedded in the object.
 *
 * `static_vector` is intended for Aerial hot paths that need vector-like size
 * management without dynamic allocation. It mirrors the core `std::inplace_vector`
 * shape expected in C++26 while remaining a self-contained C++20 utility.
 *
 * All bounds-sensitive operations are narrow contracts. Violating a precondition
 * triggers `gsl_Expects`, which is configured by `aerial_common` to throw
 * `gsl_lite::fail_fast` in tests.
 *
 * @tparam T Element type. Active elements are constructed in embedded storage.
 * @tparam Capacity Maximum number of elements, fixed at compile time.
 */
template <typename T, std::size_t Capacity>
class static_vector final
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    static_vector() noexcept = default;

    /**
     * @brief Construct @p count value-initialized elements.
     * @param[in] count Number of elements to construct.
     * @pre count <= capacity().
     */
    explicit static_vector(const size_type count)
    {
        gsl_Expects(count <= capacity());
        construct_with_cleanup([&]() {
            while (size_ < count)
            {
                static_cast<void>(emplace_back());
            }
        });
    }

    /**
     * @brief Construct @p count copies of @p value.
     * @param[in] count Number of elements to construct.
     * @param[in] value Element value to copy into each constructed slot.
     * @pre count <= capacity().
     */
    static_vector(const size_type count, const value_type &value)
    {
        gsl_Expects(count <= capacity());
        construct_with_cleanup([&]() {
            while (size_ < count)
            {
                static_cast<void>(emplace_back(value));
            }
        });
    }

    /**
     * @brief Construct elements from an initializer list.
     * @param[in] values Element values to copy into the vector.
     * @pre values.size() <= capacity().
     */
    static_vector(std::initializer_list<value_type> values)
    {
        gsl_Expects(values.size() <= capacity());
        construct_with_cleanup([&]() {
            for (const auto &value : values)
            {
                static_cast<void>(emplace_back(value));
            }
        });
    }

    /**
     * @brief Copy-construct from another static_vector.
     * @param[in] other Vector whose active elements are copied.
     */
    static_vector(const static_vector &other)
    {
        construct_with_cleanup([&]() {
            for (const auto &value : other)
            {
                static_cast<void>(emplace_back(value));
            }
        });
    }

    /**
     * @brief Move-construct from another static_vector.
     * @param[in] other Vector whose active elements are moved.
     *
     * This constructor is noexcept when value_type is nothrow move-constructible.
     */
    static_vector(static_vector &&other) noexcept(std::is_nothrow_move_constructible_v<value_type>)
    {
        construct_with_cleanup([&]() {
            for (auto &value : other)
            {
                static_cast<void>(emplace_back(std::move(value)));
            }
        });
    }

    static_vector &operator=(const static_vector &other)
    {
        if (this == &other)
        {
            return *this;
        }

        clear();
        for (const auto &value : other)
        {
            static_cast<void>(emplace_back(value));
        }
        return *this;
    }

    static_vector &operator=(static_vector &&other) noexcept(
            std::is_nothrow_move_constructible_v<value_type>)
    {
        if (this == &other)
        {
            return *this;
        }

        clear();
        for (auto &value : other)
        {
            static_cast<void>(emplace_back(std::move(value)));
        }
        return *this;
    }

    ~static_vector() noexcept
    {
        static_assert(
                std::is_nothrow_destructible_v<value_type>,
                "static_vector requires nothrow-destructible value_type");
        clear();
    }

    [[nodiscard]] iterator begin() noexcept
    {
        return data();
    }

    [[nodiscard]] const_iterator begin() const noexcept
    {
        return data();
    }

    [[nodiscard]] const_iterator cbegin() const noexcept
    {
        return begin();
    }

    [[nodiscard]] iterator end() noexcept
    {
        if constexpr (Capacity == 0U)
        {
            return nullptr;
        }
        else
        {
            return data() + size_;
        }
    }

    [[nodiscard]] const_iterator end() const noexcept
    {
        if constexpr (Capacity == 0U)
        {
            return nullptr;
        }
        else
        {
            return data() + size_;
        }
    }

    [[nodiscard]] const_iterator cend() const noexcept
    {
        return end();
    }

    [[nodiscard]] reverse_iterator rbegin() noexcept
    {
        return reverse_iterator{end()};
    }

    [[nodiscard]] const_reverse_iterator rbegin() const noexcept
    {
        return const_reverse_iterator{end()};
    }

    [[nodiscard]] const_reverse_iterator crbegin() const noexcept
    {
        return rbegin();
    }

    [[nodiscard]] reverse_iterator rend() noexcept
    {
        return reverse_iterator{begin()};
    }

    [[nodiscard]] const_reverse_iterator rend() const noexcept
    {
        return const_reverse_iterator{begin()};
    }

    [[nodiscard]] const_reverse_iterator crend() const noexcept
    {
        return rend();
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return size_ == 0U;
    }

    [[nodiscard]] bool full() const noexcept
    {
        return size_ == capacity();
    }

    [[nodiscard]] size_type size() const noexcept
    {
        return size_;
    }

    [[nodiscard]] static constexpr size_type capacity() noexcept
    {
        return Capacity;
    }

    [[nodiscard]] static constexpr size_type max_size() noexcept
    {
        return capacity();
    }

    /**
     * @brief Return mutable reference to an active element.
     * @param[in] index Element index.
     * @return Reference to element at @p index.
     * @pre index < size().
     */
    [[nodiscard]] reference operator[](const size_type index)
    {
        gsl_Expects(index < size_);
        return *ptr_at(index);
    }

    /**
     * @brief Return const reference to an active element.
     * @param[in] index Element index.
     * @return Const reference to element at @p index.
     * @pre index < size().
     */
    [[nodiscard]] const_reference operator[](const size_type index) const
    {
        gsl_Expects(index < size_);
        return *ptr_at(index);
    }

    /**
     * @brief Return first active element.
     * @return Reference to first element.
     * @pre !empty().
     */
    [[nodiscard]] reference front()
    {
        gsl_Expects(!empty());
        return (*this)[0U];
    }

    /**
     * @brief Return first active element.
     * @return Const reference to first element.
     * @pre !empty().
     */
    [[nodiscard]] const_reference front() const
    {
        gsl_Expects(!empty());
        return (*this)[0U];
    }

    /**
     * @brief Return last active element.
     * @return Reference to last element.
     * @pre !empty().
     */
    [[nodiscard]] reference back()
    {
        gsl_Expects(!empty());
        return (*this)[size_ - 1U];
    }

    /**
     * @brief Return last active element.
     * @return Const reference to last element.
     * @pre !empty().
     */
    [[nodiscard]] const_reference back() const
    {
        gsl_Expects(!empty());
        return (*this)[size_ - 1U];
    }

    [[nodiscard]] pointer data() noexcept
    {
        if constexpr (Capacity == 0U)
        {
            return nullptr;
        }
        else
        {
            return raw_ptr_at(0U);
        }
    }

    [[nodiscard]] const_pointer data() const noexcept
    {
        if constexpr (Capacity == 0U)
        {
            return nullptr;
        }
        else
        {
            return raw_ptr_at(0U);
        }
    }

    /**
     * @brief Construct an element at the end of the active range.
     * @param[in] args Constructor arguments forwarded to T.
     * @return Reference to the inserted element.
     * @pre size() < capacity().
     */
    template <typename... Args>
    [[nodiscard]]
    reference emplace_back(Args &&...args)
    {
        gsl_Expects(size_ < capacity());
        auto *obj = std::construct_at(raw_ptr_at(size_), std::forward<Args>(args)...);
        ++size_;
        return *obj;
    }

    /**
     * @brief Copy an element to the end of the active range.
     * @param[in] value Element value to append.
     * @pre size() < capacity().
     */
    void push_back(const value_type &value)
    {
        static_cast<void>(emplace_back(value));
    }

    /**
     * @brief Move an element to the end of the active range.
     * @param[in] value Element value to append.
     * @pre size() < capacity().
     */
    void push_back(value_type &&value)
    {
        static_cast<void>(emplace_back(std::move(value)));
    }

    /**
     * @brief Destroy the last active element.
     * @pre !empty().
     */
    void pop_back()
    {
        gsl_Expects(!empty());
        --size_;
        destroy_at(size_);
    }

    void clear() noexcept
    {
        while (size_ > 0U)
        {
            --size_;
            destroy_at(size_);
        }
    }

    /**
     * @brief Resize by destroying trailing elements or default-constructing new ones.
     * @param[in] count Target active size.
     * @pre count <= capacity().
     */
    void resize(const size_type count)
    {
        gsl_Expects(count <= capacity());
        while (size_ > count)
        {
            pop_back();
        }
        while (size_ < count)
        {
            static_cast<void>(emplace_back());
        }
    }

    /**
     * @brief Resize by destroying trailing elements or copy-constructing new ones.
     * @param[in] count Target active size.
     * @param[in] value Value used for newly-created elements.
     * @pre count <= capacity().
     */
    void resize(const size_type count, const value_type &value)
    {
        gsl_Expects(count <= capacity());
        while (size_ > count)
        {
            pop_back();
        }
        while (size_ < count)
        {
            static_cast<void>(emplace_back(value));
        }
    }

private:
    static constexpr size_type storage_capacity = Capacity == 0U ? 1U : Capacity;

    template <std::invocable ConstructLoop>
    void construct_with_cleanup(ConstructLoop &&construct_loop)
    {
        try
        {
            std::forward<ConstructLoop>(construct_loop)();
        }
        catch (...)
        {
            clear();
            throw;
        }
    }

    [[nodiscard]] pointer raw_ptr_at(const size_type index) noexcept
    {
        return reinterpret_cast<pointer>(storage_ + (index * sizeof(value_type)));
    }

    [[nodiscard]] const_pointer raw_ptr_at(const size_type index) const noexcept
    {
        return reinterpret_cast<const_pointer>(storage_ + (index * sizeof(value_type)));
    }

    [[nodiscard]] pointer ptr_at(const size_type index) noexcept
    {
        return std::launder(raw_ptr_at(index));
    }

    [[nodiscard]] const_pointer ptr_at(const size_type index) const noexcept
    {
        return std::launder(raw_ptr_at(index));
    }

    void destroy_at(const size_type index) noexcept
    {
        if constexpr (!std::is_trivially_destructible_v<value_type>)
        {
            std::destroy_at(ptr_at(index));
        }
    }

    alignas(value_type) std::byte storage_[storage_capacity * sizeof(value_type)]{};
    size_type size_{0U};
};

} // namespace aerial

#endif // AERIAL_CONTAINERS_STATIC_VECTOR_HPP

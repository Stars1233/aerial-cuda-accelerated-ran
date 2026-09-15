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

#if !defined(SCF_5G_FAPI_TTI_DISPATCH_HPP_INCLUDED_)
#define SCF_5G_FAPI_TTI_DISPATCH_HPP_INCLUDED_

#include <array>
#include <concepts>
#include <cstdint>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

#include "aerial/casts/casts.hpp"
#include "scf_5g_fapi_slot_concepts.hpp"
#include "scf_5g_fapi_tags.hpp"
#include "scf_5g_fapi.h"
#include "nvlog.h"
#include "nvlog_fmt.hpp"

namespace scf_5g_fapi
{


// ---------------------------------------------------------------------------
// DispatchResult
// ---------------------------------------------------------------------------

/**
 * Result returned by TtiDispatch::dispatch_pdus().
 *
 * @tparam N  Number of parsers in the TtiDispatch instantiation.
 */
template<std::size_t N>
struct DispatchResult
{
    uint16_t processed{0};       //!< PDUs matched and forwarded to a parser.
    uint16_t skipped_unknown{0}; //!< PDUs with unrecognised pdu_type (logged, skipped).

    /**
     * Per-parser hit count.
     *
     * Index matches the parser's position in the tuple
     * (i.e. TtiDispatch::parser_pdu_type<I>() identifies entry [I]).
     */
    std::array<uint16_t, N> per_parser_counts{};
};

// ---------------------------------------------------------------------------
// TtiDispatch
// ---------------------------------------------------------------------------

/**
 * Generic TTI PDU dispatcher.
 *
 * Holds one parser per PDU type and routes each PDU to the matching parser
 * via a short-circuit fold over std::index_sequence — no virtual dispatch,
 * no std::function overhead.
 *
 * This class is exposed in a public header to allow direct unit-testing of
 * the dispatch logic independently of the slot processors.
 *
 * @tparam Enum     PDU-type enum (scf_fapi_dl_tti_pdu_type_t or
 *                  scf_fapi_ul_tti_pdu_type_t).
 * @tparam Parsers  Pack of final parser classes, each satisfying
 *                  PduParser<Enum>.
 */
template<typename Enum, PduParser<Enum>... Parsers>
class TtiDispatch
{
public:
    static constexpr std::size_t parser_count = sizeof...(Parsers);
    using result_t = DispatchResult<parser_count>;

    explicit TtiDispatch(Parsers... ps) noexcept : parsers_{std::move(ps)...} {}

    // ------------------------------------------------------------------
    // Compile-time introspection (used by slot processors for validation)
    // ------------------------------------------------------------------

    /**
     * Returns the pdu_type enum value of the parser at compile-time index I.
     *
     * @tparam I  Zero-based index of the parser in the Parsers pack.
     * @return    The pdu_type enum constant for parser I.
     *            Return value must be checked.
     */
    template<std::size_t I>
    [[nodiscard]] static constexpr Enum parser_pdu_type() noexcept
    {
        return std::tuple_element_t<I, std::tuple<Parsers...>>::pdu_type;
    }

    /**
     * Returns the tuple index for parser type P.
     *
     * @tparam P  Parser type to locate; must be one of the Parsers.
     * @return    Zero-based parser index in the dispatch tuple.
     *            Return value must be checked.
     */
    template<typename P>
    [[nodiscard]] static consteval std::size_t parser_index() noexcept
    {
        constexpr std::size_t idx =
            parser_index_impl<P>(std::index_sequence_for<Parsers...>{});
        static_assert(idx < parser_count,
                      "Parser type is not part of this TtiDispatch");
        return idx;
    }

    /**
     * Returns a reference to the parser of type P.
     *
     * Useful for calling pre-dispatch setup methods (e.g. setup_cell()).
     *
     * @tparam P  Parser type to retrieve; must be one of the Parsers.
     * @return    Reference to the parser instance of type P.
     *            Return value must be checked.
     */
    template<typename P>
    [[nodiscard]] P& get_parser() noexcept { return std::get<P>(parsers_); }

    // ------------------------------------------------------------------
    // Runtime dispatch
    // ------------------------------------------------------------------

    /**
     * Walk the PDU payload span and dispatch each PDU to its matching parser.
     *
     * Unknown pdu_type values are logged and skipped (not treated as errors).
     * Malformed PDU headers (pdu_size zero, smaller than the generic header,
     * or larger than remaining payload) abort the walk and increment skipped_unknown.
     * The caller is responsible for validating message integrity before calling
     * this function.
     *
     * @param[in] sfn       System Frame Number from the TTI request header.
     * @param[in] slot      Slot number from the TTI request header.
     * @param[in] payload   Byte span covering the PDU array (req->payload onwards).
     * @param[in] num_pdus  Number of PDUs declared in the TTI request header.
     * @return              DispatchResult with processed/skipped counts and per-parser hits.
     *                      Return value must be checked.
     */
    [[nodiscard]] result_t dispatch_pdus(uint16_t sfn, uint16_t slot,
                                          std::span<const uint8_t> payload,
                                          uint16_t num_pdus) noexcept
    {
        return walk_pdus(sfn, slot, payload, num_pdus,
            [this, sfn, slot](const scf_fapi_generic_pdu_info_t* hdr, result_t& r) noexcept {
                const auto type = static_cast<Enum>(
                    static_cast<std::underlying_type_t<Enum>>(hdr->pdu_type));
                if (dispatch_one(type, sfn, slot, hdr, r.per_parser_counts,
                                 std::index_sequence_for<Parsers...>{}))
                {
                    ++r.processed;
                }
                else
                {
                    ++r.skipped_unknown;
                    NVLOGW_FMT(detail::k_tag,
                               "TTI dispatch: unknown pdu_type={} sfn={} slot={} — skipped",
                               static_cast<unsigned>(hdr->pdu_type), sfn, slot);
                }
            });
    }

    /**
     * Walk the PDU payload span and dispatch only PDUs whose type is in ActiveTypes.
     *
     * PDUs of types not listed in ActiveTypes are silently skipped (not logged).
     * For each active-type PDU, only the parser whose compile-time pdu_type matches
     * is invoked — inactive parsers are compile-time no-ops in the fold.
     *
     * @tparam ActiveTypes  One or more PDU type enum values to dispatch.
     * @param[in] sfn       System Frame Number from the TTI request header.
     * @param[in] slot      Slot number from the TTI request header.
     * @param[in] payload   Byte span covering the PDU array (req->payload onwards).
     * @param[in] num_pdus  Number of PDUs declared in the TTI request header.
     * @return              DispatchResult with processed/skipped counts and per-parser hits.
     *                      Return value must be checked.
     */
    template<Enum... ActiveTypes>
    [[nodiscard]] result_t dispatch_pdus_typed(uint16_t sfn, uint16_t slot,
                                               std::span<const uint8_t> payload,
                                               uint16_t num_pdus) noexcept
    {
        return walk_pdus(sfn, slot, payload, num_pdus,
            [this, sfn, slot](const scf_fapi_generic_pdu_info_t* hdr, result_t& r) noexcept {
                const auto type = static_cast<Enum>(
                    static_cast<std::underlying_type_t<Enum>>(hdr->pdu_type));
                if (((type == static_cast<Enum>(ActiveTypes)) || ...))
                {
                    if (dispatch_one_typed<ActiveTypes...>(type, sfn, slot, hdr,
                                                           r.per_parser_counts,
                                                           std::index_sequence_for<Parsers...>{}))
                    {
                        ++r.processed;
                    }
                }
            });
    }

private:
    std::tuple<Parsers...> parsers_;

    template<typename P, std::size_t... Is>
    [[nodiscard]] static consteval std::size_t
    parser_index_impl(std::index_sequence<Is...>) noexcept
    {
        std::size_t idx = parser_count;
        ((std::is_same_v<P, std::tuple_element_t<Is, std::tuple<Parsers...>>>
              ? (idx = Is, true)
              : false) || ...);
        return idx;
    }

    template<typename P>
    [[nodiscard]] static bool parse_pdu(P& parser,
                                        uint16_t sfn,
                                        uint16_t slot,
                                        const typename P::pdu_t& pdu) noexcept
    {
        return parser.parse(sfn, slot, pdu);
    }

    /**
     * Walk the PDU payload, apply header and size guards, and invoke @p fn for each valid PDU.
     *
     * Owns all iteration, Guard 1 (header size), Guard 2 (pdu_size range), and pointer-advance
     * logic — @p dispatch_pdus and @p dispatch_pdus_typed are thin wrappers that supply the
     * per-PDU dispatch step as a lambda.
     *
     * @tparam DispatchFn  Callable with signature `void(const scf_fapi_generic_pdu_info_t*, result_t&)`.
     * @param[in]     sfn        System Frame Number.
     * @param[in]     slot       Slot number.
     * @param[in]     payload    Byte span covering the PDU array.
     * @param[in]     num_pdus   Number of PDUs declared in the TTI request header.
     * @param[in,out] fn         Called for each well-formed PDU header; mutates the result.
     * @return                   DispatchResult with processed/skipped counts.
     */
    template<typename DispatchFn>
        requires std::is_nothrow_invocable_v<DispatchFn,
                                             const scf_fapi_generic_pdu_info_t*, result_t&>
    [[nodiscard]] result_t walk_pdus(uint16_t sfn, uint16_t slot,
                                     std::span<const uint8_t> payload,
                                     uint16_t num_pdus,
                                     DispatchFn&& fn) noexcept
    {
        result_t result{};
        const auto* cur = payload.data();
        const auto* const payload_end = payload.data() + payload.size();

        for (uint16_t i = 0; i < num_pdus; ++i)
        {
            const auto remaining_bytes =
                static_cast<std::size_t>(payload_end - cur);

            // Guard 1: enough bytes to read the generic header.
            if (remaining_bytes < sizeof(scf_fapi_generic_pdu_info_t))
            {
                ++result.skipped_unknown;
                NVLOGW_FMT(detail::k_tag,
                           "TTI dispatch: sfn={} slot={} payload truncated — "
                           "{} bytes remain, need {} for PDU header; aborting walk",
                           sfn, slot, remaining_bytes,
                           sizeof(scf_fapi_generic_pdu_info_t));
                break;
            }

            const auto* hdr =
                aerial::casts::assume_cast<const scf_fapi_generic_pdu_info_t>(cur);

            // Guard 2: pdu_size must cover the header and fit within the payload.
            if (hdr->pdu_size < sizeof(scf_fapi_generic_pdu_info_t)
                || static_cast<std::size_t>(hdr->pdu_size) > remaining_bytes)
            {
                ++result.skipped_unknown;
                NVLOGW_FMT(detail::k_tag,
                           "TTI dispatch: sfn={} slot={} pdu_type={} "
                           "invalid pdu_size={} (remaining={}); aborting walk",
                           sfn, slot, static_cast<unsigned>(hdr->pdu_type),
                           hdr->pdu_size, remaining_bytes);
                break;
            }

            // pdu_size includes the 4-byte generic header (pdu_type + pdu_size fields),
            // matching on_dl_tti_request which advances by pdu.pdu_size only.
            fn(hdr, result);
            cur += hdr->pdu_size;
        }

        return result;
    }

    /**
     * Dispatch a single PDU to the first parser whose pdu_type matches.
     *
     * Uses a short-circuit || fold over std::index_sequence so each parser
     * index I is a compile-time constant — no runtime idx variable, no
     * ordering hazard.
     *
     * @tparam Is            Index sequence matching the Parsers pack.
     * @param[in] type       PDU type enum value to match.
     * @param[in] sfn        System Frame Number.
     * @param[in] slot       Slot number.
     * @param[in] hdr        Pointer to the generic PDU info header.
     * @param[in,out] counts Per-parser hit counters; incremented on successful parse.
     * @return               true if a matching parser was found and parse() succeeded;
     *                       false if no parser matched or parse() returned false.
     */
    template<std::size_t... Is>
    [[nodiscard]] bool dispatch_one(Enum type,
                                    uint16_t sfn, uint16_t slot,
                                    const scf_fapi_generic_pdu_info_t* hdr,
                                    std::array<uint16_t, parser_count>& counts,
                                    std::index_sequence<Is...>) noexcept
    {
        return (... || [&] {
            auto& p = std::get<Is>(parsers_);
            if (static_cast<Enum>(p.pdu_type) != type) { return false; }
            constexpr auto k_min_size =
                sizeof(scf_fapi_generic_pdu_info_t)
                + sizeof(typename std::decay_t<decltype(p)>::pdu_t);
            if (hdr->pdu_size < k_min_size) [[unlikely]] {
                NVLOGW_FMT(detail::k_tag,
                           "TTI dispatch: sfn={} slot={} pdu_type={} pdu_size={} < {} (min sizeof pdu_t); skipping",
                           sfn, slot, static_cast<unsigned>(hdr->pdu_type),
                           hdr->pdu_size, k_min_size);
                return false;
            }
            using P = std::decay_t<decltype(p)>;
            if (!parse_pdu(p, sfn, slot,
                           *aerial::casts::assume_cast<typename P::pdu_t>(hdr->pdu_config))) { return false; }
            ++counts[Is];   // incremented only on successful parse
            return true;
        }());
    }

    /**
     * Dispatch a single PDU only to the parser in ActiveTypes whose pdu_type matches.
     *
     * Parsers not in ActiveTypes are compile-time no-ops via if constexpr.
     * The short-circuit || fold stops at the first matching parser.
     *
     * @tparam ActiveTypes  Compile-time active PDU type set.
     * @tparam Is           Index sequence matching the Parsers pack (deduced).
     * @param[in] type       PDU type enum value to match.
     * @param[in] sfn        System Frame Number.
     * @param[in] slot       Slot number.
     * @param[in] hdr        Pointer to the generic PDU info header.
     * @param[in,out] counts  Per-parser hit counters; incremented on successful parse.
     * @return              true if a matching parser was found and parse() succeeded;
     *                      false if no parser matched or parse() returned false.
     */
    template<Enum... ActiveTypes, std::size_t... Is>
    [[nodiscard]] bool dispatch_one_typed(Enum type,
                                          uint16_t sfn, uint16_t slot,
                                          const scf_fapi_generic_pdu_info_t* hdr,
                                          std::array<uint16_t, parser_count>& counts,
                                          std::index_sequence<Is...>) noexcept
    {
        return (... || [&]() -> bool {
            using P = std::tuple_element_t<Is, std::tuple<Parsers...>>;
            // Compile-time gate: parsers not in ActiveTypes are eliminated here.
            if constexpr (!((P::pdu_type == ActiveTypes) || ...)) { return false; }
            auto& p = std::get<Is>(parsers_);
            if (static_cast<Enum>(p.pdu_type) != type) { return false; }
            constexpr auto k_min_size =
                sizeof(scf_fapi_generic_pdu_info_t) + sizeof(typename P::pdu_t);
            if (hdr->pdu_size < k_min_size) [[unlikely]] {
                NVLOGW_FMT(detail::k_tag,
                           "TTI dispatch: sfn={} slot={} pdu_type={} pdu_size={} < {} (min sizeof pdu_t); skipping",
                           sfn, slot, static_cast<unsigned>(hdr->pdu_type),
                           hdr->pdu_size, k_min_size);
                return false;
            }
            if (!parse_pdu(p, sfn, slot,
                           *aerial::casts::assume_cast<typename P::pdu_t>(hdr->pdu_config))) { return false; }
            ++counts[Is];   // incremented only on successful parse
            return true;
        }());
    }
};

} // namespace scf_5g_fapi

#endif // SCF_5G_FAPI_TTI_DISPATCH_HPP_INCLUDED_

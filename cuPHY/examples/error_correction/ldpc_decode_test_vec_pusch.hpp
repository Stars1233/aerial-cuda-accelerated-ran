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

#if !defined(LDPC_DECODE_TEST_VEC_PUSCH_HPP_INCLUDED_)
#define LDPC_DECODE_TEST_VEC_PUSCH_HPP_INCLUDED_

#include <string>
#include "ldpc_decode_test_vec.hpp"

/** Parameters selecting a transport block from a PUSCH HDF5 test vector. */
struct test_vec_pusch_params
{
    /** Constructs a PUSCH test-vector selector. */
    test_vec_pusch_params(const char*     fname,
                          cuphyDataType_t llr_type,
                          int             tb_index,
                          int             num_cw_lim) :
      filename(fname),
      LLRtype(llr_type),
      TB_index(tb_index),
      num_cw_limit(num_cw_lim)
    {
    }
    const char*     filename;      ///< Input HDF5 filename.
    cuphyDataType_t LLRtype;       ///< LLR element type.
    int             TB_index;      ///< Zero-based transport-block index.
    int             num_cw_limit;  ///< Optional codeword limit; non-positive selects all.
};

/** Loads LDPC test vectors from PUSCH HDF5 test-vector files. */
class ldpc_decode_test_vec_pusch : public ldpc_decode_test_vec
{
public:
    /** Constructs a vector loader for the selected PUSCH transport block. */
    ldpc_decode_test_vec_pusch(const test_vec_pusch_params& fparams);
    /** Destroys the vector loader. */
    ~ldpc_decode_test_vec_pusch() override = default;
    /** Returns a descriptive label for the source vector. */
    [[nodiscard]] const char* desc() const override;
    /** Loads the selected PUSCH vector and derives its LDPC configuration. */
    virtual void generate() override;
private:
    /** Derives the LDPC configuration from the loaded vector. */
    void populate_config();

    std::string        filename_;       ///< Source input filename.
    int                TB_index_;       ///< Zero-based transport-block index.
    int                num_cw_limit_;   ///< Optional limit on processed codewords.
    cuphy::tensor_desc limit_desc_;     ///< LLR descriptor for the selected codeword range.

    int                nCb_;            ///< Number of codeblocks.
    int                BG_;             ///< Base graph (1 or 2).
    int                Kb_;             ///< Segmentation parameter used to derive @ref Zc_.
    int                Zc_;             ///< LDPC lifting size.
};

#endif // !defined(LDPC_DECODE_TEST_VEC_PUSCH_HPP_INCLUDED_)

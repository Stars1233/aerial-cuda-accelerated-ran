/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "utils.hpp"
#include "stdio.h"
#include <algorithm>
#include <ctype.h>


// The main "everybody quit" switch. Can be triggered by any core.
static std::atomic_bool force_quit{0};
static std::atomic_bool timer_start{0};

static std::atomic_uint64_t glob_slot_counter{0};

inline bool compare_approx(const __half& a, const __half& b, const float tolf)
{
#if CUDART_VERSION < 12020
    float af = __half2float(a);
    float bf = __half2float(b);
    float diff = fabs(af - bf);
    //float m = std::max(fabs(af), fabs(bf));
    //float ratio = (diff >= tolf) ? (diff / m) : diff;
    return (diff <= tolf);
#else
    const __half tolerance = __float2half(tolf);
    __half diff = __habs(a - b);
    //__half m = __hmax(__habs(a), __habs(b));
    //__half ratio = (diff >= tolerance) ? (diff / m) : diff;
    return (diff <= tolerance);
#endif
}

int compare_approx_buffer(uint8_t * rx_buff, uint8_t * tv_buff, size_t length)
{
    const int NUM_HALF = length / sizeof(__half);
    const __half* rx = reinterpret_cast<const __half*>(rx_buff);
    const __half* tv = reinterpret_cast<const __half*>(tv_buff);
    // float f_rx;
    // float f_tv;
    for(int i = 0; i < NUM_HALF; ++i)
    {
        if(!compare_approx(rx[i], tv[i]))
        {
            // printf("%s rx[i] %f,  tv[i] %f\n", __FUNCTION__, __half2float(rx[i]), __half2float(tv[i]));
            return 1;
        }
    }
    return 0;
}

int reorder_from_network_order(int num)
{
	/*
		s[0] = 1 -> v4.x[15:8]
		s[1] = 0    v4.x[7:0]
		s[2] = 3    v4.x[31:24]
		s[3] = 2    v4.x[23:16]
		*/
	int result = ((num >> 8) & 0xFF)  | ((num & 0xFF) << 8) | ((num >> 8) & 0x00FF0000) | ((num << 8) & 0xFF000000);
	return result;
}

void unpackInput(const uint8_t *input,
                int offset,
                int32_t vi[4],
                int32_t vq[4],
                int32_t &compParam,
                int32_t compbits)
{
    if (compbits < 16)
    {
        if (compbits == 9)
        {
            vi[0] = (input[offset] << 1) | (input[offset + 1] >> 7);              // 8 + 1, remains 7
            vq[0] = ((input[offset + 1] & 0x7f) << 2) | (input[offset + 2] >> 6); // 7 + 2, remains 6
            vi[1] = ((input[offset + 2] & 0x3f) << 3) | (input[offset + 3] >> 5); // 6 + 3, remains 5
            vq[1] = ((input[offset + 3] & 0x1f) << 4) | (input[offset + 4] >> 4); // 5 + 4, remains 4
            vi[2] = ((input[offset + 4] & 0x0f) << 5) | (input[offset + 5] >> 3); // 4 + 5, remains 3
            vq[2] = ((input[offset + 5] & 0x07) << 6) | (input[offset + 6] >> 2); // 3 + 6, remains 2
            vi[3] = ((input[offset + 6] & 0x03) << 7) | (input[offset + 7] >> 1); // 2 + 7, remains 1
            vq[3] = ((input[offset + 7] & 0x01) << 8) | input[offset + 8];        // 1 + 8
        }
        else if (compbits == 14)
        {
            vi[0] = (input[offset] << 6) | (input[offset + 1] >> 2);                                             // 8 + 6, remains 2
            vq[0] = ((input[offset + 1] & 0x03) << 12) | (input[offset + 2] << 4) | (input[offset + 3] >> 4);    // 2 + 8 + 4, remains 4
            vi[1] = ((input[offset + 3] & 0x0f) << 10) | (input[offset + 4] << 2) | (input[offset + 5] >> 6);    // 4 + 8 + 2, remains 6
            vq[1] = ((input[offset + 5] & 0x3f) << 8) | input[offset + 6];                                       // 6 + 8
            vi[2] = (input[offset + 7] << 6) | (input[offset + 8] >> 2);                                         // 8 + 6, remains 2
            vq[2] = ((input[offset + 8] & 0x03) << 12) | (input[offset + 9] << 4) | (input[offset + 10] >> 4);   // 2 + 8 + 4, remains 4
            vi[3] = ((input[offset + 10] & 0x0f) << 10) | (input[offset + 11] << 2) | (input[offset + 12] >> 6); // 4 + 8 + 2, remains 6
            vq[3] = ((input[offset + 12] & 0x3f) << 8) | input[offset + 13];                                     // 6 + 8
        }
    }
}

// Only applicable when compbits != 16
void unpackInputPrb(const uint8_t *input,
                 int32_t prbid,
                 int32_t prbStride,
                 uint32_t laneid,
                 int32_t vi[4],
                 int32_t vq[4],
                 int32_t &compParam,
                 int32_t compbits)
{
    int offset = prbid * prbStride;
    if (compbits < 16)
    {
        compParam = input[offset];
        offset += laneid * compbits + 1;
        unpackInput(input, offset, vi, vq, compParam, compbits);
    }
}

// Only for dl_bit_width != 16
void decompress_PRB(__half* decompressed_PRB_output, const uint8_t* input, int PRB_index, int PRB_stride, int dl_bit_width, float beta)
{
    for (int laneid = 0; laneid < 3; laneid++) {
        int32_t vi[4], vq[4];
        int32_t shift;
        unpackInputPrb(input, PRB_index, PRB_stride, laneid, vi, vq, shift, dl_bit_width);

        // Expand the values back to 32-bit integers
        // shift left first then right to propagate the sign bits
        for (int i = 0; i < 4; i++)
        {
            vi[i] = (vi[i] << (32 - dl_bit_width)) >> (32 - dl_bit_width - shift);
            vq[i] = (vq[i] << (32 - dl_bit_width)) >> (32 - dl_bit_width - shift);
        }

        // We'll be writing vectors of 128-bit = 8 x half values
        union u128
        {
            int4 v4;
            half vh[8];
        } vec;
        int4 *output_vec =  (int4 *)decompressed_PRB_output;

        // Apply beta scaling factor in FP32, then convert to FP16
        //    float beta = 3316293.250000;
        for (int i = 0; i < 4; i++)
        {
            vec.vh[2 * i] = (half)((float)vi[i] / beta);
            vec.vh[2 * i + 1] = (half)((float)vq[i] / beta);
        }
        output_vec[laneid] = vec.v4;
    }
}

/* tv_buff is provided uncompressed irrespective of the dl_bit_width value, so we only need to decompress the rx_buff.
   Doing so also ensures the beta_dl scaling applied in cuphydriver is consistent with the value used in the ru-emulator.
 */
int decompress_and_compare_approx_buffer(uint8_t * rx_buff, uint8_t * tv_buff, size_t length, int dl_bit_width, float beta, int flow, int symbol, int prb_start)
{
    int bytes_per_PRB = (dl_bit_width == BFP_NO_COMPRESSION) ? 48 : (3*dl_bit_width + 1);
    int n_PRB = length / bytes_per_PRB; // number of PRBs
    int32_t prbstride = bytes_per_PRB;
    const uint8_t* rx_input = reinterpret_cast<const uint8_t*>(rx_buff);
    const uint8_t* tv_input = reinterpret_cast<const uint8_t*>(tv_buff);
    int tv_prbstride = 48; // no compression

    for (int i = 0; i < n_PRB; i++) {
	__half decompressed_rx_PRB_output[SUBCARRIERS_PER_PRB*2];
        if(dl_bit_width == BFP_NO_COMPRESSION) //16-bits
        {
            memcpy(decompressed_rx_PRB_output, &rx_buff[i*prbstride], 2*SUBCARRIERS_PER_PRB*sizeof(__half));
        }
        else
        {
            decompress_PRB(decompressed_rx_PRB_output, rx_input, i, prbstride, dl_bit_width, beta);
        }

#if 0
        __half tv_PRB_output[SUBCARRIERS_PER_PRB*2];
        memcpy(tv_PRB_output, &tv_input[i*tv_prbstride], 2*12*sizeof(__half));
#else
        const __half* tv_PRB_output = reinterpret_cast<const __half*>(tv_buff + i*tv_prbstride);
#endif

	//Decompressed data for that PRB rx vs. TV
	for (int j = 0; j < SUBCARRIERS_PER_PRB; j++) {
#if 0
                //if ((symbol == 3) && (flow == 0) && (prb_start == 0) && (i == 0))

		printf("Flow %d, symbol %d, PRB %d, rel. PRB %d, RE %d is rx {%f + j %f} vs TV {%f + j %f}\n",
		       flow, symbol, prb_start+i, i, j,
		       __half2float(decompressed_rx_PRB_output[2*j]), __half2float(decompressed_rx_PRB_output[2*j + 1]),
		       __half2float(tv_PRB_output[2*j]), __half2float(tv_PRB_output[2*j + 1]));
#endif

                if(!compare_approx(decompressed_rx_PRB_output[2*j],
                                   tv_PRB_output[2*j]))
                {
                    // printf("%s rx[i] %f,  tv[i] %f\n", __FUNCTION__, __half2float(rx[i]), __half2float(tv[i]));
                    return 1;
	        }

                if(!compare_approx(decompressed_rx_PRB_output[2*j + 1],
                                   tv_PRB_output[2*j + 1]))
                {
                    return 1;
	        }
        }
    }
    return 0;
}


void unpackInputBfwBundle(const uint8_t *input,
                 uint32_t laneid,
                 int32_t vi[4],
                 int32_t vq[4],
                 int32_t compParam,
                 int32_t compbits)
{
    int offset = 0;
    if (compbits < 16)
    {
        compParam = input[offset];
        offset += laneid * compbits;
        unpackInput(input, offset, vi, vq, compParam, compbits);
    }
};

// Assume L_TRX is divisible by 4 to optimize unpacking with 4 int I/Q pairs
void decompress_BFW_bundle(__half* decompressed_bfw_output, const uint8_t* input, int dl_bit_width, float beta, int L_TRX, int compParam)
{
    int32_t vi[4], vq[4];
    constexpr int LANE_SIZE = 4;
    for (int laneid = 0; laneid < L_TRX / LANE_SIZE; laneid++) {
        unpackInputBfwBundle(input, laneid, vi, vq, compParam, dl_bit_width);

//        if(laneid == 1) {for(int n=0;n<9;n++) {printf("0x%02X ",input[laneid*dl_bit_width+n]);}printf("\n");}
        // Expand the values back to 32-bit integers
        // shift left first then right to propagate the sign bits
        for (int i = 0; i < 4; i++)
        {
//            if(laneid == 1) {printf("0x%08X 0x%08X ",vi[i],vq[i]);}
            vi[i] = (vi[i] << (32 - dl_bit_width)) >> (32 - dl_bit_width - compParam);
            vq[i] = (vq[i] << (32 - dl_bit_width)) >> (32 - dl_bit_width - compParam);
        }
//        if(laneid == 1) {printf("\n");}

        // We'll be writing vectors of 128-bit = 8 x half values
        union u128
        {
            int4 v4;
            half vh[8];
        } vec;
        int4 *output_vec =  (int4 *)decompressed_bfw_output;

        // Apply beta scaling factor in FP32, then convert to FP16
        //    float beta = 3316293.250000;
        for (int i = 0; i < 4; i++)
        {
            vec.vh[2 * i] = (half)((float)vi[i] / beta);
            vec.vh[2 * i + 1] = (half)((float)vq[i] / beta);
        }
        output_vec[laneid] = vec.v4;
    }
}

// We are operating under the assumption that only BFP9 is supported for BFW, the length == dl_bit_width * L_TRX * 2 is  byte aligned
int decompress_and_compare_approx_bfw_bundle_buffer(uint8_t * rx_buff, uint8_t rx_exp, uint8_t * tv_buff, uint8_t tv_exp, size_t length, int dl_bit_width, float beta, int flow, int symbol_id, int bundle_index, bool decompress_tv, uint16_t numGnbAnt)
{
    const uint8_t* rx_input = reinterpret_cast<const uint8_t*>(rx_buff);
    const uint8_t* tv_input = reinterpret_cast<const uint8_t*>(tv_buff);

    __half decompressed_rx_PRB_output[ORAN_SECT_EXT_11_L_TRX*2];
    __half decompressed_tv_PRB_output[ORAN_SECT_EXT_11_L_TRX*2];

    static constexpr float tol = 1.0f/float(1<<9);

    decompress_BFW_bundle(decompressed_rx_PRB_output, rx_input, dl_bit_width, beta, numGnbAnt, rx_exp);

    if(decompress_tv)
    {
        decompress_BFW_bundle(decompressed_tv_PRB_output, tv_input, dl_bit_width, beta, numGnbAnt, tv_exp);
    }
    else
    {
        //todo unpack TV?
        memcpy(decompressed_tv_PRB_output, tv_input, 2*numGnbAnt*sizeof(__half));
    }

    for (int j = 0; j < numGnbAnt; j++) {
#if 0
            //if ((symbol == 3) && (flow == 0) && (prb_start == 0) && (i == 0))
        printf("Flow %d, symbol %d, bundle %d, RE %d is rx {%f + j %f} vs TV {%f + j %f}\n",
                flow, symbol_id, bundle_index, j,
                __half2float(decompressed_rx_PRB_output[2*j]), __half2float(decompressed_rx_PRB_output[2*j + 1]),
                __half2float(decompressed_tv_PRB_output[2*j]), __half2float(decompressed_tv_PRB_output[2*j + 1]));
#endif

        if(!compare_approx(decompressed_rx_PRB_output[2*j], decompressed_tv_PRB_output[2*j],tol) ||
            !compare_approx(decompressed_rx_PRB_output[2*j + 1], decompressed_tv_PRB_output[2*j + 1],tol))
        {
#if 1
            printf("Flow %d, symbol %d, bundle %d, RE %d is rx {%f + j %f} vs TV {%f + j %f}\n",
                    flow, symbol_id, bundle_index, j,
                    __half2float(decompressed_rx_PRB_output[2*j]), __half2float(decompressed_rx_PRB_output[2*j + 1]),
                    __half2float(decompressed_tv_PRB_output[2*j]), __half2float(decompressed_tv_PRB_output[2*j + 1]));
#endif
            return 1;
        }

        // if(!compare_approx(decompressed_rx_PRB_output[2*j + 1],
        //                     decompressed_tv_PRB_output[2*j + 1]))
        // {
        //     return 1;
        // }
    }
    return 0;
}

/// Unpack 9-byte input data into 8 16-bit samples (sign-extended 9-bit)
/// Note: Current implementation assumes BFP9.  Implementation currently only for little-endian
inline SampleSet_t unpack_samples(__uint128_t data)
{
	static constexpr uint32_t bitmask18b = 0x3FFFF;    // Bitmask 18 bits
	static constexpr uint64_t bitcastu16x2 = 0x800001; // Multiplicand to cast results on to 32-bit boundaries
	static constexpr uint8_t sign_ext_bits = (16-9);
	static constexpr uint8_t samples_per_loop = 2;
	static constexpr uint8_t bits_per_loop = 9*samples_per_loop;
	// Reorder bytes for little-endian
#ifdef __clang__
	data = __uint128_t(__builtin_shufflevector(u8x16(data), u8x16(data), 8,7,6,5,4,3,2,1,0,9,9,9,9,9,9,9));
#else
	static constexpr u8x16 reorder = {8,7,6,5,4,3,2,1,0,9,9,9,9,9,9,9}; // only care about lowest 9 bytes
	data = __uint128_t(__builtin_shuffle(u8x16(data), reorder));
#endif
	SampleSet_t samples = {0};
	for(int i=0;i<8;i+=samples_per_loop)
	{
		uint64_t loopSamples = data & bitmask18b;
		loopSamples      = loopSamples * bitcastu16x2;
		samples.w[7-i]   = loopSamples; // write out in reverse order to undo little-endian ordering
		samples.w[6-i]   = (loopSamples>>32);
		data = data >> bits_per_loop;
	}
	// Sign-extend elements
	samples.vec = samples.vec << sign_ext_bits;
	samples.vec = samples.vec >> sign_ext_bits;
	return samples;
}

/// Perform fixed-point comparison on packed BFP data
/// Note: Current implementation assumes BFP9
int fixedpt_bundle_compare(uint8_t* rx_buf, uint8_t* tv_buf, int nBytes, uint8_t rx_exp, uint8_t tv_exp)
{
	int res = 0;
	int bytes = 0;
    constexpr int BFP_SIZE = 9;
	// Handle 8 samples (BFP_SIZE bytes) at a time
	SampleSet_t rx_samples,tv_samples,delta;
	__uint128_t rx_data = 0;
	__uint128_t tv_data = 0;
    /*
	// TODO assume exponents are the same for now

	rx_buf++;
	tv_buf++;
    */
    int8_t rx_shift = rx_exp - tv_exp;
    int8_t tv_shift = 0;
    if(rx_shift < 0)
    {
        tv_shift = -rx_shift;
        rx_shift = 0;
    }

	for(int i=1;i<nBytes;i+=BFP_SIZE)
	{
		// Assume host byte order for TV & Rx data
		memcpy(&rx_data, rx_buf, BFP_SIZE);
		memcpy(&tv_data, tv_buf, BFP_SIZE);
        //NVLOGD_FMT(501,"Rx Samples: 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X}",rx_buf[0],rx_buf[1],rx_buf[2],rx_buf[3],rx_buf[4],rx_buf[5],rx_buf[6],rx_buf[7]);
        //NVLOGD_FMT(501,"TV Samples: 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X}",tv_buf[0],tv_buf[1],tv_buf[2],tv_buf[3],tv_buf[4],tv_buf[5],tv_buf[6],tv_buf[7]);
        if(tv_data == rx_data)
        {
            continue;
        }
		rx_samples = unpack_samples(rx_data);
        rx_samples.vec <<= rx_shift;
//		if(i==10) {printf("Rx Samples: 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X\n",rx_samples.w[0],rx_samples.w[1],rx_samples.w[2],rx_samples.w[3],rx_samples.w[4],rx_samples.w[5],rx_samples.w[6],rx_samples.w[7]);}
		tv_samples = unpack_samples(tv_data);
        tv_samples.vec <<= tv_shift;
//		if(i==10) {printf("TV Samples: 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X\n",tv_samples.w[0],tv_samples.w[1],tv_samples.w[2],tv_samples.w[3],tv_samples.w[4],tv_samples.w[5],tv_samples.w[6],tv_samples.w[7]);}
		delta.vec =  rx_samples.vec - tv_samples.vec;
//		if(i==10) {printf("Delta:      0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X\n",delta.w[0],delta.w[1],delta.w[2],delta.w[3],delta.w[4],delta.w[5],delta.w[6],delta.w[7]);}
		// absolute value
		{
		i16x8 neg = delta.vec < 0;
		delta.vec = (neg ^ delta.vec) + (neg & 1);
		}
//		if(i==10) {printf("Abs Delta:  0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X\n",delta.w[0],delta.w[1],delta.w[2],delta.w[3],delta.w[4],delta.w[5],delta.w[6],delta.w[7]);}
		delta.vec = delta.vec > 4; // Compare against threshold
//		if(i==10) {printf("Compare >1: 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X\n",delta.w[0],delta.w[1],delta.w[2],delta.w[3],delta.w[4],delta.w[5],delta.w[6],delta.w[7]);}
		res |= delta.data != 0; // Set true if threshold is exceeded
		if(res)
		{
			NVLOGD_FMT(501,"Compare Failed on bytes {} to {}",i,i+BFP_SIZE);
			//NVLOGC_FMT(501,"Failed UnPacked Rx Samples: 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X}",rx_samples.w[0],rx_samples.w[1],rx_samples.w[2],rx_samples.w[3],rx_samples.w[4],rx_samples.w[5],rx_samples.w[6],rx_samples.w[7]);
			//NVLOGC_FMT(501,"Failed UnPacked TV Samples: 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X}",tv_samples.w[0],tv_samples.w[1],tv_samples.w[2],tv_samples.w[3],tv_samples.w[4],tv_samples.w[5],tv_samples.w[6],tv_samples.w[7]);
            NVLOGD_FMT(501,"Rx Samples(Failed): 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X}",rx_buf[0],rx_buf[1],rx_buf[2],rx_buf[3],rx_buf[4],rx_buf[5],rx_buf[6],rx_buf[7]);
            NVLOGD_FMT(501,"TV Samples(Failed): 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X}",tv_buf[0],tv_buf[1],tv_buf[2],tv_buf[3],tv_buf[4],tv_buf[5],tv_buf[6],tv_buf[7]);
			break;
		}
		rx_buf+=9;
		tv_buf+=9;
	}
	return res;
}


void get_glob_fss(struct fssId& fss, int max_slot_id)
{
    uint64_t slot_counter = glob_slot_counter.load();
    slot_counter %= (uint64_t)ORAN_MAX_FRAME_ID * ORAN_MAX_SUBFRAME_ID * (uint64_t)(max_slot_id + 1);
    fss.frameId = slot_counter / (ORAN_MAX_SUBFRAME_ID * (max_slot_id + 1));
    slot_counter -= (uint64_t)fss.frameId * (ORAN_MAX_SUBFRAME_ID * (max_slot_id + 1));
    fss.subframeId =  (slot_counter) / (max_slot_id + 1);
    slot_counter -= (uint64_t)fss.subframeId * (max_slot_id + 1);
    fss.slotId = slot_counter;
}

void set_glob_fss(struct fssId fss, int max_slot_id)
{
    uint64_t slot_counter = (uint64_t)fss.frameId * ORAN_MAX_SUBFRAME_ID * (max_slot_id + 1) + (uint64_t)fss.subframeId * (max_slot_id + 1) + (uint64_t)fss.slotId;
    glob_slot_counter.store(slot_counter);
}

void increment_glob_fss(int max_slot_id)
{
    ++glob_slot_counter;

}

bool check_force_quit()
{
    return force_quit.load();
}

void set_force_quit()
{
    force_quit.store(true);
}

bool check_timer_start()
{
    return timer_start.load();
}

void set_timer_start()
{
    timer_start.store(true);
}

void signal_handler(int signum)
{
    if (signum == SIGINT || signum == SIGTERM || signum == SIGUSR1) {
        if (check_force_quit()) {
            fprintf(stderr,"Signal %d received; quitting the hard way\n", signum);
            exit(EXIT_FAILURE);
        }
        fprintf(stdout, "Signal %d received, preparing to exit...\n", signum);
        set_force_quit();
        //usleep(20000);
    }
}

void signal_setup()
{
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGUSR1, signal_handler);
}

void do_throw(std::string const& what)
{
    re_err(AERIAL_RU_EMULATOR_EVENT, "Throwing exception: {}", what.c_str());
    throw std::runtime_error(what);
}

std::string affinity_cpu_list()
{
    sb out;
    cpu_set_t s;
    int ret = pthread_getaffinity_np(pthread_self(), sizeof(s), &s);
    if (ret != 0)
        do_throw("pthread_getaffinity_np() failed");
    int count = CPU_COUNT(&s);
    int printed = 0;
    for (int i = 0; printed < count; ++i) {
        if (CPU_ISSET(i, &s)) {
            if (printed)
                out << "," << i;
            else
                out << i;
            ++printed;
        }
    }
    return out;
}

int set_max_thread_priority()
{
/*
    pthread_t t = pthread_self();
    struct sched_param schedprm;
    
    schedprm.sched_priority = sched_get_priority_max(SCHED_FIFO);
    int ret = pthread_setschedparam(t, SCHED_FIFO, &schedprm);
    if (ret != 0)
        do_throw("Could not set max thread priority");
    int schedpol;
    int ret = pthread_getschedparam(t, &schedpol, &schedprm);
    if (ret != 0)
        do_throw("Could not get thread scheduling info");

    if (schedpol != SCHED_FIFO)
        do_throw("Failed to apply SCHED_FIFO policy");

       Removing FIFO max priority because it causes system to hang in K8 setup.
       Ideally everything should still run fine with the ISOCPUs set.
       Also causes some CPU stall. This is also removed in cuPHYDriver.

        [Thu May 21 09:08:29 2020] rcu: INFO: rcu_preempt self-detected stall on CPU
        [Thu May 21 09:08:29 2020] rcu: 	2-....: (1 GPs behind) idle=e26/1/0x4000000000000002 softirq=894762853/894762854 fqs=14999
        [Thu May 21 09:08:29 2020] rcu: 	 (t=60002 jiffies g=1045915465 q=9857)
        [Thu May 21 09:08:29 2020] NMI backtrace for cpu 2
        [Thu May 21 09:08:29 2020] CPU: 2 PID: 3419 Comm: Consumer Tainted: P           OE     5.0.0-37-lowlatency #40~18.04.1-Ubuntu
        [Thu May 21 09:08:29 2020] Hardware name: Dell Inc. PowerEdge R740/06G98X, BIOS 2.4.8 11/26/2019
        [Thu May 21 09:08:29 2020] Call Trace:
        [Thu May 21 09:08:29 2020]  <IRQ>
        [Thu May 21 09:08:29 2020]  dump_stack+0x63/0x85
        [Thu May 21 09:08:29 2020]  nmi_cpu_backtrace+0x94/0xa0
        [Thu May 21 09:08:29 2020]  ? lapic_can_unplug_cpu+0xa0/0xa0
        [Thu May 21 09:08:29 2020]  nmi_trigger_cpumask_backtrace+0xf9/0x140
        [Thu May 21 09:08:29 2020]  arch_trigger_cpumask_backtrace+0x19/0x20
        [Thu May 21 09:08:29 2020]  rcu_dump_cpu_stacks+0x9e/0xdd
        [Thu May 21 09:08:29 2020]  rcu_check_callbacks+0x6eb/0x8e0
        [Thu May 21 09:08:29 2020]  ? account_user_time+0xa8/0xb0
        [Thu May 21 09:08:29 2020]  ? tick_sched_do_timer+0x60/0x60
        [Thu May 21 09:08:29 2020]  update_process_times+0x2f/0x60
        [Thu May 21 09:08:29 2020]  tick_sched_handle+0x25/0x70
        [Thu May 21 09:08:29 2020]  tick_sched_timer+0x3c/0x80
        [Thu May 21 09:08:29 2020]  __hrtimer_run_queues+0x10f/0x2d0
        [Thu May 21 09:08:29 2020]  hrtimer_interrupt+0xe7/0x240
        [Thu May 21 09:08:29 2020]  smp_apic_timer_interrupt+0x6f/0x160
        [Thu May 21 09:08:29 2020]  apic_timer_interrupt+0xf/0x20
        [Thu May 21 09:08:29 2020]  </IRQ>
        [Thu May 21 09:08:29 2020] RIP: 0033:0x7ffef97ef6ac
        [Thu May 21 09:08:29 2020] Code: 2d 81 e9 ff ff 48 8b 0d 82 e9 ff ff 0f ae e8 0f 31 44 8b 23 45 39 e3 0f 84 05 01 00 00 44 8b 1b 45 85 db 75 d9 eb ba 0f ae e8 <0f> 31 48 c1 e2 20 48 09 c2 48 85 d2 49 8b 40 08 48 8b 0d c5 c9 ff
        [Thu May 21 09:08:29 2020] RSP: 002b:00007f4a8130cb10 EFLAGS: 00000246 ORIG_RAX: ffffffffffffff13
        [Thu May 21 09:08:29 2020] RAX: 0000000000000001 RBX: 0000000000000000 RCX: 0000000000000000
        [Thu May 21 09:08:29 2020] RDX: 0000000000000000 RSI: 00007f4a8130cb70 RDI: 0000000000000000
        [Thu May 21 09:08:29 2020] RBP: 00007f4a8130cb30 R08: 00007ffef97ec0a0 R09: 00007ffef97ec080
        [Thu May 21 09:08:29 2020] R10: 00000000c6dfccc4 R11: 000000005ec64537 R12: 0000000000000001
        [Thu May 21 09:08:29 2020] R13: 0000000000000010 R14: 00007f4a8b6277f8 R15: 00007f4a3813fc30
*/
    return 0;
}

uint32_t adler32_naive(void* buffer, int size)
{
    unsigned char* buf = (unsigned char*)buffer;
    uint32_t a = 1;
    uint32_t b = 0;
    for(int i = 0; i < size; ++i)
    {
        a = (a + buf[i]) % ADLER32_MOD;
        b = (b + a) % ADLER32_MOD;
    }
    return (b << 16) | a;
}

//https://github.com/SheetJS/js-adler32/blob/master/adler32.js

uint32_t adler32(void* buffer, int size)
{
    unsigned char* buf = (unsigned char*)buffer;
    uint32_t a = 1;
    uint32_t b = 0;
    uint32_t m;
    for(int i = 0; i < size;) {
        m = std::min(size-i, 3850)+i;
        for(;i < m;i++) {
            a += buf[i]&0xFF;
            b += a;
        }
        a = (15 * (a >> 16) + (a & 65535));
        b = (15 * (b >> 16) + (b & 65535));
    }
    return ((b % ADLER32_MOD) << 16) | (a % ADLER32_MOD);
}

//https://github.com/dingwentao/GPU-lossless-compression/blob/master/cuda-bsc/libbsc/adler32/adler32.cpp
#define BASE 65521UL
#define NMAX 5552

#define DO1(buf, i) { sum1 += (buf)[i]; sum2 += sum1; }
#define DO2(buf, i) DO1(buf, i); DO1(buf, i + 1);
#define DO4(buf, i) DO2(buf, i); DO2(buf, i + 2);
#define DO8(buf, i) DO4(buf, i); DO4(buf, i + 4);
#define DO16(buf)   DO8(buf, 0); DO8(buf, 8);
#define MOD(a)      a %= BASE

uint32_t adler32_optimized(unsigned char* buffer, int size)
{
    uint32_t sum1 = 1;
    uint32_t sum2 = 0;

    while (size >= NMAX)
    {
        for (int i = 0; i < NMAX / 16; ++i)
        {
            DO16(buffer); buffer += 16;
        }
        MOD(sum1); MOD(sum2); size -= NMAX;
    }

    while (size >= 16)
    {
        DO16(buffer); buffer += 16; size -= 16;
    }

    while (size > 0)
    {
        DO1(buffer, 0); buffer += 1; size -= 1;
    }

    MOD(sum1); MOD(sum2);

    return sum1 | (sum2 << 16);
}


void write_qam_to_file(std::string filename, void* buffer, size_t size)
{
    unsigned char* buf = (unsigned char*)buffer;
    FILE *fp;
    fp=fopen(filename.c_str(), "wb");
    if(fp == NULL)
    {
        do_throw("Failed to open file");
    }
    fwrite(buf, sizeof(buf[0]), size, fp);
    fclose(fp);
}

void print_header_info(struct oran_packet_header_info& header_info)
{
    re_cons("sym {:2d} startPrb {:3d} numPrb {:3d} flow {:3d} flow_index {:3d}",
            header_info.symbolId,
            header_info.startPrb,
            header_info.numPrb,
            header_info.flowValue,
            header_info.flow_index);
}


ul_cell_cpu_assignment_array get_cell_cpu_assignment(int num_cells, int num_cores, bool enable_mmimo, bool is_srs, int min_cores_per_cell_mmimo)
{
    NVLOGC_FMT(NVLOG_TAG_BASE_RU_EMULATOR, "get_cell_cpu_assignment: num_cells: {}, num_cores: {}", num_cells, num_cores);
    ul_cell_cpu_assignment_array ul_cell_cpu_assignment;

    if(num_cores > MAX_RU_THREADS)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_RU_EMULATOR, AERIAL_INVALID_PARAM_EVENT,"ERROR: num_cores: {}, MAX_RU_THREADS: {}", num_cores, MAX_RU_THREADS);
        do_throw(fmt::format("num_cores ({}) exceeds MAX_RU_THREADS ({})", num_cores, MAX_RU_THREADS));
    }

    if(enable_mmimo && !is_srs)
    {
        if(num_cores / num_cells < min_cores_per_cell_mmimo)
        {
            NVLOGW_FMT(NVLOG_TAG_BASE_RU_EMULATOR, "WARNING: num UL cores given less than desirable for mmimo case, it will not be performant for peak patterns. num_cores: {}, num_cells: {}, min_cores_per_cell: {}", num_cores, num_cells, min_cores_per_cell_mmimo);
            do_throw(fmt::format("num UL cores given less than desirable for mmimo case, it will not be performant for peak patterns. num_cores: {}, num_cells: {}, min_cores_per_cell: {}", num_cores, num_cells, min_cores_per_cell_mmimo));
        }
    }

    if(num_cores > num_cells)
    {
        auto max_num_cores_per_cell = (num_cores + num_cells - 1) / num_cells;
        auto num_cells_with_max_cores = num_cores % num_cells;
        if(num_cells_with_max_cores == 0)
        {
            num_cells_with_max_cores = num_cells;
        }

        int core_id = 0;
        for(int cell = 0; cell < num_cells_with_max_cores; cell++)
        {
            for(int i = 0; i < max_num_cores_per_cell; i++)
            {
                ul_cell_cpu_assignment[core_id].thread_id = core_id;
                ul_cell_cpu_assignment[core_id].start_cell_index = cell;
                ul_cell_cpu_assignment[core_id].num_cells_per_core = 1;
                core_id++;
            }
        }


        for(int cell = num_cells_with_max_cores; cell < num_cells; cell++)
        {
            for(int i = 0; i < max_num_cores_per_cell - 1; i++)
            {
                ul_cell_cpu_assignment[core_id].thread_id = core_id;
                ul_cell_cpu_assignment[core_id].start_cell_index = cell;
                ul_cell_cpu_assignment[core_id].num_cells_per_core = 1;
                core_id++;
            }
        }

        if(unlikely(core_id != num_cores))
        {
            NVLOGE_FMT(NVLOG_TAG_BASE_RU_EMULATOR, AERIAL_INVALID_PARAM_EVENT,"ERROR: core_id: {}, num_cores: {}, num_cells: {}, max_num_cores_per_cell: {}, num_cells_with_max_cores: {}", core_id, num_cores, num_cells, max_num_cores_per_cell, num_cells_with_max_cores);
            exit(1);
        }
    }
    else if(num_cores < num_cells)
    {
        auto max_num_cells_per_core = (num_cells + num_cores - 1) / num_cores;
        auto num_cores_with_max_cells = num_cells % num_cores;
        if(num_cores_with_max_cells == 0)
        {
            num_cores_with_max_cells = num_cores;
        }

        int cell_id = 0;
        for(int core_id = 0; core_id < num_cores_with_max_cells; core_id++)
        {
            ul_cell_cpu_assignment[core_id].thread_id = core_id;
            ul_cell_cpu_assignment[core_id].start_cell_index = cell_id;
            ul_cell_cpu_assignment[core_id].num_cells_per_core = max_num_cells_per_core;
            cell_id += max_num_cells_per_core;
        }

        for(int core_id = num_cores_with_max_cells; core_id < num_cores; core_id++)
        {
            ul_cell_cpu_assignment[core_id].thread_id = core_id;
            ul_cell_cpu_assignment[core_id].start_cell_index = cell_id;
            ul_cell_cpu_assignment[core_id].num_cells_per_core = max_num_cells_per_core - 1;
            cell_id += max_num_cells_per_core - 1;
        }

        if(unlikely(cell_id != num_cells))
        {
            NVLOGE_FMT(NVLOG_TAG_BASE_RU_EMULATOR, AERIAL_INVALID_PARAM_EVENT,"ERROR: cell_id: {}, num_cores: {}, num_cells: {}, max_num_cells_per_core: {}, num_cores_with_max_cells: {}", cell_id, num_cores, num_cells, max_num_cells_per_core, num_cores_with_max_cells);
            exit(1);
        }
    }
    else // num_cores == num_cells
    {
        for(int i = 0; i < num_cores; i++)
        {
            ul_cell_cpu_assignment[i].thread_id = i;
            ul_cell_cpu_assignment[i].start_cell_index = i;
            ul_cell_cpu_assignment[i].num_cells_per_core = 1;
        }
    }

    for(int i = 0; i < num_cores; i++)
    {
        NVLOGC_FMT(NVLOG_TAG_BASE_RU_EMULATOR, "ul_cell_cpu_assignment[{}].thread_id: {}, start_cell_index: {}, num_cells_per_core: {}", i, ul_cell_cpu_assignment[i].thread_id, ul_cell_cpu_assignment[i].start_cell_index, ul_cell_cpu_assignment[i].num_cells_per_core);
    }

    return ul_cell_cpu_assignment;
}

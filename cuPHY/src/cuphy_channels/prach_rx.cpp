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

#include "prach_rx.hpp"
#include "cuphy.h"
#include "cuphy_internal.h"
#include "tensor_desc.hpp"

#include <vector>
#include <cassert>
#include <complex>
#include <limits>
#include <numeric>

#define CHECK_CONFIG 1 // runtime check enabled

using namespace cuphy;

cuphyStatus_t CUPHYWINAPI cuphyValidatePrachParams(cuphyPrachStatPrms_t const* pPrachParams)
{
    return PrachRx::validateStaticParams(pPrachParams);
}

cuphyStatus_t CUPHYWINAPI cuphyCreatePrachRx(cuphyPrachRxHndl_t* pPrachRxHndl, cuphyPrachStatPrms_t const* pStatPrms)
{
    if(pPrachRxHndl == nullptr || pStatPrms == nullptr)
    {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "handle {:p} or pStatPrms {:p} is null",
        static_cast<void*>(pPrachRxHndl), const_cast<void*>(static_cast<void const*>(pStatPrms)));
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    
    return cuphy::tryCallableAndCatch([&]
    {
        cuphyStatus_t status;
        if(pStatPrms->pDbg->enableApiLogging)
        {
            PrachRx::printStatApiPrms(pStatPrms);
        }
        PrachRx* new_pipeline = new(std::nothrow) PrachRx(pStatPrms, &status);
        if(status != CUPHY_STATUS_SUCCESS)
        {
            *pPrachRxHndl = nullptr;
            delete new_pipeline;
            return status;
        }
        if(new_pipeline == nullptr)
        {
            return CUPHY_STATUS_ALLOC_FAILED;
        }
        *pPrachRxHndl = new_pipeline;
        return CUPHY_STATUS_SUCCESS;
    });
}

#if 0
const void* cuphyGetMemoryFootprintTrackerPrachRx(cuphyPrachRxHndl_t prachRxHndl)
{
    if(prachRxHndl == nullptr)
    {
        return nullptr;
    }
    PrachRx* pipeline_ptr  = static_cast<PrachRx*>(prachRxHndl);
    return pipeline_ptr->getMemoryTracker();
}
#endif

const void* PrachRx::getMemoryTracker()
{
    return &m_memoryFootprint;
}

cuphyStatus_t CUPHYWINAPI cuphySetupPrachRx(cuphyPrachRxHndl_t prachRxHndl, cuphyPrachDynPrms_t* pDynPrms)
{
    if(pDynPrms == nullptr)
    {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "Cannot perform setup with null pDynPrms");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    
    return cuphy::tryCallableAndCatch([&]
    {
        PUSH_RANGE("cuphySetupPrachRx", 1);
        PrachRx* pipeline_ptr  = static_cast<PrachRx*>(prachRxHndl);
        if(pDynPrms->pDbg->enableApiLogging)
        {
            PrachRx::printDynApiPrms(pDynPrms);
        }
        auto status = pipeline_ptr->expandParameters(pDynPrms);
        POP_RANGE;
        return status;
    });
}

cuphyStatus_t CUPHYWINAPI cuphyRunPrachRx(cuphyPrachRxHndl_t prachRxHndl)
{
    if(prachRxHndl == nullptr)
    {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "Run called with null handle");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    
    return cuphy::tryCallableAndCatch([&]
    {
        PUSH_RANGE("cuphyRunPrachRx", 2);
        PrachRx* pipeline_ptr  = static_cast<PrachRx*>(prachRxHndl);
        auto temp = pipeline_ptr->Run();
        POP_RANGE;
        return temp;
    });
}

cuphyStatus_t CUPHYWINAPI cuphyWriteDbgBufSynchPrach(cuphyPrachRxHndl_t pPrachRxHndl, cudaStream_t cuStream)
{
    if(nullptr == pPrachRxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
        PrachRx* pipeline_ptr  = static_cast<PrachRx*>(pPrachRxHndl);
        pipeline_ptr->writeDbgBufSynch(cuStream);
        //p->copyOutputToCPU(cuStream);
        //p->printInfo();
    });
}

cuphyStatus_t CUPHYWINAPI cuphyDestroyPrachRx(cuphyPrachRxHndl_t prachRxHndl)
{
    if(prachRxHndl == nullptr)
    {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "Destroy called with null handle");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    PrachRx* pipeline_ptr  = static_cast<PrachRx*>(prachRxHndl);
    delete pipeline_ptr;
    return CUPHY_STATUS_SUCCESS;
}

enum PreambleFormat
{
    ZERO,
    ONE,
    TWO,
    THREE,
    A1,
    A2,
    A3,
    B1,
    B4,
    C0,
    C2,
    A1_B1,
    A2_B2,
    A3_B3,

};

// 3GPP 38.211 (V15.4) Table 6.3.3.2-3
// only Preamble format is needed in cuPHY code
static constexpr PreambleFormat table_prachCfg_FR1TDD[] = 
    {ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 
    ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE, ONE, ONE, ONE, ONE, ONE, TWO, TWO, 
    TWO, TWO, TWO, TWO, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, 
    THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, 
    A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A2, A2, A2, A2, A2, A2, A2, 
    A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, 
    A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, B4, B4, B4, 
    B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, C0, C0, C0, C0, C0, C0, 
    C0, C0, C0, C0, C0, C0, C0, C0, C0, C0, C0, C0, C0, C0, C2, C2, C2, C2, C2, C2, C2, C2, C2, C2, C2, C2, C2, 
    C2, C2, C2, C2, C2, C2, C2, C2, C2, A1_B1, A1_B1, A1_B1, A1_B1, A1_B1, A1_B1, A1_B1, A1_B1, A1_B1, A1_B1, 
    A1_B1, A1_B1, A1_B1, A1_B1, A1_B1, A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, 
    A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, A3_B3, A3_B3, A3_B3, A3_B3, A3_B3, A3_B3, A3_B3, A3_B3, A3_B3, A3_B3, 
    A3_B3, A3_B3, A3_B3, A3_B3, A3_B3};

// 3GPP 38.211 (V15.4) Table 6.3.3.2-2
// only Preamble format is needed in cuPHY code
static constexpr PreambleFormat table_prachCfg_FR1FDD[] = 
    {ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 
    ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE, ONE, ONE, ONE, ONE, ONE, ONE, 
    ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, TWO, TWO, TWO, 
    TWO, TWO, TWO, TWO, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, 
    THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, THREE, 
    A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1_B1, A1_B1, A1_B1, 
    A1_B1, A1_B1, A1_B1, A1_B1, A1_B1, A1_B1, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, 
    A2, A2, A2, A2, A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, A3, A3, A3, A3, A3, 
    A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3_B3, A3_B3, A3_B3, A3_B3, A3_B3, A3_B3, 
    A3_B3, A3_B3, A3_B3, A3_B3, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, 
    B1, B1, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, C0, C0, C0, 
    C0, C0, C0, C0, C0, C0, C0, C0, C0, C0, C0, C0, C0, C0, C2, C2, C2, C2, C2, C2, C2, C2, C2, C2, C2, C2, 
    C2, C2, C2, C2, C2, C2, C2, C2};

// 3GPP 38.211 (V15.4) Table 6.3.3.2-4
// only Preamble format is needed in cuPHY code
static constexpr PreambleFormat table_prachCfg_FR2[] = 
    {A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, 
    A1, A1, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, 
    A2, A2, A2, A2, A2, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, 
    A3, A3, A3, A3, A3, A3, A3, A3, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, B1, 
    B1, B1, B1, B1, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, B4, 
    B4, B4, B4, B4, B4, B4, B4, B4, B4, C0, C0, C0, C0, C0, C0, C0, C0, C0, C0, C0, C0, C0, C0, C0, C0, C0, C0, 
    C0, C0, C0, C0, C0, C0, C0, C0, C0, C0, C0, C2, C2, C2, C2, C2, C2, C2, C2, C2, C2, C2, C2, C2, C2, C2, C2, 
    C2, C2, C2, C2, C2, C2, C2, C2, C2, C2, C2, C2, C2, A1_B1, A1_B1, A1_B1, A1_B1, A1_B1, A1_B1, A1_B1, A1_B1, 
    A1_B1, A1_B1, A1_B1, A1_B1, A1_B1, A1_B1, A1_B1, A1_B1, A1_B1, A1_B1, A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, 
    A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, A2_B2, A3_B3, A3_B3, A3_B3, A3_B3, 
    A3_B3, A3_B3, A3_B3, A3_B3, A3_B3, A3_B3, A3_B3, A3_B3, A3_B3, A3_B3, A3_B3, A3_B3, A3_B3, A3_B3};

// 3GPP TS 38.211 V15.4.0 Table 6.3.3.1-3
static constexpr uint16_t table_logIdx2u_839[] = 
    {129, 710, 140, 699, 120, 719, 210, 629, 168, 671, 84, 755, 105, 734, 93, 746, 70, 769, 60, 779, 2, 837, 1, 838, 
    56, 783, 112, 727, 148, 691, 80, 759, 42, 797, 40, 799, 35, 804, 73, 766, 146, 693, 31, 808, 28, 811, 30, 809, 27, 
    812, 29, 810, 24, 815, 48, 791, 68, 771, 74, 765, 178, 661, 136, 703, 86, 753, 78, 761, 43, 796, 39, 800, 20, 819, 
    21, 818, 95, 744, 202, 637, 190, 649, 181, 658, 137, 702, 125, 714, 151, 688, 217, 622, 128, 711, 142, 697, 122, 717, 
    203, 636, 118, 721, 110, 729, 89, 750, 103, 736, 61, 778, 55, 784, 15, 824, 14, 825, 12, 827, 23, 816, 34, 805, 37, 802, 
    46, 793, 207, 632, 179, 660, 145, 694, 130, 709, 223, 616, 228, 611, 227, 612, 132, 707, 133, 706, 143, 696, 135, 704, 
    161, 678, 201, 638, 173, 666, 106, 733, 83, 756, 91, 748, 66, 773, 53, 786, 10, 829, 9, 830, 7, 832, 8, 831, 16, 823, 47, 
    792, 64, 775, 57, 782, 104, 735, 101, 738, 108, 731, 208, 631, 184, 655, 197, 642, 191, 648, 121, 718, 141, 698, 149, 690, 
    216, 623, 218, 621, 152, 687, 144, 695, 134, 705, 138, 701, 199, 640, 162, 677, 176, 663, 119, 720, 158, 681, 164, 675, 
    174, 665, 171, 668, 170, 669, 87, 752, 169, 670, 88, 751, 107, 732, 81, 758, 82, 757, 100, 739, 98, 741, 71, 768, 59, 780, 
    65, 774, 50, 789, 49, 790, 26, 813, 17, 822, 13, 826, 6, 833, 5, 834, 33, 806, 51, 788, 75, 764, 99, 740, 96, 743, 97, 
    742, 166, 673, 172, 667, 175, 664, 187, 652, 163, 676, 185, 654, 200, 639, 114, 725, 189, 650, 115, 724, 194, 645, 195, 
    644, 192, 647, 182, 657, 157, 682, 156, 683, 211, 628, 154, 685, 123, 716, 139, 700, 212, 627, 153, 686, 213, 626, 215, 
    624, 150, 689, 225, 614, 224, 615, 221, 618, 220, 619, 127, 712, 147, 692, 124, 715, 193, 646, 205, 634, 206, 633, 116, 
    723, 160, 679, 186, 653, 167, 672, 79, 760, 85, 754, 77, 762, 92, 747, 58, 781, 62, 777, 69, 770, 54, 785, 36, 803, 32, 
    807, 25, 814, 18, 821, 11, 828, 4, 835, 3, 836, 19, 820, 22, 817, 41, 798, 38, 801, 44, 795, 52, 787, 45, 794, 63, 776, 
    67, 772, 72, 767, 76, 763, 94, 745, 102, 737, 90, 749, 109, 730, 165, 674, 111, 728, 209, 630, 204, 635, 117, 722, 188, 
    651, 159, 680, 198, 641, 113, 726, 183, 656, 180, 659, 177, 662, 196, 643, 155, 684, 214, 625, 126, 713, 131, 708, 219, 
    620, 222, 617, 226, 613, 230, 609, 232, 607, 262, 577, 252, 587, 418, 421, 416, 423, 413, 426, 411, 428, 376, 463, 395, 
    444, 283, 556, 285, 554, 379, 460, 390, 449, 363, 476, 384, 455, 388, 451, 386, 453, 361, 478, 387, 452, 360, 479, 310, 
    529, 354, 485, 328, 511, 315, 524, 337, 502, 349, 490, 335, 504, 324, 515, 323, 516, 320, 519, 334, 505, 359, 480, 295, 
    544, 385, 454, 292, 547, 291, 548, 381, 458, 399, 440, 380, 459, 397, 442, 369, 470, 377, 462, 410, 429, 407, 432, 281, 
    558, 414, 425, 247, 592, 277, 562, 271, 568, 272, 567, 264, 575, 259, 580, 237, 602, 239, 600, 244, 595, 243, 596, 275, 
    564, 278, 561, 250, 589, 246, 593, 417, 422, 248, 591, 394, 445, 393, 446, 370, 469, 365, 474, 300, 539, 299, 540, 364, 
    475, 362, 477, 298, 541, 312, 527, 313, 526, 314, 525, 353, 486, 352, 487, 343, 496, 327, 512, 350, 489, 326, 513, 319, 
    520, 332, 507, 333, 506, 348, 491, 347, 492, 322, 517, 330, 509, 338, 501, 341, 498, 340, 499, 342, 497, 301, 538, 366, 
    473, 401, 438, 371, 468, 408, 431, 375, 464, 249, 590, 269, 570, 238, 601, 234, 605, 257, 582, 273, 566, 255, 584, 254, 
    585, 245, 594, 251, 588, 412, 427, 372, 467, 282, 557, 403, 436, 396, 443, 392, 447, 391, 448, 382, 457, 389, 450, 294, 
    545, 297, 542, 311, 528, 344, 495, 345, 494, 318, 521, 331, 508, 325, 514, 321, 518, 346, 493, 339, 500, 351, 488, 306, 
    533, 289, 550, 400, 439, 378, 461, 374, 465, 415, 424, 270, 569, 241, 598, 231, 608, 260, 579, 268, 571, 276, 563, 409, 
    430, 398, 441, 290, 549, 304, 535, 308, 531, 358, 481, 316, 523, 293, 546, 288, 551, 284, 555, 368, 471, 253, 586, 256, 
    583, 263, 576, 242, 597, 274, 565, 402, 437, 383, 456, 357, 482, 329, 510, 317, 522, 307, 532, 286, 553, 287, 552, 266, 
    573, 261, 578, 236, 603, 303, 536, 356, 483, 355, 484, 405, 434, 404, 435, 406, 433, 235, 604, 267, 572, 302, 537, 309, 
    530, 265, 574, 233, 606, 367, 472, 296, 543, 336, 503, 305, 534, 373, 466, 280, 559, 279, 560, 419, 420, 240, 599, 258, 
    581, 229, 610};

// 3GPP TS 38.211 V15.4.0 Table 6.3.3.1-4
static constexpr uint16_t table_logIdx2u_139[] = 
    {1, 138, 2, 137, 3, 136, 4, 135, 5, 134, 6, 133, 7, 132, 8, 131, 9, 130, 10, 129, 11, 128, 12, 127, 13, 126, 14, 125, 15, 
    124, 16, 123, 17, 122, 18, 121, 19, 120, 20, 119, 21, 118, 22, 117, 23, 116, 24, 115, 25, 114, 26, 113, 27, 112, 28, 111, 
    29, 110, 30, 109, 31, 108, 32, 107, 33, 106, 34, 105, 35, 104, 36, 103, 37, 102, 38, 101, 39, 100, 40, 99, 41, 98, 42, 97, 
    43, 96, 44, 95, 45, 94, 46, 93, 47, 92, 48, 91, 49, 90, 50, 89, 51, 88, 52, 87, 53, 86, 54, 85, 55, 84, 56, 83, 57, 82, 
    58, 81, 59, 80, 60, 79, 61, 78, 62, 77, 63, 76, 64, 75, 65, 74, 66, 73, 67, 72, 68, 71, 69, 70};

// 3GPP TS 38.211 V15.4.0 Table 6.3.3.1-5
// only unrestricted set (prach_cuphy_params->restrictedSet == 0) needed
static constexpr uint16_t table_NCS_1p25k[] = {0, 13, 15, 18, 22, 26, 32, 38, 46, 59, 76, 93, 119, 167, 279, 419};

// 3GPP TS 38.211 V15.4.0 Table 6.3.3.1-7
// only unrestricted set (prach_cuphy_params->restrictedSet == 0) needed
static constexpr uint16_t table_NCS_15kplus[] = {0, 2, 4, 6, 8, 10, 12, 13, 15, 17, 19, 23, 27, 34, 46, 69};

struct kBar_entry
{
    uint16_t L_RA;
    uint8_t delta_f;
    uint8_t kbar;
    float delta_f_RA;
};

// 3GPP 38.211 (V15.4) Table 6.3.3.2-1
static constexpr kBar_entry kBar_table[] = {{839,15,7,1.25},
                                            {839,30,1,1.25},
                                            {839,60,133,1.25},
                                            {839,15,12,5},
                                            {839,30,10,5},
                                            {839,60,7,5},
                                            {139,15,2,15},
                                            {139,30,2,15},
                                            {139,60,2,15},
                                            {139,15,2,30},
                                            {139,30,2,30},
                                            {139,60,2,30},
                                            {139,60,2,60},
                                            {139,120,2,60},
                                            {139,60,2,120},
                                            {139,120,2,120}};

static PreambleFormat getPreambleFormat(uint8_t prachCfgIdx, uint8_t FR, uint8_t duplex)
{
    if(FR == 1 && duplex == 1)
    {
        // 3GPP 38.211 (V15.4) Table 6.3.3.2-3 table_prachCfg_FR1TDD
        return table_prachCfg_FR1TDD[prachCfgIdx];
    }
    else if(FR == 1 && duplex == 0)
    {
        // 3GPP 38.211 (V15.4) Table 6.3.3.2-2 table_prachCfg_FR1FDD
        return table_prachCfg_FR1FDD[prachCfgIdx];
    }
    else
    {
        // 3GPP 38.211 (V15.4) Table 6.3.3.2-4 table_prachCfg_FR2
        return table_prachCfg_FR2[prachCfgIdx];
    }
}

static uint16_t findZcPar(uint8_t prmbIdx, uint16_t rootSequenceIndex, uint16_t L_RA, uint16_t N_CS)
{
    // Derive logIdx from prmbIdx
    uint16_t logIdx = rootSequenceIndex;
    if(N_CS == 0)
    {
        logIdx = (logIdx + prmbIdx) % (L_RA - 1);
    }
    else
    {
        uint16_t countCS = L_RA/N_CS;
        logIdx = (logIdx + prmbIdx/countCS) % (L_RA - 1);
    }

    // logical root mapping
    if(L_RA == 839)
    {
        assert(logIdx <= 837);
        return table_logIdx2u_839[logIdx];
    }
    else if(L_RA == 139)
    {
        assert(logIdx <= 137);
        return table_logIdx2u_139[logIdx];
    }
    else
    {
        assert(0); // L_RA length is not supported
        return 0;
    }
}

// Lookup table for thro based on L_RA, mu and N_ant values
// layout for the table is as follows
// thr0[MU_COUNT][L_RA_COUNT][N_ANT_COUNT]
static constexpr float thr0[2][2][6] = {{{11.0f, 11.0f, 8.5f, 8.5f, 8.5f, 7.0f}, 
                                        {10.0f, 10.0f, 8.0f, 8.0f, 8.0f, 6.0f}},
                                        {{12.0f, 12.0f, 9.5f, 9.5f, 9.5f, 7.5f},
                                        {10.0f, 10.0f, 8.5f, 8.5f, 8.5f, 6.0f}}};

static float getthr0(uint8_t mu, uint16_t L_RA, uint32_t N_ant)
{
    assert(L_RA == 139 || L_RA == 839);
    assert(mu == 0 || mu == 1);
    int l_ra_index = L_RA == 139 ? 0 : 1;
    int n_ant_index = N_ant > 5 ? 5 : N_ant - 1;

    return thr0[mu][l_ra_index][n_ant_index];
}

static uint16_t findq(uint16_t u, uint16_t L_RA)
{
    uint16_t q = 0;
    for(q = 0; q < L_RA; ++q)
    {
        if((q * u) % L_RA == 1)
            break;
    }

    if(q == L_RA)
    {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, "Can not find q < L_RA ({})",L_RA);
    }

    return q;

}

static void genZcPreamble(__half2* y_u_ref_host, uint16_t L_RA, uint16_t* uvalues, uint32_t uCount)
{
    assert(L_RA <= 839);
    double invsqrt_lra = 1.0f / sqrt((double)L_RA);

    for(int iu = 0; iu < uCount; ++iu)
    {
        uint16_t u = uvalues[iu];

        uint16_t q = findq(u, L_RA);

        std::complex<double> x_u[839];

        for(int i = 0; i < L_RA; ++i)
        {
            const double pi = std::acos(-1);
            x_u[i] = exp(std::complex<double>(0.0F, -1.0F * pi * ((double)u * (double)i * (double)(i+1))/(double)L_RA));
        }

        std::complex<double> sum = std::accumulate(x_u, x_u + L_RA, std::complex<double>(0,0));

        for(int m = 0; m < L_RA; ++m)
        {
            std::complex<double> y_uv = sum * x_u[0] * std::conj(x_u[q * m % L_RA]) * invsqrt_lra;
            y_u_ref_host[iu * L_RA + m] = __half2(y_uv.real(), y_uv.imag());
        }
    }
}

/** @brief: Populate PrachParams from cuphyPrachCellStatPrms_t.
 *  @param[in] prach_cuphy_params: pointer to configuration paramters for PRACH
 *  @param[out] prach_params: pointer to paramters for PRACH receiver processing
 *  @param[out] d_y_u_ref: device pointer to reference sequence. Size of this is prach_params->L_RA (faster dim)  * prach_params->uCount (slower dim)
 *                          ownership of pointer lies with client code and it is expected to de-allocate the memory
 *  @param[in] strm: CUDA stream for asynchronous launch
 *  @return cuphy status
 */
static cuphyStatus_t deriveParams(const cuphyPrachCellStatPrms_t* prach_cuphy_params, 
                                const cuphyPrachOccaStatPrms_t* pOccaPrms,
                                PrachParams* prach_params, 
                                cuphy::buffer<__half2, cuphy::device_alloc>* d_y_u_ref,
                                cudaStream_t strm,
                                cuphyMemoryFootprint* pMemoryFootprint=nullptr)
{
    static_assert(sizeof(table_prachCfg_FR1FDD) / sizeof(PreambleFormat) == 256);
    static_assert(sizeof(table_prachCfg_FR1TDD) / sizeof(PreambleFormat) == 256);
    static_assert(sizeof(table_prachCfg_FR2) / sizeof(PreambleFormat) == 256);
    static_assert(sizeof(table_logIdx2u_839) / sizeof(uint16_t) == 838);
    static_assert(sizeof(table_logIdx2u_139) / sizeof(uint16_t) == 138);
    static_assert(sizeof(table_NCS_1p25k) / sizeof(uint16_t) == 16);
    static_assert(sizeof(table_NCS_15kplus) / sizeof(uint16_t) == 16);
    static_assert(sizeof(kBar_table) / sizeof(kBar_entry) == 16);

    // Parameter validation is now handled by validateStaticParams()

    PreambleFormat preambleFormat = getPreambleFormat(prach_cuphy_params->configurationIndex, 
                                                        prach_cuphy_params->FR,
                                                        prach_cuphy_params->duplex);

    uint32_t N_ant = prach_cuphy_params->N_ant;
    uint8_t mu = prach_cuphy_params->mu;

    uint32_t delta_f_RA;
    uint16_t L_RA;
    uint32_t Nfft;
    uint32_t N_rep;
    uint16_t N_CS;

    switch(preambleFormat)
    {
        case PreambleFormat::ZERO:
            delta_f_RA = 1250;
            L_RA = 839;
            Nfft = 1024;
            N_rep = 1;
            N_CS = table_NCS_1p25k[pOccaPrms->prachZeroCorrConf];
            break;
        case PreambleFormat::B4:
            delta_f_RA = 15000 * (1 << mu);
            L_RA = 139;
            Nfft = 256;
            N_rep = 12;
            N_CS = table_NCS_15kplus[pOccaPrms->prachZeroCorrConf];
            break;
        default:
            // This should not happen as validation is done in validateStaticParams()
            NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, "Unexpected preambleFormat {} - validation should have caught this", +preambleFormat);
            return CUPHY_STATUS_INTERNAL_ERROR;
    }

    // generate ZC sequence and preamble
    constexpr uint8_t Nprmb = PRACH_MAX_NUM_PREAMBLES;
    uint16_t uvalues[Nprmb];

    uint16_t oldu = std::numeric_limits<unsigned short>::max();
    uint32_t uCount = 0;
    
    for (uint8_t prmbIdx = 0; prmbIdx < Nprmb; ++prmbIdx)
    {
        uint16_t u = findZcPar(prmbIdx, pOccaPrms->prachRootSequenceIndex, L_RA, N_CS);
        assert(u != std::numeric_limits<unsigned short>::max());

        if(u != oldu)
        {
            uvalues[uCount] = u;
            ++uCount;
            oldu = u;
        }
    }

    // allocate tensor for y_u_ref on host memory and gpu memory
    cuphy::buffer<__half2, cuphy::pinned_alloc> h_y_u_ref(L_RA * uCount);
    *d_y_u_ref = cuphy::buffer<__half2, cuphy::device_alloc>(L_RA * uCount, pMemoryFootprint);

    // populate y_u_ref tensor on CPU
    genZcPreamble(h_y_u_ref.addr(), L_RA, uvalues, uCount);

    // transfer data to GPU memory
    cudaError_t result = cudaMemcpyAsync(d_y_u_ref->addr(), h_y_u_ref.addr(), sizeof(__half2) * L_RA * uCount, cudaMemcpyHostToDevice, strm);
    if(cudaSuccess != result)
    {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "CUDA Runtime Error: {}:{}:{}", 
                   __FILE__, __LINE__, cudaGetErrorString(result));
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "L_RA {}, uCount {}",L_RA, uCount);

        return CUPHY_STATUS_MEMCPY_ERROR;
    }

    // Set preamble detection SNR threshold
    // Test with preamble format 0 and B4, mu = 0 and 1.

    int M = sizeof(kBar_table) / sizeof(kBar_entry);

    // find kBar for subcarrier offset in unit of delta_f_RA
    bool find_flag = false;
    uint8_t kBar = 0;
    uint32_t delta_f = 15000 * (1 << mu);
    for(int m = 0; m < M; ++m)
    {
        if(L_RA == kBar_table[m].L_RA && delta_f_RA == kBar_table[m].delta_f_RA * 1000 &&
            delta_f == kBar_table[m].delta_f * 1000)
        {
            kBar = kBar_table[m].kbar; // preamble first subcarrier shift
            find_flag = true;
        }
    }

    if(!find_flag)
    {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, "kBar table error ... ");
        return CUPHY_STATUS_INTERNAL_ERROR;
    }

    constexpr int N_nc = PRACH_NUM_NON_COHERENT_COMBINING_GROUPS;

    // Add derived paramters into PrachParams
    prach_params->L_RA = L_RA;
    prach_params->uCount = uCount;
    prach_params->N_rep = N_rep;
    prach_params->Nfft = Nfft;
    prach_params->N_nc = N_nc;
    prach_params->kBar = kBar;
    prach_params->delta_f_RA = delta_f_RA;
    prach_params->N_CS = N_CS;
    prach_params->mu = mu;
    prach_params->N_ant = N_ant;

    result = cudaStreamSynchronize(strm);
    if(cudaSuccess != result)
    {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "CUDA Runtime Error: {}:{}:{}", 
                   __FILE__, __LINE__, cudaGetErrorString(result));
        return CUPHY_STATUS_MEMCPY_ERROR;
    }

    return CUPHY_STATUS_SUCCESS;
}

/** @brief: Return workspace size, in bytes, needed for all configuration parameters
 *          and intermediate computations of the PRACH receiver. Does not allocate any space.
 *  @param[in] h_prach_params: pointer to configuration paramters for PRACH
 *  @param[in] prach_complex_data_type: PRACH receiver data type identifier: CUPHY_C_32F or CUPHY_C_16F
 *  @return workspace size in bytes
 */
static size_t getReceiverWorkspaceSize(const PrachParams * prach_params,
                                       cuphyDataType_t prach_complex_data_type) 
{
    const int Nfft = prach_params->Nfft;
    const int N_ant = prach_params->N_ant;
    const int uCount = prach_params->uCount;
    const int N_nc = prach_params->N_nc;

    int fft_buffer_elements = Nfft * N_ant * uCount * N_nc;
    int prach_element_size = sizeof(cuComplex);
    if (prach_complex_data_type == CUPHY_C_16F)
    {
        prach_element_size = sizeof(__half2);
    }

    int pdp_buffer_elements = CUPHY_PRACH_RX_NUM_PREAMBLE * N_ant;

    if (prach_complex_data_type == CUPHY_C_32F)
        return (sizeof(unsigned int) + sizeof(float) + sizeof(float) * N_ant + pdp_buffer_elements * sizeof(prach_pdp_t<float>) + fft_buffer_elements * prach_element_size + sizeof(prach_det_t<float>));
    else
        return (sizeof(unsigned int) + sizeof(float) + sizeof(float) * N_ant + pdp_buffer_elements * sizeof(prach_pdp_t<__half>) + fft_buffer_elements * prach_element_size + sizeof(prach_det_t<__half>));
}

template <fmtlog::LogLevel log_level>
static void printPrachCellStatPrms(cuphyPrachCellStatPrms_t const* pCellPrms)
{
    NVLOG_FMT(log_level, NVLOG_PRACH,"N_ant: {}", pCellPrms->N_ant);
    NVLOG_FMT(log_level, NVLOG_PRACH,"configurationIndex: {}", pCellPrms->configurationIndex);
    NVLOG_FMT(log_level, NVLOG_PRACH,"restrictedSet: {}", pCellPrms->restrictedSet);
    NVLOG_FMT(log_level, NVLOG_PRACH,"FR: {}", pCellPrms->FR);
    NVLOG_FMT(log_level, NVLOG_PRACH,"duplex: {}", pCellPrms->duplex);
    NVLOG_FMT(log_level, NVLOG_PRACH,"mu: {}", pCellPrms->mu);
}

template <fmtlog::LogLevel log_level>
static void printPrachOccaDynPrms(cuphyPrachOccaDynPrms_t const* pOccaPrms)
{
    NVLOG_FMT(log_level, NVLOG_PRACH,"occaPrmStatIdx: {} ", pOccaPrms->occaPrmStatIdx);
    NVLOG_FMT(log_level, NVLOG_PRACH,"occaPrmDynIdx: {} ", pOccaPrms->occaPrmDynIdx);
    NVLOG_FMT(log_level, NVLOG_PRACH,"force_thr0: {}", pOccaPrms->force_thr0);
}

template <fmtlog::LogLevel log_level>
void PrachRx::printStatApiPrms(cuphyPrachStatPrms_t const* pStatPrms)
{
    for(int i=0;i<pStatPrms->nMaxCells;i++){
        NVLOG_FMT(log_level, NVLOG_PRACH,"Cell [{}]:",i);
        printPrachCellStatPrms<log_level>(pStatPrms->pCellPrms);
    }
    for(int i=0;i<pStatPrms->nMaxOccaProc;i++){
        NVLOG_FMT(log_level, NVLOG_PRACH,"Occasion [{}]:",i);
        NVLOG_FMT(log_level, NVLOG_PRACH," cellPrmStatIdx: {}",pStatPrms->pOccaPrms[i].cellPrmStatIdx);
        NVLOG_FMT(log_level, NVLOG_PRACH," prachRootSequenceIndex: {}",pStatPrms->pOccaPrms[i].prachRootSequenceIndex);
        NVLOG_FMT(log_level, NVLOG_PRACH," prachZeroCorrConf: {}",pStatPrms->pOccaPrms[i].prachZeroCorrConf);
    }
}

template <fmtlog::LogLevel log_level>
void PrachRx::printDynApiPrms(cuphyPrachDynPrms_t* pDynPrm)
{
    NVLOG_FMT(log_level, NVLOG_PRACH,"processing mode: {}", pDynPrm->procModeBmsk);
    for(int i=0; i<pDynPrm->nOccaProc; i++)
    {
        NVLOG_FMT(log_level, NVLOG_PRACH,"Occasion [{}]:",i);
        printPrachOccaDynPrms<log_level>(&pDynPrm->pOccaPrms[i]);
    }
}

cuphyStatus_t PrachRx::validateStaticParams(cuphyPrachStatPrms_t const* pStatPrms)
{
    // Basic null pointer checks
    if (pStatPrms == nullptr)
    {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "{} Invalid input: pStatPrms is null", __FUNCTION__);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if (pStatPrms->pCellPrms == nullptr)
    {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "{} Invalid input: pCellPrms is null", __FUNCTION__);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if (pStatPrms->pOccaPrms == nullptr)
    {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "{} Invalid input: pOccaPrms is null", __FUNCTION__);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    const uint16_t nMaxCells = pStatPrms->nMaxCells;
    const uint16_t nMaxOccasions = pStatPrms->nMaxOccaProc;

    // Calculate total cell occasions across all cells
    uint16_t nTotCellOcca = 0;
    for (uint16_t i = 0; i < nMaxCells; ++i)
    {
        nTotCellOcca += pStatPrms->pCellPrms[i].nFdmOccasions;
    }

    // Check if total cell occasions exceed max occasions
    if (nTotCellOcca > nMaxOccasions)
    {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, 
                   "Cell occasions {} greater than nMaxOccasions {}", 
                   nTotCellOcca, nMaxOccasions);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // Validate each cell and its occasions
    for (int prachCell = 0; prachCell < nMaxCells; ++prachCell)
    {
        const cuphyPrachCellStatPrms_t* cellPrms = &pStatPrms->pCellPrms[prachCell];

        // Validate cell parameters
        if (cellPrms->restrictedSet != 0)
        {
            NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, 
                       "restrictedSet ({}) is not supported (must be 0)", 
                       cellPrms->restrictedSet);
            return CUPHY_STATUS_NOT_SUPPORTED;
        }

        if (cellPrms->mu > 1)
        {
            NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, 
                       "mu ({}) is not supported (must be 0 or 1)", 
                       cellPrms->mu);
            return CUPHY_STATUS_NOT_SUPPORTED;
        }

        // Validate preamble format is supported
        PreambleFormat preambleFormat = getPreambleFormat(cellPrms->configurationIndex, 
                                                          cellPrms->FR, 
                                                          cellPrms->duplex);
        
        if (preambleFormat != PreambleFormat::ZERO && preambleFormat != PreambleFormat::B4)
        {
            NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, 
                       "preambleFormat {} is not supported", 
                       +preambleFormat);
            return CUPHY_STATUS_NOT_SUPPORTED;
        }

        // Validate occasions for this cell
        for (int occa = 0; occa < cellPrms->nFdmOccasions; ++occa)
        {
            const int prachOccasion = occa + cellPrms->occaStartIdx;
            
            if (prachOccasion >= nMaxOccasions)
            {
                NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, 
                           "Occasion index {} exceeds nMaxOccasions {}", 
                           prachOccasion, nMaxOccasions);
                return CUPHY_STATUS_INVALID_ARGUMENT;
            }

            const cuphyPrachOccaStatPrms_t* occaPrms = &pStatPrms->pOccaPrms[prachOccasion];

            // Validate occasion parameters
            if (occaPrms->prachZeroCorrConf >= 16)
            {
                NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, 
                           "prachZeroCorrConf ({}) exceeds maximum value (15)", 
                           occaPrms->prachZeroCorrConf);
                return CUPHY_STATUS_INVALID_ARGUMENT;
            }

            // Additional validation for root sequence index based on preamble format
            const uint16_t L_RA = (preambleFormat == PreambleFormat::ZERO) ? 839 : 139;
            if (occaPrms->prachRootSequenceIndex >= (L_RA - 1))
            {
                NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, 
                           "prachRootSequenceIndex ({}) exceeds maximum for L_RA ({})", 
                           occaPrms->prachRootSequenceIndex, L_RA - 1);
                return CUPHY_STATUS_INVALID_ARGUMENT;
            }
        }
    }

    return CUPHY_STATUS_SUCCESS;
}

PrachRx::PrachRx(cuphyPrachStatPrms_t const* pStatPrms, cuphyStatus_t* status) : m_cudaDeviceArchInfo(get_cuda_device_arch())
{
    // Validate static parameters first
    *status = validateStaticParams(pStatPrms);
    if (*status != CUPHY_STATUS_SUCCESS)
    {
        return;
    }

    pStatPrms->pOutInfo->pMemoryFootprint = &m_memoryFootprint; // update  static parameter field that points to the cuphyMemoryFootprintTracker object for this channel

    nMaxCells = pStatPrms->nMaxCells;
    nMaxOccasions = pStatPrms->nMaxOccaProc;

    // find total occaions across all cells
    nTotCellOcca = 0;
    for(uint16_t i = 0; i < nMaxCells; ++i)
    {
        nTotCellOcca +=  pStatPrms->pCellPrms[i].nFdmOccasions;
    }

    staticParam.resize(nMaxOccasions);
    activeOccasions.clear();
    activeOccasions.resize(nMaxOccasions,0);
    // initialize with 1 as by default all FFT nodes are enabled
    prevActiveOccasions.clear();
    prevActiveOccasions.resize(nMaxOccasions, 1);

    cuphy::stream cuphyStream(cudaStreamNonBlocking);
    cudaStream_t strm = cuphyStream.handle();

    h_staticParam = cuphy::buffer<PrachDeviceInternalStaticParamPerOcca, cuphy::pinned_alloc>(nMaxOccasions);
    h_dynParam = cuphy::buffer<PrachInternalDynParamPerOcca, cuphy::pinned_alloc>(nMaxOccasions);

    const int nFftSizes = sizeof(PRACH_SUPPORTED_FFT_SIZES) / sizeof(PRACH_SUPPORTED_FFT_SIZES[0]);
    const int maxFftsPerOccasion = MAX_N_ANTENNAS_SUPPORTED * PRACH_MAX_NUM_PREAMBLES * PRACH_NUM_NON_COHERENT_COMBINING_GROUPS;
    h_fftPointers = cuphy::buffer<cuFloatComplex *, cuphy::pinned_alloc>(nFftSizes * nMaxOccasions * maxFftsPerOccasion);

    d_staticParam = cuphy::buffer<PrachDeviceInternalStaticParamPerOcca, cuphy::device_alloc>(nMaxOccasions, &m_memoryFootprint);
    d_dynParam = cuphy::buffer<PrachInternalDynParamPerOcca, cuphy::device_alloc>(nMaxOccasions, &m_memoryFootprint);

    for(int prachCell = 0; prachCell < nMaxCells; ++prachCell)
    {
        for(int occa = 0; occa < pStatPrms->pCellPrms[prachCell].nFdmOccasions; ++occa)
        {
            int prachOccasion  = occa + pStatPrms->pCellPrms[prachCell].occaStartIdx;
            PrachParams& prach_params = staticParam[prachOccasion].prach_params;
            cuphy::buffer<__half2, cuphy::device_alloc>& d_y_u_ref = staticParam[prachOccasion].d_y_u_ref;

            *status = deriveParams(&(pStatPrms->pCellPrms[prachCell]), &(pStatPrms->pOccaPrms[prachOccasion]), &prach_params, &d_y_u_ref, strm, &m_memoryFootprint);

            if(*status != CUPHY_STATUS_SUCCESS)
            {
                return;
            }

            // Allocate workspace size: includes config parameters as well as space for intermediate results
            size_t prach_workspace_size = getReceiverWorkspaceSize(&prach_params, CUPHY_C_32F); // in bytes

            cuphy::buffer<float, cuphy::device_alloc>& prach_workspace_buffer = staticParam[prachOccasion].prach_workspace_buffer;
            prach_workspace_buffer = std::move(buffer<float, device_alloc>(div_round_up(prach_workspace_size, sizeof(float)), &m_memoryFootprint));

            staticParam[prachOccasion].enableUlRxBf = pStatPrms->enableUlRxBf;
            
            h_staticParam[prachOccasion] = {prach_params, prach_workspace_buffer.addr(), d_y_u_ref.addr(), pStatPrms->enableUlRxBf};

            const int N_ant = prach_params.N_ant;

            maxAntenna = maxAntenna >= N_ant ? maxAntenna : N_ant;

            const int N_rep = prach_params.N_rep;
            const int L_RA = prach_params.L_RA;

            // O-RAN FH sends 144 or 864 samples instead of 139 or 839 samples
            const int L_ORAN = (L_RA == 139) ? 144 : 864;

            // align L_ORAN so that same warp doesn't have samples for two different antennas
            // this allows us to use shuffle reduction
            unsigned int align_l_oran = ((L_ORAN * N_rep + 31) >> 5) << 5;
            max_l_oran_ant = std::max(max_l_oran_ant, align_l_oran * N_ant);

            const int uCount = prach_params.uCount;
            max_ant_u = max_ant_u >= N_ant * uCount ? max_ant_u : N_ant * uCount;

            const int Nfft = prach_params.Nfft;
            max_nfft = max_nfft >= Nfft ? max_nfft : Nfft;

            const int N_CS = prach_params.N_CS;
            int zoneSize = N_CS*Nfft/L_RA;
            int zoneSizeExt = 1 << (int(log2f((zoneSize)))+1);

            max_zoneSizeExt = max_zoneSizeExt >= zoneSizeExt ? max_zoneSizeExt : zoneSizeExt;
            activeOccasions[prachOccasion] = 1;
        }
    }

    cudaError_t result = cudaMemcpyAsync(d_staticParam.addr(), h_staticParam.addr(), sizeof(PrachDeviceInternalStaticParamPerOcca) * nMaxOccasions, 
                                        cudaMemcpyHostToDevice, strm);
    if(cudaSuccess != result)
    {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "CUDA Runtime Error: {}:{}:{}", 
                   __FILE__, __LINE__, cudaGetErrorString(result));
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "nMaxOccasions {}", nMaxOccasions);
        *status = CUPHY_STATUS_MEMCPY_ERROR;
    }
    cuphyStream.synchronize();

    // GraphNodeType::FFTNode is the last entry in the enum and we create PRACH_NUM_SUPPORTED_FFT_SIZES nodes
    // for FFTs, so we create GraphNodeType::FFTNode + PRACH_NUM_SUPPORTED_FFT_SIZES total nodes.
    nodes.resize(GraphNodeType::FFTNode + PRACH_NUM_SUPPORTED_FFT_SIZES);
    // initialize graph nodes with nOccaProc as nMaxOccasions
    nPrevOccaProc = nTotCellOcca;
    *status = cuphyPrachCreateGraph(&graph, &graphInstance, nodes, strm,
                            d_dynParam.addr(), 
                            d_staticParam.addr(),
                            staticParam.data(),
                            h_fftPointers.addr(),
                            h_fftInfo,
                            (uint32_t*)numDetectedPrmb.pAddr,
                            (uint32_t*)prmbIndexEstimates.pAddr,
                            (float*)prmbDelayEstimates.pAddr,
                            (float*)prmbPowerEstimates.pAddr,
                            (float*)antRssi.pAddr,
                            (float*)rssi.pAddr,
                            (float*)interference.pAddr,
                            nTotCellOcca,
                            nMaxOccasions,
                            maxAntenna,
                            max_l_oran_ant,
                            max_ant_u,
                            max_nfft,
                            max_zoneSizeExt,
                            activeOccasions,
                            m_cudaDeviceArchInfo.computeCapability);
    //reset for the setup/run phase                        
    maxAntenna = 0;
    max_l_oran_ant = 0;
    max_ant_u = 0;

    m_outputPrms.debugOutputFlag = false;
    if(nullptr != pStatPrms->pDbg)
    {
        if(nullptr != pStatPrms->pDbg->pOutFileName)
        {
            m_outputPrms.debugOutputFlag = true;
            m_outputPrms.outHdf5File     = hdf5hpp::hdf5_file::open(pStatPrms->pDbg->pOutFileName);
        }
    }

    if(PRINT_GPU_MEMORY_CUPHY_CHANNEL == 1)
    {
        m_memoryFootprint.printMemoryFootprint(this, "PRACH");
    }
}

PrachRx::~PrachRx()
{
}

// PRACH Setup
cuphyStatus_t PrachRx::expandParameters(cuphyPrachDynPrms_t* pDynPrms)
{
    cuStream = pDynPrms->cuStream;
    nOccaProc = pDynPrms->nOccaProc;
    procModeBmsk = pDynPrms->procModeBmsk;

    if(nOccaProc > nTotCellOcca)
    {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, 
                   "number of occasions {} cannot exceed total cell occasions {}", nOccaProc, nTotCellOcca);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    std::for_each(activeOccasions.begin(), activeOccasions.end(), 
                    [](char& val) {val = 0;});

    for(int i = 0; i < nOccaProc; ++i)
    {
        uint16_t occaPrmDynIdx = pDynPrms->pOccaPrms[i].occaPrmDynIdx;
        uint16_t occaPrmStaticIdx = pDynPrms->pOccaPrms[i].occaPrmStatIdx;
        activeOccasions[occaPrmStaticIdx] = 1;

        uint32_t L_RA = staticParam[occaPrmStaticIdx].prach_params.L_RA;
        uint32_t N_ant = staticParam[occaPrmStaticIdx].prach_params.N_ant;
        float thr0 = pDynPrms->pOccaPrms[i].force_thr0;
        if(thr0 == 0)
        {
            uint32_t mu = staticParam[occaPrmStaticIdx].prach_params.mu;
            if(staticParam[occaPrmStaticIdx].enableUlRxBf)
            {
                N_ant = pDynPrms->pOccaPrms[i].nUplinkStreams;
            }
            thr0 = getthr0(mu, L_RA, N_ant);
        }
        
        ///////////////////////////////////////
        maxAntenna = maxAntenna >= N_ant ? maxAntenna : N_ant;

        const int N_rep = staticParam[occaPrmStaticIdx].prach_params.N_rep;

        // O-RAN FH sends 144 or 864 samples instead of 139 or 839 samples
        const int L_ORAN = (L_RA == 139) ? 144 : 864;

        // align L_ORAN so that same warp doesn't have samples for two different antennas
        // this allows us to use shuffle reduction
        unsigned int align_l_oran = ((L_ORAN * N_rep + 31) >> 5) << 5;
        max_l_oran_ant = std::max(max_l_oran_ant, align_l_oran * N_ant);

        const int uCount = staticParam[occaPrmStaticIdx].prach_params.uCount;
        max_ant_u = max_ant_u >= N_ant * uCount ? max_ant_u : N_ant * uCount;
        /////////////////////////////////////

        h_dynParam[i] = {(__half2*)(pDynPrms->pDataIn->pTDataRx[occaPrmDynIdx].pAddr), occaPrmStaticIdx, occaPrmDynIdx, thr0, pDynPrms->pOccaPrms[i].nUplinkStreams};
    }

    cudaError_t result = cudaMemcpyAsync(d_dynParam.addr(), h_dynParam.addr(), sizeof(PrachInternalDynParamPerOcca) * nOccaProc, 
                                        cudaMemcpyHostToDevice, cuStream);
    if(cudaSuccess != result)
    {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "CUDA Runtime Error: {}:{}:{}", 
                   __FILE__, __LINE__, cudaGetErrorString(result));
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "nOccaProc {}", nOccaProc);
        return CUPHY_STATUS_MEMCPY_ERROR;
    }

    numDetectedPrmb = pDynPrms->pDataOut->numDetectedPrmb;
    prmbIndexEstimates = pDynPrms->pDataOut->prmbIndexEstimates;
    prmbDelayEstimates = pDynPrms->pDataOut->prmbDelayEstimates;
    prmbPowerEstimates = pDynPrms->pDataOut->prmbPowerEstimates;
    antRssi = pDynPrms->pDataOut->antRssi;
    rssi = pDynPrms->pDataOut->rssi;
    interference = pDynPrms->pDataOut->interference;

    // update graph if graph processing mode
    if(procModeBmsk & PRACH_PROC_MODE_WITH_GRAPH)
    {
        cuphyStatus_t status =  cuphyPrachUpdateGraph(graphInstance, nodes,
                                d_dynParam.addr(),
                                d_staticParam.addr(),
                                h_dynParam.addr(),
                                staticParam.data(),
                                h_fftPointers.addr(),
                                h_fftInfo,
                                (uint32_t*)numDetectedPrmb.pAddr,
                                (uint32_t*)prmbIndexEstimates.pAddr,
                                (float*)prmbDelayEstimates.pAddr,
                                (float*)prmbPowerEstimates.pAddr,
                                (float*)antRssi.pAddr,
                                (float*)rssi.pAddr,
                                (float*)interference.pAddr,
                                prev_numDetectedPrmb,
                                prev_prmbIndexEstimates,
                                prev_prmbDelayEstimates,
                                prev_prmbPowerEstimates,
                                prev_antRssi,
                                prev_rssi,
                                prev_interference,
                                nMaxOccasions,
                                nPrevOccaProc,
                                nOccaProc,
                                maxAntenna,
                                max_l_oran_ant,
                                max_ant_u,
                                max_nfft,
                                max_zoneSizeExt,
                                activeOccasions,
                                prevActiveOccasions);
        if(CUPHY_STATUS_SUCCESS != status)
        {
            pDynPrms->pStatusOut->status = cuphyPrachStatusType_t::CUPHY_PRACH_STATUS_GRAPH_UPDATE_ERROR;
            pDynPrms->pStatusOut->ueIdx = MAX_UINT16;
            pDynPrms->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
        return CUPHY_STATUS_SUCCESS;
    }
    else
    {
        return CUPHY_STATUS_SUCCESS;
    }
}

cuphyStatus_t PrachRx::Run()
{
    // launch graph if graph processing mode
    if(procModeBmsk & PRACH_PROC_MODE_WITH_GRAPH)
    {
        MemtraceDisableScope md; // Disable temporarily
        return cuphyPrachLaunchGraph(graphInstance, cuStream);
    }
    else
    {
        return cuphyPrachReceiver(d_dynParam.addr(), 
                                d_staticParam.addr(),
                                h_dynParam.addr(),
                                staticParam.data(),
                                h_fftPointers.addr(),
                                h_fftInfo,
                                (uint32_t*)numDetectedPrmb.pAddr,
                                (uint32_t*)prmbIndexEstimates.pAddr,
                                (float*)prmbDelayEstimates.pAddr,
                                (float*)prmbPowerEstimates.pAddr,
                                (float*)antRssi.pAddr,
                                (float*)rssi.pAddr,
                                (float*)interference.pAddr,
                                nOccaProc,
                                maxAntenna,
                                max_l_oran_ant,
                                max_ant_u,
                                max_nfft,
                                max_zoneSizeExt,
                                m_cudaDeviceArchInfo.computeCapability,
                                cuStream);
    }
}

void PrachRx::writeDbgBufSynch(cudaStream_t cuStream)
{
    if(m_outputPrms.debugOutputFlag)
    {
        for(int occa = 0; occa < h_staticParam.size(); occa++)
        {
            std::string name = "Stat_Occasion_Param_" + std::to_string(occa);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, name.c_str(), CUPHY_R_32U, sizeof(PrachParams)/sizeof(uint32_t),&h_staticParam[occa].prach_params);
        }
        for(int occa = 0; occa < h_dynParam.size(); occa++)
        {
            std::string name = "Dyn_Occasion_Data_" + std::to_string(occa);
            const int L_ORAN = ((h_staticParam[occa].prach_params.L_RA == 139) ? 144 : 864) * h_staticParam[occa].prach_params.N_rep;
            const int N_ant = h_staticParam[occa].prach_params.N_ant;
            cuphy::tensor_ref tDynData;
            tDynData.desc().set(CUPHY_C_16F, L_ORAN, N_ant, cuphy::tensor_flags::align_tight);
            tDynData.set_addr(h_dynParam[occa].dataRx);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, tDynData, tDynData.desc(), name.c_str(), cuStream);
        }
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, numDetectedPrmb,    "numDetectedPrmb",    cuStream);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, prmbIndexEstimates, "prmbIndexEstimates", cuStream);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, prmbDelayEstimates, "prmbDelayEstimates", cuStream);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, prmbPowerEstimates, "prmbPowerEstimates", cuStream);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, antRssi,            "antRssi",            cuStream);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, rssi,               "rssi",               cuStream);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, interference,       "interference",       cuStream);
    }
}

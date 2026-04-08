<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

**Aerial cuMAC Tests**

All cuMAC Tests are included in the cumac_tests.sh. Current support cuMAC Multi-cell Scheduling for 4T4R and 64TR, SRS, DRL MCS selection, TDL and CDL tests.

**How to run cuMAC tests:**

1. Test Execution

   export cuBB_SDK=/opt/nvidia/cuBB && ${cuBB_SDK}/cuMAC/scripts/cumac_tests.sh -t <test> -l <log> -g <gpu>

   Usage: cumac_tests.sh -t <test> -l <log> -g <gpu>
      test: 4t4r,tdl,cdl,drl,srs,64tr
      log: log folder
      gpu: gpu device id, if not provided, then will be 0

  ```
  Example:
  export cuBB_SDK=/opt/nvidia/cuBB && ${cuBB_SDK}/cuMAC/scripts/cumac_tests.sh -t 4t4r -l /home/aerial/nfs/Log
  export cuBB_SDK=/opt/nvidia/cuBB && ${cuBB_SDK}/cuMAC/scripts/cumac_tests.sh -t drl -l /home/aerial/nfs/Log
  export cuBB_SDK=/opt/nvidia/cuBB && ${cuBB_SDK}/cuMAC/scripts/cumac_tests.sh -t tdl -l /home/aerial/nfs/Log
  export cuBB_SDK=/opt/nvidia/cuBB && ${cuBB_SDK}/cuMAC/scripts/cumac_tests.sh -t cdl -l /home/aerial/nfs/Log
  export cuBB_SDK=/opt/nvidia/cuBB && ${cuBB_SDK}/cuMAC/scripts/cumac_tests.sh -t 64tr -l /home/aerial/nfs/Log
  export cuBB_SDK=/opt/nvidia/cuBB && ${cuBB_SDK}/cuMAC/scripts/cumac_tests.sh -t srs -l /home/aerial/nfs/Log
  ```

2. Test Result
  In the console log, you will see 'TEST PASS' for each case, liking example for 64tr tests

  ```
  2024-12-04 08:33:02,725 - __main__ - INFO - Complete to run cuMAC 64tr tests on 64T64R DL 2C_100UEPerCell_gpuAllocType1_64tr.
  2024-12-04 08:33:02,725 - __main__ - INFO - Checking the log file /home/aerial/nfs/Log/SCF/Container/L0/cuMAC/20241204_083253_cuMAC_64tr_test_0001_main.log ......
  2024-12-04 08:33:02,726 - __main__ - INFO - 64tr TEST PASS.
  ```
**Current Supported cuMAC tests:**

1. 4T4R Multi-cell Scheduling

   ***Test Params:*** cuMAC/scripts/cumac_tv_parameters.csv
   
   ***TV Generation:***
   ```
   DL TV: ./build/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection –t 1
   UL TV: ./build/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection –d 0 –t 1
   ```
   ***Test Command for DL example:***
   ```
   UE selection: ./build/examples/tvLoadingTest/tvLoadingTest –i [path to TV] –g 2 –d 1 –m 01000
   PRG allocation: ./build/examples/tvLoadingTest/tvLoadingTest –i [path to TV] –g 2 –d 1 –m 00100
   Layer selection: ./build/examples/tvLoadingTest/tvLoadingTest –i [path to TV] –g 2 –d 1 –m 00010
   MCS selection: ./build/examples/tvLoadingTest/tvLoadingTest –i [path to TV] –g 2 –d 1 –m 00001
   ```
   
2. SRS tests
   ***Test Command:***
   ```
   compute-sanitizer --error-exitcode 1 --tool memcheck --leak-check full ./build/cuMAC/examples/multiCellSrsScheduler/multiCellSrsScheduler ./cuMAC/examples/multiCellSrsScheduler/srs_scheduler_testing_config.yaml
   ```
3. DRL MCS Selection tests

   ***Test Command:***
   ```
   ./build/examples/drlMcsSelection/drlMcsSelection -i [path to aerial_sdk/cuMAC/testVectors/mlSim] -m [path to model.onnx file] -g [GPU device #]
   ```
4. 64TR Multi-cell Scheduling

  ***Test Params:*** cuMAC/scripts/cumac_64tr_combinations.csv

  ***Test Command:***
  ```
  compute-sanitizer --tool memcheck ./build/cuMAC/examples/multiCellMuMimoScheduler/multiCellMuMimoScheduler -c cuMAC/examples/multiCellMuMimoScheduler/config.yaml
  ```
5. TDL tests

   ***Test Params:*** cuMAC/scripts/cumac_tdl_tv_parameters.csv
   
   ***Test Command:***
   ```
   ./build/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection -d 1 -b 0 -f1
   ./build/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection -d 1 -b 0 -f2
   ```
6. CDL tests

   ***Test Params:*** Same as the TDL's file: cuMAC/scripts/cumac_tdl_tv_parameters.csv
   
   ***Test Command:***
   ```
   ./build/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection -d 1 -b 0 -f3
   ./build/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection -d 1 -b 0 -f4
   ```
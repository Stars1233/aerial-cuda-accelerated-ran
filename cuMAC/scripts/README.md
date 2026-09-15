**Aerial cuMAC Tests**

All cuMAC Tests are included in the cumac_tests.sh. Current support cuMAC Multi-cell Scheduling for 4T4R and 64TR, SRS, DRL MCS selection, TDL, CDL and muMimoUeGrpL2 tests.

**How to run cuMAC tests:**

1. Test Execution

   export cuBB_SDK=/opt/nvidia/cuBB && ${cuBB_SDK}/cuMAC/scripts/cumac_tests.sh -t <test> -l <log> -g <gpu>

   Usage: cumac_tests.sh -t <test> -l <log> -g <gpu> -m <true|false>
      test: 4t4r,tdl,cdl,drl,srs,64tr,muuegrp
      log: log folder
      gpu: gpu device id, if not provided, then will be 0
      -m: smoke mode (true|false, default false); runs a reduced subset (4t4r and 64tr only)

  ```
  Example:
  export cuBB_SDK=/opt/nvidia/cuBB && ${cuBB_SDK}/cuMAC/scripts/cumac_tests.sh -t 4t4r -l /home/aerial/nfs/Log
  export cuBB_SDK=/opt/nvidia/cuBB && ${cuBB_SDK}/cuMAC/scripts/cumac_tests.sh -t drl -l /home/aerial/nfs/Log
  export cuBB_SDK=/opt/nvidia/cuBB && ${cuBB_SDK}/cuMAC/scripts/cumac_tests.sh -t tdl -l /home/aerial/nfs/Log
  export cuBB_SDK=/opt/nvidia/cuBB && ${cuBB_SDK}/cuMAC/scripts/cumac_tests.sh -t cdl -l /home/aerial/nfs/Log
  export cuBB_SDK=/opt/nvidia/cuBB && ${cuBB_SDK}/cuMAC/scripts/cumac_tests.sh -t 64tr -l /home/aerial/nfs/Log
  export cuBB_SDK=/opt/nvidia/cuBB && ${cuBB_SDK}/cuMAC/scripts/cumac_tests.sh -t 64tr -l /home/aerial/nfs/Log -m true   # 64tr smoke subset
  export cuBB_SDK=/opt/nvidia/cuBB && ${cuBB_SDK}/cuMAC/scripts/cumac_tests.sh -t srs -l /home/aerial/nfs/Log
  export cuBB_SDK=/opt/nvidia/cuBB && ${cuBB_SDK}/cuMAC/scripts/cumac_tests.sh -t muuegrp -l /home/aerial/nfs/Log
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
4. 64TR Multi-cell MU-MIMO Channel-Model Tests (GT-10416)

  Standalone 64T64R MU-MIMO channel-model validation sweep driven by `cumac_64tr_test.py`.
  Jointly validates the GPU-based 3GPP 38.901 channel model, multi-cell interference,
  EESM PHY abstraction and channel-estimation error modeling. Each combination runs the
  scheduler binary, then the `cellStatAnalysis.py` post-analysis; a case passes only if
  both print their PASS line. If the arch-specific `build.$(uname -m)` folder is missing,
  the cuBB SDK is built first via `testBenches/phase4_test_scripts/build_aerial_sdk.sh`.

  ***Test Params:***
  - Full coverage (128 combos): `cuMAC/scripts/cumac_64tr_chanmodel_combinations.csv`
  - Smoke subset (68 combos = 1 & 3-cell full sweep + four 6-cell sanity cases, `-m true`): `cuMAC/scripts/cumac_64tr_chanmodel_smoke_combinations.csv`

  ***Test Command:*** (`cumac_tests.sh -t 64tr` invokes this)
  ```
  # full coverage
  python3 cumac_64tr_test.py --execute --log-dir <log>/64tr
  # smoke subset (cumac_tests.sh -t 64tr -m true)
  python3 cumac_64tr_test.py --smoke --execute --log-dir <log>/64tr
  ```
  Under the hood each combination runs:
  ```
  ./build.$(uname -m)/cuMAC/examples/multiCellMuMimoScheduler/multiCellMuMimoScheduler -c cuMAC/examples/multiCellMuMimoScheduler/config.yaml -t <slots> -l
  ```

  ***Environment variables:*** `CUMAC_SIM_SLOTS` (slots for `-t`, default 1000),
  `CUMAC_TEST_TIMEOUT` (per-combination timeout in s, default 3600),
  `CUBB_BUILD_TIMEOUT` (cuBB build timeout in s when `build.<arch>` is missing, default 7200).
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
7. MU-MIMO UE Group L2 Integration tests

   Validates the cuMAC 64T64R MU-MIMO scheduler API with L1/L2 memory sharing support.
   Runs 3 binaries (`l1_muUeGrp_test`, `cumac_muUeGrp_test`, `l2_muUeGrp_test`) via `cuMAC/examples/muMimoUeGrpL2Integration/run_cumac_muuegrp_test.sh` (under `${cuBB_SDK}`) and verifies all 7
   success patterns in the combined log.


   ***Test Params:*** Generated by `run_cumac_muuegrp_test.sh` into `cuMAC/scripts/muuegrp_tv_parameters.csv`

   ***Parameters:***

   | Parameter | Values |
   |-----------|--------|
   | `ENABLE_L1_L2_MEM_SHARING` | false / true |
   | `TDD_PATTERN` | SDDDS / SSSSS |
   | `NUM_CELL` | 1 / 6 |
   | `NUM_BS_ANT_PORT` | 32 / 64 |
   | `NUM_SRS_UE_PER_CELL` | 32 / 128 / 256 |
   | `NUM_TIME_SLOTS` | 20 / 50 |
   | `NUM_SUBBAND` | 1 / 4 |
   | `NUM_PRG_SAMP_PER_SUBBAND` | 1 / 2 |
   | `NUM_SRS_UE_PER_SLOT` | 8 / 32 |
   | `MAX_NUM_UE_SCHEDULED_PER_CELL_TTI` | 16 / 64 |
   | `MAX_NUM_UE_FOR_GRP_PER_CELL` | 32 / 64 |

   ***Config file:*** `cuMAC/examples/muMimoUeGrpL2Integration/yamlConfigFiles/config.yaml`

   ***Test Command:***

   Use the automation runner under the cuBB SDK tree (set `cuBB_SDK` and optional `LOG_PATH` as needed):

   ```
   ${cuBB_SDK}/cuMAC/examples/muMimoUeGrpL2Integration/run_cumac_muuegrp_test.sh -c muUeGrp_L2_Integration -v <tv_index>
   ```

   ***Example:***
   ```
   ${cuBB_SDK}/cuMAC/examples/muMimoUeGrpL2Integration/run_cumac_muuegrp_test.sh -c muUeGrp_L2_Integration -v 0001
   ${cuBB_SDK}/cuMAC/examples/muMimoUeGrpL2Integration/run_cumac_muuegrp_test.sh -c muUeGrp_L2_Integration -v 0241
   ```

   ***Pass Criteria (all 7 patterns must appear in log):***
   ```
   L2-MAIN: test completed successfully
   L2-cuMAC RECV: test completed successfully
   cuMAC-MAIN: test completed successfully
   cuMAC-MAIN: UE pairing solution CPU verification - test completed successfully
   cuMAC-L2 RECV: test completed successfully
   L1-MAIN: test completed successfully
   L1-L2 RECV: test completed successfully
   ```

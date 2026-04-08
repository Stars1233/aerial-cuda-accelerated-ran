# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


def check_ref_mismatch(lines, cell_count=None):
    """Parse buffer-XX.txt lines for per-channel reference-check errors.

    Channels checked:
      PDSCH  - CRC Error Count, LDPC Error Count, mismatched symbols
      PUSCH  - Error CBs, Mismatched CBs, MismatchedCRC CBs/TBs
      PRACH  - "========> Test PASS/FAIL"
      PDCCH  - "PDCCH REFERENCE CHECK: PASSED/FAILED!", "====> TV ... Test PASS/FAIL"
      CSIRS  - "CSIRS REFERENCE CHECK: PASSED/FAILED!"
      PUCCH  - "PUCCH REFERENCE CHECK: PASSED/FAILED!", per-format mismatch counts
      SSB    - "SSB REFERENCE CHECK: PASSED/FAILED!"
      cuMAC  - "cuMAC REFERENCE CHECK at slot N: PASSED/FAILED"

    Returns True if all checks pass, False otherwise.
    """
    label = f"[ref_check {cell_count} cells]" if cell_count is not None else "[ref_check]"

    # Per-channel state: {channel: {"seen": bool, "pass": bool, "errors": [str]}}
    channels = {}

    def _ch(name):
        if name not in channels:
            channels[name] = {"seen": False, "pass": True, "errors": []}
        return channels[name]

    for line in lines:
        # -- PDSCH: CRC errors --
        m = re.search(r"CRC Error Count:\s*(\d+)", line)
        if m:
            ch = _ch("PDSCH")
            ch["seen"] = True
            count = int(m.group(1))
            if count > 0:
                ch["pass"] = False
                ch["errors"].append(f"CRC Error Count: {count}")

        # -- PDSCH: LDPC errors --
        m = re.search(r"LDPC Error Count:\s*(\d+)", line)
        if m:
            ch = _ch("PDSCH")
            ch["seen"] = True
            count = int(m.group(1))
            if count > 0:
                ch["pass"] = False
                ch["errors"].append(f"LDPC Error Count: {count}")

        # -- PDSCH: mismatched symbols --
        if "PDSCH" in line or "PDSCH_TX" in line:
            m = re.search(r"Found\s+(\d+)\s+mismatched symbols", line)
            if m:
                ch = _ch("PDSCH")
                ch["seen"] = True
                count = int(m.group(1))
                if count > 0:
                    ch["pass"] = False
                    ch["errors"].append(f"mismatched symbols: {count}")

        # -- PUSCH: CB / TB errors --
        if "PUSCH_RX" in line or "PUSCH" in line:
            m = re.search(r"Error CBs\s+(\d+).*Mismatched CBs\s+(\d+).*MismatchedCRC CBs\s+(\d+)", line)
            if m:
                ch = _ch("PUSCH")
                ch["seen"] = True
                err_cbs = int(m.group(1))
                mis_cbs = int(m.group(2))
                mis_crc_cbs = int(m.group(3))
                if err_cbs > 0:
                    ch["pass"] = False
                    ch["errors"].append(f"Error CBs: {err_cbs}")
                if mis_cbs > 0:
                    ch["pass"] = False
                    ch["errors"].append(f"Mismatched CBs: {mis_cbs}")
                if mis_crc_cbs > 0:
                    ch["pass"] = False
                    ch["errors"].append(f"MismatchedCRC CBs: {mis_crc_cbs}")

            m = re.search(r"MismatchedCRC TBs\s+(\d+)", line)
            if m:
                ch = _ch("PUSCH")
                ch["seen"] = True
                count = int(m.group(1))
                if count > 0:
                    ch["pass"] = False
                    ch["errors"].append(f"MismatchedCRC TBs: {count}")

        # -- PRACH: Test PASS / FAIL --
        if "PRACH_RX" in line or "PRACH" in line:
            if "========> Test PASS" in line:
                _ch("PRACH")["seen"] = True
            if "========> Test FAIL" in line:
                ch = _ch("PRACH")
                ch["seen"] = True
                ch["pass"] = False
                ch["errors"].append("Test FAIL")

        # -- PDCCH: REFERENCE CHECK + TV Test --
        if "PDCCH REFERENCE CHECK: PASSED" in line:
            _ch("PDCCH")["seen"] = True
        if "PDCCH REFERENCE CHECK: FAILED" in line:
            ch = _ch("PDCCH")
            ch["seen"] = True
            ch["pass"] = False
            ch["errors"].append("REFERENCE CHECK FAILED")
        if "PDCCH_TX" in line and "Test PASS" in line:
            _ch("PDCCH")["seen"] = True
        if "PDCCH_TX" in line and "Test FAIL" in line:
            ch = _ch("PDCCH")
            ch["seen"] = True
            ch["pass"] = False
            ch["errors"].append("TV Test FAIL")

        # -- CSIRS: REFERENCE CHECK --
        if "CSIRS REFERENCE CHECK: PASSED" in line:
            _ch("CSIRS")["seen"] = True
        if "CSIRS REFERENCE CHECK: FAILED" in line:
            ch = _ch("CSIRS")
            ch["seen"] = True
            ch["pass"] = False
            ch["errors"].append("REFERENCE CHECK FAILED")

        # -- PUCCH: REFERENCE CHECK + format mismatches --
        if "PUCCH REFERENCE CHECK: PASSED" in line:
            _ch("PUCCH")["seen"] = True
        if "PUCCH REFERENCE CHECK: FAILED" in line:
            ch = _ch("PUCCH")
            ch["seen"] = True
            ch["pass"] = False
            ch["errors"].append("REFERENCE CHECK FAILED")
        m = re.search(r"PUCCH format\s+(\d+):\s+found\s+(\d+)\s+mismatches\s+out of\s+(\d+)", line)
        if m:
            ch = _ch("PUCCH")
            ch["seen"] = True
            fmt, mis, total = m.group(1), int(m.group(2)), int(m.group(3))
            if mis > 0:
                ch["pass"] = False
                ch["errors"].append(f"format {fmt}: {mis}/{total} mismatches")

        # -- SSB: REFERENCE CHECK --
        if "SSB REFERENCE CHECK: PASSED" in line:
            _ch("SSB")["seen"] = True
        if "SSB REFERENCE CHECK: FAILED" in line:
            ch = _ch("SSB")
            ch["seen"] = True
            ch["pass"] = False
            ch["errors"].append("REFERENCE CHECK FAILED")

        # -- cuMAC: REFERENCE CHECK --
        if "cuMAC REFERENCE CHECK" in line:
            if "PASSED" in line:
                _ch("cuMAC")["seen"] = True
            if "FAILED" in line:
                ch = _ch("cuMAC")
                ch["seen"] = True
                ch["pass"] = False
                m_slot = re.search(r"at slot\s+(\d+)", line)
                slot_str = f" at slot {m_slot.group(1)}" if m_slot else ""
                ch["errors"].append(f"REFERENCE CHECK FAILED{slot_str}")

    # Build summary
    all_pass = True
    seen_channels = [name for name, st in channels.items() if st["seen"]]
    failed_channels = [name for name, st in channels.items() if st["seen"] and not st["pass"]]

    if failed_channels:
        all_pass = False

    channel_order = ["PDSCH", "PUSCH", "PRACH", "PDCCH", "CSIRS", "PUCCH", "SSB", "cuMAC"]
    status_parts = []
    for name in channel_order:
        if name in channels and channels[name]["seen"]:
            status_parts.append(f"{name}: {'PASS' if channels[name]['pass'] else 'FAIL'}")

    summary = ", ".join(status_parts) if status_parts else "no channels detected"

    if not seen_channels:
        print(f"{label} FAIL (no channels detected -- check buffer output)")
        return False, {"ref_check_pass": False, "ref_check_channels": {}}

    if all_pass:
        print(f"{label} PASS ({summary})")
    else:
        print(f"{label} FAIL ({summary})")
        for name in channel_order:
            if name in channels and channels[name]["seen"] and not channels[name]["pass"]:
                for e in channels[name]["errors"]:
                    print(f"  - {name}: {e}")

    # Build JSON-serializable per-channel results
    ref_check_results = {}
    for name in channel_order:
        if name in channels and channels[name]["seen"]:
            ref_check_results[name] = {
                "pass": channels[name]["pass"],
                "errors": channels[name]["errors"],
            }

    return all_pass, {"ref_check_pass": all_pass, "ref_check_channels": ref_check_results}

#!/bin/bash

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

# Unit tests for execute_retry_or_die (and related) in includes.sh
# Run: ./unit_tests.sh   or   bash unit_tests.sh

set -e
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGFILE=/dev/null
DRYRUN=0
VERBOSE=0
SHOW_TIME=0
source "$_SCRIPT_DIR/includes.sh"
LOG_NEEDS_SUDO=0

PASS=0
FAIL=0

assert_eq() {
    local expected="$1"
    local actual="$2"
    local name="${3:-}"
    if [[ "$expected" == "$actual" ]]; then
        echo "  PASS: $name"
        PASS=$((PASS + 1))
        return 0
    else
        echo "  FAIL: $name (expected '$expected', got '$actual')"
        FAIL=$((FAIL + 1))
        return 1
    fi
}

# Stub get_system_status so "fail after all retries" test doesn't dump logs
get_system_status() { :; }

echo "--- execute_retry_or_die: succeed on first try ---"
execute_retry_or_die 3 'exit 0'
assert_eq 0 $? "succeed on first try"

echo "--- execute_retry_or_die: fail twice then succeed (attempts in file) ---"
f=$(mktemp); echo 0 > "$f"
# n=0: fail (exit 1); n=1: fail (exit 1); n=2: succeed (exit 0)
execute_retry_or_die 3 'n=$(cat "$f"); echo $((n+1)) > "$f"; exit $((n < 2))'
assert_eq 0 $? "fail twice then succeed"
assert_eq "3" "$(cat "$f")" "exactly 3 attempts run"
rm -f "$f"

echo "--- execute_retry_or_die: fail all 3 times then exit ---"
exc=0
out=$(bash -c "source '$_SCRIPT_DIR/includes.sh' 2>/dev/null; LOGFILE=/dev/null; LOG_NEEDS_SUDO=0; get_system_status() { :; }; execute_retry_or_die 3 'exit 1'" 2>&1) || exc=$?
assert_eq 1 $exc "exit code 1 after all retries"
echo "$out" | grep -q "attempt 1/3" && assert_eq 0 0 "log shows attempt 1/3" || ((FAIL++))
echo "$out" | grep -q "attempt 3/3" && assert_eq 0 0 "log shows attempt 3/3" || ((FAIL++))
echo "$out" | grep -q "failed after 3 attempts" && assert_eq 0 0 "log shows failed after 3 attempts" || ((FAIL++))

echo "--- execute_retry_or_die: dry-run does not run command ---"
DRYRUN=1
out=$(execute_retry_or_die 3 'exit 1'; echo "OK")
DRYRUN=0
# Dry-run must not run the command (no "Executing (attempt" line) and must log DRY-RUN
if echo "$out" | grep -q "Executing (attempt"; then
    FAIL=$((FAIL + 1)); echo "  FAIL: dry-run should not run command"
else
    assert_eq 0 0 "dry-run did not run command"
fi
echo "$out" | grep -q "DRY-RUN" && assert_eq 0 0 "dry-run logs DRY-RUN" || { FAIL=$((FAIL+1)); echo "  FAIL: dry-run should log DRY-RUN"; }
echo "$out" | grep -q "OK" && assert_eq 0 0 "dry-run returned and printed OK" || { FAIL=$((FAIL+1)); echo "  FAIL: expected OK in output"; }

echo ""
echo "Result: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]]

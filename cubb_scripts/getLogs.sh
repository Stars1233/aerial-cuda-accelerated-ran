#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

getLogsFile=getLogs.log
exec > >(tee ${getLogsFile}) 2>&1

#Figure out container name
if [[ ! -v AERIAL_CUBB_CONTAINER ]]; then
   aerialContainer=$(docker ps | grep -i cubb | head -1 | awk '{ print $NF }')
   echo "Getting log information from container: $aerialContainer"
   echo "If this is not what you expect, set the AERIAL_CUBB_CONTAINER environment variable"
else
   aerialContainer=$(docker ps | grep -w $AERIAL_CUBB_CONTAINER | head -1 | awk '{ print $NF }')
   if [ ! -z $aerialContainer ] && [ $aerialContainer == $AERIAL_CUBB_CONTAINER ]; then
      echo "Getting log information from container: $aerialContainer as specified in AERIAL_CUBB_CONTAINER"
   else
      echo "User-specified container $AERIAL_CUBB_CONTAINER is not running, exiting"
      exit 1
   fi
fi

if [[ ! -v L2_CONAINER ]]; then
   l2Container=$(docker ps | grep -i c_oai_aerial | head -1 | awk '{ print $NF }')
   echo "Getting log information from container: $l2Container"
   echo "If this is not what you expect, set the L2_CONAINER environment variable"
else
   l2Container=$(docker ps | grep -w $L2_CONAINER | head -1 | awk '{ print $NF }')
   if [ ! -z $l2Container ] && [ $l2Container == $L2_CONAINER ]; then
      echo "Getting log information from container: $l2Container as specified in L2_CONAINER"
   else
      echo "User-specified container $L2_CONAINER is not running, exiting"
      exit 1
   fi
fi

l2Logfile=/var/log/aerial/oai.log
destFolder=$(pwd)

dateRunStart=$(docker exec -it $aerialContainer bash -c 'stat -c '%.19w' /tmp/phy.log')
dateRunStart=$(date --date "$dateRunStart 5 minute ago" +"%Y-%m-%d %H:%M:%S")

dateRunEnd=$(docker exec -it $aerialContainer bash -c 'stat -c '%.19y' /tmp/phy.log')
dateRunEnd=$(date --date "$dateRunEnd 5 minute" +"%Y-%m-%d %H:%M:%S")

runDate=$(docker exec -it $aerialContainer bash -c 'date -ur /tmp/phy.log +"%Y-%m-%dT%H%M%SZ"')
cleanedDate=${runDate//[$'\t\r\n']}

logFolder=${destFolder}/$cleanedDate
echo "Creating $logFolder based the end of the run"
mkdir -p $logFolder && cd $_

echo "Gathering syslogs for period during run, from $dateRunStart to $dateRunEnd."
journalctl --since "$dateRunStart" --until "$dateRunEnd" | gzip > syslog.txt.gz

echo "Gathering system information"
systemCheck=systemChecks.txt
echo "Inside container: " > $systemCheck
cubbSdkPath=$(docker exec $aerialContainer printenv cuBB_SDK)
if [ -n $cubbSdkPath ]; then
   cubbSdkPath=/opt/nvidia/cuBB
   echo "cuBB_SDK unset in container, using default $cubbSdkPath"
fi
docker exec -it $aerialContainer bash -c "apt update > /dev/null && yes | apt install dmidecode"
docker exec -it $aerialContainer bash -c "yes | pip3 install psutil -q -q"
docker exec -it $aerialContainer bash -c "python3 $cubbSdkPath/cuPHY/util/cuBB_system_checks/cuBB_system_checks.py  -bcdegilmnps --ptp --sys"  | sed 's/\r//' >> $systemCheck
docker cp $aerialContainer:$cubbSdkPath/cuPHY/util/cuBB_system_checks/cuBB_system_checks.py .

sudo apt update > /dev/null && yes | sudo apt install dmidecode
sudo pip3 install psutil dmidecode -q
printf "\n\n\nOutside container:\n" >> $systemCheck
#python3 cuBB_system_checks.py >> $systemCheck
sudo python3 cuBB_system_checks.py  -bcdegilmnps --ptp --sys  >> $systemCheck
printf "\n\n/proc/cmdline:\n" >> $systemCheck
cat /proc/cmdline >> $systemCheck
printf "\n\nlscpu:\n" >> $systemCheck
lscpu >> $systemCheck

rt_runtime=/proc/sys/kernel/sched_rt_runtime_us
printf "\n$rt_runtime: " >> $systemCheck
cat $rt_runtime >> $systemCheck

printf "\n\nOutput of docker inspect $aerialContainer" >> $systemCheck
docker inspect $aerialContainer >> $systemCheck

rm -f serverInfo.txt
for file in /sys/class/dmi/id/*
do
   if [ -r $file ] && [ ! -d $file ]; then
     name=$(basename $file)
     echo -n "$name: " >> serverInfo.txt
     cat $file >> serverInfo.txt
   fi
done

echo "Copying phy.log from L1"
docker cp $aerialContainer:/tmp/phy.log .

#Get config file name from phy.log
cuphyConfig=$(grep "Config file:" phy.log | awk '{print $NF}')

echo "Copying cuphycontroller config file $cuphyConfig"
docker cp $aerialContainer:$cuphyConfig .

l2AdapterConfig=$(grep l2adapter_filename ${cuphyConfig##*/}  | awk '{print $NF}')

echo "Copying l2 adapter file conf $l2AdapterConfig"
docker cp $aerialContainer:`dirname $cuphyConfig`/$l2AdapterConfig .

echo "Copying nvipc.pcap"

docker exec -it $aerialContainer /bin/bash -c "[[ -d build.$(uname -m) ]] && cd  build.$(uname -m) || cd build; sudo -E ./cuPHY-CP/gt_common_libs/nvIPC/tests/pcap/pcap_collect; mv nvipc.pcap ../" > pcap_collect.log
docker cp $aerialContainer:$cubbSdkPath/nvipc.pcap .
echo "gzipping nvipc.pcap"
gzip -f nvipc.pcap

docker exec -it $aerialContainer /bin/bash -c "git --version && git -C $cuBB_SDK status && git -C $cuBB_SDK diff" > $gitLog
docker exec -it $aerialContainer /bin/bash -c "find ./ -name CMakeCache.txt -exec echo {} \; -exec cat {}  \; " > CMakeCaches.txt

if [ -e $l2Logfile ]; then
   echo "Copying oai.log from L2"
   cp $l2Logfile .
   sed -i 's/\x1B\[[0-9;]\{1,\}[A-Za-z]//g' oai.log

   printf "\n\nOutput of docker inspect $l2Container" >> $systemCheck
   docker inspect $l2Container >> $systemCheck

   l2Config=$(grep 'get parameters from libconfig' oai.log | awk '{ print $9}')
   echo "Copying L2 configuration file $l2Config"
   l2WorkDir=$(docker exec $l2Container bash -c "pwd")
   docker cp $l2Container:$l2WorkDir/cmake_targets/ran_build/build/$l2Config . || l2failure=true
   if [ $l2failure ]
   then
      echo "Unable to save l2 configuration file"
   fi
else
   echo "$l2Logfile doesn't exist, can't save"
fi

if [ "$(sysctl -n kernel.dmesg_restrict)" -eq "0" ]
then
   dmesg --time-format iso > dmesg.txt
else
   echo "dmesg restricted" > dmesg.txt
fi

systemctl status nvidia-persistenced.service >> systemctl-status.txt

cp ${destFolder}/$getLogsFile .
tar -cvzf ${logFolder}.tgz ../$cleanedDate
echo "Logs saved in ${logFolder}.tgz"

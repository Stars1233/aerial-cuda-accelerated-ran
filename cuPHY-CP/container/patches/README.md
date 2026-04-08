
# This is the setup

```
$ ls -lart
total 28
drwxr-xr-x 77 XXX dip       12288 Sep 25 13:27 ..
drwxrwxr-x  5 XXX XXX  4096 Oct  2 18:51 .
drwxrwxr-x 18 XXX XXX  4096 Oct  2 18:59 aerial_sdk
drwxrwxr-x 14 XXX XXX  4096 Oct  2 21:37 internals
drwxrwxr-x 19 XXX XXX  4096 Oct  3 08:43 externals
```

These has been checked out from
ssh://git@gitlab-master.nvidia.com:12051/gputelecom-external
```
$ ls -lart internals/
total 68
drwxrwxr-x  5 XXX XXX  4096 Oct  2 18:51 ..
drwxrwxr-x 12 XXX XXX  4096 Oct  2 18:52 benchmark
drwxrwxr-x  9 XXX XXX  4096 Oct  2 18:52 libyang-cpp
drwxrwxr-x 12 XXX XXX  4096 Oct  2 18:52 libyang
drwxrwxr-x  6 XXX XXX  4096 Oct  2 18:53 modeling
drwxrwxr-x 16 XXX XXX  4096 Oct  2 18:53 cli11
drwxrwxr-x  9 XXX XXX  4096 Oct  2 18:53 gsl-lite
drwxrwxr-x 19 XXX XXX 16384 Oct  2 18:53 cmake-modules
drwxrwxr-x  9 XXX XXX  4096 Oct  2 18:54 yaml-cpp
drwxrwxr-x  6 XXX XXX  4096 Oct  2 18:54 wise_enum
drwxrwxr-x  6 XXX XXX  4096 Oct  2 18:54 fmtlog
drwxrwxr-x 14 XXX XXX  4096 Oct  2 21:37 .
drwxrwxr-x  6 XXX XXX  4096 Oct  2 22:18 backward-cpp
drwxrwxr-x 12 XXX XXX  4096 Oct  2 22:20 mimalloc
```

These has been checked out from github
```
$ ls -lart externals/
total 88
drwxrwxr-x  9 XXX XXX 4096 Oct  2 18:44 libyang-cpp
drwxrwxr-x 12 XXX XXX 4096 Oct  2 18:44 libyang
drwxrwxr-x 16 XXX XXX 4096 Oct  2 18:44 CLI11
drwxrwxr-x  5 XXX XXX 4096 Oct  2 18:44 cmake_modules
drwxrwxr-x  9 XXX XXX 4096 Oct  2 18:45 yaml-cpp
drwxrwxr-x  6 XXX XXX 4096 Oct  2 18:45 wise_enum
drwxrwxr-x  5 XXX XXX 4096 Oct  2 18:45 fmtlog
drwxrwxr-x 17 XXX XXX 4096 Oct  2 18:46 eigen
drwxrwxr-x 10 XXX XXX 4096 Oct  2 18:46 clickhouse-cpp
drwxrwxr-x  5 XXX XXX 4096 Oct  2 18:51 ..
drwxrwxr-x 20 XXX XXX 8192 Oct  2 19:01 grpc
drwxrwxr-x 12 XXX XXX 4096 Oct  2 19:03 benchmark
drwxrwxr-x  8 XXX XXX 4096 Oct  2 19:05 googletest
drwxrwxr-x 12 XXX XXX 4096 Oct  2 19:07 libyaml
drwxrwxr-x 12 XXX XXX 4096 Oct  2 20:17 prometheus-cpp
drwxrwxr-x  9 XXX XXX 4096 Oct  2 20:19 pybind11
-rw-rw-r--  1 XXX XXX 4662 Oct  2 21:38 README
drwxrwxr-x  6 XXX XXX 4096 Oct  2 21:39 backward-cpp
drwxrwxr-x 12 XXX XXX 4096 Oct  2 21:42 mimalloc
drwxrwxr-x 12 XXX XXX 4096 Oct  3 09:31 gsl-lite
drwxrwxr-x 19 XXX XXX 4096 Oct  3 08:43 .
```

# mimalloc

[submodule "cuPHY-CP/external/mimalloc"]
	path = cuPHY-CP/external/mimalloc
	url = ../../gputelecom-external/mimalloc.git
	shallow = true
	ignore = dirty

 ab2162fa7f2e9b38c1c2373e753cc1e6f078451b cuPHY-CP/external/mimalloc (heads/memtrace)

```
$ cd internals/mimalloc
$ git log
commit ab2162fa7f2e9b38c1c2373e753cc1e6f078451b (HEAD -> memtrace, origin/memtrace, origin/HEAD)
Author: Tim Martin
Date:   Tue Aug 22 16:48:55 2023 +0000

    Remove extraneous url

...

commit f2712f4a8f038a7fb4df2790f4c3b7e3ed9e219b (origin/master)
Merge: f819dbb 8d6a9df
Author: Daan Leijen <daan@microsoft.com>
Date:   Thu Apr 14 16:54:04 2022 -0700

    Merge branch 'dev' into dev-slice

$ cd externals/mimalloc
git checkout f2712f4a8f038a7fb4df2790f4c3b7e3ed9e219b

$ cd internals/mimalloc

git diff f2712f4a8f038a7fb4df2790f4c3b7e3ed9e219b > ../../aerial_sdk/cuPHY-CP/container/patches/mimalloc.patch
```



# Googletest

[submodule "cuPHY/external/googletest"]
	path = cuPHY/external/googletest
	url = https://github.com/google/googletest.git
	shallow = true
	ignore = dirty

 b514bdc898e2951020cbdca1304b75f5950d1f59 cuPHY/external/googletest (release-1.8.0-3420-gb514bdc8)

Updated to:
    #branch='v1.17.0',
    commit='52eb8108c5bdec04579160ae17225d66034bd723',

# libyaml

[submodule "cuPHY/external/libyaml"]
	path = cuPHY/external/libyaml
	url = https://github.com/yaml/libyaml.git
	shallow = true
	ignore = dirty

 2c891fc7a770e8ba2fec34fc6b545c672beb37e6 cuPHY/external/libyaml (0.2.5)

Updated to:
    #branch='0.2.5',
    commit='2c891fc7a770e8ba2fec34fc6b545c672beb37e6',


# prometheus

[submodule "cuPHY-CP/external/prometheus-cpp"]
	path = cuPHY-CP/external/prometheus
	url = https://github.com/jupp0r/prometheus-cpp.git
	shallow = true
	ignore = dirty

 342de5e93bd0cbafde77ec801f9dd35a03bceb3f cuPHY-CP/external/prometheus (v0.13.0)

Updated to:
    #branch='v1.3.0',
    commit='e5fada43131d251e9c4786b04263ce98b6767ba5',



# grpc

[submodule "cuPHY-CP/external/grpc"]
	path = cuPHY-CP/external/grpc
	#url = ssh://git@gitlab-master.nvidia.com:12051/gputelecom-external/grpc-flat.git
	url = ../../gputelecom-external/grpc-flat.git
	branch = v1.58.0
	shallow = true
	ignore = dirty

 2f5e276eda21138b4d0fac44053170ffce6731a1 cuPHY-CP/external/grpc (heads/v1.35.0-1-g2f5e276e)

Updated to:
    #branch='v1.75.0',
    commit='093085cc925e0d5aa6e92bc29e917f9bdc00add2',


# benchmark

[submodule "cuPHY/external/benchmark"]
	path = cuPHY/external/benchmark
	url = ../../gputelecom-external/benchmark.git
	shallow = true
	ignore = dirty

 e73915667c21faccd7019c6da8ab083b0264db13 cuPHY/external/benchmark (heads/main)

Updated to:
    #branch='v1.9.4',
    commit='eddb0241389718a23a42db6af5f0164b6e0139af',

# libyang-cpp

[submodule "cuPHY-CP/external/libyang-cpp"]
	path = cuPHY-CP/external/libyang-cpp
	url = ../../gputelecom-external/libyang-cpp.git
	branch = v1.1.0_for_aerial
	shallow = true

 06cc5c73f5ca73a5a9c1f95f55d05340f72f70d7 cuPHY-CP/external/libyang-cpp (v1.1.0_for_aerial)

Updated to:
    #branch='v4',
    commit='249da7280864fbda5fccb340b455b7000ebfe67d',


# libyang

[submodule "cuPHY-CP/external/libyang"]
	path = cuPHY-CP/external/libyang
	url = ../../gputelecom-external/libyang.git
	branch = v2.1.111
	shallow = true

 8b0b910a2dcb7360cb5b0aaefbd1338271d50946 cuPHY-CP/external/libyang (v2.1.111)

Updated to:
    #branch='v3.13.5',
    commit='efe43e3790822a3dc64d7d28db935d03fff8b81f',


# pybind11

[submodule "pyaerial/external/pybind11"]
	path = pyaerial/external/pybind11
	url = https://github.com/pybind/pybind11

 869cc1ff085dd405635b00eb46e5c84f50f26099 pyaerial/external/pybind11 (v2.11.0-76-g869cc1ff)

Updated to:
    #branch='v3.0.1',
    commit='f5fbe867d2d26e4a0a9177a51f6e568868ad3dc8',

# eigen

[submodule "cuMAC/eigen"]
	path = cuMAC/eigen
	url = https://gitlab.com/libeigen/eigen.git

 126ba1a166090dd5605995921d964c6a6e9e0a88 cuMAC/eigen (before-3.4-1012-g126ba1a16)

Updated to:
    #branch='v2.6.0',
    commit='69195246a3b39542c397ef27df9f46ec4a4bf206',

# clickhouse-cpp

[submodule "cuPHY-CP/external/clickhouse-cpp"]
	path = cuPHY-CP/external/clickhouse-cpp
	url = https://github.com/ClickHouse/clickhouse-cpp.git

 0fb483543b313a0979b4dbd130f834352a034ba8 cuPHY-CP/external/clickhouse-cpp (v2.5.1-51-g0fb4835)

Updated to:
    #branch='v2.6.0',
    commit='69195246a3b39542c397ef27df9f46ec4a4bf206',

# cli11

[submodule "cuPHY/external/cli11"]
	path = cuPHY/external/cli11
	url = ../../gputelecom-external/cli11.git

 f4f225d9a233583604fc1bf6e8d1fa0ccd60c652 cuPHY/external/cli11 (v1.7.1-512-gf4f225d)

Updated to:
    #branch='v2.5.0',
    commit='4160d259d961cd393fd8d67590a8c7d210207348',

# gsl-lite

[submodule "cuPHY/external/gsl-lite"]
	path = cuPHY/external/gsl-lite
	url = ../../gputelecom-external/gsl-lite.git

 bd9eb162d42d8ae6a7e86902ca7060247d71ac41 cuPHY/external/gsl-lite (heads/master)

Updated to:
    #branch='v1.0.1',
    commit='56dab5ce071c4ca17d3e0dbbda9a94bd5a1cbca1',

# cmake-modules

[submodule "cmake/cmake-modules"]
	path = cmake/cmake-modules
	url = ../../gputelecom-external/cmake-modules.git

 1b450496c5d11fdcad8b000843d0c516e1eaa59f cmake/cmake-modules (heads/main)

Updated to:
    #branch='0.5.2',
    commit='3f8318d8f673e619023e9a526b6ee37536be1659',

# yaml-cpp

[submodule "cuPHY/external/yaml-cpp"]
	path = cuPHY/external/yaml-cpp
	url = ../../gputelecom-external/yaml-cpp.git

 9ce5a25188d83b43dd5cd633f2975be10f5d6608 cuPHY/external/yaml-cpp (0.8.0-62-g9ce5a25)

Updated to:
    #branch='0.8.0',
    commit='f7320141120f720aecc4c32be25586e7da9eb978',

# wise_enum

[submodule "cuPHY/external/wise_enum"]
	path = cuPHY/external/wise_enum
	url = ../../gputelecom-external/wise_enum.git

 34ac79f7ea2658a148359ce82508cc9301e31dd3 cuPHY/external/wise_enum (3.1.0)

Updated to:
    #branch='3.1.0',
    commit='34ac79f7ea2658a148359ce82508cc9301e31dd3',

# fmtlog

[submodule "cuPHY/external/fmtlog"]
	path = cuPHY/external/fmtlog
	url = ../../gputelecom-external/fmtlog.git
        branch = main-internal

 2a51592e4a234c1300e24aa8bab5e323e470ad2e cuPHY/external/fmtlog (remotes/origin/main-internal)

Patched:
```
$ cd internal/fmtlog
$ git checkout main-internal
$ git log
commit 2a51592e4a234c1300e24aa8bab5e323e470ad2e (HEAD -> main-internal, origin/main-internal)
Author: Yacob (Kobi) Cohen-Arazi 
Date:   Wed Jan 1 18:00:40 2025 -0800

    Updated URL in fmt submodule

    - git submodule set-url -- fmt  ../../gputelecom-external/fmt
    - git submodule sync
    - git submodule update --init --recursive

...

commit acd521b1a64480354136a745c511358da1ec7dc5 (origin/main)
Author: Rao Meng <raomeng@localhost.localdomain>
Date:   Fri May 31 16:32:23 2024 +0800

    upgrade fmtlib to 10.2.1

$ cd external/fmtlog
$ git checkout acd521b1a64480354136a745c511358da1ec7dc5
$ git log -n1
commit acd521b1a64480354136a745c511358da1ec7dc5 (HEAD, tag: v2.2.2)
Author: Rao Meng <raomeng@localhost.localdomain>
Date:   Fri May 31 16:32:23 2024 +0800

    upgrade fmtlib to 10.2.1

$ cd internals/fmtlog
$ sed -i 's|../../gputelecom-external/fmt|https://github.com/fmtlib/fmt.git|g' .gitmodules
$ git diff acd521b1a64480354136a745c511358da1ec7dc5 > ../../aerial_sdk/cuPHY-CP/container/patches/fmtlog.patch
```


# fmt

[submodule "fmt"]
    path = fmt
    url = ../../gputelecom-external/fmt

 e69e5f977d458f2650bb346dadf2ad30c5320281 fmt (10.2.1)

Patched:
```

$ cd external/fmt
$ git checkout e69e5f977d458f2650bb346dadf2ad30c5320281


# backward-cpp
from mimalloc
https://gitlab-master.nvidia.com/gputelecom-external/backward-cpp/-/tree/65a769ffe77cf9d759d801bc792ac56af8e911a3


Updated to:
    #branch='v1.6',
    commit='3bb9240cb15459768adb3e7d963a20e1523a6294'


# ldpc_decoder_cubin

[submodule "cuPHY/src/cuphy/error_correction/ldpc_decoder_cubin"]
	path = cuPHY/src/cuphy/error_correction/ldpc_decoder_cubin
	#url = ssh://git@gitlab-master.nvidia.com:12051/gputelecom/ldpc_decoder_cubin.git
	url = ../ldpc_decoder_cubin.git
	branch = develop

 251252f7c6be6489df4ad88e387f583b3d618f58 cuPHY/src/cuphy/error_correction/ldpc_decoder_cubin (QA_VER_d1_0.0.135-10-g251252f)

# modeling

[submodule "cuPHY-CP/external/modeling"]
	path = cuPHY-CP/external/modeling
	#url = ssh://git@gitlab-master.nvidia.com:12051/gputelecom-external/modeling.git
	url = ../../gputelecom-external/modeling.git
	shallow = true
	branch = o_ran_ru_fh_only_for_aerial
	ignore = dirty

 93e9576776f131b8402ef337e97d6f9178e766ec cuPHY-CP/external/modeling (heads/master-1-g93e9576)

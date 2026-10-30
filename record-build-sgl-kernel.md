# RTX 2080Ti 上编译 sgl-kernel 完整记录

> 目标：在双卡 NVIDIA RTX 2080Ti（Compute Capability 7.5 / sm75）+ CUDA 12.8 + PyTorch 2.9.1+cu128 环境下，从源码编译 `sgl-kernel`，使其包含 sm75 cubin，从而让 sglang v0.5.7 能在该硬件上运行。

---

## 一、环境与问题

### 1.1 硬件/软件

| 项目 | 版本/型号 |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 Ti × 2 |
| Compute Capability | 7.5 (sm75 / Turing) |
| CUDA | 12.8 |
| PyTorch | 2.9.1+cu128 |
| Python | 3.10.12 |
| OS | Ubuntu 22.04.5 LTS |
| sglang | v0.5.7 (editable: `/root/tmp01/sglang/python`) |
| sgl-kernel | 0.3.20 (源码: `/root/tmp01/sglang/sgl-kernel`) |

### 1.2 原始失败原因

直接使用：

```bash
pip uninstall -y sgl_kernel
pip install -e "python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --no-build-isolation
```

会拉取 PyPI 上预编译的 `sgl-kernel==0.3.20` wheel，该 wheel **没有 sm75 cubin**，导致 2080Ti 运行时报 `no kernel image is available`。

### 1.3 源码编译又遇到的上游问题

| 问题 | 原因 | 解决 |
|---|---|---|
| `sgl-kernel/CMakeLists.txt` 原 `sm_75` 被删除 | PR #9207 移除 | 加回 `-gencode=arch=compute_75,code=sm_75` |
| DeepGEMM 提交哈希 `54f99a8...` 不存在 | `sgl-project/DeepGEMM` 被 force-push | 改为当前存在的 HEAD `348a285...` |
| Triton / DeepGEMM 等 8 个仓库网络克隆失败 | GitHub 不稳定 | 手动 clone 到本地，用 `FETCHCONTENT_SOURCE_DIR_*` 指向 |
| mscclpp 编译报 `Could NOT find NUMA` | 缺 `libnuma-dev` | `apt install libnuma-dev` |
| DeepGEMM 编译缺 `elfutils/libdwfl.h` | 缺 `libdw-dev` | `apt install libdw-dev libelf-dev` |
| DeepGEMM 编译缺 `<format>` | GCC 11 无 C++20 `<format>` | 安装 GCC 13 |
| DeepGEMM 要求 CUDA Driver/Runtime API >= 12.9 | 本机 CUDA 12.8 | **禁用 DeepGEMM**（AWQ 模型用不到 FP8/MXFP4） |
| editable 安装后 `import sgl_kernel` 找不到 `.so` | scikit-build-core 没有把 .so 同步到源码树 | 手动创建 symlink |

---

## 二、准备工作

### 2.1 安装系统依赖

```bash
# NUMA（mscclpp 需要）
apt-get update
apt-get install -y libnuma-dev

# elfutils / libdw（DeepGEMM 的 deep_jit 需要，若不禁用 DeepGEMM）
apt-get install -y libdw-dev libelf-dev

# GCC 13（DeepGEMM 的 deep_jit 需要 C++20 <format>；若不禁用 DeepGEMM）
# Ubuntu 22.04 默认没有 gcc-13，需要加 PPA
apt-get install -y software-properties-common
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get update
apt-get install -y gcc-13 g++-13
```

### 2.2 安装 Python 构建依赖

```bash
pip install --upgrade cmake scikit-build-core wheel build
```

### 2.3 手动准备 8 个第三方仓库

由于网络问题，把以下仓库全部 clone 到 `/root/tmp01/sglang/3rdparty_repos`，并初始化 submodule（DeepGEMM 需要 `deep_jit` 子模块）：

```bash
mkdir -p /root/tmp01/sglang/3rdparty_repos
cd /root/tmp01/sglang/3rdparty_repos

# 1) cutlass
git clone --no-checkout https://github.com/NVIDIA/cutlass.git repo-cutlass
cd repo-cutlass && git checkout 57e3cfb47a2d9e0d46eb6335c3dc411498efa198 && cd ..

# 2) DeepGEMM（注意：原 CMakeLists 指定的 54f99a8... 已失效）
git clone --no-checkout https://github.com/sgl-project/DeepGEMM.git repo-deepgemm
cd repo-deepgemm
git checkout 348a285dbb5a19be34aaedfe36fd5951fc3f2639
# DeepGEMM 依赖 deep_jit 子模块
git config --global --add safe.directory /root/tmp01/sglang/3rdparty_repos/repo-deepgemm
git submodule update --init --recursive
cd ..

# 3) fmt
git clone --no-checkout https://github.com/fmtlib/fmt.git repo-fmt
cd repo-fmt && git checkout 553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28 && cd ..

# 4) triton
git clone --no-checkout https://github.com/triton-lang/triton.git repo-triton
cd repo-triton && git checkout v3.5.1 && cd ..

# 5) flashinfer
git clone --no-checkout https://github.com/flashinfer-ai/flashinfer.git repo-flashinfer
cd repo-flashinfer && git checkout bc29697ba20b7e6bdb728ded98f04788e16ee021 && cd ..

# 6) sgl-attn（flash-attention）
git clone --no-checkout https://github.com/sgl-project/sgl-attn.git repo-flash-attention
cd repo-flash-attention && git checkout f20a52329482ddca4a627b2f028f88c2959ee299 && cd ..

# 7) mscclpp
git clone --no-checkout https://github.com/microsoft/mscclpp.git repo-mscclpp
cd repo-mscclpp && git checkout 51eca89d20f0cfb3764ccd764338d7b22cd486a6 && cd ..

# 8) fast-hadamard-transform
git clone --no-checkout https://github.com/sgl-project/fast-hadamard-transform.git repo-fast-hadamard-transform
cd repo-fast-hadamard-transform && git checkout 48f3c13764dc2ec662ade842a4696a90a137f1bc && cd ..
```

---

## 三、修改 sgl-kernel/CMakeLists.txt

### 3.1 加回 sm75 gencode

文件：`/root/tmp01/sglang/sgl-kernel/CMakeLists.txt`

在 `if (ENABLE_BELOW_SM90)` 块内加回一行：

```cmake
if (ENABLE_BELOW_SM90)
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_75,code=sm_75"   # ← 加回这行
        "-gencode=arch=compute_80,code=sm_80"
        "-gencode=arch=compute_89,code=sm_89"
    )
    ...
endif()
```

### 3.2 修正 DeepGEMM 仓库哈希

原内容：

```cmake
FetchContent_Declare(
    repo-deepgemm
    GIT_REPOSITORY https://github.com/sgl-project/DeepGEMM
    GIT_TAG        54f99a8af537b3c6eb4819b69907ccbe2b600792
    GIT_SHALLOW    OFF
)
```

改为：

```cmake
FetchContent_Declare(
    repo-deepgemm
    GIT_REPOSITORY https://github.com/sgl-project/DeepGEMM
    GIT_TAG        348a285dbb5a19be34aaedfe36fd5951fc3f2639
    GIT_SHALLOW    OFF
)
```

### 3.3 禁用 DeepGEMM 编译（关键）

当前 DeepGEMM / deep_jit 要求：
- CUDA Driver/Runtime API >= 12.9（本机 12.8 不满足）
- C++20 `<format>`（GCC 13 才支持）

而 RTX 2080Ti + AWQ 模型根本用不到 DeepGEMM 的 FP8/MXFP4 功能，因此直接禁用它：

```cmake
# ============================ Extra Install: DeepGEMM (JIT) ============================= #
option(SGL_KERNEL_ENABLE_DEEPGEMM "Build DeepGEMM JIT Python bindings" OFF)

if (SGL_KERNEL_ENABLE_DEEPGEMM)
    # ... 原有 DeepGEMM 编译/安装逻辑 ...
endif()
```

**注**：如果未来 CUDA 升级到 12.9+ 且需要 FP8，可把 `OFF` 改回 `ON`，并确保 `repo-deepgemm/third-party/deep_jit` 子模块已初始化、GCC >= 13、已安装 `libdw-dev`。

---

## 四、编译 sgl-kernel

### 4.1 进入目录并卸载旧版本

```bash
cd /root/tmp01/sglang/sgl-kernel
pip uninstall -y sgl-kernel
```

### 4.2 执行编译

```bash
CMAKE_ARGS="
  -DCMAKE_C_COMPILER=/usr/bin/x86_64-linux-gnu-gcc-13
  -DCMAKE_CXX_COMPILER=/usr/bin/x86_64-linux-gnu-g++-13
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11
  -DTorch_DIR=/usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch
  -DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS=/root/tmp01/sglang/3rdparty_repos/repo-cutlass
  -DFETCHCONTENT_SOURCE_DIR_REPO-DEEPGEMM=/root/tmp01/sglang/3rdparty_repos/repo-deepgemm
  -DFETCHCONTENT_SOURCE_DIR_REPO-FMT=/root/tmp01/sglang/3rdparty_repos/repo-fmt
  -DFETCHCONTENT_SOURCE_DIR_REPO-TRITON=/root/tmp01/sglang/3rdparty_repos/repo-triton
  -DFETCHCONTENT_SOURCE_DIR_REPO-FLASHINFER=/root/tmp01/sglang/3rdparty_repos/repo-flashinfer
  -DFETCHCONTENT_SOURCE_DIR_REPO-FLASH-ATTENTION=/root/tmp01/sglang/3rdparty_repos/repo-flash-attention
  -DFETCHCONTENT_SOURCE_DIR_REPO-MSCCLPP=/root/tmp01/sglang/3rdparty_repos/repo-mscclpp
  -DFETCHCONTENT_SOURCE_DIR_REPO-FAST-HADAMARD-TRANSFORM=/root/tmp01/sglang/3rdparty_repos/repo-fast-hadamard-transform
" \
CUDA_HOME=/usr/local/cuda \
CMAKE_POLICY_VERSION_MINIMUM=3.5 \
CMAKE_BUILD_PARALLEL_LEVEL=32 \
MAX_JOBS=32 \
pip install -e . --no-build-isolation --config-settings=build-dir=build
```

说明：
- `--config-settings=build-dir=build`：使用持久化 build 目录，后续增量编译更快。
- `CMAKE_BUILD_PARALLEL_LEVEL=32 MAX_JOBS=32`：并行任务数，按机器资源调整。
- `CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11`：nvcc 仍用 GCC 11，避免 CUDA 代码因 GCC 13 变化而全量重编。
- 若不需要 DeepGEMM，可去掉 GCC 13 相关设置，直接用系统默认 GCC 11。

### 4.3 修复 editable 安装的 .so 路径

scikit-build-core 的 editable 安装不会自动把编译好的 `.so` 同步到源码树 `python/sgl_kernel/`，而 `sgl_kernel/load_utils.py` 又通过 `Path(__file__).parent` 去该目录找 `common_ops`。

手动创建 symlink：

```bash
cd /root/tmp01/sglang/sgl-kernel/python/sgl_kernel
mkdir -p sm90 sm100
ln -sf /root/tmp01/sglang/sgl-kernel/build/flash_ops.abi3.so       ./flash_ops.abi3.so
ln -sf /root/tmp01/sglang/sgl-kernel/build/flashmla_ops.abi3.so    ./flashmla_ops.abi3.so
ln -sf /root/tmp01/sglang/sgl-kernel/build/spatial_ops.abi3.so     ./spatial_ops.abi3.so
ln -sf /root/tmp01/sglang/sgl-kernel/build/sm90/common_ops.abi3.so ./sm90/common_ops.abi3.so
ln -sf /root/tmp01/sglang/sgl-kernel/build/sm100/common_ops.abi3.so ./sm100/common_ops.abi3.so
```

---

## 五、验证

### 5.1 确认 sm75 cubin 已打入

```bash
SO90=/root/tmp01/sglang/sgl-kernel/python/sgl_kernel/sm90/common_ops.abi3.so
SO100=/root/tmp01/sglang/sgl-kernel/python/sgl_kernel/sm100/common_ops.abi3.so

cuobjdump --list-elf $SO90  | grep sm_75
cuobjdump --list-elf $SO100 | grep sm_75
```

预期输出包含 `sm_75` 的 ELF。

### 5.2 import 与基础 kernel 测试

```bash
python3 - <<'PY'
import torch, sgl_kernel
print("loaded:", sgl_kernel.common_ops.__file__)

# rmsnorm 冒烟（fp16，sm75 路径）
x = torch.randn(4, 4096, dtype=torch.float16, device="cuda")
w = torch.ones(4096, dtype=torch.float16, device="cuda")
out = sgl_kernel.rmsnorm(x, w, 1e-6)
print("rmsnorm OK", out.shape, out.dtype)
PY
```

成功输出类似：

```text
loaded: /root/tmp01/sglang/sgl-kernel/python/sgl_kernel/sm100/common_ops.abi3.so
rmsnorm OK torch.Size([4, 4096]) torch.float16
```

### 5.3 确认 sglang 仍可用

```bash
python3 -c "import sglang; print('sglang version:', sglang.__version__)"
```

---

## 六、启动 sglang server（2080Ti 注意）

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
    --model-path /root/tmp01/data/models/Qwen2.5-14B-Instruct-AWQ \
    --tp 2 \
    --dtype float16 \
    --host 127.0.0.1 \
    --port 30000 \
    --context-length 8192 \
    --mem-fraction-static 0.80 \
    --attention-backend triton
```

关键参数：
- `--attention-backend triton`：flashinfer 的 JIT kernel 在 Turing 上不完全可用，必须显式指定 triton。
- `--dtype float16`：2080Ti 无 BF16 硬件支持，sglang 会自动强制 fp16，也可手动指定。
- 若 flashinfer sampling 在 sm75 上报错，可追加 `--sampling-backend pytorch`。

---

## 七、已知限制

1. **FP8 / BF16 / Marlin / FA3 / FlashMLA 相关功能在 2080Ti 上不可用**：这是硬件限制，与本次编译无关。
2. **DeepGEMM 被禁用**：因此 FP8/MXFP4 量化模型的相关加速不可用。本记录针对 AWQ（INT4）模型，不受影响。
3. **部分 sampling kernel 可能在 sm75 上 launch 失败**：如 `top_k_top_p_sampling_from_logits` 可能报 `too many resources requested for launch`，此时用 `--sampling-backend pytorch` 回退。
4. **GCC 13 来自 PPA**：如果环境重置，需要重新安装。

---

## 八、文件变更清单

| 文件 | 变更 |
|---|---|
| `/root/tmp01/sglang/sgl-kernel/CMakeLists.txt` | 加回 sm75 gencode；修正 DeepGEMM 哈希；新增 `SGL_KERNEL_ENABLE_DEEPGEMM` 选项并默认 OFF |
| `/root/tmp01/sglang/sgl-kernel/python/sgl_kernel/*.abi3.so` | symlink 到 `build/` 下的编译产物 |
| `/root/tmp01/sglang/sgl-kernel/python/sgl_kernel/sm90/*.abi3.so` | symlink 到 `build/sm90/` |
| `/root/tmp01/sglang/sgl-kernel/python/sgl_kernel/sm100/*.abi3.so` | symlink 到 `build/sm100/` |
| `/root/tmp01/sglang/need_git_clone_repos.txt` | 第三方仓库手动 clone 清单 |
| `/root/tmp01/sglang/record-build-sgl-kernel.md` | 本记录文件 |

---

## 九、可选：重新触发增量编译

后续若修改了 `sgl-kernel` 源码，只需执行：

```bash
cd /root/tmp01/sglang/sgl-kernel
CMAKE_ARGS="
  -DCMAKE_C_COMPILER=/usr/bin/x86_64-linux-gnu-gcc-13
  -DCMAKE_CXX_COMPILER=/usr/bin/x86_64-linux-gnu-g++-13
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11
  -DTorch_DIR=/usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch
  -DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS=/root/tmp01/sglang/3rdparty_repos/repo-cutlass
  -DFETCHCONTENT_SOURCE_DIR_REPO-DEEPGEMM=/root/tmp01/sglang/3rdparty_repos/repo-deepgemm
  -DFETCHCONTENT_SOURCE_DIR_REPO-FMT=/root/tmp01/sglang/3rdparty_repos/repo-fmt
  -DFETCHCONTENT_SOURCE_DIR_REPO-TRITON=/root/tmp01/sglang/3rdparty_repos/repo-triton
  -DFETCHCONTENT_SOURCE_DIR_REPO-FLASHINFER=/root/tmp01/sglang/3rdparty_repos/repo-flashinfer
  -DFETCHCONTENT_SOURCE_DIR_REPO-FLASH-ATTENTION=/root/tmp01/sglang/3rdparty_repos/repo-flash-attention
  -DFETCHCONTENT_SOURCE_DIR_REPO-MSCCLPP=/root/tmp01/sglang/3rdparty_repos/repo-mscclpp
  -DFETCHCONTENT_SOURCE_DIR_REPO-FAST-HADAMARD-TRANSFORM=/root/tmp01/sglang/3rdparty_repos/repo-fast-hadamard-transform
" \
CUDA_HOME=/usr/local/cuda \
CMAKE_POLICY_VERSION_MINIMUM=3.5 \
CMAKE_BUILD_PARALLEL_LEVEL=32 \
MAX_JOBS=32 \
pip install -e . --no-build-isolation --config-settings=build-dir=build
```

ninja 会自动只编译改动过的文件。

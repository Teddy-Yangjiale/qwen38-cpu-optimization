# Qwen3.8-Flash-Next CPU 推理优化

本仓库保存校内超算比赛的 llama.cpp 优化源码，供队友查看、构建和复核。当前开发分支为 `qwen38-mmid-optimization`，已包含 grouped MMID GEMM 和 packing 任务分配优化。

## 获取代码

```bash
git clone --branch qwen38-mmid-optimization https://github.com/Teddy-Yangjiale/qwen38-cpu-optimization.git
cd qwen38-cpu-optimization
```

仓库包含完整 llama.cpp 源码；上游项目介绍保存在 [README.upstream.md](README.upstream.md)，许可证见 [LICENSE](LICENSE)。

## 我们修改了哪里

| 内容 | 文件 | 作用 |
| --- | --- | --- |
| 当前优化实现 | [ggml/src/ggml-cpu/repack.cpp](ggml/src/ggml-cpu/repack.cpp) | 在 CPU repack 的 MUL_MAT_ID 路径中按专家聚合输入，用已有 GEMM 处理四行分组，用 GEMV 处理尾部；按线程步长分配 packing 任务 |
| 第一阶段补丁 | [0001-repack-mmid-grouped-gemm.patch](optimization/patches/0001-repack-mmid-grouped-gemm.patch) | 在上游基线上引入 grouped MMID GEMM |
| 第二阶段补丁 | [0002-mmid-pack-stride.patch](optimization/patches/0002-mmid-pack-stride.patch) | 在第一阶段上减少各线程遍历其他线程任务的开销 |
| 合成算子检查 | [test_mmid_repack.cpp](optimization/test_mmid_repack.cpp) | 检查 Q4_K MUL_MAT_ID 输出，覆盖不同 token 数、路由和输入行数 |
| 数值比较 | [compare_dumps.py](optimization/compare_dumps.py) | 比较 float32 输出；拒绝空文件、长度错误和 NaN/Inf |
| 调度差分检查 | [check_mmid_pack_schedule.py](optimization/check_mmid_pack_schedule.py) | 比较两个 packing 循环的任务归属、顺序和覆盖范围 |

上游基线提交为 `d81235049384534c167caea52b85a694f6103d14`；优化实现提交为 `d3f633dfec8b4bced5b78a8fe4b3fd50b5593126`。两份补丁依次应用可重建该提交的 `repack.cpp`。当前分支已经应用这两份补丁，直接构建即可。

## 构建和验证

在目标 CPU 环境中按 [上游构建说明](docs/build.md) 构建。CPU 构建示例：

```bash
cmake -B build -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build --config Release -j 4
```

合成算子工具须链接同一份构建生成的 ggml 库。以下示例用于提供 CPU_REPACK 后端的 Linux CPU 环境，运行时应确认输出包含 `CPU_REPACK`，且后端支持相应 Q4_K 内核：

```bash
c++ -O2 -std=c++17 optimization/test_mmid_repack.cpp \
  -Iggml/include -Lbuild/bin -Wl,-rpath,"$PWD/build/bin" \
  -lggml -lggml-base -lggml-cpu -o build/test_mmid_repack
GGML_REPACK_MMID_GEMV=1 build/test_mmid_repack 64 /tmp/mmid-gemv.bin 8 1 16 0 1 hotspot
env -u GGML_REPACK_MMID_GEMV build/test_mmid_repack 64 /tmp/mmid-grouped.bin 8 1 16 0 1 hotspot
python3 optimization/compare_dumps.py /tmp/mmid-gemv.bin /tmp/mmid-grouped.bin
```

参数顺序为：tokens、输出文件、线程数、是否使用 repack、总专家数、计时重复次数、输入行数、路由（`hotspot` 或 `balanced`）。默认比较容差为 `atol=1e-3, rtol=1e-3`，通过仅表示这个合成输入符合该容差。需要另外覆盖尾部、512 专家、38 线程、两种路由及输入行数 1/10，并完成模型级正确性检查。

仅验证第二阶段 packing 任务调度时，可在本机执行以下命令。需要 Python 3、Git 和支持 AddressSanitizer/UndefinedBehaviorSanitizer 的 clang++：

```bash
repo_dir="$PWD"
check_dir="$(mktemp -d)"
git -C "$check_dir" init -q
mkdir -p "$check_dir/ggml/src/ggml-cpu"
git show d81235049384534c167caea52b85a694f6103d14:ggml/src/ggml-cpu/repack.cpp > "$check_dir/ggml/src/ggml-cpu/repack.cpp"
git -C "$check_dir" apply "$repo_dir/optimization/patches/0001-repack-mmid-grouped-gemm.patch"
python3 optimization/check_mmid_pack_schedule.py \
  "$check_dir/ggml/src/ggml-cpu/repack.cpp" ggml/src/ggml-cpu/repack.cpp "$check_dir/result"
```

该检查执行 52,800 次逐线程比较。它移除了实际量化和内存读写，仅证明所覆盖输入的循环调度一致；不能替代完整内核、PPL 或性能测试。

## 比赛测试口径和当前状态

比赛目标平台为单节点双路 Intel Xeon Gold 6148（40 物理核、2 个 NUMA 节点），正式推理使用 38 线程，模型为四分片 Qwen3.8-Flash-Next-UD-Q4_K_XL。性能工作负载：

| 项目 | 固定参数 |
| --- | --- |
| PP2048 | `-p 2048 -n 0` |
| TG128@D2048 | `-p 0 -n 128 -d 2048` |
| 公共参数 | `-t 38 -b 2048 -ub 512 -ctk f16 -ctv f16 -ngl 0 -r 10` |

实际参赛须以组委会最新规则为准，PPL 使用其独立规定的参数。对比需要同节点、同构建条件和可比缓存/预热状态的配对重复，保留原始样本；吞吐、波动及能耗分别解释。

当前源码是开发候选：packing 调度差分检查已通过；第二阶段改动仍待目标节点完整构建、算子检查、模型 PPL 和同节点 PP/TG 配对验证。历史上其他构建的成绩不能直接视为当前提交的成绩。本仓库暂未发布当前提交的有效实测能耗或最终性能提升数字。

完整模型、原始比赛日志和带账号/服务器路径的本地调度脚本未纳入本次共享目录。队友可以先审阅实现及运行本地调度检查，再根据获准的集群环境配置测试。

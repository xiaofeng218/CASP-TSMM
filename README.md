# CASP-TSMM

Tall-Skinny Matrix Multiplication (TSMM) 评测与优化项目。

本项目计算：

```text
C = A^T * B
A: k x m
B: k x n
C: m x n
数据类型: double
```

评测程序同时支持行主序和列主序存储。默认运行 required 四组任务，并分别测试 row-major 和 col-major 两种布局。

## 目标平台

目标计算节点：

```text
CPU: Intel Xeon Platinum 9242
核心数: 96
NUMA 节点: 4
指令集: AVX-512
调度系统: Slurm
```

当前推荐流程是：

```text
登录节点编译 -> Slurm 提交 -> 计算节点运行
```

也就是说，`scripts/submit_slurm.sh` 不再负责编译，只负责申请计算节点资源、设置运行环境、执行 benchmark 并收集结果。

## 目录结构

```text
CASP-TSMM/
|-- src/
|   |-- benchmark.cpp       # 评测入口、计时、正确性检查、JSON 输出
|   |-- tsmm.hpp            # 算子接口、布局辅助函数
|   |-- tsmm_registry.cpp   # 算子自动注册
|   |-- reference.cpp       # MKL/OpenBLAS dgemm 参考实现，或内置 fallback
|   `-- tsmm/
|       |-- naive.cpp           # 简单串行参考实现，不注册到 benchmark
|       `-- opt.cpp             # 综合优化算子，默认提交算子
|-- scripts/
|   |-- collect_gflops.py   # 汇总 GFLOPS 到 CSV/JSON
|   |-- run_local.sh        # Linux 本地运行脚本
|   |-- run_local.ps1       # Windows PowerShell 本地运行脚本
|   `-- submit_slurm.sh     # Slurm 提交脚本，只运行，不编译
|-- web/
|   |-- index.html          # 结果展示页面
|   `-- server.py           # 轻量级本地 Web 服务，不需要 Docker
`-- Makefile
```

生成文件不属于源码，例如：

```text
obj/
web/results/
logs/
benchmark
benchmark.exe
```

## 算子版本

| 名称 | 说明 |
| --- | --- |
| `reference` | 使用 MKL/OpenBLAS `dgemm` 的参考实现；总是作为基准运行 |
| `opt` | 默认提交的综合优化版本，位于 `src/tsmm/opt.cpp` |
| `naive` | 简单串行参考实现，位于 `src/tsmm/naive.cpp`，不注册到 benchmark |

默认 benchmark 只会测试：

```text
reference
opt
```

`src/tsmm/naive.cpp` 只作为参考模板保留，没有调用 `REGISTER_TSMM_IMPL`，所以不会出现在默认结果里。其他同学新增算子时，只需要新增 `.cpp` 并调用 `REGISTER_TSMM_IMPL`，benchmark 就会自动把它和 `reference`、`opt` 一起测试。

## 集群编译

在登录节点进入项目目录：

```bash
cd /path/to/CASP-TSMM
```

推荐使用 Intel 编译环境和 MKL：

```bash
module purge
module load intel/2022.1
module load python/3.8.6
```

编译：

```bash
make clean
make CXX=icpc BLAS=mkl AVX512=1 -j16
```

`-j16` 表示最多同时启动 16 个编译任务。登录节点上不建议直接使用不带数字的 `-j`，也不建议使用太大的并行数。

编译完成后检查：

```bash
ls -lh obj/benchmark
```

备用 OpenBLAS 编译方式：

```bash
module purge
module load gcc/10.2.0
module load openblas/0.3.17-ips18
module load python/3.8.6

make clean
make BLAS=openblas AVX512=1 -j8
```

## 集群运行

提交前必须已经存在可执行文件：

```bash
ls -lh obj/benchmark
```

提交 required 四组任务：

```bash
bash scripts/submit_slurm.sh
```

提交 required + optional 全部任务：

```bash
bash scripts/submit_slurm.sh --all
```

脚本默认申请：

```text
partition: 不默认指定，使用集群默认分区
nodes: 1
ntasks: 1
cpus-per-task: 96
memory: 64G
time: 02:00:00
NUMA nodes: 0-3
```

脚本会分别运行：

```text
--layout row
--layout col
```

运行时使用：

```text
numactl --cpunodebind=0-3 --interleave=all
OMP_NUM_THREADS=96
OMP_PROC_BIND=spread
OMP_PLACES=cores
MKL_NUM_THREADS=96
MKL_DYNAMIC=FALSE
```

如果需要覆盖 Slurm 参数，可以在命令前加环境变量：

```bash
sinfo
PARTITION=实际分区名 CPUS_PER_TASK=96 MEM=64G TIME_LIMIT=02:00:00 bash scripts/submit_slurm.sh
```

如果 `sinfo` 显示存在默认分区，通常直接运行 `bash scripts/submit_slurm.sh` 即可，不需要设置 `PARTITION`。

如果只想快速测试，可以把 benchmark 参数继续传给脚本，例如：

```bash
bash scripts/submit_slurm.sh --warmup 1 --runs 1
```

这些额外参数会传给 `obj/benchmark`。

## 查看任务和结果

提交后脚本会输出任务号，例如：

```text
Submitted job 324xxxxx
Monitor: squeue -j 324xxxxx
Log: tail -f logs/tsmm_324xxxxx.out
```

查看排队或运行状态：

```bash
squeue -j 324xxxxx
```

实时查看标准输出：

```bash
tail -f logs/tsmm_324xxxxx.out
```

如果任务失败，查看错误日志：

```bash
cat logs/tsmm_324xxxxx.err
```

每次运行的结果目录为：

```text
web/results/<job_id>_<timestamp>/
```

重点查看：

```bash
cat web/results/<run-id>/gflops.csv
cat web/results/<run-id>/gflops_summary.json
```

其中：

```text
gflops.csv           # 每个任务、每个算子的 GFLOPS 和加速比
gflops_summary.json  # 汇总结果，包括几何平均加速比
results_row_*.json   # row-major 原始结果
results_col_*.json   # col-major 原始结果
```

## Benchmark 参数

`obj/benchmark` 支持：

```text
--required-only       只运行 required 四组问题
--all                 运行 required + optional 全部问题
--layout row|col      指定 row-major 或 col-major
--output-dir DIR      输出结果到目录，文件名自动带 layout 和时间戳
--output PATH         指定完整输出文件路径
--warmup N            预热次数，默认 10
--runs N              正式计时次数，默认 20
--no-correctness      跳过和 reference 的正确性对比
```

大规模任务会自动限制预热和正式计时次数，避免运行时间过长。

默认只测试：

```text
reference
opt
```

如果要添加新的实现，把新的 `.cpp` 放到 `src/tsmm/` 并注册即可，Makefile 会自动编译。

手动 smoke test 示例：

```bash
./obj/benchmark --required-only --layout row --warmup 1 --runs 1
```

正式性能测试请使用 Slurm 脚本，而不是在登录节点直接运行。

## 本地运行

Linux 本地快速运行：

```bash
bash scripts/run_local.sh --required-only
```

运行全部任务：

```bash
bash scripts/run_local.sh --all
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_local.ps1
```

不依赖 BLAS 的功能测试：

```bash
make BLAS=none
./obj/benchmark --required-only --layout row --warmup 1 --runs 1
```

## Web 展示

启动本地结果页面：

```bash
make web
```

然后打开：

```text
http://localhost:8080
```

也可以直接用本地运行脚本生成结果并启动页面：

```bash
bash scripts/run_local.sh --required-only
```

## 添加新算子

在 `src/tsmm/` 下新增一个 `.cpp` 文件，实现符合 `TsmmKernel` 签名的函数，然后注册：

```cpp
#include "../tsmm.hpp"

void tsmm_my_kernel(int m, int n, int k,
                    const double* A,
                    const double* B,
                    double* C,
                    Layout layout) {
    // compute C = A^T * B
}

REGISTER_TSMM_IMPL("my_kernel", tsmm_my_kernel);
```

Makefile 会自动编译 `src/tsmm/*.cpp`，benchmark 会自动发现注册的算子。

## 常见问题

如果编译时报：

```text
MKLROOT is not set
```

说明 MKL 环境没有加载好。优先尝试：

```bash
module purge
module load intel/2022.1
module load python/3.8.6
echo $MKLROOT
```

如果仍然为空，尝试：

```bash
module purge
module load oneAPI/2022.1
module load python/3.8.6
echo $MKLROOT
```

如果提交后报：

```text
Missing executable: ./obj/benchmark
```

说明提交前还没有编译，先执行：

```bash
make CXX=icpc BLAS=mkl AVX512=1 -j8
```

如果运行时报 MKL 动态库找不到，确认提交脚本里加载了和编译时一致的 Intel/oneAPI 模块。

1. 请你完成下面的大作业，搭建一个评测框架，然后生成并优化TSMM算子，把各个版本的算子单独存储在cpp文件中，把评测脚本和算法实现分隔开，算法文件中只保留预处理函数和计算函数，降低代码耦合度。评测时需要与MKL库对比验证正确性。
2. 测评时先预热10轮，再运行20次取平均，最终指标是各个任务的GFLOPS；另外计算相对于MKL库中gemm算子的加速比，然后取几何平均，作为参考。
3. 搭建一个网页展示，实时展示测评结果，注意目标机器上不能使用docker。
4. 你先在本机生成算子，但是最终目标是优化在指定机器上的性能，后续我将手动在目标机器上进行性能评测与分析，我会把结果反馈给你，你再根据结果进一步优化。
5. 在本机可以直接运行测试，但是在目标机器需要使用slurm提交任务，使用4个NUMA node的所有CPU并行计算。

## 作业要求：

Course Project: TSMM Multiplication optimization
- 课程大作业：Tall-Skinny Matrix Multiplication (TSMM) 优化
- 目标：通过优化瘦高矩阵乘法（TSMM），掌握性能分析方法与工具、并行编程技术、访存优化技术等，锻炼解决实际问题的综合能力
- TSMM定义：
  $$
  C = A^T \times B,\ A \in R^{k \times m}, B \in R^{k \times n}, C \in R^{m \times n}
  $$
  required : $(m,n,k) = (4000,16000,128),(8,16,16000),(32,16000,16),(144,144,144)$
  optional : $(m,n,k) = (16,12344,16),(4,64,606841),(442,193,11),(40,1127228,40)$
- 假定：A、B、C均是稠密矩阵，双精度（60分）
- 实验平台：统一提供Intel平台

Course Project: TSMTTSM Multiplication optimization
- 作业要求：
- 阶段一：基础实现与分析
  - 串行实现：用C/C++编写TSMM的串行版本，支持行主序/列主序存储
  - 与库函数MKL/openBLAS等对比
  - 用性能模型、prof工具等分析性能及其优化的潜在空间
- 阶段二：并行优化
  - 多线程优化（OpenMP），设计并行分块策略，对比不同调度策略的性能差异，分析扩展性等
  - 访存模式优化，设计blocking等策略提升访存效率，并给出理论分析
  - 内核汇编优化（可选）
  - 最终优化结果与库函数MKL/openBLAS等对比
- 提交要求：
  - 代码：完整可编译的C/C++代码（30%，正确性&性能）
  - 报告：PDF/WORD格式，至少包含上述阶段内容（40%，完整性&创新性）
  - 答辩：第19周大作业答辩，分组制作PPT展示优化方法和结果分析等（30%）
  - 分组：自由组队，每组7~8人。代码&报告每个人都要单独提交一份，答辩ppt只需要小组提交一份。

## 目标CPU信息

```
[t6s008866@ln2%bscc-t6 ~]$ srun lscpu
srun: job 32437165 queued and waiting for resources
srun: job 32437165 has been allocated resources
Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                96
On-line CPU(s) list:   0-95
Thread(s) per core:    1
Core(s) per socket:    48
Socket(s):             2
NUMA node(s):          4
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 85
Model name:            Intel(R) Xeon(R) Platinum 9242 CPU @ 2.30GHz
Stepping:              7
CPU MHz:               3099.890
CPU max MHz:           3800.0000
CPU min MHz:           1000.0000
BogoMIPS:              4600.00
Virtualization:        VT-x
L1d cache:             32K
L1i cache:             32K
L2 cache:              1024K
L3 cache:              36608K
NUMA node0 CPU(s):     0-23
NUMA node1 CPU(s):     24-47
NUMA node2 CPU(s):     48-71
NUMA node3 CPU(s):     72-95
Flags:                 fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch epb cat_l3 cdp_l3 invpcid_single ssbd mba rsb_ctxsw ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm cqm mpx rdt_a avx512f avx512dq rdseed adx smap clflushopt clwb intel_pt avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req pku ospke avx512_vnni md_clear spec_ctrl intel_stibp flush_l1d arch_capabilities
```
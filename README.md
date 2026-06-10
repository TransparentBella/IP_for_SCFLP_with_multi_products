# 论文复现：Two-Stage Capacitated Facility Location with Multilevel Capacities

本目录复现论文：

- `Formulation and solution of a two-stage capacitated facility location problem with multilevel capacities`

代码目标分两层：

1. 严格复现论文的两个 ILP 模型。
2. 严格复现论文的 MAAT 算法框架：`Stage 1 聚合+局部搜索`，`Stage 2 原问题局部搜索`，`Stage 3 在候选仓库集合上精确求解`。

## 目录结构

- `src/scflp_multilevel/instance.py`
  - 统一实例数据结构。
- `src/scflp_multilevel/exact_models.py`
  - 论文 Model A 与 Model B 的 Gurobi 精确模型。
- `src/scflp_multilevel/maat.py`
  - 论文 MAAT 算法与局部搜索。
- `src/scflp_multilevel/random_instance.py`
  - 论文风格随机实例生成器。
- `src/scflp_multilevel/teacher_adapter.py`
  - 老师提供 Excel 的兼容转换入口。
- `run_reproduction.py`
  - 运行脚本。

## 与论文一致的部分

- Model A 变量与目标函数：
  - `Q_j`
  - `Y_jpd`
  - `X_ijp`
  - `Xhat_jkp`
- Model B 变量与目标函数：
  - `Y_jd`
  - `X_ijp`
  - `Xhat_jkp`
- MAAT 主流程：
  - `rho` 按论文式(20)/(21)计算。
  - `mu = beta * rho`
  - Stage 1 每轮保留上轮最好仓库集合，再随机补足聚合候选集。
  - 每轮聚合问题先精确求解，再执行论文 Fig.2 的局部搜索。
  - Stage 2 对原问题再做一次局部搜索。
  - Stage 3 把候选仓库集合缩减为 `S*` 后做精确求解。
- 局部搜索：
  - 使用论文中的 first-improvement swap。
  - 候选仓库按最近已开仓分配。
  - 计算 `d_hat_s = max_j in N_s d_tilde_j`
  - 仅当 `d_js < gamma * d_hat_s` 时才评估替换。
  - MPTP 子问题在局部搜索中按论文放松为连续流，并令每个候选仓库使用最大容量。

## 论文未完全给出的地方

论文完整给出了模型和算法，但没有完整公开所有随机数据公式。缺失主要包括：

- 坐标采样区间的排版在 PDF 抽取中损坏。
- 随机实例的固定成本生成公式未完全列出。
- Model B 的产品体积与体积容量生成规则未完全列出。

因此我把这部分显式参数化，并在代码中写死为可追溯规则：

- 坐标：整数均匀分布，默认在 `[0, n_customers]`
- 需求：`U{1,5}`
- Model A 容量层：默认 `(30, 50, 70)`
- Model B 体积层：默认 `(180, 280, 400)`
- 产品体积：整数均匀分布 `U{1,4}`
- 运输成本：欧氏距离乘以产品费率
- 固定成本：按平均运输成本标定，使固定成本与运输成本量级接近

这些规则不影响“论文算法复现”，但会影响“论文随机表格数值逐项一致”。因此本实现严格复现算法，不宣称逐表重建原文全部随机数据。

## 老师数据的处理方式

`data_100200400.xlsx` / `data_200400800.xlsx` 只包含：

- 单产品
- 单层容量
- 无坐标
- 无产品体积

这不足以直接严格运行论文 MAAT。为此，`teacher_adapter.py` 做了显式兼容转换：

- 令 `|P| = 1`
- 由原始仓库容量生成 3 个容量层
- 由原始固定成本生成 3 个容量层固定成本
- 随机补齐坐标，只用于 MAAT 的邻域半径限制

所以：

- 若目标是严格按论文算法复现，推荐使用 `random_instance.py`
- 若目标是尽量利用老师数据，使用 `teacher_adapter.py`，但需接受其不是论文原始数据结构

## 运行

优先使用项目内环境：

```powershell
.\.venv311\python.exe .\IP_for_SCFLP_with_multi_products\run_reproduction.py --source random --model both --run-maat --plants 5 --depots 12 --customers 24 --products 5
```

只跑 Model A：

```powershell
.\.venv311\python.exe .\IP_for_SCFLP_with_multi_products\run_reproduction.py --source random --model a --run-maat
```

使用老师数据：

```powershell
.\.venv311\python.exe .\IP_for_SCFLP_with_multi_products\run_reproduction.py --source teacher --teacher-file .\data_100200400.xlsx --model a
```

使用老师数据并跑 MAAT：

```powershell
.\.venv311\python.exe .\IP_for_SCFLP_with_multi_products\run_reproduction.py --source teacher --teacher-file .\data_100200400.xlsx --model a --run-maat --exact-time-limit 1800 --output-name teacher_data_100200400_model_a_maat.json
```

## 批量实验与断点续跑

批量任务定义在 `batch_tasks.json`。

- `run_batch_resume.py`
  - 顺序执行任务
  - 每个任务单独写一个 JSON
  - 已完成任务会自动跳过
  - 每完成一个任务都会刷新 `outputs/batch_summary.csv`
- `run_batch_background.ps1`
  - 后台启动批量任务
  - 标准输出写入 `outputs/batch_stdout.log`
  - 标准错误写入 `outputs/batch_stderr.log`

前台执行：

```powershell
.\.venv311\python.exe .\IP_for_SCFLP_with_multi_products\run_batch_resume.py --task-file .\IP_for_SCFLP_with_multi_products\batch_tasks.json --output-dir .\IP_for_SCFLP_with_multi_products\outputs
```

仅刷新汇总表：

```powershell
.\.venv311\python.exe .\IP_for_SCFLP_with_multi_products\run_batch_resume.py --output-dir .\IP_for_SCFLP_with_multi_products\outputs --summary-only
```

后台执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\IP_for_SCFLP_with_multi_products\run_batch_background.ps1
```

## 论文参数对应

论文第 4 节给出的默认 MAAT 参数：

- `T = 10`
- `beta = 1.5`
- `tau = 150 s`
- `epsilon = 0.5%`
- `tau' = 108 s`
- `tau_hat = 0.25 * n * |P|`
- `gamma = 2.5`

代码中：

- `MAATConfig.iterations = 10`
- `MAATConfig.beta = 1.5`
- `MAATConfig.aggregated_time_limit = 150`
- `MAATConfig.aggregated_mip_gap = 0.005`
- `MAATConfig.final_time_limit = 108`
- `MAATConfig.local_search_time_limit = None` 时自动使用 `0.25 * n * |P|`
- `MAATConfig.gamma = 2.5`

对于 Model B，论文还给出：

- `|P| = 10` 时 `beta = 2`
- `|P| = 5` 时 `beta = 3`

脚本在 `--model b` 且 `--run-maat` 时按这一规则设置。

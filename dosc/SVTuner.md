# 项目名称

**英文：**
SVTuner: a plug-in framework for spatial-variability- and type-aware tuning of cell-to-spot mapping
**中文对应：**
SVTuner：一种面向空间变异性与细胞类型感知的细胞–空间位点映射调谐插件框架
**简称：**SVTuner





# 项目骨架

------

## 一、三大部分 & 8（7+1）阶段总览

#### **部分划分**

- **Part A：基线 & 共同地基**
  - **阶段 0：数据准备 & 环境验证 baseline（非正式）**
  - **阶段 1：统一预处理（sc + ST）**
  
- **Part B：Plus 模型设计 & 集成**
  - **阶段 2：模块一——SVG+HVG 动态加权插件**
  - **阶段 3：模块二——细胞类型不匹配（对齐 & unknown-aware）插件**
  - **阶段 4：CytoSPACE 正式映射：baseline vs plus 成对运行**
  
- **Part C：结果验证 & 展示**

  - **新增：模拟数据设计阶段**

  - **阶段 5：模拟数据实验（有真值）**
  - **阶段 6：真实数据实验（无真值，间接指标）**
  - **阶段 7：可视化整合 & 论文结构规划**



#### 整套项目的“树状流程图”

```c
main.py  （总指挥：通过 --samples / --stages / --backends 控制整条流水线）
├─ 真实数据线：sample = real_brca
│   ├─ Stage 0：数据集选择 + 环境测试
│   │   ├─ 选择 BRCA scRNA + Visium BRCA ST 作为真实数据
│   │   ├─ 用原始 CytoSPACE 跑一遍基线映射
│   │   └─ 确认环境、依赖、日志都 OK（记录 baseline 结果）
│   │
│   ├─ Stage 1：真实数据预处理（R + Python）
│   │   ├─ R/Seurat：QC + 温和过滤（防止 SVG 被过滤掉）
│   │   ├─ 归一化 / 找公共基因 / 生成标准化 sc/ST 矩阵
│   │   └─ 输出：real_brca 的 sc_processed / st_processed
│   │
│   ├─ Stage 2：模块一（SVG + HVG 动态加权）
│   │   ├─ 在 Stage1 处理后的数据上计算 SVG 指标
│   │   ├─ 和 HVG 信息结合，得到每个基因的综合权重 gene_weights
│   │   ├─ 选出插件基因集合 G_plugin（final_weight Top-K）
│   │   └─ 产出：G_plugin + gene_weights
│   │       （→ 后续在 CytoSPACE-plus 里构造加权表达矩阵 / cost，确保“SVG 真正进入 cost 空间”）
│   │
│   ├─ Stage 3：模块二（类型不匹配 / Unknown 插件）
│   │   ├─ 计算 sc 类型与 ST 表达之间的相似度矩阵
│   │   ├─ 合并相似类型 / 标记置信度低的类型为 Unknown 候选
│   │   ├─ 生成类型映射表（orig_type → plugin_type）
│   │   └─ 生成 spot×plugin_type 的先验矩阵 type_prior_matrix
│   │       （→ 后续在 CytoSPACE-plus 中改写 F_base → F_aligned，影响候选池和约束）
│   │
│   ├─ Stage 4：映射 Orchestrator（多 backend，支持多配置 / 多 seed / 多 λ）
│   │   ├─ 对每个 backend（当前主要是 CytoSPACE）：
│   │   │   ├─ baseline 模式：
│   │   │   │   ├─ 不用 Stage2/3 插件，使用原始表达矩阵（或传统 HVG）构建 cost
│   │   │   │   └─ 跑一遍官方 CytoSPACE 映射，得到干净 baseline（单配置、单 seed 即可）
│   │   │   └─ plus 模式（CytoSPACE-plus）：
│   │   │       ├─ 使用 Stage2 输出：
│   │   │       │   ├─ 只取 G_plugin 基因
│   │   │       │   └─ 用 gene_weights.final_weight 构造加权表达矩阵
│   │   │       │       → X_sc_plugin, X_st_plugin 进入 CytoSPACE cost 计算
│   │   │       ├─ 使用 Stage3 输出：
│   │   │       │   ├─ 用类型对齐矩阵 M 得到 F_aligned
│   │   │       │   └─ 用 type_prior_matrix 约束 / 引导候选细胞池
│   │   │       ├─ 主 assignment：
│   │   │       │   └─ 基于加权 cost matrix C_plus + F_aligned + UMI 先验求解一次映射
│   │   │       ├─ 映射后 SVG 空间 refine（轻量，可开关）：
│   │   │       │   ├─ 定义 SVG 空间重构 loss：
│   │   │       │   │   - 用映射得到的 cell→spot 结果重建 ST 表达
│   │   │       │   │   - 特别对插件 SVG 基因，计算重构 vs 实测的空间误差（如 SVG_recon_loss）
│   │   │       │   ├─ 加邻域正则（neighbor smoothness）：
│   │   │       │   │   - 邻近 spot 的 type 组成或表达不至于“剧烈抖动”
│   │   │       │   └─ 在小步长 / 少迭代下，对 cell_spot_matrix 做微调 refine
│   │   │       └─ 支持多配置 / 多 seed / 多 λ：
│   │   │           ├─ 如：
│   │   │           │   - 不同随机种子（seed）
│   │   │           │   - 不同 SVG/HVG 权重组合（α/β）
│   │   │           │   - 不同 SVG_recon vs neighbor 正则的 λ 比例
│   │   │           └─ 对每组配置分别输出一套 plus 结果目录（plus_cfg1 / plus_cfg2 / …）
│   │   └─ 输出（每种模式 × 每个配置）：
│   │       ├─ cell_assignment_*.csv      （细胞主落点）
│   │       ├─ cell_spot_matrix_*.npz     （cell×spot 软分配）
│   │       └─ spot_type_fraction_*.csv   （spot×type 预测分布）
│   │           （→ Stage5/6 统一只吃这“三件套”，其余 cost / refine 细节存为诊断文件）
│   │
│   ├─ Stage 6：真实数据验证（无真值）
│   │   ├─ 对 baseline & 各 plus 配置（多 seed / 多 λ）：
│   │   │   ├─ 利用映射结果重建 ST 表达，与实测 ST 做对比
│   │   │   │   ├─ 按基因：整体重构相关性（重点看 SVG / marker 基因）
│   │   │   │   └─ 按 spot：整体表达结构相关性、空间自相关
│   │   │   ├─ SVG 空间重构指标：
│   │   │   │   ├─ 对插件 SVG 基因，评估重建表达的空间模式 vs 实测（SVG_recon_score）
│   │   │   │   └─ 观察“空间 pattern 是否被破坏 / 修复”
│   │   │   ├─ 稳定性 & 邻域一致性：
│   │   │   │   ├─ 多 seed / 多配置下，cell_assignment / spot_type_fraction 的一致度
│   │   │   │   └─ 邻近 spot type 组成变化是否更平滑、是否出现明显伪斑块
│   │   │   ├─ marker–type 一致性：类型占比 vs 对应 marker 表达
│   │   │   ├─ Unknown 使用是否合理：是否集中在拟合差 / 类型缺口 / 冲突区域
│   │   │   └─ 若 backend=CytoSPACE：再看 cost、约束 RMSE 等（保证不违背原方法约束精神）
│   │   └─ 综合上述，形成“真实数据上的 plus 配置评分 & 仲裁建议”：
│   │       ├─ 给出每个 plus_cfg 的综合得分（SVG_recon + 稳定性 + 生物一致性）
│   │       └─ 标记推荐配置（推荐作为主结果写入论文）
│   │
│   └─ Stage 7：报告整合（真实 + 模拟一起）
│       ├─ 收集所有阶段的指标：
│       │   ├─ Stage2：SVG/HVG 权重分布、被过滤 SVG 的敏感性分析
│       │   ├─ Stage3：类型对齐矩阵、Unknown 诊断、类型覆盖率
│       │   ├─ Stage4：多配置 / 多 seed / 多 λ 的运行记录（含 refine 相关诊断）
│       │   ├─ Stage5：模拟场景下的真值评估 ＋ 最优 plus 配置（有真值仲裁）
│       │   └─ Stage6：真实数据上的间接指标 ＋ plus 配置仲裁结果
│       ├─ 生成论文级图表：
│       │   ├─ Fig1：流程图 & 数据集概览
│       │   ├─ Fig2：SVG 插件效果（SVG vs HVG、空间模式 / SVG_recon 指标）
│       │   ├─ Fig3：类型插件效果（相似度矩阵、先验分布、Unknown 分布）
│       │   ├─ Fig4：模拟数据评估（有真值，多配置比较 + 仲裁出来的最佳 plus）
│       │   └─ Fig5：真实数据评估（无真值，侧重 SVG 空间一致性 + 稳定性）
│       └─ 输出：Table1–5 + Stage7_summary_CN.md 报告
│
├─ SimGen 模块：模拟数据构建（独立于阶段 0–7）
│   ├─ 读取真实 BRCA sc 表达 + 类型注释
│   ├─ World / Ref 拆分
│   │   ├─ World：只用于构造模拟 ST（真世界细胞）
│   │   └─ Ref：只作为映射时看到的 sc 参考库
│   ├─ 为 World 细胞设计空间布局
│   │   ├─ 割成 block（不同类型有不同高占比区）
│   │   └─ 随机赋 2D 坐标（带空间结构和过渡带）
│   ├─ 构造 spot 网格并聚合表达 → sim_st_expression
│   ├─ 输出模拟真值：
│   │   ├─ sim_truth_cell_spot.csv
│   │   └─ sim_truth_spot_type_fraction.csv
│   └─ 生成多个场景（S0 完全匹配 / M1 sc 缺 type / M2 ST 缺 type）
│
└─ 模拟场景流水线：sample = S0_matched / M1...（每个场景跑一次）
    ├─ Stage 1：对 SimGen 的 sc/ST 做同样的预处理（R + Python）
    ├─ Stage 2：在模拟数据上重算 SVG/HVG，检查过滤对 SVG 的影响（同真实线逻辑）
    ├─ Stage 3：在模拟数据中应用类型插件，特别考察 mismatch 场景
    ├─ Stage 4：对该场景做 baseline / plus 映射（同真实线的 Orchestrator 逻辑）
    │   ├─ baseline：单配置，记录基准误差
    │   └─ plus：支持多配置 / 多 seed / 多 λ，输出多套 plus 结果（含 SVG refine）
    ├─ Stage 5：有真值的模拟评估（5A/5B）
    │   ├─ 通用指标：
    │   │   ├─ spot×type L1 / JS 距离（对比 sim_truth_spot_type_fraction）
    │   │   ├─ rare type 虚假扩散程度
    │   │   └─ missing→Unknown vs →wrong type 的比例
    │   ├─ CytoSPACE 专用：
    │   │   ├─ 映射 cost（total / mean per cell）
    │   │   └─ spot 总细胞数 & spot×type 约束 RMSE
    │   └─ SVG / 仲裁相关指标：
    │       ├─ 对插件 SVG 基因的空间重构 vs 真值（有真值版 SVG_recon）
    │       ├─ 多配置 / 多 seed 间的稳定性对比
    │       └─ 选出“真值误差最小 + SVG 空间最好”的 plus 配置，作为该场景推荐解
    └─ Stage 7：这些模拟结果（含“最佳 plus 配置标记”）也会被 Stage7 报告整合进去

```

## 封板配置（S0/M1）

### S0（Plus best：Part2 + Part3 v1+quota）
- 配置文件：`configs/datasets/S0_matched_dataset.yaml`
- 关键开关：
  - `svg_post_enabled: true`
  - `type_posterior_enabled: true`
  - `type_post_mode: local_marker_swap`
  - `type_post_max_changed_cells: 12`
  - `type_post_high_hurt_threshold: 0.1`
  - `type_post_max_high_hurt_actions: 2`
- 复现建议：`--config_id S0_freeze_best`

### M1（Safe：baseline-compatible）
- 配置文件：`configs/datasets/M1_sc_missing_Bcell.yaml`
- 关键开关：
  - `strict_config: true`
  - `svg_post_enabled: false`
  - `type_posterior_enabled: false`
  - `spot_weight_mode: none`
- 复现建议：`--config_id M1_freeze_safe`

------





## 二、逐阶段设计 + “关节处”说明

下面按阶段来写，每个阶段都包含：

- 目标
- 输入
- 核心操作
- 输出
- 🔗 **关键关节**（和前后阶段的衔接）

------

### 0️⃣ 阶段 0：数据准备 & 环境验证 baseline（非正式）

**目标：**

- 选定要用的 **scRNA + ST 数据集**；
- 跑通 CytoSPACE，确认环境 / 接口 / 文件格式；
- 可以跑一版 **exploratory baseline 映射**，用于熟悉方法，不一定写进论文。

**输入：**

- 原始 sc 数据（count matrix / h5 / 其他格式）
- 原始 ST 数据（Visium / MERFISH / 其他）

**核心操作：**

- 整理成 CytoSPACE 接收的最简形式，跑一遍“原作者默认流程”；
- 记录：
  - 需要的输入文件格式；
  - 命令行 / 函数调用方式；
  - 核心输出文件结构。

**输出：**

- `cytospace_raw_run/`（示例运行目录）
- 非正式：`raw_baseline_assigned_locations.csv`（只用于熟悉）

**🔗 关键关节：**

- 为后续阶段 1 的预处理，明确**我们最终希望的输入格式**；
- 为阶段 4 的正式 baseline + plus 提供：
  - 参数模板
  - 运行脚本模板（比如 `run_cytospace(mode='baseline')`）

> 心态上：阶段 0 更多是“练习场 +环境验证”，正式对照组 baseline 会在阶段 4 再跑一遍。

------

### 1️⃣ 阶段 1：统一预处理（sc + ST）

**目标：**
 为后续所有阶段提供一套统一、干净、尽量保留信息的表达矩阵和元数据。

**输入：**

- 原始 sc 数据
- 原始 ST 数据

**核心操作：**

1. **scRNA 预处理**
   - 细胞过滤：低 UMI / 高线粒体比例等；
   - 基因初步过滤：超低表达、垃圾基因；
   - 归一化、log transform、必要的批次校正；
   - 初步 HVG 标记（`hvg_flag` 列，只是“传统 HVG”参考）；
   - 初步 cell type 注释（允许 `unknown`）。
2. **ST 预处理**
   - spot 过滤：低 UMI、低信号等；
   - 基因过滤：与 sc 兼容的基因子集；
   - 归一化、log transform；
   - 计算每个 spot 的 UMI 总量 `UMI_total`（用于后面估细胞数）。
3. **重要：基因过滤 + SVG 敏感性准备**
   - 先记录：
     - `genes_before_filter.txt`：最初出现过的基因集合；
   - 进行一个 **温和的基因过滤**（剔除极低表达和垃圾基因），得到：
     - `G_clean`：过滤后的基因列表；
     - `genes_after_filter.txt`：保存下来；
   - 为阶段 2 的 SVG/HVG 敏感性分析埋点。

**输出：**

- `sc_processed.h5ad`
  - `X_sc (C × G_clean)`
  - `cell_type`（含 `unknown`）
  - `hvg_flag`（传统 HVG 标记）
- `st_processed.h5ad`
  - `X_st (S × G_clean)`
  - `UMI_total`
  - `spatial_coords`
- `shared_genes.txt`（sc & ST 交集，基本 = `G_clean`）
- `genes_before_filter.txt`
- `genes_after_filter.txt`

**🔗 关键关节：**

- **→ 阶段 2（模块一）的基因输入：**
   HVG & SVG 都将在 **同一套 `G_clean` 上计算**，保证不会“各算各的子集”。
- **→ 阶段 3/4 的基础表达矩阵：**
   后续所有 mapping（baseline & plus）都将从 `sc_processed.h5ad` / `st_processed.h5ad` 读取。
- **→ SVG 过滤敏感性分析：**
   `genes_before_filter.txt` vs `genes_after_filter.txt` 为后续检测“是否误杀了潜在 SVG/HVG”提供信息。

------



------

### 2️⃣ 阶段 2：模块一——SVG+HVG 动态加权插件

**目标：**
 在**同一套过滤后的基因集合 `G_clean`** 上，计算 HVG & SVG 分数，并合成为一个可调的基因权重：
 1）在 Stage4 中**直接改写 CytoSPACE 看到的表达空间 / cost 矩阵**，让算法在映射时真正“看见” SVG；
 2）为 Stage5/6 的 **SVG 空间重构指标 + 仲裁指标**提供统一的基因评分基准。

------

**输入：**

- `sc_processed.h5ad`
  - 含 `X_sc (C × G_clean)`
  - 含 `cell_type`、`hvg_flag` 等基础信息
- `st_processed.h5ad`
  - 含 `X_st (S × G_clean)`
  - 含 `spatial_coords`、`UMI_total` 等
- `genes_before_filter.txt` / `genes_after_filter.txt`
  - 阶段 1 过滤前后基因列表，用于 SVG/HVG 过滤敏感性分析

------

**核心操作：**

1. **计算 HVG 分数（在 `G_clean` 上）**
   - 可用 scanpy 等标准方法，对每个基因得到 `HVG_score(g)`。
   - 直观：这个基因在“细胞之间”差异大不大。
2. **计算 SVG 分数（在 ST 上）**
   - 使用 `X_st` + 空间坐标，按选定方法（如 SPARK / Moran’s I 等）对每个基因计算 `SVG_score(g)`。
   - 直观：这个基因在“空间上”是否呈现成片结构（某些区域亮、某些区域暗）。
3. **HVG & SVG 归一化 + 动态加权**
   - 对 `HVG_score`、`SVG_score` 分别做标准化（z-score 或 0–1）：
     - `HVG_norm(g)`
     - `SVG_norm(g)`
   - 合成最终权重（全局示意）：
     - `w_gene(g) = α · HVG_norm(g) + β · SVG_norm(g)`
   - `α, β` 作为超参数（默认可 α=β=0.5），以后可以在 main.py 中调节：
     - 偏重空间结构：减小 α、增大 β。
   - **这一条是后面“改写 cost”与“SVG 仲裁”的统一基准**：所有映射和评估都在同一套 `w_gene` 上比较。
4. **选出插件基因集合 `G_plugin` 并准备给 Stage4/5/6 使用的元信息**
   - 典型做法：按 `w_gene(g)` 从大到小排序，取 Top-K 作为 `G_plugin`。
   - 在内部为每个基因打标：
     - 是否在 `G_plugin` 中；
     - 对应的 `HVG_score / SVG_score / HVG_norm / SVG_norm / w_gene`。
   - 保留**全基因的 HVG/SVG 分数和最终权重**，方便 Stage5/6 后续读取，用来计算：
     - SVG 空间重构损失（哪些 SVG 在映射后被重构得更好）；
     - 多次 run 之间的 SVG 仲裁指标（哪条映射链路在 SVG 维度上“最一致 / 最好”）。
5. **SVG/HVG 过滤敏感性分析（利用 `genes_before_filter` / `genes_after_filter`）**
   - 回答：**“阶段 1 的基因过滤有没有误杀大量 SVG/HVG 候选？”**
   - 基本思路：
     - 在 `genes_before_filter.txt` 对所有基因粗算一版 SVG/HVG 候选；
     - 看前 N 个候选里，有多少在 `genes_after_filter.txt` 中被删掉；
   - 输出一个简单报告，用于判断：
     - 如果大量高 SVG/HVG 候选被扔掉 → 需要回头调 Stage1 过滤；
     - 如果损失很少 → 可以认为 Stage1 的过滤对 SVG/HVG 相对安全。

------

**输出：**

- `gene_weights.csv`
  - 每个基因一行，包含（至少）：
    - `gene`
    - `HVG_score` / `SVG_score`
    - `HVG_norm` / `SVG_norm`
    - `final_weight (w_gene)`
    - （可选）是否进入 `G_plugin`、是否为 Seurat HVG 等标记
- `selected_genes.txt`（或 `plugin_genes.txt`）
  - 插件基因集合 `G_plugin`（比如按 `w_gene` Top-K）
- `hvg_svg_filter_sensitivity.json`
  - 记录过滤前后潜在 HVG/SVG 候选的损失比例等信息，作为对 Stage1 的反馈信号

------

**🔗 关键关节：**

- **→ 阶段 4：表达矩阵 / cost 入口被改写**
  - 在 plus 分支中构造：
    - `X_sc_plugin = X_sc[:, G_plugin] * w_gene`
    - `X_st_plugin = X_st[:, G_plugin] * w_gene`
  - **CytoSPACE plus 分支的 cost 矩阵完全建立在这两个“加权表达空间”之上**，
     等价于在 cost 中按 `w_gene` 放大 SVG 贡献，让算法真正“用上” SVG。
- **→ 阶段 1：过滤质量的反馈回路**
  - 若 `hvg_svg_filter_sensitivity.json` 显示“过滤误杀严重”，
     → 可以回到 Stage1 调低过滤强度，再重跑 Stage1+Stage2。
- **→ 阶段 5 / 6：SVG 空间重构 & 仲裁的评分基准**
  - Stage5/6 在比较 **不同 seed / 不同 λ / baseline vs plus** 时，
     会基于 Stage2 提供的 `SVG_score(g)`、`w_gene(g)` 计算：
    - “SVG 空间重构是否更好”（SVG 模式是否被更好保留）；
    - “哪条映射链路在 SVG 维度上表现最佳”（SVG 仲裁）。
  - 这样可以保证：
    - **我们后面选出来的“最优映射”，确实是在同一套 SVG 评分空间下表现最好的那条。**



------

### 3️⃣ 阶段 3：模块二——细胞类型不匹配（对齐 & unknown-aware）插件

**目标：**
 在细胞类型层面对 sc 与 ST 之间的 mismatch 进行诊断和对齐，并专门处理 `unknown` 类型。

**输入：**

- `sc_processed.h5ad`（含 `cell_type`，含 `unknown`）
- 用于 ST 端类型信息的来源：
  - deconvolution 的初始 `F_base (S × T_sc)`，或
  - marker 基因集 / 其他先验生成的 ST-type profile。

**核心操作：**

1. **类型相似度计算**
   - 基于 sc 各类型的 mean expression、marker，实现 type×type 相似度矩阵。
2. **类型对齐矩阵 `M` 构建**
   - 生成 `M (T_sc × T_st)`，每个元素为“sc_type i 与 st_type j 的兼容度（0–1）”。
3. **unknown-aware 分析**
   - 对 `unknown` 细胞 / cluster：
     - 算它们与各 ST-type 的 soft similarity 分布（candidate type）；
     - 看它们映射后在空间上的聚集情况；
   - 输出一个诊断报告：
     - 哪些 unknown 可能对应某类 ST 类型的缺口；
     - 哪些区域主要由 unknown 解释，提示“参考数据盲点”。
4. **类型覆盖率 & 缺口报告**
   - 基于 M 和 F_base：分析哪些 ST-type 能被 sc-type（含 unknown）充分解释，哪些存在明显缺口。

**输出：**

- `type_alignment_matrix.csv`（`M`）
- `unknown_diagnostics.json`（unknown 的候选类型、空间分布特征）
- `type_coverage_report.html`（或 markdown）

**🔗 关键关节：**

- **→ 阶段 4：spot×type 组合被改写**
  - 初始 spot composition `F_base (S×T_sc)` 通过 `M` 得到对齐版：
    - `F_aligned = F_base · M`
  - CytoSPACE 将基于 `F_aligned` + UMI 总量构建候选细胞池。
- **→ 阶段 6：类型覆盖 & unknown 作为评估指标来源**
  - 真实数据实验中，会用你在这里算出的覆盖率 / unknown 分布来解释映射结果的合理性。



------

### 4️⃣ 阶段 4：映射编排 & CytoSPACE 后端——baseline / plus / SVG-refine 多配置运行

**目标：**
 在**完全统一的预处理与参数前提**下，由 Stage4 的 Orchestrator 统一调度 `CytoSPACEBackend`，系统性地产生：

- **baseline 分支**：不接插件（或只用传统 HVG 设定），作为对照；
- **plus 分支**：接入
  - 阶段 2 的 **SVG+HVG 基因权重（Step1：加权 cost）**，
  - 阶段 3 的 **类型重标记 + 类型先验矩阵**；
- （可选）**plus-refine 分支**：在 plus 映射结果上做
   **SVG 空间重构 loss + 邻域微调（Step2）**；

并支持在**多配置 / 多 seed / 多 λ** 下反复运行，为后续 Stage5/6 的
 **“SVG 指标仲裁（Step3）”** 提供足够丰富的候选映射结果。

------

**输入：**

- 阶段 1 输出：
  - `sc_processed.h5ad`
  - `st_processed.h5ad`
- 阶段 2 输出（SVG+HVG 插件）：
  - `selected_genes.txt` / `plugin_genes.txt`
  - `gene_weights.csv`（含 `w_gene`、SVG/HVG 分数等，用于加权 cost）
- 阶段 3 输出（类型插件）：
  - `cell_type_relabel.csv`（最终 `plugin_type`）
  - `type_prior_matrix.csv`（spot×type 先验矩阵）
- 阶段 4 自身的运行配置：
  - `stage4_config.yaml` / `stage4_config.json`（示意）
    - 使用哪些 backend（当前至少有 `CytoSPACEBackend`）
    - baseline / plus 是否都跑
    - plus 分支的参数网格：
      - 不同 SVG 权重策略（是否只用 `G_plugin`、是否同时保留其他基因）
      - 不同 λ（SVG 空间 loss 权重）
      - 不同随机 seed（稳定性对比）

------

**核心操作：**

1. **Orchestrator：统一调度映射后端**
   - 读取 `stage4_config` 中的：
     - backend 列表（当前：`CytoSPACEBackend`）
     - `seeds`、`lambda_svg`、`config_id` 等超参数组合；
   - 对每个 backend、每个 `(config_id, seed, λ)` 组合：
     - 调用 `run_baseline(...)`（可只跑一次或少数几次，用于对照与稳定性分析）；
     - 调用 `run_plus(...)` 生成 plus 映射结果；
     - 如开启 refine，再调用 `run_plus_refine(...)` 或单独的 `svg_refine(...)`。
2. **CytoSPACE baseline 分支（不接插件）**
   - 使用来自 Stage1 的 `sc_processed.h5ad` / `st_processed.h5ad` 作为输入；
   - 基因空间：
     - 使用默认的 HVG / 全基因设置（不读 `gene_weights.csv`）；
   - 成本函数：
     - 使用原始 CytoSPACE cost（相关 / 欧氏距离等，不带 SVG 加权）；
   - 输出典型文件（每个 seed 一套）：
     - `cell_assignment_baseline_seed{seed}.csv`
     - `spot_type_matrix_baseline_seed{seed}.csv`
     - baseline 版 CytoSPACE 日志 / 诊断图。
3. **CytoSPACE plus 分支——接入插件 + Step1：加权 cost**
   - 基因空间：
     - 从 Stage2 读取 `plugin_genes` + `w_gene`，构造：
       - `X_sc_plugin  = X_sc[:, G_plugin] * w_gene`
       - `X_st_plugin  = X_st[:, G_plugin] * w_gene`
     - 或按配置决定是否同时保留少量非插件基因。
   - 类型与先验：
     - 从 Stage3 读取：
       - 细胞类型：`plugin_type`（替换原始类型标签）
       - spot×type 先验矩阵：`type_prior_matrix`
   - 成本函数（Step1 落地点）：
     - 使用**加权相关 / 加权距离**构造 cell–spot cost 矩阵，使 SVG 权重真正进入优化目标；
   - 对每个 `(seed, λ)` 运行 `run_plus(...)`，输出：
     - `cell_assignment_plus_backend=cytospace_config={id}_seed={s}_lambda={λ}.csv`
     - 对应的 spot×type 组合矩阵、CytoSPACE 诊断结果等。
4. **SVG 空间重构 + 邻域微调（Step2：可选 refine 子模块）**
   - 对每个 plus 映射结果，额外执行 `svg_refine` 子模块：
     - 利用 Stage2 的 SVG 列表和 ST 表达，重构每个 SVG 在空间上的预测表达；
     - 与真实 ST 的 SVG 表达比较（相关 / 邻域差分 / Moran’s I）得到 **SVG 空间 loss**；
     - 在空间邻域内尝试局部细胞交换，只接受能降低 SVG 空间 loss 的调整；
   - 输出：
     - `cell_assignment_plus_refined_backend=cytospace_config={id}_seed={s}_lambda={λ}.csv`
     - 以及对应的 “refined” 版 spot×type 矩阵。
5. **多配置 / 多 seed / 多 λ 的运行记录（为 Step3 做准备）**
   - Stage4 **不直接“判优劣”**，而是：
     - 对每次 baseline / plus / plus-refine 运行，记录一条 `run_id`；
     - 把 `(backend, config_id, seed, lambda_svg, 使用的 gene 选择策略等)` 写入
        `stage4_run_manifest.csv` / `stage4_run_manifest.json`；
   - Stage5/6 会根据这个 manifest 去读取对应映射结果，
      结合 SVG 空间指标 + 全局指标，在后续完成真正的 **“SVG 指标仲裁（Step3）”**。

------

**输出：**

- 目录结构示意（以 Cytospace 为例）：

```text
result/
  stage4_mapping/
    cytospace/
      baseline/
        cell_assignment_baseline_seed*.csv
        spot_type_matrix_baseline_seed*.csv
        ...
      plus_svg_type/
        config_{id}_seed_{s}_lambda_{λ}/
          cell_assignment_plus.csv
          cell_assignment_plus_refined.csv   # 若启用 refine
          spot_type_matrix_plus*.csv
          svg_refine_log.json                # 可选：记录 SVG loss 变化
  stage4_run_manifest.csv / .json
```

------

**🔗 关键关节：**

- **→ 阶段 5（模拟数据）：有真值下的 baseline vs plus(+refine) 对照**
  - Stage5 会：
    - 读取 `stage4_run_manifest`，遍历不同 seed / λ / 配置；
    - 对每个 run 计算 **带真值的映射准确率 + SVG 空间重构指标**；
    - 再基于这些指标去做：
      - baseline vs plus / plus-refine 的对比；
      - 哪一个 plus-config 在 SVG 与真值对齐上表现最好。
- **→ 阶段 6（真实数据）：无真值下的 SVG 指标仲裁（Step3 落地点）**
  - 对真实 BRCA 数据：
    - 读取所有 plus / plus-refine 运行；
    - 计算 SVG 空间重构相关、空间自相关保持度、marker–type 一致性等；
    - 综合全局指标（表达重构、CytoSPACE cost/约束）做 **多配置仲裁**，
       选出“SVG 表现最稳健 / 最合理”的一条映射链路，作为论文主结果。
- **→ 阶段 7：核心论文图的原始数据来源**
  - 所有与“方法有效性相关”的图：
    - baseline vs plus / plus-refine 的 SVG 空间图对比；
    - 不同 seed/λ 下 SVG 指标分布与最终被选中 run 的位置；
    - 映射准确率 / 覆盖度 / 稳定性指标曲线；
  - 都将直接基于 Stage4 产生的这些映射结果 + manifest。



------



------

### SimGen 阶段：模拟数据构建（World/Ref 拆分 + 场景生成）

**目标：**
 用真实 BRCA scRNA + 类型注释做模板，**单独构建带真值的模拟场景**（S0 / M1 / M2…），为后面的 Stage5“有真值评估”提供输入。
 SimGen 只负责“造世界”，**不负责跑映射、算指标**。

**输入：**

- 真实 BRCA 的 sc 表达与类型信息（来自 Stage0/1 准备好的数据）；
- （可选）真实 ST 的坐标 / 组织结构，用来参考空间布局；
- 每个场景的配置文件：`configs/simgen/S0_matched.yaml`, `M1_sc_missing_Bcell.yaml`, …
  - 里面定义：
    - 使用哪些 cell type 做 World / Ref；
    - 哪些类型在 sc 缺失 / 哪些在 ST 缺失；
    - World 细胞的空间布局规则（块、梯度、过渡带等）。

**核心操作：**

1. **World / Ref 拆分：**
   - 从真实 sc 中抽取一部分作为 **World 细胞集**（只用于造模拟 ST，永远不直接参与映射）；
   - 剩余部分作为 **Ref 参考集**（后面做映射时当作 sc 参考）。
2. **给 World 细胞设计空间布局：**
   - 把空间切成若干 block / 区域（例如 Tumor 高占比区、免疫富集区、混合区）；
   - 按场景规则，把不同类型的 World 细胞随机放到这些区域；
   - 确保：
     - S0 场景：各类型都有合理的支持区域；
     - M1/M2 场景：人为制造 “sc 缺 type / ST 缺 type / 稀有型” 等情况。
3. **构造模拟 ST spot：**
   - 在 World 坐标上铺一层 ST 网格（spot）；
   - 按 spot 采样附近 World 细胞并累加基因表达 → `sim_st_expression`；
   - 统计每个 spot 的 **真值类型构成** → `sim_truth_spot_type_fraction`；
   - 记录每个 World 细胞真正落在哪个 spot → `sim_truth_cell_spot`。
4. **多场景生成：**
   - 针对每个 `scenario_id`（S0 / M1 / M2…）重复 1–3 步；
   - 保证：
     - 场景定义符合预期（比如“sc 缺 B cell”的场景里，Ref 里确实没有 B cell）；
     - 输出格式与真实数据兼容，能直接喂给 Stage1–4 跑。

**输出：**

对每个 `scenario_id`，在 `data/sim/<scenario_id>/` 下生成：

- `sim_sc_expression.csv`（⚠️Query：Stage4 要映射的 sc，等价于 stage1_exported 的 sc）
- `sim_sc_metadata.csv`（Query 元数据，建议含 `celltype` 列）
- `sim_ref_sc_expression.csv`（可选：Ref，仅分析用，不参与 Stage4）
- `sim_ref_sc_metadata.csv`（可选：Ref，仅分析用）
- `sim_st_expression.csv`
- `sim_st_coordinates.csv`
- `sim_truth_spot_type_fraction.csv`（spot×type 真值）
- `sim_truth_cell_spot.csv`（World cell→spot 真值，用于解释/审计 World 造 ST 过程）
- `sim_truth_query_cell_spot.csv`（⚠️Query cell→spot 真值：Stage5 的 cell-level 真值口径）
- `scenario_meta.json`（记录 missing_types、world_fraction 等元信息）
  - 建议包含：`role_map`、`cell_level_mode="query"`、`query_truth_policy`、`truth_files`（含别名）

**🔗 关键关节：**

- **→ Stage1–4：**
   每个场景在 pipeline 里就像一个新的 sample（`sample_id="S0_matched"` 等），后续直接跑 Stage1–4，产生 baseline / plus 映射结果。
- **→ Stage5：**
   Stage5 **只读取**这些模拟 sc/ST 和真值文件做评估，
   **完全不再关心“模拟是怎么造出来的”** —— SimGen 和 Stage5 职责彻底分离。

------

### 5️⃣ 阶段 5：基于模拟数据的映射验证实验（有真值）

**目标：**
 在 **SimGen 已经生成好的模拟场景（带 cell→spot 真值）** 上，对比：

1. baseline vs plus（SVG+HVG cost + 类型插件）的映射质量；
2. plus 在 **SVG 空间模式**、**类型/Unknown 行为**、**稳定性（seed / λ / 配置）** 上是否有明确优势；
3. 为真实数据的 Stage6 提供一块“**安全且有效的超参数区间**”（比如合适的 SVG 权重 λ、推荐配置组合）。

**输入：**

- 来自 **SimGen** 的每个场景文件（按 `scenario_id`）：
  - `sim_sc_expression.csv`, `sim_sc_metadata.csv`
  - `sim_st_expression.csv`, `sim_st_coordinates.csv`
  - `sim_truth_spot_type_fraction.csv`
  - `sim_truth_query_cell_spot.csv`（cell-level 真值；若存在别名 `cell_true_spot.csv` 也可作为兼容读取）
  - `sim_truth_cell_spot.csv`（World 真值，仅用于审计/解释，可选读取）
  - `scenario_meta.json`
- 对每个场景跑完 Stage1–4 后的输出（按 backend / baseline / plus）：
  - `result/<scenario_id>/stage4_mapping/<backend>/baseline/...`
  - `result/<scenario_id>/stage4_mapping/<backend>/plus_svg_type/...`
  - 对 CytoSPACE 额外有：cost 矩阵、约束信息等（可选）。

**核心操作：**

1. **按场景 × backend 读取映射结果 + 真值**

   - 对每个 `(scenario_id, backend)`：
     - 读取 baseline / plus 的：
       - `cell_assignment_*.csv`（细胞主落点）
       - `cell_spot_matrix_*.npz`（cell×spot 分布）
       - `spot_type_fraction_*.csv`（spot×type 分布）
     - 同时读取：
       - 真值 `sim_truth_cell_spot.csv`、`sim_truth_spot_type_fraction.csv`
       - 场景元信息 `scenario_meta.json`（哪个类型是缺失 / 稀有等）。

2. **计算通用“有真值”指标（5B 部分，全 backend 共享）**

   主要分 3 大类：

   - **映射准确度：**
     - 细胞级 top1 / topK 准确率（预测落点 vs 真值 spot）；
     - spot×type 构成的 L1 / JS / 相关性；
     - 稀有类型的虚假扩散程度（false positive spots 数、分布熵）。
   - **SVG / 表达结构相关指标：**
     - SVG 基因的真实 vs 重构表达相关性分布；
     - SVG 相关的 Moran’s I / 空间自相关是否被保持或提高；
        -（可选）SVG 空间表达图的 SSIM 等图像结构指标。
   - **类型 / Unknown 行为：**
     - 在 “sc 缺 type / ST 缺 type” 的场景中：
       - 错误类型分配的比例 vs “标为 Unknown”的比例；
       - 真值不存在类型的 false positive rate；
       - Unknown 在空间上是否集中于拟合差 / 真值不支持的区域。

3. **计算 CytoSPACE 专用指标（5A 部分，只对 backend=cytospace）**

   - **成本相关：**
     - `mean_cost_per_cell`，total cost；
     - baseline vs plus 的 cost 分布变化。
   - **约束满足度：**
     - spot 总 cell 数 RMSE；
     - spot×type cell 数 RMSE；
     - 检查 plus 在“尊重 CytoSPACE 自己的数量/类型约束”上是否更好或至少不变坏。

4. **稳定性与超参数扫描（seed / 配置 / λ）**

   - 对同一场景，用不同：
     - **seed**：随机初始化；
     - **配置**：如 SVG 选基策略、是否只用插件基因、是否启用 refine 等；
     - **λ**：SVG 空间 loss 权重强弱；
   - 计算：
     - baseline / plus 在多次 run 之间的映射一致性（例如 Jaccard / overlap）；
     - SVG 指标、类型指标、cost 指标随 λ / 配置的变化曲线；
   - 找出一块“**plus 效果好、且行为稳定**”的参数区间，做成推荐（给 Stage6 用）。

**输出：**

- 对每个场景和 backend：
  - `result/<scenario_id>/stage5_eval/generic/metrics_generic.json`
  - 若 backend=cytospace：
     `result/<scenario_id>/stage5_eval/cytospace_specific/metrics_cytospace_specific.json`
- 汇总表：
  - `stage5_scenario_summary.json`：每个场景 baseline vs plus 的主指标对比；
  - `stage5_config_recommendation.json`：推荐的 λ 范围、配置组合、是否启用 refine 等。

**🔗 关键关节：**

- **→ Stage6（真实数据验证）：**
  - Stage5 的 `stage5_config_recommendation.json` 告诉 Stage6：
     “哪些 plus 配置在模拟世界（有真值）里表现最好 / 最稳”，
     Stage6 就优先在这些配置上跑 BRCA 真实数据，再用无真值指标做最终仲裁。
- **→ Stage7（论文图表）：**
  - 所有“有真值下 baseline vs plus 对比”的主结果图（accuracy/JS/Unknown 行为/稳定性曲线）都来自 Stage5；
  - 真实数据的 Stage6 结果则补上“现实可用性”的一半故事，两者在 Stage7 里一起拼成完整结果部分。

------





------

### 6️⃣ 阶段 6：真实数据实验（无真值）——SVG-aware 仲裁与全局评估

**目标：**
 在真实 BRCA（及后续扩展数据）上，在**没有 ground truth 的前提下**，利用：

- Stage2 给出的 **SVG/HVG 权重体系**；
- Stage3 的 **类型重标记 + 类型先验**；
- Stage4 的 **baseline / plus / plus-refine 多配置映射结果**；
- Stage5 推荐的 **超参数安全区间**；

对所有映射结果进行系统评估，并通过 **SVG 空间指标 + 全局指标的“仲裁规则”（Step3 落地）**，选出：

1. 论文中重点展示的 **canonical plus 映射链路**；
2. 对照的 baseline 映射；
3. 说明 **Step1+Step2（SVG 加权 cost + SVG 空间 refine）在真实数据上是否持续带来收益**。

------

**输入：**

- 真实 BRCA 的：
  - `sc_processed.h5ad`（含类型、质量控制信息）；
  - `st_processed.h5ad`（表达矩阵 + 组织切片坐标等）。
- Stage2 输出（SVG+HVG 插件）：
  - `gene_weights.csv`（含 SVG/HVG 分数 + `w_gene`）；
  - `SVG_gene_list.txt` / `plugin_genes.txt`（用于定义 SVG 指标计算的基因集合）。
- Stage3 输出：
  - `cell_type_relabel.csv`（`plugin_type`）；
  - `type_prior_matrix.csv`（spot×type 先验）。
- Stage4 输出（真实数据上的 baseline / plus / plus-refine 映射结果）：
  - `result/BRCA/stage4_mapping/...`
  - `stage4_run_manifest.csv/.json`（记录 backend / config_id / seed / λ / 是否 refine 等）。
- Stage5 输出（基于模拟的配置推荐）：
  - `stage5_config_recommendation.json`
    - 建议的 λ 范围、SVG 选基策略、是否启用 refine、推荐 seed 数量等。

------

**核心操作：**

1. **确定在真实数据上需要评估的“配置集合”**

   - 从 `stage5_config_recommendation.json` 中读取推荐的：
     - 一个或几个优选的 `config_id`（例如某种 SVG 选基策略 + cost 选择 + 是否用 refine）；
     - 一段 λ 范围（比如 [0.1, 0.5]）；
     - 建议的 seed 数量（例如每个配置跑 3~5 个 seed）。
   - 在 `stage4_run_manifest` 中筛选出符合这些条件的 run：
     - `backend == cytoSPACE`（暂时以 CytoSPACE 为主）；
     - `config_id ∈ 推荐集合`；
     - `lambda_svg ∈ 推荐范围`；
     - seed 数量满足要求；
     - 是否启用 refine（如果 Stage5 认为 refine 有稳定收益，则优先选用 refine 版本）。
   - 同时选取一小批 baseline run（原始 Cytospace，不接插件）作为对照。

2. **对每个 run 计算“无真值”下的通用评估指标**

   **2.1 表达重构 & SVG 空间模式**

   - 基因表达层面：
     - 全基因 / HVG 维度上：
       - 预测 ST 表达 vs 实测 ST 表达的相关性 / RMSE / R²；
     - **SVG 专属：**
       - 使用 Stage2 的 SVG 列表 + `w_gene`：
         - 计算每个 SVG 基因的空间相关（预测 vs 实测）；
         - SVG 的 Moran’s I / Geary’s C 等空间自相关是否被保留或增强；
         - （可选）SSIM / UQI 等图像结构指标，衡量空间 pattern 是否更贴近组织结构。
     - 输出：
       - SVG 指标的分布（中位数、均值、分位数），与 baseline 比较。

   **2.2 类型分布 & Unknown 行为**

   - 细胞类型在空间上的分布合理性（基于已有生物学知识/病理区域）：
     - 类型在肿瘤区 vs 免疫区 vs 其他区域的 enrichment（可以定义感兴趣区域或用粗略组织掩膜）；
     - 稀有类型是否只在少数合理区域出现，而不是大范围误扩散。
   - Unknown 的使用情况：
     - Unknown 占比是否过高/过低；
     - Unknown 是否集中在表达/类型拟合差的区域（例如预测表达残差高的spot）；
     - baseline vs plus：是否存在 **“原来瞎分配，现在更诚实地变成 Unknown 或被收缩掉”** 的现象。
   - 若有已知 marker 基因与类型的对应关系：
     - 在 plus 映射下，类型富集区域的 marker 表达是否更一致；
     - 可用 simple enrichment score / AUC 等方式量化。

   **2.3 CytoSPACE 专用指标（backend-specific）**

   - cost 分布（mean / median cost per cell）；
   - 约束满足情况：
     - spot 总 cell 数与预期值的 RMSE / MAE；
     - spot×type cell 数与 type_prior 之间的偏离情况；
   - baseline vs plus：
     - 检查插件是否在不违反或甚至改善约束满足度的前提下，提升 SVG/类型行为。

3. **稳定性与 SVG-aware 仲裁（Step3 真正落地）**

   - 对每个 `(config_id, λ)`，跨 seed 对比：
     - cell→spot 映射的一致性（例如每个细胞的主落点是否稳定、spot×type 组合是否稳定）；
     - SVG 空间指标的波动（SVG 空间相关 / Moran’s I / SSIM 的方差）。
   - 在推荐配置集合内部，对所有 run 建立一个“仲裁表”：
     - 行：run_id（backend+config+seed+λ）；
     - 列：
       - SVG 空间相关（均值 / 中位数）；
       - SVG Moran’s I 保持度；
       - 总体表达重构相关；
       - 类型/Unknown 指标（如 Unknown 是否合理、类型是否过度扩散）；
       - CytoSPACE cost / 约束指标。
   - 设计一个 **SVG-aware 仲裁规则**（示意）：
     - 排序优先级例如：
       1. SVG 空间相关（中位数）必须不低于 baseline 且尽可能高；
       2. 总体表达重构相关不能显著劣于 baseline；
       3. 类型/Unknown 行为“合理”（不过度 Unknown，也不过度乱分）；
       4. CytoSPACE cost/约束指标在可接受范围内；
       5. 跨 seed 的稳定性良好（指标方差较小）。
   - 根据上述规则，选出：
     - 1–2 个 **canonical plus run**（可来自不同 λ/seed 但在表现上近似）；
     - 1 个 baseline run 作为主对照；
     - 如有必要，可保留 1 个“极端 λ”结果用于展示 trade-off。

4. **结果封装：为 Stage7 图表 & 论文文本准备数据**

   - 对最终被选中的 run：
     - 将其 cell→spot 映射、spot×type 组合、重构表达矩阵等，拷贝/链接到：
       - `result/BRCA/stage6_selected/baseline/…`
       - `result/BRCA/stage6_selected/plus_best/…`
       - `result/BRCA/stage6_selected/plus_alt/…（可选）`
   - 输出一份总结 JSON：
     - `stage6_arbitration_summary.json`：
       - 记录每个 run 的 SVG / 全局指标；
       - 标记哪几个 run 被选为 final；
       - 给出仲裁规则的关键阈值与排序逻辑（方便在论文方法中复述）。
   - 为 Stage7 预先组织好若干“图表素材包”：
     - SVG 空间表达三联图（baseline / plus-best / plus-refine-best）；
     - 多 run SVG 指标分布（箱线图），标出最终选中的 run；
     - 类型分布 / Unknown 热图；
     - cost / 约束对比条形图或散点。

------

**输出：**

- 每个 run 的无真值评估结果：
  - `stage6_eval/run_{id}/metrics_generic.json`（表达 + SVG + 类型 + Unknown 等）；
  - `stage6_eval/run_{id}/metrics_cytospace_specific.json`（cost / 约束指标）。
- 整体仲裁总结：
  - `stage6_arbitration_summary.json`：
    - run 列表 + 指标 + 被选中的 canonical run ID + 仲裁规则说明。
- 最终选中映射的“集中输出”目录：
  - `stage6_selected/baseline/...`
  - `stage6_selected/plus_best/...`
  - `stage6_selected/plus_alt/...`（如保留）。
- （可选）中间诊断图：
  - SVG 空间指标分布图、类型/Unknown 分布可视化、cost 与表达相关性的散点图等。

------

**🔗 关键关节：**

- **→ 承接 Stage4 + Stage5：**
  - Stage4 提供大量 baseline / plus(+refine) 候选映射；
  - Stage5 用模拟世界筛出了“值得在真实数据上重点评估的配置区间”；
  - Stage6 在真实 BRCA 上，用无真值指标 + SVG-aware 仲裁规则，从这些候选中挑出**真正可靠的 plus 映射**。
- **→ 输往 Stage7：**
  - Stage6 给 Stage7 提供：
    - 经过仲裁的 canonical plus 映射结果；
    - baseline 对照；
    - 一整套可视化素材（SVG 模式、类型、Unknown、稳定性等）。
  - Stage7 只需要基于这些结果组织图表和文本，不再关心底层仲裁逻辑细节。

------

这样 Stage2 / Stage4 / SimGen / Stage5 / Stage6 的顶层骨架就完全闭环了：

- Step1（加权 cost）在 Stage4 落地，
- Step2（SVG 空间 refine）在 Stage4 的 plus-refine 子模块落地，
- Step3（SVG-aware 仲裁）在 Stage6 完整实现。



------

### 7️⃣ 阶段 7：可视化整合 & 论文结构

**目标：**
 把整个方法与实验结果收束成一套清晰的“论文故事”和图表。

**输入：**

- 各阶段的中间结果 & 指标 & 图

**核心操作：**

- 方法示意图：
  - 整体框架图（类似我们现在的“骨架 + 关节”）；
  - 两个插件的结构图（模块一、模块二）；
- 结果图：
  - 模拟数据对比图（baseline vs 各插件组合）；
  - 真实数据指标对比图；
  - 稳定性、类型覆盖、unknown 诊断等关键可视化。
- 论文结构：
  - Methods：整体框架 + 插件一 + 插件二 + 实验设计；
  - Results：模拟 + 真实 + ablation + 稳定性；
  - Discussion：优点、局限、未来扩展。

**🔗 关键关节：**

- 这一步是 **“把所有前面阶段产出的数据流，转成读者可理解的故事流”**，为投稿做准备。

------





## 三、简短总结

> - **骨架：**
>    0–1 搭地基；2–3 做两个通用插件（SVG+HVG、类型对齐）；4 把插件硬接到 CytoSPACE；5–7 用模拟 + 真实数据 + 可视化完整证明它们的价值。
> - **关节：**
>   - 阶段 1 的温和过滤 + 保存过滤前后基因列表，为阶段 2 的 SVG/HVG 敏感性分析提供依据；
>   - 阶段 2 通过**改写表达矩阵**（子集+加权）直接控制 CytoSPACE 的 cost matrix；
>   - 阶段 3 通过**改写 spot×type 组成矩阵**并影响候选细胞池，让类型对齐真正进入 assignment；
>   - 阶段 4 产出的 baseline vs plus 成对结果，是阶段 5–7 所有实验与图表的共同起点。











# 项目各阶段详细设计

------

## 阶段 0 总结：数据选择 + 环境验证

### 1. 阶段 0 的目标

阶段 0 只做两件事：

1. **选定后续所有阶段要用的“主数据集”**
2. **确认本地环境可以正常运行 CytoSPACE，并成功输出结果和日志**

------

### 2. 为什麽选 HER2⁺ 乳腺癌（BRCA）这套 sc + Visium 数据？

这套数据非常适合作为我们项目的“主战场”，原因主要有：

1. **有配对的 scRNA + 空间转录组**
   - 单细胞：乳腺癌的 scRNA 图谱，细胞类型丰富（肿瘤细胞、免疫细胞、基质细胞等）。
   - ST：同一 HER2⁺ 乳腺癌样本的 Visium FFPE 数据，Spot 数量足够，质量可用。
      → 完全符合我们“sc → ST 映射”的项目场景。
2. **就是 CytoSPACE 官方示例数据，格式现成**
   - 已经有 `brca_scRNA_GEP.txt / celllabels.txt / STdata_GEP.txt / coordinates.txt` 四个核心输入文件，
   - 已经成功用它跑出 `brca_assigned_locations.csv` 等结果，
      → 非常适合阶段 0 的“环境可行性测试”。
3. **空间结构和肿瘤微环境（TME）都很丰富**
   - 肿瘤区 vs 间质区 vs 免疫浸润区，空间异质性明显；
   - 病理图像、spot 坐标等信息完整。
      → 这对我们后面做 SVG、空间指标、可视化都很有利。

------

### 3. 它为什么适配我们的两个模块化设计？

#### 模块一：SVG + HVG 动态加权插件

- 我们需要：**spot×gene 表达矩阵 + 空间坐标** 来计算 SVG。
- 这套数据里有：
  - Visium 的 filtered count 矩阵（`...filtered_feature_bc_matrix.h5`），
  - spot 坐标 + 组织掩膜（`tissue_positions_list.csv` + `scalefactors_json.json`），
- 同时肿瘤样本本身空间异质性强，很多基因在肿瘤 vs 免疫/基质区域会呈现典型空间模式。

👉 所以：
 这套数据既有**技术条件**（完整 Visium 输出）又有**生物条件**（强空间异质性），
 非常适合用来设计和检验 SVG+HVG 的加权策略。

------

#### 模块二：细胞类型不匹配 / unknown-aware 插件

- 单细胞端：
  - `brca_scRNA_celllabels.txt` 提供了肿瘤、免疫、基质等多种 cell type；
  - 实际上这种肿瘤图谱里经常会有“亚型很多、边界模糊、甚至 unknown 的情况”。
- 空转端：
  - 我们可以通过 deconvolution 或其它方法为每个 spot 建一个初始 type 组成（F_base），
  - 再用模块二做类型对齐、unknown 诊断、覆盖率分析。

👉 肿瘤场景本身就是 “类型复杂 + 注释不完全 + sc/ST 类型体系不完全一致” 的典型例子，
 非常适合作为 **模块二（类型不匹配插件）** 的主要展示舞台。

------

### 4. 阶段 0 的结论

- **数据集方面：**
   我们正式选定 **HER2⁺ 乳腺癌（BRCA）sc + Visium** 作为整个项目的主数据集，
   同时它天然适配：
  - 模块一：SVG+HVG 动态加权
  - 模块二：细胞类型对齐 + unknown-aware 诊断
- **环境方面：**
   已经用这套数据成功跑通 CytoSPACE 示例，拿到了输出文件，
   → 可以认定：**阶段 0 的两个目标已经完成 ✅**，



## 阶段一详细设计

------

### 一、阶段一在整个项目里的角色

在整个项目里，阶段 1 的职责只有一句话：

> **把原始的 scRNA + ST 数据，整理成一套标准、干净、可复用的矩阵和元数据，并且把“过滤前后基因集合”记录好，供后面 SVG 插件、类型插件和 CytoSPACE 使用。**

几个关键点：

- **实现语言：** 阶段 1 的核心逻辑用 **R + Seurat** 完成（借鉴 CytoSPACE / Cell2Spatial 的实践）；
- **主控语言：** 仍然是 **Python 的 `main.py`**，通过命令行调用 R；
- **输出形式：** 一组统一命名的 **CSV/TXT 文件**，后续所有 Python 模块只认这些文件，不关心内部用的是 R。

------

### 二、阶段一的输入 / 输出设计

#### 1. 输入（来自阶段 0 准备的数据目录）

假设数据目录为：`data/brca/`，我们约定：

- `brca_scRNA_GEP.txt`
- `brca_scRNA_celllabels.txt`
- `brca_STdata_GEP.txt`
- `brca_STdata_coordinates.txt`

------

#### 2. 输出（阶段一输出目录，例如 `result/stage1/`）

**单细胞部分（sc）**

1. `sc_expression_normalized.csv`
2. `sc_metadata.csv`
3. `sc_hvg_genes.txt`

**空间 ST 部分（st）**

1. `st_expression_normalized.csv`
2. `st_coordinates.csv`

**SVG 敏感性埋点（关键）**

1. `st_all_genes.txt`
   - 空间数据在任何基因过滤之前的**完整基因列表**
2. `st_filtered_genes.txt`
   - 经过我们“温和过滤 + HVG 等操作之后仍然保留的基因列表”
   - 专门用来在阶段 2 对比“有多少潜在 SVG 被过滤掉”

------

### 三、与 main.py 的接口设计

在阶段 1 上，`main.py` 负责两件事：

1. 决定 **过滤温和程度**（比如 mild / medium / strict），转换成一个字符串参数；
2. 用 `subprocess` 调用 R 脚本 `preprocess_stage1.R`，并把：
   - `data_dir`
   - `out_dir`
   - `filter_mode`
      传进去。

示意设计：

```python
# src/pipeline.py (举例)
import subprocess
import os

def run_stage1_preprocess(data_dir: str, out_dir: str, filter_mode: str = "mild"):
    """
    调用 R 的阶段一预处理脚本。
    filter_mode: 'mild' / 'medium' / 'strict'
    """
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "Rscript",
        os.path.join("scripts", "preprocess_stage1.R"),
        "--data_dir", data_dir,
        "--out_dir", out_dir,
        "--filter_mode", filter_mode,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # 把 R 的输出保存在日志里
    with open(os.path.join(out_dir, "stage1_preprocess.log"), "w", encoding="utf-8") as f:
        f.write(result.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Stage 1 preprocess failed, see {out_dir}/stage1_preprocess.log")
```

然后在 `main.py` 顶层，解析命令行参数：

```python
# main.py (结构示意)
import argparse
from src.pipeline import run_stage1_preprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--work_dir', type=str, default='result')
    parser.add_argument('--filter_mode', type=str, default='mild', choices=['mild', 'medium', 'strict'])
    args = parser.parse_args()

    stage1_dir = os.path.join(args.work_dir, "stage1")
    run_stage1_preprocess(data_dir=args.data_dir,
                          out_dir=stage1_dir,
                          filter_mode=args.filter_mode)

if __name__ == '__main__':
    main()
```

> 这样：
>
> - 你只要在命令行跑：
>    `python main.py --data_dir data/brca --filter_mode mild`
> - 过滤温和程度自动传给 R，且所有输出在 `result/stage1` 目录里。

------

### 四、R 脚本内部设计

我们现在说的是 **`preprocess_stage1.R` 的“结构和逻辑”**，不是逐行代码。

#### 1. 步骤 0：解析命令行参数

使用 `optparse`（或 `argparse` for R）：

- 接收：
  - `--data_dir`
  - `--out_dir`
  - `--filter_mode`（mild / medium / strict）

逻辑：

1. 从 `opt$data_dir` 读取四个输入文件；
2. 确保 `opt$out_dir` 存在（`dir.create`）；
3. 把参数内容写一段进日志（方便追踪）。

#### 2. 步骤 1：将 filter_mode 转换为具体过滤阈值（温和度设计）

我们之前讨论的“温和过滤”在这里正式写进设计。

设计一个函数：

- 输入：`filter_mode`（字符串）
- 输出：`min_features`、`min_counts` 两个阈值

示例设计（可再调）：

| 模式   | min_features | min_counts | 含义                     |
| ------ | ------------ | ---------- | ------------------------ |
| mild   | 100          | 300        | 非常温和，只去掉极差细胞 |
| medium | 200          | 500        | Seurat 常用阈值附近      |
| strict | 500          | 1000       | 比较严格，追求高质量     |

伪代码：

```r
get_filter_params <- function(filter_mode) {
  if (filter_mode == "mild") {
    list(min_features = 100, min_counts = 300)
  } else if (filter_mode == "medium") {
    list(min_features = 200, min_counts = 500)
  } else if (filter_mode == "strict") {
    list(min_features = 500, min_counts = 1000)
  } else {
    stop(paste("Unknown filter_mode:", filter_mode))
  }
}
```

> 这样，**“过滤温和程度”这个抽象概念被设计成一个很清晰的接口**：
>  Python → 传一个字符串；
>  R → 查表得出具体数值 → 用在 subset 里。

#### 3. 步骤 2：scRNA 预处理逻辑

1. **读取矩阵 + metadata：**
   - `brca_scRNA_GEP.txt` → `sc_mat`
   - `brca_scRNA_celllabels.txt` → `celltype`
2. **CreateSeuratObject + 过滤：**
   - 使用 `min_features` 和 `min_counts` 为阈值做细胞过滤；
   - 这是“温和”的地方。
3. **归一化：**
   - `NormalizeData(sc_obj)`（例如默认 log-normalize）
4. **HVG 选择：**
   - `FindVariableFeatures(sc_obj, selection.method = 'vst', nfeatures = 2000)`
   - 得到 `VariableFeatures(sc_obj)`，输出到 `sc_hvg_genes.txt`
5. **导出：**
   - `GetAssayData(sc_obj, slot='data')` → `sc_expression_normalized.csv`
   - `meta.data` → `sc_metadata.csv`（其中包含 celltype）

**注意**：
 这里我们**不做任何 SVG 相关的事**，SVG 的计算发生在阶段 2。
 阶段 1 的职责：准备好干净的矩阵 + HVG 信息。

#### 4. 步骤 3：ST 预处理逻辑 + SVG 敏感性埋点

1. **读取 ST 表达 & 坐标：**
   - `brca_STdata_GEP.txt` → `st_mat`
   - `brca_STdata_coordinates.txt` → `coords`
2. **记录“所有基因”（敏感性分析用）：**
   - `rownames(st_mat)` 写进 `st_all_genes.txt`
   - 这是“**过滤前**”的基因集合 G_before。
3. **创建 Seurat 对象 + 稍微过滤 spot（可选）：**
   - 根据总 counts 过滤极低质量 spot（可以用固定阈值或与 filter_mode 共用简化版），但这部分可以非常温和，甚至在 V1 可以不开。
4. **标准化 + HVG：**
   - `NormalizeData(st_obj)`
   - `FindVariableFeatures(st_obj, ...)`
   - 得到 `VariableFeatures(st_obj)` 和 `rownames(st_obj)` 的当前集合
5. **记录“过滤后基因集合”：**
   - `rownames(st_obj)` 写成 `st_filtered_genes.txt`
   - 这就是我们后面要对比的 G_after。
6. **导出：**
   - 归一化表达矩阵 → `st_expression_normalized.csv`
   - 坐标表整合 → `st_coordinates.csv`

**关键设计思想**：

> 阶段 1 不直接计算 SVG，但通过 “`st_all_genes.txt` vs `st_filtered_genes.txt`”
>  为阶段 2 提供了一个“**过滤对 SVG 潜在候选基因影响**”的窗口。
>  后面我们可以统计：
>  有多少未来可能被判为 SVG 的基因其实在阶段 1 就被扔掉了。

------

### 五、给后续阶段的“契约”：阶段一保证什么？

从阶段 2 开始，你的所有 Python 模块，都可以假设：

1. 在 `result/stage1/` 下存在一整套标准化文件：

   - `sc_expression_normalized.csv`
   - `sc_metadata.csv`
   - `sc_hvg_genes.txt`
   - `st_expression_normalized.csv`
   - `st_coordinates.csv`
   - `st_all_genes.txt`
   - `st_filtered_genes.txt`

2. 这些文件的特点：

   - **基因命名统一**：
      sc 和 st 都来自同一套基因名体系，可以用 gene 交集做统一。
   - **过滤程度可控**：
     - 过滤强度通过 `filter_mode` 已记录在日志中；
     - 如果将来你发现“过滤太狠了”，可以简单地改 `filter_mode` 并重跑阶段 1。
   - **SVG 敏感性信息完备**：
     - 任何阶段 2 的 SVG 分析，都可以拿 G_before vs G_after 看“损失了多少潜在 SVG”。

3. 对 main.py 来说：

   - 阶段 1 只是

     > “调用一个黑盒 R 模块，传入 `filter_mode`，得到一堆约定好的文件”。

   - 你后续如果想把预处理改成 100% Python 也没问题，只要保证**输出文件格式不变**，下游就完全不用改。







------

## 阶段二：SVGs

------

### 一、阶段二在项目里的位置（一句话版）

> **阶段二 = 在统一过滤后的基因集合上，为每个基因打一个“HVG+SVG 综合分数”，得到 `gene_weights.csv` 和 `G_plugin`，既作为 Stage4 里 \*加权 cost（Step1）\* 的根基，也作为 Stage5/6 里 \*SVG 空间指标 + 仲裁（Step3）\* 的评分基准。**

- HVG：这个基因在“细胞之间”差异大不大？（分细胞类型厉不厉害）
- SVG：这个基因在“空间上”是不是成片亮？（有空间结构没有）
- 最终：
  - 得到每个基因的 `final_weight(g)`；
  - 选出一批插件基因集合 `G_plugin`；
  - 后面 **CytoSPACE-plus 会在 cost 里按 `final_weight` 放大这些基因（Step1）**；
  - 同时 Stage5/6 会用 `SVG_score` / `final_weight` 来衡量“哪条映射链路在 SVG 维度上最好”（Step3）。

------

### 二、阶段二的输入 / 输出（超简表）

**输入（来自阶段一 & 数据准备）：**

1. `sc_expression_normalized.csv`
   - 行：cell，列：gene（已做基本归一化）
2. `st_expression_normalized.csv`
   - 行：spot，列：gene（与 sc 取交集）
3. `sc_meta.csv`
   - 包含 cell_type / 质控信息 / 是否保留等
4. `st_coordinates.csv`
   - 每个 spot 的 (x, y)，用于算 SVG
5. `sc_hvg_genes.txt`
   - Seurat/R 预处理输出的 HVG 列表（仅当参考）

**输出（给后续阶段用）：**

1. `gene_weights.csv`
   - 每个基因一行，主要字段：
     - `gene`：基因名
     - `hvg_score`：HVG 原始分数
     - `svg_score`：SVG 原始分数
     - `hvg_norm`：0~1，HVG 归一化
     - `svg_norm`：0~1，SVG 归一化
     - `final_weight`：综合权重 **（Stage4 构造加权 cost 的真正系数）**
     - `selected_in_plugin`：是否被选入 `G_plugin`（0/1）
     - `is_sc_hvg`：是否为 Seurat HVG（0/1）
2. `plugin_genes.txt`
   - 插件基因集合 `G_plugin`：按 `final_weight` 排序取 Top-K。
3. `svg_filter_sensitivity.json`（监控用）
   - 简单记录：前 N 个最强 SVG 候选里，有多少被阶段一过滤掉了。

> 后面：
>
> - Stage4 会读取 `gene_weights.csv` + `plugin_genes.txt`，在 **构造 cell–spot cost 矩阵时对 SVG 加权（Step1）**；
> - Stage5/6 会读取 `svg_score` / `final_weight`，在 **做 SVG 空间指标与多 run 仲裁（Step3）** 时使用同一套评分体系。

------

### 三、核心思路（超抽象版）

你可以把阶段二想成 3 步：

1. **在公共基因集合上，分别算 “HVG 分数” 和 “SVG 分数”。**
2. **把 HVG / SVG 分数归一化后用 (α, β) 混合，得到综合权重 `final_weight(g)`。**
3. **按 `final_weight` 选出 Top-K 基因作为插件集合 `G_plugin`，并把这些信息写成标准文件，交给 Stage4/5/6。**

所以阶段二不是一个“黑盒子”，而是一个：

> **“基因评分 + 选基 + 权重导出模块”，
>  一头接收 Stage1 的 sc/ST 表达，一头输出给 Stage4 cost 和 Stage5/6 指标。**

------

### 四、算法流程（简单版拆解）

#### Step 0：加载数据 + 找公共基因

- 从阶段一目录读入 `sc_expr`、`st_expr`、`sc_meta`、`st_coords` 等文件；
- 取 sc/ST 公共基因集合：

```python
G = sorted(set(sc_expr.columns) & set(st_expr.columns))
```

后续所有计算只在 G 上做。

------

#### Step 1：计算 HVG 分数（看细胞差异）

**直观：**
 看这个基因在不同细胞之间是不是“变化很大”。

做法（示意）：

1. 对每个基因算方差（在所有 cell 上）：

```python
var_sc = sc_expr[G].var(axis=0)
```

1. 简单平滑 + 排序归一化：

```python
hvg_raw = np.log1p(var_sc)                                 # 原始 HVG 分数
hvg_norm = hvg_raw.rank(method="average") / len(hvg_raw)   # 0~1
```

1. 用 `sc_hvg_genes.txt` 标记 `is_sc_hvg`（仅记录，不强行改分）。

> 注：这里不需要把 HVG 做得特别复杂，关键是提供一个**稳定可调的“细胞变异度”参考分数**，后面通过 (α, β) 混合来调节 HVG vs SVG 的权重。

------

#### Step 2：计算 SVG 分数（看空间结构）

**直观：**
 看这个基因在组织上是不是“某一块区域亮、另一块区域暗”。

做法（示意）：

1. 根据 `st_coordinates.csv` 构建邻居关系（比如 k=6 的 kNN）：
   - 每个 spot 有一串“邻居 spot”。
2. 对每个基因 g：
   - 拿它在所有 spots 上的表达向量 `x_g`；
   - 如果这个基因几乎都不表达（比如只有 <5% spot 非零）→ 直接记为 `svg_score=0`；
   - 否则用一个简单的空间自相关统计（例如 Moran’s I / Geary’s C）：

```python
svg_raw[g] = morans_I(x_g, neighbors)
```

1. 对所有 `svg_raw` 做排序归一化：

```python
svg_norm = svg_raw.rank(method="average") / len(svg_raw)   # 0~1
```

> 这里的 `svg_norm` 就是 **“在当前 ST 上，这个基因的空间性有多强”**。
>  它一方面会进入 `final_weight`，另一方面会在 Stage5/6 的 **SVG 空间重构指标** 中作为权重或基因筛选标准。

------

#### Step 3：融合 HVG / SVG，得到 final_weight

**直观：**
 我们不想只看 HVG，也不想只看 SVG，而是用一个可调的混合：

[
 \text{final_weight}(g) = \alpha \cdot \text{hvg_norm}(g) + \beta \cdot \text{svg_norm}(g)
 ]

- Stage2 的脚本里会暴露超参数：
  - `alpha`（偏重 HVG 时大一点）
  - `beta`（偏重 SVG 时大一点）
- 默认可以设成 α=β=0.5，后续在 Stage5/6 里探索不同组合时再调。

示意代码：

```python
final_weight = alpha * hvg_norm + beta * svg_norm
```

最后，将所有基因整理成 `gene_weights.csv`：

- `gene`
- `hvg_score`, `svg_score`
- `hvg_norm`, `svg_norm`
- `final_weight`
- `is_sc_hvg` 等辅助标记

> 这一列 `final_weight` 是后面整个项目的关键点之一：
>
> - 在 Stage4 中，它决定 **成本函数里哪几个基因被放大（Step1）**；
> - 在 Stage5/6 中，它决定 **SVG 空间指标里哪些基因更被“当一回事”（Step3）**。

------

#### Step 4：选插件基因集合 G_plugin（Top-K）

用一个最简单稳妥的策略：

1. 按 `final_weight` 从大到小排序；
2. 取前 K 个基因作为 `G_plugin`，K 为命令行参数 `--svg_topk`（默认如 1000）。

```python
plugin_genes = final_weight.sort_values(ascending=False).index[:topk]
```

然后：

- 写入 `plugin_genes.txt`；
- 在 `gene_weights.csv` 中给这些基因打标 `selected_in_plugin = 1`。

> 后面 Stage4 的 plus 分支会优先在 `G_plugin` 上构造加权 cost，
>  若配置允许，也可在 `G_plugin` 之外保留一小部分其他基因作为“背景”。

------

#### Step 5：SVG 过滤敏感性（简单版）

用来回答：**“阶段一过滤有没把好多 SVG 候选滤掉？”**

简版流程：

1. 从原始 `brca_STdata_GEP.txt` 算一次“全基因 SVG 分数”（粗略版）。
2. 取前 N 个 SVG 最强基因（比如 N=500）。
3. 看这 500 个里面，有多少**不在** `st_filtered_genes.txt` 里 → 就是被过滤掉的 SVG 候选数。
4. 把数字和部分基因名写进 `svg_filter_sensitivity.json`。

之后你就可以看：

- 如果“被过滤掉的潜在 SVG 太多” → 说明 Stage1 过滤太狠，需要回去调；
- 如果“损失很少” → Stage1 对 SVG 基本友好，可以放行。

> 这个模块本身不直接影响 Stage4/5/6 的运算结果，
>  但它是保护 SVG 信号不被 Stage1 误杀的一个“报警器”。

------

### 五、和 main.py / 阶段四 / 阶段五 / 阶段六的关系（简单版）

#### 1）main.py 如何调用阶段二？

在 `main.py` 里，可以有参数：

- `--enable_svg_plugin`：开关
- `--svg_alpha`、`--svg_beta`：HVG / SVG 权重
- `--svg_topk`：选多少个基因

示意：

```python
if args.enable_svg_plugin:
    run_stage2_svg(
        data_dir=args.data_dir,
        stage1_dir=stage1_dir,
        out_dir=stage2_dir,
        alpha=args.svg_alpha,
        beta=args.svg_beta,
        topk=args.svg_topk,
    )
```

#### 2）阶段四（映射）如何用阶段二结果？——对应 Step1：加权 cost

在阶段四的 CytoSPACE-plus 分支中：

1. 读入：
   - `sc_expression_normalized.csv`、`st_expression_normalized.csv`
   - `plugin_genes.txt`
   - `gene_weights.csv`
2. 根据插件基因和权重构造“加权表达空间”，然后在此基础上构造 **加权 cost 矩阵**：

```python
plugin_genes = ...  # 从 plugin_genes.txt 读
weights = ...       # 从 gene_weights.csv 读 final_weight

# 1）表达空间加权（用于 CytoSPACE 内部的相似度计算）
X_sc = sc_expr[plugin_genes].values * weights.loc[plugin_genes, "final_weight"].values
X_st = st_expr[plugin_genes].values * weights.loc[plugin_genes, "final_weight"].values

# 2）在构造 cell–spot cost 时，使用 Weighted Pearson / Weighted L2，
#    实质上就是用这些权重去“放大 SVG/HVG 合成高分基因”的贡献。
```

1. 用 `X_sc` 和 `X_st` 去跑 CytoSPACE-plus 映射（plus 分支）：
   - baseline 分支仍然可以用未加权的 HVG 或全基因，作为对照。

> 这样就保证了：
>
> - 阶段二输出的 `final_weight` **真正写进了 cost 函数（Step1）**，
> - 而不是只在“输入矩阵”层面做一个轻飘飘的装饰。

#### 3）阶段五 / 六如何用阶段二结果？——对应 Step3：SVG-aware 仲裁基准

- Stage5（模拟数据，有真值）：
  - 会读取 `gene_weights.csv` 和 Stage2 中记录的 `svg_score` / `final_weight`；
  - 在比较 baseline vs plus / plus-refine 时：
    - 专门统计 **SVG 基因（或者按 `final_weight` 加权的基因集合）** 的空间重构质量；
    - 在多 seed / 多配置 / 多 λ 的 run 之间，通过这些 SVG 指标来判断：
      - 哪些配置在“有真值世界里”对 SVG pattern 的重构最好、最稳定。
- Stage6（真实数据，无真值）：
  - 同样会以 Stage2 定义的 SVG 列表 + `final_weight` 为评分基准；
  - 在多 run 之间计算：
    - SVG 空间相关 / Moran’s I 保持度 / SSIM 等；
  - 再结合表达重构、类型/Unknown 行为、CytoSPACE cost/约束等指标，完成 **SVG-aware 的最终仲裁（Step3）**。

> 换句话说：
>  **阶段二定义了一套“我们认为什么是有空间价值的基因”的统一尺度，
>  这套尺度既驱动了 Stage4 的 SVG-aware cost，又驱动了 Stage5/6 的 SVG-aware 评价与选解。**











## 阶段三：细胞类型匹配问题

------

### 一、阶段三一句话概括

> **阶段三 = 细胞类型不匹配 / unknown-aware 插件**
>  做的事就是：
>
> - 看 sc 里的每种 cell type，**在 ST 里到底有没有“证据”**；
> - 对几乎不出现的类型做“Unknown 合并”；
> - 给 CytoSPACE-plus 提供一套 **更新后的类型标签 + spot×type 先验矩阵**。

简单理解：

> 我们不再盲目相信原始 celltype，而是“问一遍空间数据：
>  这些类型你认不认？”

------

### 二、阶段三用什么、产出什么？（输入 / 输出简表）

#### 输入（来自前面阶段）

来自阶段 1：

- `sc_expression_normalized.csv`：单细胞归一化表达矩阵
- `sc_metadata.csv`：包含 `cell_id` 和 `celltype`（可以有 unknown）
- `st_expression_normalized.csv`：空间归一化表达矩阵
- `st_coordinates.csv`：ST 坐标

来自阶段 2：

- `plugin_genes.txt`：插件基因集合（G_plugin）
- `gene_weights.csv`：每个基因的 `final_weight`（可用于加权相似度）

> 阶段三会只在这些 `plugin_genes` 上做类型匹配，保证和映射使用的是同一特征空间。

------

#### 输出（给阶段四用）

输出目录例如：`result/stage3_typematch/`，主要有三类文件：

1. **`type_support.csv`**
    每个原始 `celltype` 一行，告诉我们：

   - 这个类型在 sc 里有多少细胞；
   - 它在 ST 里有没有对应的区域（support_score 高 / 中 / 很低）；
   - 给出一个标签：`strong / weak / unsupported`。

2. **`cell_type_relabel.csv`**
    每个细胞一行，重写过的类型标签：

   | cell_id | orig_type | plugin_type     | status         |
   | ------- | --------- | --------------- | -------------- |
   | c1      | T_cell    | T_cell          | kept           |
   | c2      | rare_A    | Unknown_sc_only | unknown_merged |
   | c3      | B_cell    | B_cell          | downweighted   |

   - **`plugin_type`**：阶段三之后真正要给 CytoSPACE-plus 用的类型
   - **status**：标记是保留、削弱、还是合并进 Unknown

3. **`type_prior_matrix.csv`**

   - 一个 **spot × plugin_type** 的矩阵

   - 代表：每个 spot 对各个类型的“相似度/先验概率”

   - 比如某行：

     | spot_id | B_cell | T_cell | Myeloid | Unknown_sc_only |
     | ------- | ------ | ------ | ------- | --------------- |
     | spot_01 | 0.50   | 0.30   | 0.10    | 0.10            |

   这会直接作为 **CytoSPACE-plus 的“类型约束输入”**。

------

### 三、思路核心（把复杂的拆成 3 步）

你可以把阶段三想成一个“三步走”：

#### 第一步：给每种 cell type 做一个“平均画像”

- 对 sc 数据：
  - 把所有细胞按 `celltype` 分组；
  - 在 `plugin_genes` 这些基因上求均值 → 得到每个类型的**平均表达向量**。

可以理解为：

> “T_cell 大概长这样、B_cell 大概长这样、Myeloid 大概长这样……”

后面统一用这些“类型画像”去和 ST 匹配。

------

#### 第二步：问 ST：“你哪里像这些类型？”

分两层：

1. **ST 做一个粗聚类（可选但推荐）**
   - 用 `st_expression_normalized` 在 `plugin_genes` 上对 spot 做聚类（如 K=30）；
   - 计算每个 cluster 的平均表达 profile。
2. **算相似度：sc 类型 ↔ ST cluster / spot**
   - 用加权相关 / 余弦相似度，衡量：
     - 某个 sc-type 的平均表达，和某个 ST cluster（或 spot）的表达像不像。
   - 对每个 sc-type 取它在 ST 中“最像的几个 cluster”的相似度平均，得到一个 `support_score`。

然后给每个原始 `celltype` 一个等级：

- `strong`：在 ST 中明显有 “像它的区域”；
- `weak`：有一点像，但不算很强；
- `unsupported`：在 ST 中几乎找不到像它的区域。

这些结果写进 `type_support.csv`。

------

#### 第三步：重写标签 + 构建 spot×type 先验

##### 3.1 重写单细胞类型标签（unknown-aware）

根据 `support_category`：

- **strong 类型**：
  - 保留原始类型：
    - `plugin_type = 原类型名`
    - `status = kept`
- **weak 类型**：
  - 仍然保留这个类型，但将在先验中稍微减重：
    - `plugin_type = 原类型名`
    - `status = downweighted`
- **unsupported 类型**：
  - 认为当前 ST 样本基本不存在这种类型：
    - 统一并入一个未知类别：
      - `plugin_type = "Unknown_sc_only"`
      - `status = unknown_merged`

再加上原始本来就标 `unknown` 的细胞，也一并纳入 Unknown 类别（具体命名，你可以后面统一一下）。

最终得到 `cell_type_relabel.csv`，里面的 `plugin_type` 是阶段之后要真正使用的类型体系。

------

##### 3.2 构建 spot×plugin_type 的先验矩阵

接下来要给每个 spot 一个“对各类型的偏好分布”，变成 `type_prior_matrix.csv`：

大致做法：

1. 对每个 spot s 和每种 `plugin_type` t：

   - 知道 `x_s` = 该 spot 在 `plugin_genes` 上的表达向量；
   - 知道 `profile_sc[t]` = 该类型的平均表达向量；
   - 算一个相似度 `sim[t, s]`。

2. 对同一个 spot s，把所有 `sim[t, s]` 做非负化 + 归一化：

   ```python
   sim_pos = max(sim[t, s], 0)
   prior[t, s] = sim_pos / sum_t sim_pos
   ```

3. 如果这个 spot 对所有已知类型都不太像（总和太低）：

   - 就人为给 `Unknown` 一定的基础占比（比如≥0.3），剩下再按比例分给其他类型；
   - 这样我们给 ST 中“解释不了的模式”留了空间。

最后得到一个 matrix：
 **行 = spot，列 = plugin_type，值 = 先验权重**。
 写成 `type_prior_matrix.csv`。

> 阶段四在构建 CytoSPACE-plus 的约束时，就不再使用原始的 spot×type 先验，而是用这一张矩阵。

------

### 四、和 main.py / 阶段四的关系（概括）

#### main.py 这边

可以有这样的参数：

- `--enable_typematch_plugin`：是否启用阶段三
- `--typematch_strong_th` / `--typematch_weak_th`：类型支持度分层的阈值
- `--st_cluster_k`：ST 聚类数 K

调用大概类似：

```python
if args.enable_typematch_plugin:
    run_stage3_typematch(
        stage1_dir=stage1_dir,
        stage2_dir=stage2_dir,
        out_dir=stage3_dir,
        strong_th=args.typematch_strong_th,
        weak_th=args.typematch_weak_th,
        st_cluster_k=args.st_cluster_k,
    )
```

#### 阶段四（映射）这边

CytoSPACE-plus 需要用到两样东西：

1. **细胞的最终类型标签**
   - 从 `cell_type_relabel.csv` 读取 `plugin_type`；
   - 以后所有“类型”的概念，都用这个重写后的 label。
2. **spot×type 的先验矩阵**
   - 从 `type_prior_matrix.csv` 读取；
   - 在构建优化目标时，当作每个 spot 类型组成的“软约束 / 先验”。

> 这样，第二个插件就通过 “**改类型标签 + 提供类型先验**” 两条通路，真实影响了映射，而不是停留在分析层。

------









------

## 阶段四：映射编排与多后端（以 CytoSPACE 为首个 backend）

> 关键词：**插件通用 + backend 可插拔 + SVG 真正进入 cost + SVG 空间重构 refine**

------

### 一、阶段四的定位（重新强调）

从全局看：

- 阶段 2：**基因插件（SVG+HVG）**
   → 输出 `plugin_genes` + `gene_weights`（含 HVG_score / SVG_score / final_weight）
- 阶段 3：**类型插件**
   → 输出 `plugin_type` + `type_prior_matrix`
- **阶段 4：不是“CytoSPACE 阶段”，而是“映射编排 + 后端适配层 +（对 CytoSPACE 来说）SVG-aware refine 层”**

#### 阶段四的核心职责

1. 定义一个**统一的“映射后端接口”**（`MappingBackend` 抽象类）。
2. 把阶段 2/3 的插件输出，**严格注入**到后端的 baseline / plus 两条映射分支：
   - baseline：不看插件，原汁原味的后端；
   - plus：必须显式使用 `plugin_genes/gene_weights` + `plugin_type/type_prior_matrix`。
3. 对 CytoSPACE 这一后端，在 plus 分支内部做两件关键事：
   - **Step1：在 cost 里给 SVG 加权**（用 Stage2 产出的 `final_weight`，构造加权表达空间并生成 cost 矩阵 `C_plus`）；
   - **Step2：在初次映射后做 SVG 空间重构 refine**（用 SVG 基因的空间模式 + 邻域平滑，对高不确定性细胞做局部微调）。
4. 保持统一输出格式，为阶段 5 / 6（模拟 + 真实评估、以及 Step3 的 SVG 仲裁）提供输入。
5. 在编排层上支持**不同 seed / 不同 SVG 权重 λ 等配置**，方便上层脚本用多次运行来做 “多配置候选 → 阶段 5 / 6 仲裁”。

所以，设计上要分清三层：

- **插件层（Stage2/3）**：只产生通用“权重 + 类型 + 先验”的中间结果；
- **编排层（Stage4 Orchestrator）**：负责“跑哪几个 backend、以什么配置跑、结果写到哪”；
- **后端层（backends）**：每个映射方法自己的实现，**遵守统一输入/输出接口**。
   对 CytoSPACE 来说，plus 分支内部再细分为：
   **cost 加权 → 初次 assignment → SVG 空间 refine**。

------

### 二、阶段四总体架构

可以画成这样一个逻辑图（文字版）：

```text
        [阶段1 输出]
             │
        [阶段2 插件] ──► plugin_genes + gene_weights
             │
        [阶段3 插件] ──► plugin_type + type_prior_matrix
             │
             ▼
   ┌───────────────────────┐
   │  Stage4 Orchestrator  │
   │  (run_stage4_mapping) │
   └─────────┬─────────────┘
             │
   ┌─────────┴─────────────────────────────────────────────┐
   │ backends = ["cytospace", "tangram", ...]             │
   │                                                      │
   │  ┌────────────────────┐   ┌──────────────────┐       │
   │  │ CytoSPACEBackend   │   │ TangramBackend   │  ...  │
   │  │  - run_baseline    │   │  - run_baseline │       │
   │  │  - run_plus        │   │  - run_plus     │       │
   │  └────────────────────┘   └──────────────────┘       │
   └──────────────────────────────────────────────────────┘
```

#### 2.1 Orchestrator 做的事（核心）

- 按 `backends` 列表依次调用各个 backend；
- 对每个 backend：
  - 跑 baseline 分支（不接插件）；
  - 跑 plus 分支（接入 Stage2/3 插件，并为 CytoSPACE 启用 SVG-aware cost + refine）；
- 把输出写入标准路径，例如：

```text
result/
  stage4_mapping/
    cytospace/
      baseline/
      plus_svg_type/          # CytoSPACE-plus（含 SVG cost + refine 后的最终结果）
    tangram/
      baseline/
      plus_svg_type/
```

> **注意：Orchestrator 不关心“内部怎么跑 CytoSPACE/Tangram”，
>  只负责把正确的参数和路径传给 backend，拿回“三件套” + meta。**

#### 2.2 关于“多配置 / 多 seed / 多 λ”的支持方式（给后面仲裁用）

阶段四本身是**“给定一组配置，产出一组映射结果”**。
 “多配置 / 多 seed / 多 λ”的思路是：

- 在 `backend_configs["cytospace"]["plus"]` 里允许传入：
  - `seed`：控制 CytoSPACE 的随机性；
  - `svg_refine_lambda`：控制 SVG 空间重构 loss 在 refine 中的权重；
  - （将来可扩展：`neighbor_radius`、`max_refine_iter` 等）。
- 上层脚本可以：
  - 要么**多次调用 Stage4**，每次给一个不同 `(seed, svg_refine_lambda)`，输出到不同的 `work_dir` / `sample_id`；
  - 要么扩展 Orchestrator，在一次调用内部对一组 `run_configs` 做 for-loop（这属于实现细节，可以后续在代码层拓展）。

> 关键点是：
>  **Stage4 的接口已经暴露出 `seed` 和 `svg_refine_lambda` 等参数，
>  所以后续要做 “多配置候选 → 阶段 5/6 仲裁” 是可行的。**

------

### 三、上游输入与标准输出（只讲“接口”，不讲细节）

#### 3.1 上游输入（来自阶段 1/2/3）

- 阶段 1：
  - `sc_expression_normalized.csv`
  - `st_expression_normalized.csv`
  - `sc_metadata.csv`（含 `cell_id`, `celltype`）
  - `st_coordinates.csv`
  - （可选）`spot_cell_counts.csv`
- 阶段 2：
  - `plugin_genes.txt`
  - `gene_weights.csv`（至少包含：`HVG_score`, `SVG_score`, `HVG_norm`, `SVG_norm`, `final_weight`）
- 阶段 3：
  - `cell_type_relabel.csv`（含 `plugin_type`）
  - `type_prior_matrix.csv`（spot×plugin_type）

> 这一组就是**插件层与映射层之间的“稳定 API”**，
>  不管下面是 CytoSPACE 还是 Tangram，都用这几个文件。

#### 3.2 标准输出（所有 backend 必须输出的“三件套” + 可选 meta）

每个 backend，在 `baseline/` 和 `plus_svg_type/` 下都要输出：

1. `cell_assignment_*.csv`
   - 每行一个细胞，至少要有：
     - `cell_id`
     - `spot_id`
   - 推荐附带：
     - `type`（baseline 用原类型，plus 用 plugin_type）
     - `backend`（如 `"cytospace"`）
     - `mode`（`"baseline"` 或 `"plus"`）
     - `assign_score`（成本 / 打分）
2. `cell_spot_matrix_*.npz`
   - cell×spot 的指派矩阵（soft 或 one-hot）。
3. `spot_type_fraction_*.csv`
   - 每行一个 spot，列为各类型的占比（baseline 为原类型，plus 为 plugin_type）。

> **评估阶段（5/6）只需要这“三件套”就能做通用指标和 SVG 仲裁。**

**推荐（可选）的 meta 输出：**

- `mapping_meta_*.json`
  - 记录本次运行的关键配置，例如：
    - `backend`: `"cytospace"`
    - `mode`: `"plus"`
    - `seed`
    - `svg_refine_lambda`
    - `cost_metric`
    - `enable_svg_refine` 等。
- `svg_refine_stats_*.json`（仅 CytoSPACE-plus）
  - 记录 refine 前后 SVG 重构指标（例如平均 SVG gene 的相关性、loss 前后变化等），
     供阶段 6 解释用（不是 Stage5/6 必须依赖的输入，只是诊断信息）。

------

### 四、后端接口：MappingBackend 抽象类

这是“你要给其他方法留好的接口”的核心。

#### 4.1 抽象类定义（伪代码）

```python
class MappingBackend:
    """
    所有映射方法（CytoSPACE, Tangram, Cell2location ...）的统一接口。
    """

    def __init__(self, name: str):
        self.name = name  # 如 "cytospace", "tangram"

    # ------- baseline: 不使用插件 --------
    def run_baseline(
        self,
        stage1_dir: str,     # only stage1
        out_dir: str,        # e.g. result/stage4_mapping/cytospace/baseline
        config: dict | None = None,
    ) -> None:
        raise NotImplementedError

    # ------- plus: 使用 Stage2/3 插件 +（可选）SVG refine --------
    def run_plus(
        self,
        stage1_dir: str,     # 共用
        stage2_dir: str,     # gene_weights/plugin_genes
        stage3_dir: str,     # plugin_type/type_prior_matrix
        out_dir: str,        # e.g. result/stage4_mapping/cytospace/plus_svg_type
        config: dict | None = None,
    ) -> None:
        """
        要求：
        - 必须显式使用：
          1) plugin_genes + gene_weights.final_weight 影响表达空间 / cost 结构
          2) cell_type_relabel.csv → plugin_type
          3) type_prior_matrix.csv → spot×type 先验
        - 输出统一的“三件套”结果到 out_dir。
        - 对支持 SVG refine 的 backend（目前是 CytoSPACE）：
          - 可以从 config 中读取:
            * seed
            * svg_refine_lambda 等参数
          - 在初次 assignment 后执行可选的 SVG 空间重构 refine，
            再写出“三件套”与可选的 svg_refine_stats.json。
        """
        raise NotImplementedError
```

#### 4.2 将来新增 backend 时，只需做什么？

比如你以后想加 `TangramBackend`：

```python
class TangramBackend(MappingBackend):
    def __init__(self):
        super().__init__(name="tangram")

    def run_baseline(self, stage1_dir, out_dir, config=None):
        # 1. 用stage1输出构建Tangram需要的输入
        # 2. 按Tangram默认流程跑一次映射
        # 3. 把结果转换成 "三件套" 格式写入 out_dir
        ...

    def run_plus(self, stage1_dir, stage2_dir, stage3_dir, out_dir, config=None):
        # 1. 用stage2：plugin_genes + final_weight 约束 Tangram 基因空间 / loss
        # 2. 用stage3：plugin_type/type_prior_matrix 影响 Tangram 的类型先验
        # 3. 若将来也想做 SVG refine，可仿照 CytoSPACE 实现一个 Tangram 专用 refine
        # 4. 输出 "三件套"
        ...
```

> 这样 Stage4 的主体逻辑不变，
>  任何新方法只要按接口“填空”，就能自动加入后续 Stage5/6/7 的评估和可视化。

------

### 五、Orchestrator：run_stage4_mapping

#### 5.1 函数签名（伪代码）

```python
def run_stage4_mapping(
    backends: list[str],     # e.g. ["cytospace"]
    stage1_dir: str,
    stage2_dir: str,
    stage3_dir: str,
    out_root: str,           # e.g. result/real_brca/stage4_mapping
    run_baseline: bool,
    run_plus: bool,
    backend_configs: dict | None = None,  # 每个 backend 的 config（可含 seed / λ 等）
):
    ...
```

#### 5.2 内部调用逻辑

1. **初始化 backend 实例**

   ```python
   registry = {
       "cytospace": CytoSPACEBackend(),
       # 将来:
       # "tangram": TangramBackend(),
       # "cell2location": Cell2LocationBackend(),
   }
   ```

2. **循环调用**

   ```python
   for name in backends:
       backend = registry[name]
       cfg = (backend_configs or {}).get(name, {})
   
       base_dir = os.path.join(out_root, name)
   
       if run_baseline:
           out_dir = os.path.join(base_dir, "baseline")
           backend.run_baseline(
               stage1_dir=stage1_dir,
               out_dir=out_dir,
               config=cfg.get("baseline"),
           )
   
       if run_plus:
           out_dir = os.path.join(base_dir, "plus_svg_type")
           backend.run_plus(
               stage1_dir=stage1_dir,
               stage2_dir=stage2_dir,
               stage3_dir=stage3_dir,
               out_dir=out_dir,
               config=cfg.get("plus"),
           )
   ```

**关键点：**

- Orchestrator 完全在用“字符串 backend_name + 抽象接口”来驱动；
- **是否启用 SVG refine、λ 取多少、seed 取多少，全部通过 `config` 传给 backend；**
- 将来只要在 `registry` 里多注册一个 backend 实现类，就能自动参加阶段四，不用改主逻辑。

#### 5.3 “多 seed / 多 λ”在这一层的用法提示

- 最简单的使用方式：在外层脚本中写 for-loop，多次调用 `run_stage4_mapping`：
  - 每次传不同的 `backend_configs["cytospace"]["plus"]["seed"]` 和 `svg_refine_lambda`；
  - 每次用不同的 `work_dir` 或 `sample_id` 区分。
- 如果后续觉得需要，也可以在 `backend_configs["cytospace"]["plus"]` 里加一个 `runs` 列表，让 Orchestrator 内部支持多 run；
   这一点可以在真正动手写代码时按需要扩展，这里先在设计稿中留出空间。

------

### 六、CytoSPACEBackend 的具体实现（作为“首个 backend”示范）

这里是**具体到 CytoSPACE 的部分**，但要明确：
 它只是 `MappingBackend` 的一个实现，不是阶段四本身。

#### 6.1 CytoSPACEBackend.run_baseline（不使用插件）

目标：给 CytoSPACE 一条**干净的 baseline**。

> 内部做什么你已经比较熟悉，这里只把关键逻辑列出来，重点在“它不依赖 Stage2/3”。

1. 从 `stage1_dir` 读取：

   - `sc_expression_normalized.csv`
   - `st_expression_normalized.csv`
   - `sc_metadata.csv`（拿 `celltype`）
   - `spot_cell_counts.csv`（若无，则内部计算）

2. 构造 CytoSPACE 输入：

   - 细胞表达矩阵、spot 表达矩阵；
   - 每个细胞的原始 `celltype`；
   - spot-level 细胞数先验；
   - 可选：简单 marker-based spot×type 先验。

3. 在**原始表达空间**上构建 cost 矩阵。

4. 调 CytoSPACE 官方 solver，跑一遍映射。

5. 写出“三件套”到：

   ```text
   result/stage4_mapping/cytospace/baseline/
     cell_assignment_baseline.csv
     cell_spot_matrix_baseline.npz
     spot_type_fraction_baseline.csv
     (可选: cyto_cost_matrix_baseline.npz, cyto_constraints_baseline.json...)
   ```

#### 6.2 CytoSPACEBackend.run_plus（接入模块一 + 模块二 + SVG refine）

目标：用上插件，让 CytoSPACE 变成真正的 **“SVG-aware + type-aware 的 CytoSPACE-plus”**。

重点：**在三个地方插入插件信息，并在第四个地方启用 SVG 空间 refine：**

1. 表达空间 → 使用 `plugin_genes` + `final_weight` 改写 cost；
2. 类型体系 → 使用 `plugin_type`；
3. spot×type 先验 → 使用 `type_prior_matrix`；
4. 初次 assignment 后 → 使用 SVG 空间重构 loss + 邻域微调（refine）。

内部逻辑可以分步写：

##### Step 1：读入表达 + 插件结果

- 从阶段 1 读：
  - `sc_expression_normalized.csv`, `st_expression_normalized.csv`
  - `st_coordinates.csv`
- 从阶段 2 读：
  - `plugin_genes.txt`, `gene_weights.csv`
- 从阶段 3 读：
  - `cell_type_relabel.csv`, `type_prior_matrix.csv`
- 从 `config` 读（可选）：
  - `seed`（若 CytoSPACE 内部有随机行为）
  - `svg_refine_lambda`（SVG 空间重构 loss 的权重，默认为 0 表示不 refine）

##### Step 2：构建“加权表达空间”（SVG 真正进入 cost）

- 只保留 `plugin_genes ∩ sc/st 表达列`；

- 对每个基因乘 `final_weight`，得到 `sc_X_weighted`, `st_X_weighted`：

  ```python
  plugin_genes = ...
  weights = ...  # gene_weights.csv
  
  sc_X_weighted = sc_expr[plugin_genes].values * weights.loc[plugin_genes, "final_weight"].values
  st_X_weighted = st_expr[plugin_genes].values * weights.loc[plugin_genes, "final_weight"].values
  ```

> 这一步就是我们之前说的 **Step1：在 cost 里给 SVG 加权**，
>  保证 CytoSPACE 在算距离/相似度时，真的“看见”了 SVG 信号，而不是把它当普通 HVG。

##### Step 3：替换细胞类型为 plugin_type

- 从 `cell_type_relabel.csv` 中取 `plugin_type`；
- 对齐 `sc_expr` 的行；
- 在 CytoSPACE 类型相关统计中，完全使用 `plugin_type`（含 Unknown）。

##### Step 4：使用 type_prior_matrix 作为 spot×type 先验

- 读取 `type_prior_matrix`，按 ST spot 顺序对齐；
- 可结合 `spot_cell_counts` 得到期望 `expected_cells[s,t]`；
- 填入 CytoSPACE 类型/数量约束模块中，替代 baseline 的简单先验。

##### Step 5：在加权表达空间上构建 cost 矩阵

- 使用 `sc_X_weighted` vs `st_X_weighted`；
- 计算相似度 / 距离（如加权余弦、1-corr 等）；
- 得到 cell×spot 的 cost matrix `C_plus`。

##### Step 6：调用 CytoSPACE solver，得到初次 plus 结果

- 用 `C_plus` + `plugin_type` + `expected_cells` 求解；
- 得到初次的：
  - `cell_assignment_plus_raw`
  - `cell_spot_matrix_plus_raw`
  - `spot_type_fraction_plus_raw`

> 这一步只是**“带 SVG cost + 类型先验的 CytoSPACE 映射”**，
>  还没有做我们设计中的 **Step2：SVG 空间重构 refine**。

##### Step 7：SVG 空间重构 refine（可选，svg_refine_lambda > 0 时启用）

**目标：**
 在不严重破坏 CytoSPACE cost / 约束的前提下，对高不确定性细胞做局部微调，
 让映射后的空间表达模式在 SVG 基因上**更贴近真实 ST 模式**，并在局部空间上更平滑。

1. **选择用于重构的 SVG 基因子集**

   - 从 `gene_weights.csv` 中选取 SVG_score / SVG_norm 较高的一部分基因（例如前 K 个或 SVG_norm > 某阈值）；
   - 只在这个子集上计算 SVG 重构 loss。

2. **基于当前映射重构 ST 表达（只看 SVG 基因）**

   - 对每个 SVG 基因 g：
     - 用 `cell_spot_matrix_plus_raw` + sc 表达重构预测 ST 表达 `Ŝ[:, g]`；
     - 与真实 ST 表达 `S[:, g]` 做比较（如 Pearson r 或 L2 loss）。
   - 累积得到一个整体的 SVG 重构指标：
     - 例如：`svg_recon_score = Σ_g w_g * corr(S[:,g], Ŝ[:,g])`
        或对应的 loss 版本 `svg_recon_loss`。

3. **定义总目标（概念层面）**

   - 当前 CytoSPACE 的 assignment 对应一个 cost：
     - `cyto_cost = Σ_i,s A_{i,s} * C_plus[i,s]`（可以近似）
   - 我们定义一个综合目标：
     - `Total = cyto_cost + svg_refine_lambda * svg_recon_loss`
   - refine 的目标是：在不把 `cyto_cost` 拉得很离谱的前提下，尽量降低 `svg_recon_loss`。

4. **局部邻域微调策略（示意）**

   - 标记“高不确定性”的细胞：
     - 例如根据 cell-wise entropy、top1-top2 cost 差等指标选出一批候选细胞；
   - 对每个候选细胞 i：
     - 限制可选的候选 spot 在原始落点周围一个空间半径内（利用 `st_coordinates` 定义邻域）；
     - 在这些备选 spot 中尝试“换落点”或“重新分配权重”，估计：
       - Δcyto_cost
       - Δsvg_recon_loss（利用局部更新重算相关性 / loss）
     - 若 `Δcyto_cost + svg_refine_lambda * Δsvg_recon_loss < 0`，则接受这次修改。
   - 迭代若干轮（迭代次数、步长等由 config 控制），直到改善变小或达到上限。

5. **输出 refine 后的最终 plus 结果 + 统计**

   - 得到 refine 后的：

     - `cell_assignment_plus`（最终硬分配）
     - `cell_spot_matrix_plus`（最终 soft 分配）
     - `spot_type_fraction_plus`

   - 写出到：

     ```text
     result/stage4_mapping/cytospace/plus_svg_type/
       cell_assignment_plus.csv
       cell_spot_matrix_plus.npz
       spot_type_fraction_plus.csv
       cyto_cost_matrix_plus.npz              # 可选
       cyto_constraints_plus.json             # 可选
       svg_refine_stats_plus.json             # 可选：记录 refine 前后 SVG 指标变化
       mapping_meta_plus.json                 # seed / svg_refine_lambda 等
     ```

> 这样，**Step1（cost 加权）+ Step2（SVG refine）都落实在 CytoSPACEBackend.run_plus 内部**：
>
> - CytoSPACE 确实在 cost 中“看见了” SVG；
> - 细胞布局又被 SVG 空间重构 loss 进一步“拉”向更合理的空间模式；
> - 阶段 5 / 6 的 SVG 相关评估和仲裁就有了真正的抓手。

------

### 七、main.py 如何挂载阶段四（体现“多后端 + 可调 seed / λ”）

在 `main.py` 里的调用可以这样设计：

```python
if args.run_stage4:
    backends = args.backends.split(",")  # 比如 "cytospace" 或 "cytospace,tangram"

    run_stage4_mapping(
        backends=backends,
        stage1_dir=stage1_out_dir,
        stage2_dir=stage2_out_dir,
        stage3_dir=stage3_out_dir,
        out_root=os.path.join(work_dir, "stage4_mapping"),
        run_baseline=args.run_baseline_mapping,
        run_plus=args.run_plus_mapping,
        backend_configs={
            "cytospace": {
                "baseline": {
                    "cost_metric": args.mapping_cost_metric,
                    "seed": args.mapping_seed,
                },
                "plus": {
                    "cost_metric": args.mapping_cost_metric,
                    "seed": args.mapping_seed,
                    "svg_refine_lambda": args.svg_refine_lambda,
                    # 将来可以继续加:
                    # "neighbor_radius": args.svg_refine_neighbor_radius,
                    # "max_refine_iter": args.svg_refine_max_iter,
                },
            },
            # "tangram": {...} 将来可加
        },
    )
```

**这样：**

- 现在只用 `--backends cytospace`，就只跑 CytoSPACE；
- 以后加了 `TangramBackend`，你只用在命令行写 `--backends cytospace,tangram` 就可以同时跑两套方法；
- 若想做多 seed / 多 λ 的候选：
  - 外层脚本可以写 for-loop，改变 `--mapping_seed` 和 `--svg_refine_lambda`，
     得到多套 Stage4 结果，再交给 Stage5/6 做 SVG 仲裁。

------

### 八、用一句话总结“留好接口 + 让 SVG 真正发挥作用”这件事

> 阶段四不是“为 CytoSPACE 写死的一段代码”，
>  而是：
>
> - 上游：只依赖阶段 1–3 的**通用插件输出**（`plugin_genes/gene_weights/plugin_type/type_prior`），
>    同时保证 **SVG 权重直接改写了 cost 矩阵**；
> - 中间：用 `MappingBackend` 抽象了**backend 接口**，Orchestrator 只认接口，不认具体方法，
>    并通过 `config` 提供 seed / svg_refine_lambda 等参数来支持多配置运行；
> - 下游：每个 backend 必须按统一输出格式写“三件套”，
>    评估层（Stage5/6）因此可以做到**方法无关 + 对 SVG 友好（可做 Step3 仲裁）**。
>
> 目前我们只实现 `CytoSPACEBackend` 作为首个 backend，
>  但 Tangram / Cell2location 等后续方法，都可以毫无压力地“按接口填空”接进来，
>  同时也可以选择是否实现自己的 SVG refine 模块。

------







## SimGen 阶段

好，那我现在给你一份**“更新版 + 完整版” SimGen 阶段详细设计文档**，已经把你刚才特别强调的那一点——

> ✅ **只用一部分 BRCA scRNA 构造模拟 ST，拆成 World / Ref 两个子集，避免“自己造自己猜”的虚假高准确率**

完整写进整体设计里了，而不是“口头补丁”。

> 角色定位：
>  **SimGen 是一个独立的数据生成模块，用真实 BRCA scRNA 做模板，构造带真值的模拟 sc+ST 数据集，为后续阶段 5A/5B（模拟数据验证）服务。**

------

### 一、在全项目中的位置

项目整体 8（7+1）阶段回顾：

1. 阶段 0：数据选择 + 环境测试（真实 BRCA + CytoSPACE 跑一遍）
2. 阶段 1：预处理（R/Seurat + Python，得到干净的 sc/ST 矩阵）
3. 阶段 2：模块一（SVG+HVG 动态加权插件）
4. 阶段 3：模块二（细胞类型不匹配 / Unknown-aware 插件）
5. 阶段 4：映射整合（多 backend，有 CytoSPACEBackend 等）
    **SimGen：模拟数据构建模块（独立）**
6. 阶段 5：基于模拟数据的验证（5A：CytoSPACE 专用；5B：通用）
7. 阶段 6：真实数据验证
8. 阶段 7：可视化 & 报告

**注意关键点：**

- SimGen **不算“阶段 5”本身**，而是为阶段 5 准备输入的**前置模块**；
- 阶段 5A/5B 把 SimGen 产出的模拟数据**当成新的样本**，跑 1–4 + 做评估。

------

### 二、设计目标（包括你刚才特别强调的点）

SimGen 的目标是构造一批“高仿真 + 可控 + 有真值”的模拟 ST 数据场景，用于严谨地验证：

- **模块一（SVG+HVG 动态加权）**是否能提升空间映射质量；
- **模块二（类型不匹配插件）**在“sc/ST 类型不匹配”的场景中是否表现更好；
- 而且保证结果不因为设计“太简单”而虚假偏好我们的方法。

为此，我们明确几个设计要求：

1. **表达真实**
   - 使用真实 BRCA scRNA 的表达矩阵作为“模式模板”，不自己造高斯表达；
   - 保留真实的基因相关性、marker pattern。
2. **只用“部分原始数据造世界”，避免过拟合**
   - 不会直接用“全体 BRCA scRNA”同时做 ST 和 sc 参考。
   - 显式拆成两个子集：
     - **World 子集**：只在 SimGen 内部用来构造 ST（模拟组织空间里的细胞）；
     - **Ref 子集**：只作为后续映射阶段所看到的“sc 参考库”。
   - 这样就不会出现“同一批细胞既造 ST，又被拿来当 sc 参考”的“自嗨式完美映射”。
3. **空间结构清晰**
   - 为 World 子集的每个细胞赋予 2D 坐标 `(sim_x, sim_y)`；
   - 不同类型在空间上有分区 / 过渡带，而不是随机撒点。
4. **可以生成 Visium 风格 ST**
   - 将 World 子集中细胞按坐标聚合为 spot；
   - 生成 ST 表达矩阵，记录每个 spot 内的真值类型构成。
5. **可控地构造“类型不匹配场景”**
   - 支持构建：
     - 完全匹配（S0）——sc 和 ST 类型体系一致；
     - sc 缺类型（M1）——sc 参考不包含某些类型，ST 仍含；
     - ST 缺类型（M2）——sc 很全，但 ST 空间几乎不出现某类型。
   - 用来专门考察 **插件二（Unknown-aware 类型插件）** 的效果。
6. **输出格式兼容我们的 pipeline**
   - 模拟 sc / ST 数据文件格式与真实数据尽量一致；
   - 阶段 1–4 可以直接使用，不需要特例代码。

------

### 三、输入设计

#### 3.1 必需输入

假设真实数据输入路径为：

```text
data/real_brca/
  sc_expression.csv
  sc_metadata.csv
```

1. `sc_expression.csv`
   - 维度：`N_all_cells × N_genes`
   - 行索引：cell_id
   - 列索引：gene_name
   - 可以是：
     - 原始 count；
     - 或 log-normalized 表达（只要你在整个项目中认为这是“sc 表达”的统一格式即可）。
2. `sc_metadata.csv`
   - 每行一个细胞，至少包含：
     - `cell_id`
     - `celltype`（或 `cluster` + 我们已有的 type 注释）

> 这两份数据来自 CytoSPACE 提供的 BRCA scRNA，可以在预处理环节中适当清洗，但 SimGen 本身只依赖这二者。

#### 3.2 配置（YAML）

建议单独的配置文件，例如：

```text
configs/simgen_S0.yaml
configs/simgen_M1.yaml
configs/simgen_M2.yaml
```

配置内容（示例）：

```yaml
scenario_id: "S0_matched"
description: "完全匹配场景"

# World / Ref 拆分
world_fraction: 0.5         # 约 50% 细胞作为 World
ref_fraction: 0.5           # 约 50% 作为 Ref (可有少量重叠或不重叠)
allow_world_ref_overlap: false  # 默认不重叠，避免自嗨

min_cells_per_type: 50      # 类型少于 50 个细胞则合并/丢弃

# 空间布局
space_width: 1000
space_height: 1000
block_nx: 5
block_ny: 5

# spot 网格
spot_grid_nx: 30
spot_grid_ny: 30
spot_radius: 80
min_cells_per_spot: 5

# 噪声
libsize_mode: "lognormal"
libsize_sigma: 0.3
noise_level: 0.1            # 控制额外噪声强度

# 类型不匹配控制
sc_missing_types: []        # S0 场景中为空
st_missing_types: []
```

在 M1 / M2 的配置里，只需修改 `sc_missing_types` / `st_missing_types` 等。

------

### 四、输出 & 目录结构设计

SimGen 可以为每个场景生成一个独立目录，例如（⚠️统一根目录：`data/sim/<scenario_id>/`）：

```text
data/
  sim/
    S0_matched/
      sim_sc_expression.csv
      sim_sc_metadata.csv
      sim_ref_sc_expression.csv          # 可选：Ref，仅分析用
      sim_ref_sc_metadata.csv            # 可选：Ref，仅分析用
      sim_st_expression.csv
      sim_st_coordinates.csv
      sim_truth_spot_type_fraction.csv
      sim_truth_cell_spot.csv            # World cell -> spot（审计/解释）
      sim_truth_query_cell_spot.csv      # ⚠️Query cell -> spot（cell-level 真值）
      cell_true_spot.csv                 # alias -> sim_truth_query_cell_spot.csv
      spot_true_type_fraction.csv        # alias -> sim_truth_spot_type_fraction.csv
      scenario_meta.json
    M1_sc_missing_Bcell/
      ...
    M2_st_missing_Tumor/
      ...
```

#### 4.1 模拟 sc（Query）部分输出

> ⚠️契约：这里的 `sim_sc_*.csv` 对应 **Query 子集**，也就是后续 Stage4 要映射的 sc（并同步写入 `data/processed/<scenario>/stage1_preprocess/exported/`）。
> `sim_ref_sc_*.csv`（若存在）才是 Ref，仅用于分析/对照，不参与映射。

##### 4.1.1 `sim_sc_expression.csv`

- 维度：`N_query_cells × N_genes`
- 行：`cell_id`（来源于原始 BRCA scRNA 的 cell_id 子集）
- 列：gene_name（与原始 sc 表达列一致，可能只保留交集基因）

Query 子集的构建方式见后文的 World/Query/Ref 拆分设定（默认 World/Query 不重叠）。

##### 4.1.2 `sim_sc_metadata.csv`

至少包含：

| 列名      | 含义                                              |
| --------- | ------------------------------------------------- |
| `cell_id` | Query 子集细胞 ID                                 |
| `type`    | Query 子集的类型注释（一般使用清洗后的 `celltype`） |
| `celltype` | `type` 的别名（建议同时写，避免下游列名歧义） |

可附加：

- `orig_type`（若对类型做了合并）
- `ref_group`（如 `"kept_in_sc"`, `"removed_in_sc"`，主要在 M1 场景有用）

------

#### 4.2 模拟 ST（World→Spot）部分输出

这些来自 World 子集，用坐标聚合而来。

##### 4.2.1 `sim_st_expression.csv`

- 维度：`N_spots × N_genes`
- 行：`spot_id`
- 列：gene_name
- 值：World 子集中，在对应 spot 覆盖范围内所有细胞的表达求和（+噪声）。

##### 4.2.2 `sim_st_coordinates.csv`

| 列名      | 含义                        |
| --------- | --------------------------- |
| `spot_id` | spot ID，形式如 `spot_0001` |
| `x`       | spot 中心 X 坐标            |
| `y`       | spot 中心 Y 坐标            |

------

#### 4.3 真值映射部分（Core）

##### 4.3.1 `sim_truth_cell_spot.csv`

重点：**每个 World 细胞真正“属于哪个 spot”**。

| 列名           | 含义                                  |
| -------------- | ------------------------------------- |
| `cell_id`      | World 子集细胞的 ID                   |
| `true_spot_id` | 该细胞最近/主要归属的 spot（无则 NA） |

> World 子集的定义很关键：
>
> - 所有用来造 ST 的细胞都属于 World；
> - 后续，映射算法看到的 sc 输入来自 **Query** 子集，不看到 World 子集本身。
>    -（是否允许 World/Query 有少量重叠，可以由配置决定，默认不重叠）

##### 4.3.2 `sim_truth_query_cell_spot.csv`

重点：**每个 Query 细胞的 cell-level 真值落点**（用于 Stage5 的 cell-level 指标）。

| 列名           | 含义                                  |
| -------------- | ------------------------------------- |
| `cell_id`      | Query 子集细胞 ID（⚠️应与 Stage4 输出的 cell_id 对齐） |
| `true_spot_id` | 该 Query 细胞的真值 spot（若该类型在 ST 缺失，可为 NA） |

> `cell_true_spot.csv` 作为兼容别名，默认指向这份 Query 真值文件。

##### 4.3.3 `sim_truth_spot_type_fraction.csv`

每个 spot 里真实类型构成：

- 行：`spot_id`
- 列：各种 `true_type`（来自 World 子集的 `true_type`）
- 值：fraction（每 spot 内该类型细胞数 / spot 总细胞数）

------

#### 4.4 场景元信息

`scenario_meta.json` 为评估阶段提供场景标签和生成参数，例如：

```json
{
  "scenario_id": "M1_sc_missing_Bcell",
  "description": "sc 参考中缺失 B_cell 类型，ST World 中仍保留 B_cell",
  "world_fraction": 0.5,
  "ref_fraction": 0.5,
  "allow_world_ref_overlap": false,
  "sc_missing_types": ["B_cell"],
  "st_missing_types": [],
  "n_world_cells": 5000,
  "n_ref_cells": 4000,
  "n_spots_total": 900,
  "space_width": 1000,
  "space_height": 1000,
  "block_nx": 5,
  "block_ny": 5,
  "spot_grid_nx": 30,
  "spot_grid_ny": 30,
  "spot_radius": 80,
  "noise_level": 0.1
}
```

------

### 五、核心设计要点：World / Ref 拆分策略

这是这次你特别强调的重点，这里单独写一节。

#### 5.1 为什么要拆 World / Ref？

如果我们直接：

- 用“全部 BRCA scRNA 细胞”来构造 ST（聚成 spot）；
- 又用“同一批细胞”作为 sc 参考去做映射，

那映射问题会变得“过于容易”：

- 映射算法可以在某种意义上找到“几乎一一对应”的匹配（表达完全一样）；
- 导致我们测出来的准确率虚高，对真实应用场景没有参考价值。

**拆分 World / Ref 的目的**：

> 逼近真实场景：
>
> - World ≈ 某块真实组织里的细胞（我们并不知道它们的具体身份，只能看到 ST）；
> - Ref ≈ 通过 scRNA 测到的一批参考细胞（来自同一组织体系，但不是“一模一样那块切片的所有细胞”）。

#### 5.2 拆分策略（建议方案）

1. **类型清洗后，先得到一批“可用细胞”列表**
   - 删除类型数太少的 cell type（比如 < 50）；
   - 若要合并类型，在这一步统一处理。
2. **按类型分层随机分配到 World / Ref**

例如：

- 对每个类型 t，有 `N_t` 个细胞；
- 配置中设定：
  - `world_fraction = 0.5`
  - `ref_fraction = 0.5`
  - `allow_world_ref_overlap = false`

则：

- 从类型 t 的细胞中随机抽取 `N_t * world_fraction` 个到 World；
- 从剩余的细胞中再抽 `N_t * ref_fraction` 个到 Ref；
- 如果 `allow_world_ref_overlap = true`，则可以在同一 pool 上采样两次（少量重叠）。

**默认我们建议**：`allow_world_ref_overlap = false`，即**World 与 Ref 完全不重叠**，最干净地避免自嗨。

1. **World 子集** 用于：

- 后续赋坐标、聚 spot，生成 `sim_st_*` 和真值映射。

1. **Ref 子集** 用于：

- 输出 `sim_sc_expression`、`sim_sc_metadata`，
- 给阶段 1–4 当“sc 输入”使用。

> 这样一来，你就可以明确写在论文里：
>  “模拟 ST 由 BRCA scRNA 的 World 子集构造，而映射算法使用的 sc 参考数据来自另一不重叠的 Ref 子集，这与 Cell2Spatial / CytoSPACE 使用部分数据构造模拟数据、另一部分数据作为参考的一般做法是一致的。”

------

### 六、SimGen 算法流程（包含 World/Ref 拆分）

下面按流程顺序来写（每一步你将来都可以写成一个函数/脚本）。

#### Step 0：载入输入并类型清洗

1. 读取 `sc_expression.csv` & `sc_metadata.csv`；
2. 删除没有类型注释的细胞；
3. 按 `min_cells_per_type` 过滤类型：
   - 若某类型细胞数 < 阈值，可以：
     - 合并到 “Other”；或
     - 直接丢弃（为了模拟更稳定）。

输出：

- `cells_use`: 可用细胞 ID 列表；
- `X_sc[cells_use, :]`: 表达矩阵；
- `type_vec[cells_use]`: 类型向量。

------

#### Step 1：World / Ref 子集拆分

1. 对每个 type t：

   ```text
   cells_t = {cells_use 中 type 为 t 的 cell}
   n_t = len(cells_t)
   
   n_world_t = floor(n_t * world_fraction)
   n_ref_t   = floor((n_t - n_world_t) * ref_fraction)    # 不重叠方案
   ```

2. 随机打乱 `cells_t`，前 `n_world_t` 个分到 World，后 `n_ref_t` 分到 Ref；

3. 合并所有类型的结果：

   - `World_cells = ⋃t World_t`
   - `Ref_cells   = ⋃t Ref_t`

4. 检查两者是否重叠（如果 `allow_world_ref_overlap=false`，必须为空交集）。

输出：

- `X_world`，`type_world`
- `X_ref`，`type_ref`

------

#### Step 2：为 World 子集设计空间布局（block-level）

1. 定义空间 `[0, W] × [0, H]`，按 `block_nx × block_ny` 划分 block；
2. 建立 block 集合 `block_{i,j}`；
3. 为每个 type t 指定其主 block（自动或配置）：
   - 简单策略：
     - 随机选若干 block 作为 type t 的“高占比区”；
     - 在其周边 block 设为混合区：
       - 例如：
         - 主 block：`80% t + 20% others`
         - 邻近 block：`50% t + 50% others`
         - 远离 block：`few % t` 或 0。
4. 得到 `block_type_mixture[block_id][type]`。

可以把这个表输出成 debug 文件 `block_layout.csv`，便于可视化。

------

#### Step 3：为 World 中每个细胞采样 2D 坐标

对每个 World 细胞：

1. 看它的 `true_type`；
2. 根据该 type 在各 block 的配比，采样一个 block（比如根据 mixture 权重多项式采样）；
3. 在该 block 对应的坐标范围内均匀采样 `(sim_x, sim_y)`。

输出：World 细胞的坐标表 `world_coords`：

| cell_id | true_type | sim_x | sim_y |
| ------- | --------- | ----- | ----- |
|         |           |       |       |

这份信息将写入 `sim_sc_metadata_world.csv`（内部中间文件，用于生成 ST & 真值）。

注意：Ref 子集**不需要坐标**，因为在我们的逻辑中：

- Ref 是“参考库”，sc 数据不需要有空间坐标；
- 在阶段 1–4 中我们只把 `sim_sc_expression` + `sim_sc_metadata` 当作 sc 输入（没有真实坐标）。

------

#### Step 4：生成 ST spot 网格

根据配置：

1. 在 `[0, W] × [0, H]` 中生成 `spot_grid_nx × spot_grid_ny` 个 spot 中心；
2. 每个 spot 分配一个 `spot_id`，以及中心坐标 `(xs, ys)`；
3. 保存为 `sim_st_coordinates.csv`。

------

#### Step 5：建立 World cell → spot 覆盖关系

1. 对每个 spot s：
   - 计算所有 World 细胞与 `(xs,ys)` 的距离；
   - 选择距离 ≤ `spot_radius` 的细胞，构成 `cells_in_spot[s]`。
2. 对每个 World 细胞 i：
   - 找到离它最近的覆盖 spot，记为 `true_spot(i)`（若没有任何 spot 覆盖，可设 NA 或剔除）。

输出：

- `sim_truth_cell_spot.csv`：
   `cell_id, true_spot_id`
- 内部变量 `cells_in_spot[s]`，后续用于生成 ST 表达和 spot 类型真值。

------

#### Step 6：生成 ST 表达矩阵（World → spot）

初始 ST 表达矩阵：

```text
for each spot s:
  for each gene g:
    sim_st_expr[s, g] = sum( X_world[cell_i, g] for cell_i in cells_in_spot[s] )
```

然后根据配置加入技术噪声：

1. library size 变异（可选）：
   - 为每个 spot 采样 `L_s ~ LogNormal(0, libsize_sigma)`；
   - `sim_st_expr[s, :] *= L_s`。
2. 额外噪声（可选）：
   - `noise_level > 0` 时，对每个 `(s,g)` 做轻微扰动：
     - Poisson：`Poisson(lambda = sim_st_expr[s,g])`
     - 或乘以 `(1 + Normal(0, sigma))`。

输出为 `sim_st_expression.csv`。

------

#### Step 7：计算 spot 类型真值分布

根据 `cells_in_spot[s]` 和 `world_coords.true_type`：

1. 对每个 spot s：
   - 统计各 `true_type` 的细胞数 `count[s, t]`；
   - 计算 fraction：`frac[s, t] = count[s,t] / sum_t count[s,t]`。
2. 输出为 `sim_truth_spot_type_fraction.csv`。

------

#### Step 8：构建 Ref 子集输出（sc 输入）

Ref 子集不会参与 ST 生成，它只负责提供“映射参考库”。

1. 从 `X_ref` 和 `type_ref` 构造：
   - `sim_sc_expression.csv`：行 = Ref 细胞，列 = 基因；
   - `sim_sc_metadata.csv`：
     - `cell_id` = Ref 细胞 ID
     - `type` = 对应类型。
2. 在 mismatch 场景（M1）中，如需删除某些类型，可在这一步直接：
   - 根据 `sc_missing_types` 列表，把这些类型的 Ref 细胞从输出中移除；
   - 真值仍然以 World 子集为准，不受影响。

------

#### Step 9：场景化生成（S0 / M1 / M2）

通过配置文件，可以生成多个不同难度的场景。

##### 9.1 场景 S0：完全匹配（S0_matched）

- 拆分 World/Ref（所有类型都参与）；
- 构造 ST（World → spot）、真值；
- Ref 输出保持全类型；
- `sc_missing_types = []`, `st_missing_types = []`；
- 协调好类型名，使得：
  - World 中 type 集合 == Ref 中 type 集合；
  - 这是“最理想、无 mismatch”的场景，用来测模块一+模块二整体 mapping 提升。

##### 9.2 场景 M1：sc 缺类型（M1_sc_missing_Bcell）

- 在配置中设定：`sc_missing_types: ["B_cell"]`；
- World：
  - 仍然包含 B_cell 类型（即 B_cell 细胞参与构造 ST）。
- Ref：
  - 按 Step 8 输出前，将 `type_ref == "B_cell"` 的细胞剔除；
  - 这样映射算法在 sc 参考中看不到 B_cell 类型。

目的：

> 考察：
>
> - baseline 是否会把 B_cell 区域的细胞错误标注为 Tumor/T_cell 等；
> - plus 是否选择把这些区域更诚实地映成 Unknown。

##### 9.3 场景 M2：ST 缺类型（M2_st_missing_Tumor）

- 在配置中设定：`st_missing_types: ["Tumor"]`；
- World：
  - 在 Step 2 的空间布局阶段，不给 Tumor 分配主 block，甚至只在极少数 block 中放一点 Tumor 细胞；
  - 这样 ST 真值图上，Tumor 类型只出现在极少数 spot。
- Ref：
  - 保留所有类型（包括 Tumor），sc 参考中仍有大量 Tumor 细胞。

目的：

> 考察：
>
> - baseline 是否会把 Tumor 细胞强行填满很多 spot（虚假扩散）；
> - plus 是否通过 type_prior_matrix 限制 Tumor 在 ST 中的出现区域，甚至把“本不该出现的 Tumor”映成 Unknown 或其他合理类型。

------

### 七、质量控制 & 验收标准

SimGen 阶段可以用以下 checklist 来判断是否“达到设计目标”：

1. **World / Ref 合理拆分**
   - `World_cells` 与 `Ref_cells` 是否无交集（如果配置如此要求）；
   - 各类型在 World 与 Ref 中都有足够 cell 数（避免某类型完全出现在一侧）。
2. **空间结构可视化检查**
   - 用 `sim_sc_metadata_world` 画出 `(sim_x, sim_y)` 上的 true_type 点图：
     - 看是否出现预期的“区域分布 + 过渡带”。
3. **ST spot QC**
   - 大部分 spot 的 `cells_in_spot` 不为空；
   - `sim_st_expression` 基本分布合理（如总 UMI 有一定变异但不极端）；
   - `sim_truth_spot_type_fraction` 热度图与直观的空间结构一致。
4. **场景正确性**
   - S0：
     - `sc_missing_types = []`；
     - World 中 type 集合 == Ref 中 type 集合。
   - M1：
     - 配置中的 `sc_missing_types` 确实在 Ref 输出里消失了；
     - World 中这些类型仍有明显存在。
   - M2：
     - `st_missing_types` 在真值 spot_type_fraction 中仅极少数出现；
     - sc 参考中仍正常存在这些 type。
5. **可被阶段 1–4 正常使用**
   - 用任一场景（如 S0）作为输入，能跑通阶段 1–4，得到 baseline / plus 的映射输出；
   - 不需要专门 hack，说明 SimGen 输出格式已经与真实数据接轨。

------

### 八、与阶段 5 的接口（简述）

最后，把 SimGen 和阶段 5 的“接口”再强调一下（你已理解，这里只是整理成一句话）：

- 阶段 5 不再关心“模拟数据怎么造”；
- 它只需要：
  - 把 `data/sim/<scenario>/sim_sc_*.csv`（⚠️Query）和 `sim_st_*.csv` 当成新的 sc/ST 输入，跑 1–4；
    - 代码侧等价于：`data/processed/<scenario>/stage1_preprocess/exported/` 下的四件套 CSV
  - 再用：
    - `sim_truth_query_cell_spot.csv`（cell-level 真值；别名 `cell_true_spot.csv`）
    - `sim_truth_spot_type_fraction.csv`（spot-level 真值；别名 `spot_true_type_fraction.csv`）
    - `scenario_meta.json`
    - （可选）`sim_truth_cell_spot.csv`（World 真值，仅用于审计/解释）
    - 对 baseline vs plus 的映射结果做指标对比（cell-level + spot-level + Missing→Unknown 等）。

------









------

## 阶段五（Stage 5）：基于模拟数据的映射验证实验

> 说明：
>
> - **SimGen** 负责“造模拟世界”（World/Ref 拆分 + 各种缺型/稀有场景）；
> - **阶段 5** 负责“在这些模拟世界里做实验 + 算指标”；
> - 本阶段既要评估 **CytoSPACE 特有的 cost/约束行为（5A）**，也要评估**所有 backend 共用的映射质量指标（5B）**；
> - **特别结合本项目的三步解题思路：**
>   1. Step1：在 Stage4 中对 `plugin_genes` 按 `final_weight` 加权，构造 SVG+HVG 加权 cost；
>   2. Step2：在 CytoSPACE-plus 中，做一次 SVG 空间重构 refine（可通过 `svg_refine_lambda` 控制强度）；
>   3. Step3：在 Stage6 中，用 SVG-aware 指标对多配置 run 做仲裁——而 Stage5 要提前在“有真值世界”里告诉我们**哪些配置值得在真实数据里重点尝试**。

------

### 一、阶段五的定位与目标

#### 1.1 在整个项目中的位置

整体链路回顾（只标和 Stage5 强相关的部分）：

1. **阶段 1：预处理**
    → 得到干净且对齐好的 `sc` / `st` 表达矩阵和 meta 信息。

2. **阶段 2：SVG+HVG 动态加权插件**
    → 输出

   - `gene_weights.csv`（包含 `hvg_score`、`svg_score`、`hvg_norm`、`svg_norm`、`final_weight` 等）；
   - `plugin_genes.txt`（按 `final_weight` 选出的基因集合 G_plugin）。
      这些结果在 Stage4 中用于 **Step1：给 SVG/HVG 综合高分基因加权，真正写进 CytoSPACE cost 里**；
      同时也在 Stage5/6 中用来定义 **SVG 空间指标的“关注基因集合”和权重**。

3. **阶段 3：类型不匹配 / Unknown-aware 插件**
    → 输出

   - `cell_type_relabel.csv`（`plugin_type`）；
   - `type_prior_matrix.csv`（spot×plugin_type 先验）。
      在 Stage4 的 plus 分支中被使用，用来让映射对“缺型 / 错型 / Unknown”更敏感。

4. **阶段 4：映射编排 + 后端适配（以 CytoSPACEBackend 为首个 backend）**

   - baseline：原版 CytoSPACE，不用插件；
   - plus：CytoSPACE-plus，内部包含：
     - **Step1：SVG+HVG 加权 cost**（用 Stage2 的 `final_weight` 和 `plugin_genes` 改写表达空间与 cost 矩阵）；
     - **类型插件**（用 Stage3 的 `plugin_type` 和 `type_prior_matrix`）；
     - **Step2：SVG 空间 refine**（可选，由 `svg_refine_lambda` 控制，针对高不确定性细胞做局部微调，让 SVG 空间 pattern 更贴近 ST 真值）。
   - Stage4 Orchestrator 支持对不同 `seed / λ / config_id` 跑**多次映射**，把所有 run 的“三件套 + meta”保存下来。

5. **SimGen：模拟世界构建模块（独立）**

   - 针对不同场景（S0、M1、M2…）生成带真值的 `sim_sc_*`、`sim_st_*` 和 `sim_truth_*`；
   - World/Ref 不重叠 → 映射算法永远只看到 Ref 细胞，ST 由 World 细胞构造。

6. **阶段 5：本阶段**

   - 输入：SimGen 真值 + Stage2/3/4 输出；

   - 任务：在“有真值世界”里全面评估 baseline vs plus（含 Step1 + Step2）在不同场景下的表现，重点关注：

     - 映射质量（spot×type、表达重构等）；
     - SVG 空间 pattern 重构质量（特别关联 Step1 和 Step2）；
     - 类型/Unknown 行为（关联模块二）；
     - 稳定性（seed / λ / config）；
     - CytoSPACE 自身 cost/约束的变化（不能被插件搞崩）。

   - 输出：

     - 每个场景、每个 backend、每个 run 的完整指标表；

     - 一份跨场景的 `stage5_config_recommendation.json`，告诉 Stage6：

       > “在有真值模拟下，哪些配置在 SVG + 类型 + 稳定性 + cost 这些维度上总体最靠谱，
       >  你在真实 BRCA 上应该重点跑哪一批 run。”

7. **阶段 6：真实数据验证（无真值）**

   - 真实 BRCA 上用 Stage5 推荐的配置集跑多次映射；
   - 用无真值的 SVG 空间指标 + 类型/Unknown 指标 + 约束指标做**最终 SVG-aware 仲裁（Step3 真正落地）**。

#### 1.2 本阶段要回答的关键问题（结合 Step1/Step2）

1. **模块一（SVG+HVG 动态加权 / Step1）到底有没有用？**
   - 在 S0 理想场景下：plus 能否在 SVG 空间指标（相关性、Moran’s I 等）上明显优于 baseline？
   - 在 M1/M2 等复杂场景下：plus 是否仍然能保持较好的 SVG 重构，而不过度损害全局表达拟合和类型分布？
2. **SVG 空间 refine（Step2）是否值得开启、配多大 λ？**
   - 对同一 config_id 和 seed：
     - λ=0（无 refine） vs λ>0（有 refine）：
       - SVG 空间指标有无显著提升？
       - 对成本、约束、映射质量是否友好？
   - 不同 λ 下是否存在“过拟合”或“强行推挤”的异常行为？
3. **类型插件在“缺型 / 错型 / 稀有型”场景中的行为是否符合预期？**
   - 特别关注：
     - `missing_to_unknown_rate`（应该高一点）；
     - `missing_to_wrongtype_rate`（应该低一点）；
     - 稀有类型在空间上的“虚假扩散”是否有所缓解。
4. **在有真值世界里，哪些配置（config_id + λ + seed 数量）表现“又准又稳”？**
   - 这些配置将被写入 Stage6 使用的 `stage5_config_recommendation.json`，
   - 作为 Step3 在真实数据上仲裁的“候选配置池”。

------

### 二、输入与输出

#### 2.1 输入（SimGen + Stage1–4）

对每个模拟场景 `scenario_id`（例如：`S0_matched`、`M1_sc_missing_Bcell` 等）：

1. **SimGen 输出：** `data/sim/<scenario_id>/`
   - `sim_sc_expression.csv`（⚠️Query：Stage4 要映射的 sc）
   - `sim_sc_metadata.csv`（Query 元数据，建议含 `celltype`）
   - `sim_ref_sc_expression.csv` / `sim_ref_sc_metadata.csv`（可选：Ref，仅分析用）
   - `sim_st_expression.csv`
   - `sim_st_coordinates.csv`
   - `sim_truth_query_cell_spot.csv`（⚠️Query cell→spot 真值；别名 `cell_true_spot.csv`）
   - `sim_truth_spot_type_fraction.csv`（spot×type 真值；别名 `spot_true_type_fraction.csv`）
   - `sim_truth_cell_spot.csv`（World cell→spot 真值，仅审计/解释用）
   - `scenario_meta.json`（场景说明 + role_map + query_truth_policy + audits 等）
2. **Stage1–3 输出：** `result/<scenario_id>/stageX_*`
   - Stage2：`gene_weights.csv`、`plugin_genes.txt`
     - 用于定义“SVG 基因集合”和权重（后续 SVG 指标的核心参考）；
   - Stage3：`cell_type_relabel.csv`、`type_prior_matrix.csv`
     - 用于区分 baseline 类型与 `plugin_type`，以及理解 type 先验。
3. **Stage4 输出：** `result/<scenario_id>/stage4_mapping/`
   - 每个 backend（首先是 `cytospace`）有：
     - baseline：
       - `cell_assignment_baseline_run*.csv`
       - `cell_spot_matrix_baseline_run*.npz`
       - `spot_type_fraction_baseline_run*.csv`
       - 可选：`cyto_cost_matrix_baseline_run*.npz`、`cyto_constraints_baseline_run*.json`、`mapping_meta_baseline_run*.json` 等；
     - plus（可能多个 run）：
       - `cell_assignment_plus_run*.csv`
       - `cell_spot_matrix_plus_run*.npz`
       - `spot_type_fraction_plus_run*.csv`
       - `mapping_meta_plus_run*.json`（至少包含 `backend`, `seed`, `config_id`, `svg_refine_lambda`, `enable_refine` 等）
       - 对 CytoSPACE-plus，可选：`svg_refine_stats_plus_run*.json`（记录 refine 前后 SVG 指标变化）。

> 注意：
>
> - “run*” 表示同一 backend + 模式下可能有多个配置（不同 seed / λ / config_id）；
> - Stage5 必须能根据 `mapping_meta_*.json` 把这些 run 的配置参数读出来，才能后面做“配置扫描与推荐”。

#### 2.2 输出（按场景 + 按 backend）

对每个 `scenario_id`：

```text
result/<scenario_id>/stage5_eval/
  generic/
    metrics_<backend>.json        # 所有通用指标的汇总
    metrics_<backend>_runs.csv    # 每个 run 一行的详细指标表
  cytospace_specific/
    metrics_cytospace.json        # 仅对 backend=cytospace 存在
```

跨场景汇总与配置推荐：

```text
result/stage5_summary/
  stage5_scenario_summary.json      # 按场景汇总 baseline vs plus 效果
  stage5_config_recommendation.json # 对各 backend 的参数推荐（给 Stage6 用）
```

------

### 三、整体流程与函数接口

#### 3.1 总控：对所有场景执行阶段五

```python
def run_stage5_for_all_scenarios(
    scenarios: list[str],
    simgen_root: str,
    result_root: str,
    backends: list[str] = ["cytospace"],
):
    for sid in scenarios:
        run_stage5_for_one_scenario(
            scenario_id=sid,
            simgen_dir=f"{simgen_root}/{sid}",
            stage4_mapping_root=f"{result_root}/{sid}/stage4_mapping",
            stage2_dir=f"{result_root}/{sid}/stage2_svg_plugin",
            stage3_dir=f"{result_root}/{sid}/stage3_type_plugin",
            out_root=f"{result_root}/{sid}/stage5_eval",
            backends=backends,
        )

    # 所有场景跑完之后，做跨场景汇总与配置推荐（为 Stage6 提供建议）
    summarize_stage5_results(
        result_root=result_root,
        scenarios=scenarios,
        backends=backends,
        out_summary_root=f"{result_root}/stage5_summary",
    )
```

#### 3.2 单场景评估：run_stage5_for_one_scenario

```python
def run_stage5_for_one_scenario(
    scenario_id: str,
    simgen_dir: str,
    stage4_mapping_root: str,
    stage2_dir: str,
    stage3_dir: str,
    out_root: str,
    backends: list[str],
):
    # 5B：通用 backend 模拟评估（不依赖 CytoSPACE 内部结构）
    eval_backend_simulation_generic(
        scenario_id=scenario_id,
        mapping_root=stage4_mapping_root,
        simgen_root=simgen_dir,
        stage2_dir=stage2_dir,
        stage3_dir=stage3_dir,
        backends=backends,
        out_root=f"{out_root}/generic",
    )

    # 5A：CytoSPACE 专用模拟评估（只对 backend=cytospace 做）
    if "cytospace" in backends:
        eval_cytospace_simulation_specific(
            scenario_id=scenario_id,
            mapping_root=f"{stage4_mapping_root}/cytospace",
            simgen_root=simgen_dir,
            out_root=f"{out_root}/cytospace_specific",
        )
```

#### 3.3 通用评估接口（5B）

```python
def eval_backend_simulation_generic(
    scenario_id: str,
    mapping_root: str,
    simgen_root: str,
    stage2_dir: str,
    stage3_dir: str,
    backends: list[str],
    out_root: str,
):
    """
    对每个 backend:
      1. 读取所有 baseline/plus run 的“三件套”+ meta（seed/λ/config_id）
      2. 读取 SimGen 真值和场景 meta
      3. 读取 Stage2 的 gene_weights（尤其是 SVG 相关信息），定义
         - SVG 基因集合 G_svg（可通过 svg_score / final_weight 选取）
         - SVG 基因权重 w_g（来自 final_weight 或 svg_norm）
      4. 计算 4.1~4.5 定义的通用指标：
         - 映射质量（spot×type、表达重构等）
         - SVG 空间重构指标（重点承载 Step1 + Step2 的效果）
         - 类型/Unknown 行为（特别看 M1/M2 场景）
         - 稳定性（多 seed / 多 λ / 多配置）
      5. 输出 metrics_<backend>.json + metrics_<backend>_runs.csv
    """
```

#### 3.4 CytoSPACE 专用评估接口（5A）

```python
def eval_cytospace_simulation_specific(
    scenario_id: str,
    mapping_root: str,   # result/<scenario>/stage4_mapping/cytospace
    simgen_root: str,
    out_root: str,
):
    """
    只对 backend=cytospace 做：
      1. 读取 cyto_cost_matrix_* / cyto_constraints_* / cyto_objective_stats_* 等
      2. 计算:
         - cost 相关指标（mean_cost_per_cell, total_cost 等）
         - 约束满足度（spot_total_count_rmse, spot_type_count_rmse 等）
         - 若存在 svg_refine_stats_plus_run*.json，则比较 refine 前后:
           * cost 变化
           * SVG 空间指标变化
           * 约束是否被破坏
      3. 输出 metrics_cytospace.json
    """
```

#### 3.5 跨场景汇总与配置推荐

```python
def summarize_stage5_results(
    result_root: str,
    scenarios: list[str],
    backends: list[str],
    out_summary_root: str,
):
    """
    汇总各 scenario 的 stage5_eval 结果：
      1. 生成 stage5_scenario_summary.json:
         - 每个场景中 baseline vs plus 的主指标对比
         - 关键 SVG 指标 / 类型行为 / 稳定性差异
      2. 对 backend=cytospace:
         - 在 (config_id, svg_refine_lambda, enable_refine) 空间内扫描
         - 找出在所有场景中表现“整体最优/最稳”的配置
         - 给出 stage5_config_recommendation.json:
            * 推荐的 λ 区间
            * 推荐的 config_id
            * 建议的 seed 数量
            * 是否建议默认开启 SVG refine
      3. Stage6 将直接读取 stage5_config_recommendation.json，
         在真实 BRCA 上优先跑这些“经过模拟验证”的配置，
         然后用无真值 SVG-aware 仲裁（Step3）选出最终 canonical 映射。
    """
```

------

### 四、通用评估指标体系（5B）

> 通用指标完全不依赖 backend 内部求解细节，只依赖：
>
> - SimGen 真值（`sim_truth_cell_spot`, `sim_truth_spot_type_fraction`）；
> - Stage4 输出的“三件套”：
>   - `cell_assignment_*.csv`
>   - `cell_spot_matrix_*.npz`
>   - `spot_type_fraction_*.csv`
> - Stage2 输出的 `gene_weights.csv`（定义 SVG 基因集合及权重）。

特别提醒一下 **World / Ref 不重叠** 的影响：

- World 子集细胞（只用于构造 ST 和真值）；
- Ref 子集细胞（映射算法的输入）；
- 因此：
  - **不能简单用“cell_id 一一对应”去算 Top-1 accuracy**；
  - cell-level 指标更多是“相对空间一致性/漂移距离”，而不是“绝对 spot id 匹配”。

#### 4.1 细胞级指标（Cell-level）

##### 4.1.1 局部空间一致性 & 空间漂移（基于类型/密度）

思路：对于一个 Ref 细胞 i，我们并不知道它在模拟世界中的“唯一真值 spot”，
 但我们知道它的类型、表达模式应该更大概率出现在某些 ST 区域。

定义：

- 对每个 Ref 细胞 i：
  - 预测 spot：`s_pred(i)`（从 `cell_assignment_*.csv` 读出）；
  - 类型（baseline 用原类型，plus 用 plugin_type）：`T(i)`；
- 从真值 `sim_truth_spot_type_fraction` 中得到：
  - `Q_s(T)`：spot s 上类型 T 的真值 fraction。

**局部一致性指标：**

- 对每个细胞 i，取其预测落点 `s_pred(i)` 的局部邻域 $\mathcal{N}(s_{pred})$（比如方圆若干个 spot）；
- 计算该邻域中 T(i) 的平均真值占比：

[
 \text{local_truth}(i) = \frac{1}{|\mathcal{N}(s_{\text{pred}})|} \sum_{s \in \mathcal{N}(s_{\text{pred}})} Q_s(T(i))
 ]

- 指标：
  - 所有细胞的 `local_truth(i)` 分布：均值/中位数/分位数；
  - baseline vs plus 的提升幅度。

**空间漂移（在类型真值中心附近与否）：**

- 对每个类型 T，基于 `Q_s(T)` 定义其“重心位置”`(x_T^*, y_T^*)`（用 spot 坐标加权平均）；

- 对每个细胞 i：

  - 取 `s_pred(i)` 的坐标 `(x_{pred}, y_{pred})`；

  - 定义漂移距离：
    $$
    [
     d_i = | (x_{pred}, y_{pred}) - (x_T^*, y_T^*) |
     ]
    $$
    

- 统计 {d_i} 的分布，比较 baseline vs plus（plus 应该让同一类型的 i 更集中于其类型重心附近）。

##### 4.1.2 （可选）cell×spot 分布的局部平滑性

如果 `cell_spot_matrix` 存的是 soft 分布，可以再定义：

- 对每个细胞 i，计算其分布的局部熵/平滑度；
- 观察 baseline vs plus 在 cells 高不确定集合中的差异。

> 这些 cell-level 指标不再追求“一对一 Top-1 正确率”，而是强调“空间上是否被合理扔进了对的区域”。

#### 4.2 spot×type 组成一致性

这里是真正“有真值”的主战场之一。

对每个 spot s：

- 真值类型分布：`Q_s(t)`，来自 `sim_truth_spot_type_fraction.csv`；
- 预测类型分布（baseline/plus）：`P_s(t)`，来自 `spot_type_fraction_baseline/plus.csv`（plus 基于 `plugin_type`）。

##### 4.2.1 L1 / L2 距离

对每个 spot s 计算：
$$
[
 L1_s = \sum_t |P_s(t) - Q_s(t)|
 ]
$$

$$
[
 L2_s = \sqrt{\sum_t (P_s(t) - Q_s(t))^2}
 ]
$$

然后在所有 spot 上统计：

- `mean_L1`, `median_L1`, `p90_L1` 等；
- `mean_L2`, `median_L2` 等。

##### 4.2.2 KL / JS 散度（带平滑）

对每个 spot s：

- 平滑后计算：
  - `KL_s = KL(Q_s || P_s)`；
  - `JS_s = JS(Q_s || P_s)`（对称）。

统计：

- `mean_KL`, `mean_JS`；
- `median_JS` 等。

##### 4.2.3 类型层面的相关性

把每个 spot 看成一个向量 `(Q_s(t))_t` 和 `(P_s(t))_t`，计算每个 spot 的相关系数 `corr_s`；
 然后统计 `mean_corr` / `median_corr`。

> 这些指标越“好”，说明 **映射在类型–空间分布层面越接近 SimGen 设定的真值模式**。

#### 4.3 表达重构与 SVG 空间重构指标（关键承载 Step1/Step2）

这里是**新项目和老项目的一个重要区别点**：
 我们明确区分 **全基因/HVG 的表达重构** 和 **SVG / 插件基因的空间重构**，
 且 SVG 部分直接关联 Stage2 的 `gene_weights` 和 Stage4 的 Step1/Step2。

##### 4.3.1 全基因 / HVG 表达重构

- 真值 ST 表达：`S_true`（来自 `sim_st_expression.csv`）；
- 根据映射重构的 ST 表达：
  - `S_pred = CellSpotMatrix × sc_expression_ref`。

指标：

- 全基因层面：
  - 每个基因的 Pearson 相关系数分布（真值 vs 预测）；
  - 全体基因的 RMSE / MAE；
- HVG 层面（如果有 `sc_hvg_genes.txt`）：
  - 只在 HVG 集合上重复上述统计。

##### 4.3.2 SVG 基因空间重构（核心，看 Step1/Step2 的作用）

- 利用 Stage2 的 `gene_weights.csv`，定义：
  - SVG 候选集合 G_svg：
    - 比如：按 `svg_score` 或 `svg_norm` 排序取 Top-K；
    - 或者取 `selected_in_plugin == 1` 的插件基因集合；
  - 对 G_svg 中每个基因 g，定义权重 `w_g`：
    - 可以直接用 `final_weight(g)` 或 `svg_norm(g)`。

对每个基因 g ∈ G_svg：

1. 从 `sim_st_expression` 中取真值空间表达 `S_true[:,g]`；

2. 从 `S_pred` 中取预测空间表达 `S_pred[:,g]`；

3. 计算：

   - 空间相关性：
     $$
     [
      corr_g = corr(S_{\text{true}}[:,g], S_{\text{pred}}[:,g])
      ]
     $$
     
- 空间自相关（Moran’s I）：
   
  - 用 ST 坐标和邻接关系计算 `I_true_g` 与 `I_pred_g`；
     - 关注 `|I_pred_g - I_true_g|` 的分布。
   
- （可选）SSIM/UQI 等图像结构指标：
   
  - 把 ST 空间插值为规则网格，计算 `SSIM_g`。

整体指标：

- 加权/不加权的分布统计：
  - `median_corr_svg`、`mean_corr_svg`；
  - `median_SSIM_svg`（如启用）、`mean_SSIM_svg`；
  - `median_delta_Moran = median(|I_pred_g - I_true_g|)` 等。

> 对 Step1/Step2 的期望行为：
>
> - **Step1（cost 加权）**：
>    在未 refine（λ=0）的条件下，plus 相对 baseline 就应该在 `median_corr_svg` 等指标上有提升；
> - **Step2（SVG refine）**：
>    对同一个 config_id/seed，当 `svg_refine_lambda` 从 0 提高到一个合适值时，
>    应该看到 SVG 空间指标进一步提升，而 cost/约束指标仍在合理范围内。
>
> Stage5 会把这些变化具体量化出来，作为后续配置推荐的依据之一。

#### 4.4 类型 / Unknown 行为（特别看 M1/M2 场景）

在 SimGen 的 M1 / M2 场景里：

- **M1：sc 缺型场景**
  - `sc_missing_types` 非空，Ref 中看不到某些类型，但 World/模拟 ST 中仍存在这些类型；
  - 用来检验“类型插件 + Unknown”在 **“参考没有这种 type 时”** 的行为。
- **M2：ST 缺型场景**
  - `st_missing_types` 非空，World/模拟 ST 中几乎没有某些类型，但 Ref 中仍存在这些类型；
  - 用来检验模块二能否避免在 ST 中“硬挤出”这些不存在的类型。

定义：

- `T_missing`：被刻意设为“缺型”的类型（对 M1/M2 分别含义不同）；
- `T_rare`：被设为“稀有型”的类型。

##### 4.4.1 missing_to_unknown / missing_to_wrongtype

以 M1（sc 缺型）为例：

- 真值中存在 `T_missing`，但 Ref 里没有这种 cell 类型。

我们关心：这些真值属于 `T_missing` 的 mass（在 spot×type 真值矩阵中）被预测成什么？

定义：

- `Mass_missing_true = Σ_s Q_s(T_missing)`
- 在预测中：
  - 分配到 Unknown 的 mass：`Mass_missing_to_unknown`
  - 分配到其他具体类型的 mass：`Mass_missing_to_wrongtype`

指标：
$$
[
 missing_to_unknown_rate = \frac{Mass_missing_to_unknown}{Mass_missing_true}
 ]
$$

$$
[
 missing_to_wrongtype_rate = \frac{Mass_missing_to_wrongtype}{Mass_missing_true}
 ]
$$



> 理想行为：
>
> - baseline：`missing_to_wrongtype_rate` 较高（把缺型 mass 硬分配给某些错误类型）；
> - plus（启用类型插件 + Unknown）：
>   - `missing_to_unknown_rate` 明显提高；
>   - `missing_to_wrongtype_rate` 明显下降。

M2 场景可以做类似定义，只是“缺”的维度从 sc 换成 ST。

##### 4.4.2 稀有类型的虚假扩散与空间熵

对于某个稀有类型 `T_rare`：

1. **虚假扩散程度：**

   - 在真值 `Q_s(T_rare)` 中：
     - 大部分 spot 该类型接近 0，只有少数 spot 有较高占比；
   - 在预测 `P_s(T_rare)` 中：
     - 统计“真值几乎为 0，但预测明显偏高”的 spot 数量：
        [
        rare_type_false_positive_spots = |{s: Q_s(T_{rare}) < \epsilon,, P_s(T_{rare}) > \tau}|
        ]
     - $\epsilon$ 可以设为 0.01，$\tau$ 可以设为 0.1 或 0.05。

2. **空间分布熵：**

   - 对真值和预测分别计算：
     $$
     [
      H^{true}(T_{rare}) = - \sum_s \tilde{Q}*s(T*{rare}) \log \tilde{Q}*s(T*{rare})
      ]
     $$

     $$
     [
      H^{pred}(T_{rare}) = - \sum_s \tilde{P}*s(T*{rare}) \log \tilde{P}*s(T*{rare})
      ]
     $$

      其中 $\tilde{Q}_s, \tilde{P}_s$ 是在所有 spot 上归一化后的分布；

   - 若 `H^{pred}` 远大于 `H^{true}`，说明该稀有型在预测中被“扩散”得太均匀、太广。

> 在 M1/M2 场景中，Stage5 会比较 baseline vs plus 的这些指标，验证类型插件是否达到了“宁可 Unknown，也不要瞎扩散”的设计目的。

#### 4.5 稳定性指标（多 seed / 多 λ / 多配置）

在同一场景、同一 backend 下，我们可能对同一配置跑多个 seed，或扫描一系列 `svg_refine_lambda`。

**4.5.1 映射结果的一致性**

对任意两个 run A 和 B：

- cell-level：
  - 统计 `cell_id` 在两次 run 中的预测 spot 是否相同的比例；
  - 或者对 cell×spot soft 分布计算 KL/JS 距离的均值。
- spot×type-level：
  - 对每个 spot 计算 `P_s^A(t)` 和 `P_s^B(t)` 的 L1 / JS；
  - 在 spot 上统计均值。

**4.5.2 SVG 指标的稳定性**

- 对每个 run 计算 `median_corr_svg`, `median_SSIM_svg`, `median_delta_Moran` 等；
- 看这些 SVG 指标在不同 seed / λ 之间的方差：
  - 方差越小，说明该配置下 SVG 行为越稳定。

> 在 Stage5 的配置推荐中，不仅看平均水平，还要看“在不同场景和 seed 之间是否稳定”。

------

### 五、CytoSPACE 专用评估（5A）

> 这部分对 backend=cytospace 特有，用来回答：
>  “我们的 SVG+HVG 加权和 SVG refine 是否把 CytoSPACE 自己的 cost/约束搞崩了？”

#### 5.1 cost 相关指标

假设 `cyto_cost_matrix_*.npz` 或 `cyto_objective_stats_*.json` 记录了：

- cell×spot 的 cost matrix（或若干压缩统计）；
- solver 优化过程中的 objective 值等。

指标示例：

1. **平均单细胞 cost**
   $$
   [
    mean_cost_per_cell = \frac{1}{N_{cell}} \sum_{i} \sum_s A_{i,s} \cdot C_{i,s}
    ]
   $$
   

   - milestone：plus 不应该无缘无故导致 cost 爆炸。

2. **总 cost**
   $$
   [
    total_cost = \sum_{i,s} A_{i,s} \cdot C_{i,s}
    ]
   $$
   

   - 可以和 baseline 对比，看 plus 是否整体 objective 稍有增加但换来更好的 SVG / 类型表现。

3. **cost 分布形状**

   - 统计不同 range cost 的 cell 数量；
   - 看是否出现极端 cost 的长尾（可能意味着某些 cell 被强行塞到很不合理的位置）。

#### 5.2 约束满足度（Constraint Satisfaction）

CytoSPACE 内部通常有：

- 每个 spot 的 cell 总数约束；
- 每个 spot×type 的 cell 数约束（或软约束）。

如果 `cyto_constraints_*.json` 提供了期望值：

1. **spot 总 cell 数 RMSE**

   - 真值/约束：`E_total(s)`；
   - 预测：`A_total(s)`（由 cell_spot_matrix 聚合得到）；

   $$
   [
    spot_total_count_rmse = \sqrt{\frac{1}{|\mathcal{S}|} \sum_s (E_{total}(s) - A_{total}(s))^2}
    ]
   $$

   

2. **spot×type cell 数 RMSE**

   - 真值/约束：`E_st(s,t)`；
   - 预测：`A_st(s,t)`；
   - 可计算：
     - `spot_type_count_rmse` 或 L1 误差的平均值。

> 概念上我们希望：
>
> - plus 版本在 cost/约束层面 **不明显劣于** baseline；
> - 理想情况：在保住或略微提升 cost/约束满足度的同时，大幅改善 SVG / 类型/Unknown / 稳定性等指标。

#### 5.3 refine 前后（λ=0 vs λ>0）的行为分析

如果 `svg_refine_stats_plus_run*.json` 记录了 refine 前后的 SVG 指标和 cost 变化：

- 可以直接画出：
  - λ 与 `median_corr_svg`、`mean_cost_per_cell`、`spot_total_count_rmse` 等指标之间的关系；
- 用于判断：
  - 是否存在一个“甜点区间”，在该 λ 范围内，SVG 明显提升，但 cost/约束尚可接受；
  - 过大的 λ 是否导致 cost 急剧上升或约束严重破坏。

------

### 六、配置扫描与 Stage6 的“前置仲裁支撑”

Stage5 不在模拟数据上做“最后仲裁”（那是 Stage6 在真实数据上的任务），
 但会提供两个直接喂给 Stage6 的关键信息：

1. 不同配置在有真值世界里的表现排名（哪些配置明显较差可以提前排除）；
2. 对“比较靠谱”的配置给出一个 λ/seed/config_id 的推荐区间。

#### 6.1 单场景内的配置排序

对每个 `scenario_id` 和每个 backend（以 CytoSPACE 为例）：

1. 汇总所有 baseline / plus run 的指标到一个 DataFrame，每行一个 run，包含：
   - `backend`, `mode`（baseline/plus）、`config_id`, `seed`, `svg_refine_lambda`, `enable_refine` 等；
   - 通用指标（spot×type、SVG、类型/Unknown、稳定性等）；
   - CytoSPACE 专用指标（cost、约束）。
2. 设定一个简单的综合排序规则，例如（仅示意）：
   - 第一级：`median_corr_svg` 越高越好；
   - 第二级：`mean_L1_spot_type` 越低越好；
   - 第三级：`missing_to_wrongtype_rate` 越低越好（在 mismatch 场景）；
   - 第四级：稳定性相关指标方差越小越好；
   - 第五级：cost/约束显著恶化时扣分。
3. 对 baseline 保留作为对照线；对 plus run 按上述规则排序，标记表现较差的配置。

#### 6.2 跨场景的配置稳定性

- 对相同 `(config_id, svg_refine_lambda, enable_refine)` 在不同场景中的表现做聚合：
  - 看某配置是否在“S0 + M1 + M2 …”中都表现不错，还是只在某类场景好；
- 对这些“跨场景表现优良”的配置集合打标签，形成推荐列表。

最终写入 `stage5_config_recommendation.json`，例如：

```json
{
  "cytospace": {
    "recommended_lambda_range": [0.1, 0.5],
    "recommended_configs": [
      {
        "config_id": "svg_cost_v1",
        "enable_refine": true,
        "svg_refine_lambda": 0.3,
        "suggested_seeds": 3,
        "notes": "S0/M1/M2 场景中 SVG 指标显著优于 baseline，类型/Unknown 行为合理，稳定性良好，cost/约束变化可接受。"
      },
      {
        "config_id": "svg_cost_v1",
        "enable_refine": false,
        "svg_refine_lambda": 0.0,
        "suggested_seeds": 3,
        "notes": "无 refine 的 plus 基线，用于对比 Step2 的增益。"
      }
    ]
  }
}
```

> Stage6 会直接读取这份 JSON，在真实 BRCA 上**只在推荐配置集合里**做多次 run，
>  然后用 **无真值的 SVG-aware 指标 + 类型/Unknown 行为 + cost/约束** 做最终的 Step3 仲裁。

------

### 七、小结：阶段五在“三步解题方案”中的角色

最后用一句话把阶段五的位置钉死：

> **在 SimGen 的有真值世界里，阶段五负责把 Step1（SVG+HVG 加权 cost）和 Step2（SVG 空间 refine）的真实效果量化出来，
>  并据此挑出一批“又准又稳”的配置，写成配置推荐文件交给阶段六，
>  让真正的 Step3（SVG-aware 仲裁）在真实数据上有据可依，而不是盲目调参。**

------











------

## 阶段六（Stage 6）：真实数据验证实验 + SVG-aware 仲裁（Step3 落地）

> **核心定位（更新版）：**
>
> 1. 在 **没有真值 cell→spot 映射** 的前提下，用真实 BRCA ST+sc 数据，
>     对比 **baseline 映射** 和 **一组 plus 候选映射（SVG+类型插件，多配置 / 多 seed / 多 λ）** 的行为；
> 2. 基于“表达拟合 + SVG 空间模式 + 类型分布合理性 + Unknown 使用 + CytoSPACE 约束”等多类**无真值/弱真值指标**，
>     对这些 plus **候选 run 做 SVG-aware 仲裁（Step3 真正落地）**，
>     选出一条“**canonical plus 映射链路**”；
> 3. 确保最终选出来的 plus 映射在真实数据上是：
>    - 表现良好（或者不比 baseline 差太多但更稳定 / 更合理）；
>    - 行为符合文献共识，不靠“奇怪的方式骗指标”。

------

### 一、阶段 6 在整体项目中的位置

整体框架（只列关键）：

- 0：选数据 + 环境测试
- 1：真实数据预处理（R + Python）
- 2：模块一 – SVG+HVG 动态加权插件
- 3：模块二 – 细胞类型不匹配 / Unknown-aware 插件
- 4：映射整合（Mapping Orchestrator + 多 backend，当前先实现 CytoSPACE）
- SimGen：模拟数据构建模块
- 5：基于模拟数据的验证（有真值）
- **6：真实数据验证 + SVG-aware 仲裁（无真值，但有生物学和技术一致性）**
- 7：可视化 + 报告整合（把阶段 5+6 的结果组合成论文 Figure & 表格）

你可以理解为：

- **阶段 5（模拟）回答**：

  > “在有真值的理想世界里，哪些配置是更好的？Step1 & Step2 在真值指标上确实有收益。”

- **阶段 6（真实数据）回答**：

  > “在真实 BRCA 上，用阶段 5 推荐的那批配置跑出的一堆 plus 候选映射里，
  >  哪一些在**无真值指标**上表现最合理？
  >  我们用 SVG-aware 仲裁，从这堆候选里选出**最终 canonical plus 映射**。”

换句话说：

- 阶段 5：在模拟数据上 **筛选配置空间**（Step3 的“前半段”）；
- 阶段 6：在真实数据上用一套 SVG-aware 指标 **做最后的仲裁 + 落盘 canonical 结果**（Step3 的“后半段”，真正决定你论文里用哪条映射）。

------

### 二、阶段 6 的输入与依赖

#### 2.1 真实数据来源（BRCA）

1. **真实 sc 数据**（来自 CytoSPACE BRCA 数据）
   - 原始文件示例：
     - `brca_scRNA_GEP.txt`
     - `brca_scRNA_celllabels.txt`
   - 阶段 1 预处理之后，得到统一格式：
     - `real_sc_expression_processed.h5ad / .csv`
     - `real_sc_metadata.csv`（至少包含 `celltype`）
2. **真实 ST 数据（Visium BRCA）**
   - 原始文件示例：
     - `brca_STdata_GEP.txt`
     - `brca_STdata_coordinates.txt`
     - SpaceRanger 输出：
        `Visium_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5` + `spatial/` 图像
   - 阶段 1 预处理之后，得到：
     - `real_st_expression_processed.h5ad / .csv`
     - `real_st_coordinates.csv`

> 阶段 6 **不再改动**这些预处理结果，只负责读取使用。

------

#### 2.2 来自阶段 2 / 3 的插件输出

- **阶段 2：SVG+HVG 动态加权插件**
  - 输出：
    - `gene_weights.csv`
      - 包含：`hvg_score`、`svg_score`、`hvg_norm`、`svg_norm`、`final_weight` 等
    - 可选：`selected_genes.csv` / `plugin_genes.txt`
      - 标记哪些基因为插件真正使用的“特征基因”。
- **阶段 3：类型不匹配 / Unknown-aware 插件**
  - 输出：
    - `plugin_type/updated_type_prior_matrix.csv`
    - `plugin_type/type_mapping_meta.json`
      - 记录：哪些原始类型被合并、哪些被标为 Unknown 候选、先验矩阵构造方式等。

> 阶段 6 中：
>
> - 在 **SVG 专用指标**、**HVG/marker 分组指标** 时，用阶段 2 的基因集合和权重做分组 / 加权；
> - 在 **类型 / Unknown 行为** 指标时，用阶段 3 的 `plugin_type` 和 `type_mapping_meta` 来解释 “Unknown 到底是谁、是怎么来的”。

------

#### 2.3 来自阶段 4 的映射结果（多 backend，多 run）

对每个 backend（当前重点是 `cytospace`），阶段 4 会输出**baseline + 多个 plus run** 的三件套和 meta 信息：

```text
result/real_brca/stage4_mapping/
  cytospace/
    baseline/
      cell_assignment_baseline.csv
      cell_spot_matrix_baseline.npz
      spot_type_fraction_baseline.csv
      cyto_cost_matrix_baseline.npz          # 可选（6A 用）
      cyto_constraints_baseline.json         # 可选
      cyto_objective_stats_baseline.json     # 可选
    plus_svg_type/
      # 注意：这里不再只有“一个 plus”，而是可能有多个 run：
      mapping_manifest_plus.json             # 列出所有 plus run 的 meta
      run_configA_seed0_lambda0.0/
        cell_assignment_plus.csv
        cell_spot_matrix_plus.npz
        spot_type_fraction_plus.csv
        cyto_cost_matrix_plus.npz
        cyto_constraints_plus.json
        cyto_objective_stats_plus.json
      run_configA_seed1_lambda0.3/
        ...
      run_configB_seed0_lambda0.3/
        ...
  tangram/    # 将来可以接入
    baseline/
    plus_svg_type/
      ...
```

说明：

- **baseline**：每个 backend 保留一个“原版基线”结果，作为 Stage6 的对照；
- **plus_svg_type**：
  - 不同的 `(config_id, svg_refine_lambda, seed, enable_refine)` 组合各对应一个子目录；
  - 一个统一的 `mapping_manifest_plus.json` 记录每个 run 的：
    - `run_id`
    - `config_id`
    - `seed`
    - `svg_refine_lambda`
    - `enable_refine`
    - 其他需要的 meta（比如是否启用了类型先验等）。

> 阶段 6 会读取这个 manifest，把**所有 plus 候选 run**都视作“备选映射”，
>  逐一计算 6B/6A 指标，然后做 SVG-aware 仲裁，最终选出一个“canonical plus run”。

------

#### 2.4 来自阶段 5 的配置推荐（Stage5_config_recommendation）

阶段 5 在模拟数据上已经对 config 空间做了“有真值世界”的筛选，输出：

```text
result/stage5_summary/
  stage5_scenario_summary.json
  stage5_config_recommendation.json
```

示例结构（只示意）：

```json
{
  "cytospace": {
    "recommended_lambda_range": [0.1, 0.5],
    "recommended_configs": [
      {
        "config_id": "svg_cost_v1",
        "enable_refine": true,
        "svg_refine_lambda": 0.3,
        "suggested_seeds": 3
      },
      {
        "config_id": "svg_cost_v1",
        "enable_refine": false,
        "svg_refine_lambda": 0.0,
        "suggested_seeds": 3
      }
    ]
  }
}
```

在 **阶段 6** 中：

1. Orchestrator 会优先根据 `stage5_config_recommendation.json` 决定：
   - 对哪些 `(config_id, λ, enable_refine)` 组合进行真实 BRCA 映射；
   - 每个组合跑多少个 seed。
2. 对于这些“推荐配置”产生的 plus run，阶段 4 会放在 `plus_svg_type/` 对应子目录，
    并在 `mapping_manifest_plus.json` 中写清楚 Meta。
3. 阶段 6 只对这些“推荐配置下的 run”做比较和仲裁；
   - 某些明显极端或“stage5 明确不推荐”的配置可以直接在真实数据中跳过。

> 这样 Stage 6 不是在整个配置空间里“瞎选”，
>  而是在 **“经模拟 + 真值验证过的一小块安全区间”** 内，
>  用真实数据的 SVG-aware 指标做最后一轮筛选与落盘。

------

### 三、阶段 6 的总体评估与仲裁思路（Step3 落地）

由于真实数据没有 cell→spot 真值，我们采用三层设计：

> 1. **多候选 run 评估**：
>     对 baseline 和所有 plus 候选 run，统一计算一套“无真值 / 弱真值”指标（6B + 6A）；
> 2. **SVG-aware 仲裁规则**：
>     在这些指标上，用一套明确的规则筛掉“明显问题 run”，
>     再在“合格集合”中，根据 SVG / 表达拟合 / 类型 / 空间结构 / cost 等综合打分排序；
> 3. **选出 canonical plus 映射**：
>     为每个 backend（至少是 CytoSPACE）选出 1 条 canonical plus run （以及可选 top-k 候选），
>     把选择结果写入 JSON，供阶段 7 和论文主结果使用。

#### 3.1 6A / 6B 两条评估线的扩展（从“baseline vs plus”到“baseline vs 一堆 plus run”）

原本你的设计里：

- **6A：CytoSPACE 专用评估**
  - cost / 约束 / 细胞数分配等；
- **6B：通用评估**
  - 只依赖 ST 表达 & 坐标 + 映射三件套。

在新方案中，两条线继续保留，只是评估对象从：

- “baseline vs（单个）plus”

扩展为：

- “baseline vs（多个 plus run）”。

即：

- baseline 做一遍，得到一组 baseline 指标向量；
- 每个 plus run（不同 config / seed / λ）也算一组指标向量；
- 后续的仲裁规则都是在这些向量上“做比较 + 选优”。

> 直观理解：
>  baseline 是一个参照点；
>  一堆 plus run 在各类指标空间里形成很多点，
>  我们希望在“距离 baseline 不太远（不出大错）”的前提下，
>  选出那些“在 SVG / 类型 / 空间结构上明显好一点”的点。

#### 3.2 SVG-aware 仲裁规则（Step3 真正落地）

仲裁过程可以拆成三步：

1. **硬过滤（Hard Filters）——先把明显不合格的 run 剔除：**

   对每个 plus run，检查：

   - 表达拟合（4.1）：
     - SVG/marker/HVG 上的中位相关性是否比 baseline **低太多**（例如下降超过预设阈值 Δ_expr）；
   - 空间结构（4.4）：
     - 邻域类型一致性、表达–类型平滑一致性是否出现明显恶化；
   - Unknown 使用（4.5）：
     - 是否出现大面积“拟合很好但 Unknown 很高”的奇怪区域；
   - CytoSPACE 约束（5.2）：
     - cost 是否暴涨；
     - spot 总细胞数 / spot×type 约束残差是否显著恶化。

   若某个 run 在以上任一维度严重违反阈值，则标记为“不合格 run”，不参与后续排序。

2. **软打分（Soft Scores）——在合格集合里做多维排序：**

   对剩下的 plus run（合格集合）定义几个“子得分”（越高越好）：

   - **ExprScore(run)**：表达拟合得分
     - 基于 SVG+marker 基因的 per-gene / per-spot 相关性提升量；
     - 可以定义为：
        `ExprScore = α1 * Δmedian_corr_svg + α2 * Δmedian_corr_marker`（相对 baseline 的增量）；
   - **SVGScore(run)**：SVG 空间模式得分
     - 基于 4.2 中的 SVG 空间相关性、结构指标；
   - **TypeScore(run)**：marker–type / 类型空间模式得分
     - 基于 4.3.1 / 4.3.2；
   - **StructScore(run)**：空间结构得分
     - 基于 4.4 邻域一致性、表达–类型平滑一致性；
   - **UnknownScore(run)**：Unknown 使用合理性
     - Unknown 在误差高 / 不一致区域的 enrichment 程度；
   - **CostScore(run)**：（仅 CytoSPACE）
     - 相对于 baseline 的 cost 变化、约束残差变化。

   然后合成一个总分：
   $$
   [
    Score(run) = w_{\text{expr}} \cdot ExprScore
    \+ w_{\text{svg}} \cdot SVGScore
    \+ w_{\text{type}} \cdot TypeScore
    \+ w_{\text{struct}} \cdot StructScore
    \+ w_{\text{unk}} \cdot UnknownScore
    \+ w_{\text{cost}} \cdot CostScore
    ]
   $$
   权重 (w_{\cdot}) 可以在阶段 6 配置文件中设定，
    并在论文里解释：我们更看重哪些维度（比如 SVG 和类型/结构）。

3. **选出 canonical run（以及备选 top-k）**

   - 对每个 backend：
     - 保留 baseline 的完整指标（不参与评分，只做参照）；
     - 在合格的 plus run 中按 Score(run) 排序；
   - 选取：
     - **canonical_plus_run**：Score 最大的一条；
     - （可选）**top_k_runs**：得分排前 k 名的一小撮 run 用于附录展示。

   并把结果写入：

```text
result/real_brca/stage6_eval/
  generic/<backend>/
    metrics_stage6_generic_runs.csv     # 每个 run 一行的指标
    metrics_stage6_generic_summary.json # 把基线 & 分布等汇总
    stage6_selection_<backend>.json     # 仲裁结果：选中的 canonical run 等
  cytospace/
    metrics_stage6_cytospace_specific.json
```

> 这样，阶段 6 不仅仅“对比 baseline vs plus”，
>  而是完成了 Step3 的最后一步：
>  **在推荐配置空间中，用 SVG-aware 指标从多条 plus 映射里选出一条最靠谱的 canonical plus 映射。**

------

### 四、通用评估指标（6B 核心，6A 也会用）

> 注意：以下所有指标都是 **per run** 计算的，
>  baseline 和每个 plus run 都会得到一整套指标向量，
>  后面仲裁会用这些向量来打分。

记号：

- 真实 ST 表达矩阵：(E \in \mathbb{R}^{S \times G})
  - S: spots 数
  - G: 基因数
- 映射得到的 cell→spot 矩阵：(A)（来自 `cell_spot_matrix`；如只有硬分配则按 one-hot 构造）
- sc 表达矩阵：(X \in \mathbb{R}^{C \times G})

#### 4.1 基本思路：预测 ST 表达 vs 实测 ST 表达

**重构方式（方法无关）：**

对每个 spot s、基因 g：
$$
[
 \hat{E}*{s,g} = \sum*{i \in \text{cells}} A_{i,s} \cdot X_{i,g}
 ]
$$
这样得到一个**预测 ST 表达矩阵** (\hat{E})，与真实 ST 表达 (E) 同维度。

接下来在三个尺度上看拟合度：

##### 4.1.1 基因层面（per-gene）

对每个基因 g：

- 计算 across spots 的 Pearson 相关性：

$$
[
 r_g = \operatorname{corr}(E[:, g], \hat{E}[:, g])
 ]
$$

然后：

- 看 `r_g` 的整体分布（箱线图 / ECDF）：
  - baseline vs 各个 plus run；
- 尤其关注几类基因子集：
  - HVG 集合；
  - SVG 集合（由阶段 2 提供）；
  - marker 基因集合（按类型手工整理或文献给出）。

**预期：**

- 对 SVG/marker/HVG 而言：
  - 合格的 plus run 至少不应在这个分布上“明显左移”；
  - 理想情况下，会略有右移（中位数稍微升高）；
- 若某个 plus run 让多数基因 `r_g` 下降明显，就在硬过滤里被淘汰。

##### 4.1.2 spot 层面（per-spot）

对每个 spot s：

- 计算 across genes 的相关性：

$$
[
 corr_spot[s] = \operatorname{corr}(E[s, :], \hat{E}[s, :])
 ]
$$

同样可先过滤低表达基因。

**含义：**

- `corr_spot` 高 → 说明该 spot 上整体表达结构被 sc 映射很好解释；
- 可以：
  - 比较 baseline vs 各个 plus run 的 `corr_spot` 分布；
  - 观察特定空间区域（比如 Tumor-rich / Immune-rich）的变化。

##### 4.1.3 汇总成表达拟合子得分 ExprScore

在仲裁时，可以从这些指标中抽一个子得分，例如：

- 对基因层面，取：
  - `Δmedian_corr_marker` = plus 中位数 − baseline 中位数；
  - `Δmedian_corr_svg`；
- 对 spot 层面：
  - `Δmedian_corr_spot`。

然后做一个加权平均变成 `ExprScore(run)`，用于 Score(run) 中的表达拟合部分。

------

#### 4.2 SVG 专用指标（对模块一 / Step1+Step2 的“对口考核”）

使用阶段 2 的 `gene_weights.csv`：

- 选出一组 SVG 候选基因 (G_{\text{svg}})，比如：
  - `svg_score` > 阈值，或
  - `selected_in_plugin == 1`（最终参与 cost 的基因）；
- 定义权重 (w_g)：可用 `final_weight(g)` 或 `svg_norm(g)`。

对每个 SVG 基因 g：

1. 真值空间表达：`E[:, g]`；
2. 预测空间表达：`\hat{E}[:, g]`；
3. 计算：
   - Pearson 相关 `corr_svg[g] = corr(E[:,g], Ē[:,g])`；
   - 可选空间结构指标（如 reshape 为 2D 图像后的 SSIM / Moran's I 等）。

汇总：

- `corr_svg[g]` 的加权中位数、均值；
- 如有 SSIM / Moran’s I，也做相应统计。

**预期：**

- 对比 baseline：
  - plus run 在 `median_corr_svg` 等指标上应不差（理想是更好）；
  - 已被阶段 5 过滤过的配置一般不会极差；
- 在仲裁阶段：
  - `SVGScore(run)` 可以直接基于 `Δmedian_corr_svg`、加权均值差等构造。

------

#### 4.3 Cell-type 相关指标（spot×type 层面）

无真值的 spot×type，利用的是“弱真值信息”：

1. **每个类型的 marker 集合 (G_t)**；
2. **映射得到的 `P_s(t)`：spot s 上类型 t 的预测占比**（来自 `spot_type_fraction`）。

##### 4.3.1 Marker–Type 一致性（Marker–Type Concordance）

对每种类型 t：

1. 真实 marker 表达：

$$
[
 M_s(t) = \frac{1}{|G_t|} \sum_{g \in G_t} E[s, g]
 ]
$$



1. 对所有 spot，计算：

$$
[
 corr_type_marker[t] = \operatorname{corr}(P_s(t), M_s(t))
 ]
$$

然后比较 baseline vs 各 plus run 的 `corr_type_marker[t]` 分布，重点关注 Tumor / Immune / Stromal 等关键类型。

**预期：**

- 对关键类型，合格的 plus run 不应让 `corr_type_marker` 明显变差；
- 若某 run 在多个关键类型上 `corr_type_marker` 大幅下降，就会在硬过滤里被剔除。

##### 4.3.2 类型空间模式（Type Spatial Patterns）

类似 SVG：

- 把 `P_s(t)` 和 `M_s(t)` 看成两张空间“热力图”；
- 可以计算：
  - 空间自相关（Moran’s I）；
  - 局域差异（例如空间卷积后的 L2 距离）；
- 这些指标可以作为 `TypeScore(run)` 的空间补充。

------

#### 4.4 空间结构指标（Spatial Structure）

##### 4.4.1 邻域类型一致性（Neighborhood Consistency）

基于 spot 邻接图（Visium 网格）：

- 对每个 spot s，看其邻域 (\mathcal{N}(s)) 中的类型分布；
- 定义：
  - 邻域内类型分布与本 spot 类型分布的 KL / JS / 相关性；
- 聚合为：
  - 全局均值 / 中位数，baseline vs plus run 比较。

**直观：**

- 若 plus 引入大量“孤岛 spot”（周围全是类型 A，自身全是类型 B），
   邻域一致性会明显变差，这类 run 在仲裁中会被扣分甚至过滤。

##### 4.4.2 表达平滑度 vs 类型平滑度的一致性

- 利用 ST 表达 E 的空间平滑度（比如每个 spot 与其邻域的表达差异）；
- 以及映射得到的 `P_s(t)` 的平滑程度；
- 比较“表达差异大但类型差异小”的区域数目（可能提示类型 over-smoothing），
   和“表达差异小但类型差异大”的区域（类型分布太跳）。

这些都可以凝练为 `StructScore(run)` 的组成部分。

------

#### 4.5 Unknown 类型使用情况（无真值视角）

Unknown 类型用得好不好，是模块二的核心考点之一。

思路：

1. 定义：
   - 每个 spot 的 Unknown 占比：`P_s(Unknown)`；
   - 每个 spot 的表达重构误差：
     - `reconstruction_error_s = ||E[s,:] - Ē[s,:]||_2` 或 1 − corr；
   - 每个 spot 的 marker–type 不一致度：
     - `marker_mismatch_s`（可以是 marker–type 相关性不够高、或绝对差的某种聚合）。
2. 分析：
   - `P_s(Unknown)` 与 `reconstruction_error_s` 的相关性；
   - `P_s(Unknown)` 与 `marker_mismatch_s` 的相关性；
   - 是否存在大量“误差很小但 Unknown 占比很高”的区域。

**解读：**

- 若 Unknown 更高的 spot 恰好是**表达拟合差 / marker 不一致严重的** spot：
  - 说明 Unknown 真正在“吸收难解释的残差”；
- 若 Unknown 在拟合很好、marker 一致的地方大面积出现：
  - 说明插件二在真实数据上有滥用风险，该 run 在仲裁中会被扣分甚至淘汰。

`UnknownScore(run)` 可以基于以上相关性和 enrichment 指标构造。

------

#### 4.6 将 4.1–4.5 指标整合为仲裁用的 run-level 特征

最终，我们为每个 run 产生一行记录，包含：

- 表达拟合指标（基因/spot/分组）；
- SVG 专用指标；
- 类型相关指标（marker–type、一致性）；
- 空间结构指标；
- Unknown 使用指标。

这些会写入：

```text
result/real_brca/stage6_eval/generic/<backend>/metrics_stage6_generic_runs.csv
```

为后续仲裁逻辑提供完整的特征输入。

------

### 五、CytoSPACE 专用评估（6A）

在 6B 指标基础上，CytoSPACEBackend 还提供内部信息：

1. **总 cost / 平均 cost**
2. **数量约束（spot 总细胞数 / spot×type 细胞数）的满足情况**
3. （可选）求解过程的收敛状态

#### 5.1 cost 相关指标

基于 `cyto_cost_matrix_*.npz` 或 `cyto_objective_stats_*.json`，对每个 run 计算：

- `total_cost(run)`：整体 objective；
- `mean_cost_per_cell(run)`：平均单细胞 cost；
- cost 分布的尾部情况（有无极端高 cost 的长尾）。

在仲裁中：

- 若 plus run 的 cost **成倍增加**，需要非常小心；
- 合格的 plus run 应在 cost 上与 baseline 同一数量级，甚至略有改善。

#### 5.2 约束满足度（Constraint Satisfaction）

若存在 `cyto_constraints_*.json`：

- 对每个 spot s，约束的总细胞数 `E_total(s)` 与预测 `A_total(s)`；
  - 得到 `spot_total_count_rmse`；
- 若有类型层约束 `E_st(s,t)` 与 `A_st(s,t)`：
  - 得到 `spot_type_count_rmse` 等。

在仲裁中：

- 约束残差明显比 baseline 大的 run，直接进入硬过滤淘汰；
- 约束残差明显更好的 run，在 `CostScore(run)` 中会被加分。

#### 5.3 将 6A 指标纳入 Score(run)

在最终 `Score(run)` 中，CytoSPACE 的 run 还会使用：

- cost 增减情况；
- 约束残差变化情况；

组合成 `CostScore(run)`，参与总分。

------

### 六、阶段 6 的脚本与接口设计（伪代码级，含仲裁）

#### 6.1 通用评估 + 仲裁入口（6B）

```python
def eval_stage6_generic(
    backend: str,                # "cytospace" / "tangram" / ...
    real_result_root: str,       # result/real_brca/stage4_mapping/<backend>
    real_data_root: str,         # data/real_brca/ (含 ST 表达和坐标)
    stage2_output: str,          # SVG gene 列表 & weights
    stage3_output: str,          # 类型先验 / Unknown 信息
    stage5_config_path: str,     # stage5_config_recommendation.json
    out_dir: str                 # result/real_brca/stage6_eval/generic/<backend>
):
    """
    1. 读取真实 ST 表达 E 和坐标 coords
    2. 读取 SVG/HVG 列表、marker gene 列表、Stage2 gene_weights
    3. 读取 Stage5 推荐配置：
       - 对 backend 对应的 config_id / lambda / enable_refine / seeds 做筛选
    4. 读取 Stage4 的 baseline 结果：
       - cell_assignment_baseline
       - cell_spot_matrix_baseline
       - spot_type_fraction_baseline
    5. 读取 Stage4 的 plus 候选结果：
       - 通过 plus_svg_type/mapping_manifest_plus.json 找到所有 run_id
       - 只保留在 Stage5 推荐配置集合中的 run（其余视为“非推荐配置”，默认不参与仲裁）
    6. 对 baseline + 每个 plus run：
       - 重建 ST 表达 Ē
       - 计算 4.1~4.5 中定义的所有通用指标
    7. 将所有 run 的指标写到 metrics_stage6_generic_runs.csv
    8. 基于这些指标，执行 SVG-aware 仲裁：
       - 硬过滤：剔除明显不合格 run
       - 软打分：根据 ExprScore / SVGScore / TypeScore / StructScore / UnknownScore 等计算 Score(run)
       - 选出 canonical_plus_run 和 top_k_runs
    9. 输出：
       - metrics_stage6_generic_runs.csv         # 每个 run 一行
       - metrics_stage6_generic_summary.json     # 各类指标分布 + baseline 对照
       - stage6_selection_<backend>.json        # 仲裁结果（canonical run + 备选 run）
    """
```

#### 6.2 CytoSPACE 专用评估入口（6A）

```python
def eval_stage6_cytospace_specific(
    real_result_root: str,    # result/real_brca/stage4_mapping/cytospace
    generic_metrics_path: str,# 上一步 metrics_stage6_generic_runs.csv
    out_dir: str              # result/real_brca/stage6_eval/cytospace
):
    """
    1. 读取 cytospace 的 baseline + plus run 的:
       - cyto_cost_matrix_*.npz
       - cyto_constraints_*.json
       - cyto_objective_stats_*.json
    2. 对每个 run 计算:
       - total_cost, mean_cost_per_cell
       - spot_total_count_rmse, spot_type_count_rmse（如有）
    3. 将这些 cyto 专用指标 merge 进
       metrics_stage6_generic_runs.csv 的每一行，得到完整的 run-level 特征表
    4. 如有必要，更新 Score(run) 中的 CostScore 部分，并重算排序（或在 6.1 中就已合并）
    5. 输出:
       - metrics_stage6_cytospace_specific.json
       - 可选绘图：cost 分布、大幅度 bad run 标记、约束残差直方图等
    """
```

在 `main.py` 中的调用关系可以类似：

```python
if args.run_stage6:
    backends = args.backends.split(",")  # e.g. "cytospace,tangram"

    for backend in backends:
        eval_stage6_generic(
            backend=backend,
            real_result_root=f"{result_root}/real_brca/stage4_mapping/{backend}",
            real_data_root=f"{data_root}/real_brca",
            stage2_output=f"{result_root}/real_brca/stage2_svg_plugin",
            stage3_output=f"{result_root}/real_brca/stage3_type_plugin",
            stage5_config_path=f"{result_root}/stage5_summary/stage5_config_recommendation.json",
            out_dir=f"{result_root}/real_brca/stage6_eval/generic/{backend}"
        )

    if "cytospace" in backends:
        eval_stage6_cytospace_specific(
            real_result_root=f"{result_root}/real_brca/stage4_mapping/cytospace",
            generic_metrics_path=f"{result_root}/real_brca/stage6_eval/generic/cytospace/metrics_stage6_generic_runs.csv",
            out_dir=f"{result_root}/real_brca/stage6_eval/cytospace"
        )
```

> 最终，`stage6_selection_cytospace.json` 中会写明：
>
> - 哪个 run_id 被选为 canonical plus 映射；
> - 它对应的 config_id / λ / seed；
> - 在关键指标上的表现（相对 baseline 的提升 / 变化）。

------

### 七、阶段 6 完成的判断标准（验收 Checklist）

1. **代码层面**

   - `eval_stage6_generic` 和 `eval_stage6_cytospace_specific` 可以在 BRCA 数据上正常运行；
   - 对 `backend="cytospace"` 至少已完成：
     - baseline + 多个 plus run 的指标计算；
     - 完整的 SVG-aware 仲裁流程（硬过滤 + 打分 + 选 canonical）。

2. **输出层面**

   - 存在并内容合理：
     - `result/real_brca/stage6_eval/generic/cytospace/metrics_stage6_generic_runs.csv`
     - `result/real_brca/stage6_eval/generic/cytospace/stage6_selection_cytospace.json`
     - `result/real_brca/stage6_eval/cytospace/metrics_stage6_cytospace_specific.json`
   - 有若干关键可视化（箱线图/散点图/热图等），用于展示：
     - baseline vs canonical plus 的差异；
     - 不同 plus run 在 SVG / marker / 类型 / Unknown / cost 等维度上的表现分布。

3. **结果层面（针对 canonical plus run）**

   - 表达拟合：
     - 对 SVG 和 marker 基因而言，canonical plus 至少不明显比 baseline 差，最好有一定提升（相关性分布向右移）；
   - marker–type 一致性：
     - 对关键 cell type（例如 Tumor / Immune），canonical plus 的 `corr_type_marker` 不劣于 baseline；
   - 空间结构：
     - canonical plus 不引入大量“噪声 island”，邻域一致性指标不显著变差；
   - Unknown 使用：
     - Unknown 占比较高的区域往往也是表达拟合差 / marker 不一致区域；
     - 没有出现大面积“莫名其妙到处都是 Unknown”的现象；
   - CytoSPACE 专用：
     - canonical plus 没有让 cost 急剧上升，也没有让约束残差恶化。

4. **论文叙述支撑（联合阶段 5 + 阶段 6）**

   - 通过阶段 5（模拟 + 真值） + 阶段 6（真实数据 + SVG-aware 仲裁），可以在论文中写出类似结论：

     > “在模拟数据上，本方法在多项有真值指标中优于 baseline，并据此确定了一块安全有效的配置区间；
     >  在真实 BRCA 数据上，我们在该区间内进行了多次映射，并通过一套 SVG-aware 仲裁指标选出了 canonical plus 映射。
     >  与原始 CytoSPACE 相比，canonical plus 映射在表达拟合、SVG 空间模式、marker–type 一致性、空间结构以及 CytoSPACE 约束满足度方面表现良好，
     >  未见明显副作用，说明本方法在真实应用场景中具有实用性和稳定性。”

------













------

## 阶段七（Stage 7）：可视化与报告整合阶段

> **核心定位：**
>
> - 把 **模块一（SVG 插件）+ 模块二（类型匹配 / Unknown 插件）** 的所有证据
>    从 **模拟数据（有真值）+ 真实数据（无真值）** 两条线，整理成一套清晰的故事。
>
> - 同时，把 **“多配置 / 多 seed / 多 λ” 多次运行的结果** 和 **Stage5/6 仲裁选出的 canonical run** 一起可视化出来，
>    明确告诉读者：
>
>   > “我们不是随便挑一条结果，而是在 SVG 空间重构 + 稳定性指标下，**系统地选出最合理的一条映射链路**。”
>
> - 最终输出：**一整套“论文级图表 + 汇总表 + 中文总结报告”**，帮助你从工程项目 → SCI 论文。

------

### 一、在整体项目中的位置和目标

#### 1.1 所有阶段快速回顾

- 0：数据选择 + 环境测试（BRCA + CytoSPACE 基线）
- 1：预处理（R/Seurat + Python）
- 2：模块一 – SVG+HVG 动态加权
- 3：模块二 – 细胞类型不匹配 / Unknown-aware 插件
- 4：映射整合（多 backend，当前以 CytoSPACE 为主）
- SimGen：模拟数据构建模块（S0/M1/M2 等）
- 5：模拟数据验证（有真值）
- 6：真实数据验证（无真值）
- **7：可视化与报告整合（本阶段）**

可以理解为：前面 0–6 阶段都在“产出数据和指标”，而阶段 7 做三件事：

#### 1.2 阶段七要达成的 3 个核心目标

1. **整理所有关键指标和图像**

   - 把阶段 2 / 3 / 5 / 6 的核心指标，从 JSON / CSV / npz 变成：
     - Figure 1 / Figure 2 / Supplementary Fig.X 级别的图；
     - 可直接放进论文 / PPT 的汇总表。
   - **特别是：**
     - 把 Stage5/6 中针对 **不同配置 / seed / λ** 的多次运行结果，一并读入；
     - 用统一的可视化（折线 / 条形 / 小提琴图）把每个配置的指标表现画出来；
     - 同时标出 Stage5/6 已经选出的 **canonical 配置 / canonical run**。

2. **形成一个“论文故事线”**

   - 按“方法 → 模块 1（SVG）→ 模块 2（类型）→ 模拟验证 → 真实验证 → 稳定性与仲裁”的逻辑，

   - 帮你构建一条清晰的 narrative：

     > “我们干了什么 → 为什么这样设计 →
     >  在 **单次运行** 维度上确实有效 →
     >  在 **多配置 / 多 seed** 维度上仍然稳健 →
     >  并且用 SVG 空间重构指标做了严谨的选解。”

3. **形成“论文提交就绪”的资源包**

   - 输出一套结构化的结果：
     - `figures/`：Fig1–Fig5 + 若干补充图（包含“多配置 / 多 seed / 多 λ”稳定性展示）；
     - `tables/`：Table1–5（关键定量结果 + canonical run 指标）；
     - `summary_json/overall_summary.json`：机器 / 脚本友好的总汇 JSON；
     - `report_markdown/Stage7_summary_CN.md`：一份中文长版总结，为写论文/毕业论文提供骨架。
   - 你后续写 SCI 时，只需要围绕 Stage7 输出的这几个模块去扩写即可。

------

### 二、阶段七用什么、产出什么？（输入 / 输出简表）

#### 2.1 输入：来自前面阶段的所有“证据”

1. **阶段 2（SVG 插件）的输出**

   - `plugin_genes.txt`：最终用于映射的插件基因集合 G_plugin；
   - `gene_weights.csv`：每个基因的 `final_weight`；
   - `svg_scores.csv`：SVG 评分结果（包括被选中与未被选中的基因）；
   - 若有：`svg_recon_metrics.json`（在 Stage4 refine / Stage5 模拟中对 SVG 空间重构做的度量）。

2. **阶段 3（类型匹配插件）的输出**

   - `type_support.csv`：类型支持度（strong / weak / unsupported）；
   - `cell_type_relabel.csv`：每个细胞的 `orig_type / plugin_type / status`；
   - `type_prior_matrix.csv`：spot×plugin_type 的先验矩阵；
   - 以及可能的 `unknown_diagnostics.json`（Unknown 使用情况统计）。

3. **阶段 5（模拟数据验证）的输出**

   - 每个模拟场景（S0/M1/M2…）、每个 backend、每种模式（baseline / plus）的：

     - 通用指标：`metrics_generic.json`；
     - CytoSPACE 专用指标：`metrics_cytospace_specific.json`。

   - **扩展用于仲裁（Step3）的一点：**

     - 若我们对同一场景、同一 backend 做了多配置 / 多 seed / 多 λ 的 sweep，
        则会出现例如：

       - `metrics_generic_run_<config_id>.json`

       - 或在一个 JSON 中包含：

         ```json
         {
           "runs": {
             "seed1_lambda0.3": {...},
             "seed2_lambda0.3": {...},
             "seed1_lambda0.5": {...}
           },
           "canonical_run_id": "seed1_lambda0.5"
         }
         ```

       - Stage7 需要识别这些结构，把“所有 run 的表现 + 被选中的 canonical run”一起可视化。

4. **阶段 6（真实数据验证）的输出**

   - 通用指标：`stage6_metrics_generic.json`，包括：
     - ST 表达重构的基因层面相关性（全基因 / SVG 子集）；
     - spot 级别表达结构相似度；
     - marker–type 一致性、空间自相关等。
   - CytoSPACE 专用：`stage6_metrics_cytospace_specific.json`：
     - cost 分布、约束 RMSE、spot 总细胞数等。
   - **与 Step3 相关的扩展：**
     - 若在真实数据上也对多 seed / 多 λ 跑了多次 CytoSPACE-plus：
       - 那么 `stage6_metrics_generic.json` 中会有类似 `runs` + `canonical_run_id` 字段；
       - 或有额外文件如 `stage6_run_scores.csv` 汇总每个 run 的得分。

5. **项目配置与元信息**

   - `config.yaml`：记录各阶段参数（包括 λ_svg、λ_type、seed 等）；
   - `experiments/` 下的配置记录（若有）；
   - 这些信息可以帮助在图中标注“这是哪一组参数 / 哪个 seed”。

#### 2.2 输出：阶段七要产出的东西

1. **Markdown 报告：`report_markdown/Stage7_summary_CN.md`**
   - 内容结构：
     - 简要方法介绍（SVTuner + 两个插件 + Cytospace-plus）；
     - SVG 模块核心结果（Fig2 系列 + 对 canonical 配置的说明）；
     - 类型匹配模块核心结果（Fig3 系列）；
     - 模拟数据评估（Fig4 系列）；
     - 真实数据评估 + 稳定性分析（Fig5 系列 + “多配置 / 多 seed / 多 λ” 可视化）；
     - 总结与展望。
2. **图像：`figures/` 目录**
   - 论文主图（Fig1–Fig5）和若干 Supplement 图（尤其是“多配置 / 多 seed / 多 λ” 对比图）。
3. **表格：`tables/` 目录**
   - Table1：数据集与预处理概要；
   - Table2：SVG 插件相关的数值总结；
   - Table3：类型插件相关的数值总结；
   - Table4：模拟数据评估（有真值）；
   - Table5：真实数据评估 + 稳定性统计。
4. **JSON 汇总：`summary_json/overall_summary.json`**
   - 机器友好的总汇结果
   - 包含：
     - 模块 1 / 模块 2 在模拟 + 真实数据上的提升幅度；
     - 每个 backend 的关键指标；
     - 若存在多 run，则包含：
       - `runs` 的列表（每个 run 的配置 + 指标）；
       - `canonical_run_id`；
       - 以及 canonical run 的各类指标。

------

### 三、阶段七的输出结构设计

#### 3.1 `figures/` 目录结构

建议按主文 / Supplement 组织：

- **主文 Fig1–Fig5（核心故事）：**
  - `Fig1_pipeline_overview.png`
    - 整体框架图：阶段 0–7 + SimGen；
    - 标出模块一 / 二 插件在 pipeline 中的位置；
    - 标出 **Step1（SVG cost 加权）+ Step2（SVG 空间重构 refine）+ Step3（多 run 仲裁）** 在流程中的位置。
  - `Fig2a_svg_vs_hvg_scatter.png`
    - HVG vs SVG 分数散点图；
    - 高亮最终 plugin_genes；
    - 可追加信息：阈值线、选择边界。
  - `Fig2b_svg_spatial_pattern.png`
    - 若干典型 SVG 基因在 ST 空间的表达图；
    - 可做 baseline 处理 vs 插件处理（加权后投影）的对比。
  - `Fig2c_svg_lambda_sweep_metrics.png`（新增，用于 Step3）
    - 横轴：不同 `λ_svg` / 配置 id；
    - 纵轴：SVG 空间重构相关性（如 `R^2_svg`）或 Moransi；
    - 每个点是一条 run（可能还有不同 seed）；
    - **用星号 /颜色高亮 Stage5/6 仲裁选出的 canonical 配置**。
  - `Fig3a_type_mapping_heatmap.png`
    - 原始 celltype 相似度矩阵 vs 插件后的 plugin_type 矩阵；
    - 高亮合并、削弱、unknown 处理的区域。
  - `Fig3b_type_prior_spatial.png`
    - spot×type 先验矩阵的空间热图；
    - baseline vs plus 对比（类型覆盖更合理、Unknown 更集中等）。
  - `Fig4a_sim_S0_spot_type_L1_boxplot.png`
    - S0 场景下，baseline vs plus 的 spot×type L1/JS 分布；
    - 展示“插件不使结果变差，甚至有整体提升”。
  - `Fig4b_sim_M1_missing_to_unknown_bar.png`
    - M1（sc 缺 type）场景中：
      - missing→Unknown vs →wrong type 的比例对比；
      - 强调 Unknown 插件的好处。
  - `Fig4c_sim_multi_run_score_vs_config.png`（新增，用于 Step3）
    - 每个点/柱代表一个配置（seed / λ 组合）的综合得分：
      - 综合 S0/M1/M2 的真值指标 + SVG 重构指标；
    - 标出 canonical run；
    - 说明“我们的最佳配置不是孤立的 outlier，而是处在较为稳定的高分区域”。
  - `Fig5a_real_ST_svg_reconstruction_corr.png`
    - 真实 BRCA 上：
      - baseline vs plus 的 ST 表达重构相关性；
      - 全基因 vs SVG 子集分别统计。
  - `Fig5b_real_ST_marker_type_concordance.png`
    - marker gene 表达 vs 推断类型占比的一致性散点 / 条形图；
    - 展示类型插件在真实数据上的生物学合理性。
  - `Fig5c_real_multi_run_stability.png`（新增，用于 Step3）
    - 在真实 BRCA 上，对多个 seed / λ 的 plus 运行：
      - 展示若干关键指标（如 SVG 重构相关性、marker–type 一致性、Unknown 分布合理性）的分布；
    - 通过箱线图 / 小提琴图 / 雷达图等形式展示；
    - 标出 canonical run 在分布中的位置（通常位于较高或较稳的区域）。
- **Supplement Figures（可选）：**
  - 更细的 ablation 图、更多 SVG/类型的可视化；
  - 若未来加入 Tangram 等 backend，也可以增加相应补充图。

#### 3.2 `tables/` 目录结构

基本沿用原来的设计，只在列上稍作扩展，**增加“配置 / run_id”相关信息**：

- `Table1_dataset_preprocess.csv`
  - 数据集概况、预处理参数。
- `Table2_svg_plugin_summary.csv`
  - 每个数据集 / 场景的 SVG 数量、权重分布、SVG 重构指标；
  - 如有多配置，可增加列：`config_id / seed / lambda_svg / svg_recon_R2`。
- `Table3_type_plugin_summary.csv`
  - 类型支持度、Unknown 比例变化等；
  - 如类型插件参数也有 sweep，可以类似增加 config 列。
- `Table4_simulation_eval.csv`
  - 每个场景（S0/M1/M2）、每个 backend、每个配置（若有）的真值指标；
  - 增加列：
    - `config_id`；
    - `overall_sim_score`（用于 Step3 仲裁的综合模拟得分）；
    - `is_canonical`（是否为最终选定配置）。
- `Table5_real_eval_and_stability.csv`
  - 真实 BRCA 上的各项指标；
  - 若有多 run：
    - 每行一个 run，含 `config_id / seed / lambda_svg / lambda_type / overall_real_score / is_canonical`。

#### 3.3 `summary_json/` 设计

仍然保留 **简单易读 + 支持多 backend** 的原则，只是在结构上预留对多 run 和 canonical 的支持。例如：

```jsonc
{
  "project": "SVTuner",
  "datasets": {
    "real_brca": {
      "backends": {
        "cytospace": {
          "canonical_run_id": "seed1_lambda0.5",
          "runs": {
            "seed1_lambda0.3": {
              "lambda_svg": 0.3,
              "lambda_type": 1.0,
              "metrics_real": {
                "svg_recon_R2": 0.82,
                "marker_type_corr": 0.75,
                "unknown_reasonable_score": 0.70
              }
            },
            "seed1_lambda0.5": {
              "lambda_svg": 0.5,
              "lambda_type": 1.0,
              "metrics_real": {
                "svg_recon_R2": 0.87,
                "marker_type_corr": 0.79,
                "unknown_reasonable_score": 0.73
              }
            }
          }
        }
      }
    },
    "sim_S0": {
      "backends": {
        "cytospace": {
          "runs": {
            "seed1_lambda0.5": {
              "metrics_sim": {
                "cell_top1_acc": 0.91,
                "spot_type_L1": 0.12,
                "overall_sim_score": 0.88
              }
            }
          }
        }
      }
    }
  }
}
```

> 说明：
>
> - 结构只是一个示例，你在实现时可以简化，但核心思想是：
>   - **能记录多 run；**
>   - **能标出 canonical run；**
>   - **能区分模拟指标 / 真实指标 / SVG 专用指标。**

------

### 四、阶段七的核心内容块设计

阶段七的可视化和文字报告，大致按照 6 个 Block 来组织，对应 Fig1–5 + 一张整体总结图。

#### 4.1 Block 1：方法概览与数据集简介（Fig1 + Table1）

- 图像内容：
  - `Fig1_pipeline_overview.png`：
    - 上半部分：SVTuner 整体框架（Stage0–7 + SimGen）；
    - 中间：两个插件（SVG 插件、类型插件）嵌入 Cytospace 的位置；
    - 下半部分：**Step1（SVG cost 加权）+ Step2（映射后 SVG 空间重构 refine）+ Step3（多 run 仲裁）** 用不同颜色标出；
      - 例如，用 1/2/3 三个标记，指出它们分别发生在 Stage4 / Stage4 refine / Stage5+6 仲裁逻辑中。
- 表格内容：
  - `Table1_dataset_preprocess.csv`：
    - BRCA 数据集描述；
    - 预处理（过滤阈值、HVG 策略、SVG 计算方式）；
    - SimGen 场景（S0/M1/M2）的基本配置。
- 文本摘要：
  - 简要复述：
    - 为什么要做 SVG 插件 + 类型插件；
    - 为什么要用模拟 + 真实两条线；
    - 为什么要做 **多配置 / 多 seed** 的稳定性验证，并用 SVG 空间指标做仲裁。

#### 4.2 Block 2：模块一（SVG 插件）结果展示（Fig2 + Table2）

重点：证明 **“SVG 插件确实改动了 Cytospace 的 cost，并且在空间模式上体现出优势”**，同时为 Step3 中的“SVG 重构得分”提供直观支撑。

- 图像内容：
  - `Fig2a_svg_vs_hvg_scatter.png`：
    - HVG score vs SVG score 散点；
    - 被选中的 plugin_genes（G_plugin）高亮；
    - 可以在图中标注部分基因名字（典型肿瘤 / 免疫 / 间质 SVG）。
  - `Fig2b_svg_spatial_pattern.png`：
    - 选 3–4 个典型 SVG 基因：
      - 左列：原始 ST 表达热图；
      - 中列：插件基因权重作用前后对 Cytospace cost 的影响示意；
      - 右列：映射后的“预测 ST”在这些基因上的空间重构。
  - `Fig2c_svg_lambda_sweep_metrics.png`（新加，衔接 Step3）
    - 横轴：不同 `λ_svg` 或配置 id；
    - 纵轴：例如 SVG 基因的 ST 重构相关性 `R^2_svg`；
    - 每个点是一条 run（可按 seed 分色）；
    - 标出 Stage5/6 仲裁得到的 **canonical λ_svg / config**：
      - 这个点既在模拟真值指标上表现好，又在真实 SVG 空间上表现好。
- 表格内容：
  - `Table2_svg_plugin_summary.csv`：
    - 基因数量、权重分布统计；
    - 不同 `λ_svg` 下的 SVG 重构指标；
    - 简单地附上“是否被仲裁选中”为 canonical。
- 文本摘要：
  - 解释：
    - SVG 插件不是“随便挑几个看起来好的基因”，而是通过明确的 SVG 分数 + HVG 约束筛选；
    - cost 确实被这些 SVG 权重影响；
    - 在多配置信息下（不同 λ_svg），我们可以看到一个“SVG 重构表现–λ”的曲线，
       并在其中选出最合适的配置（canonical）。

#### 4.3 Block 3：模块二（类型匹配 / Unknown 插件）结果展示（Fig3 + Table3）

这块基本沿用原设计，只稍微提一句：**在多配置时可以共用同一套类型插件设定，或者少量变化**，并在表格里记录。

- 图像内容：
  - `Fig3a_type_mapping_heatmap.png`：
    - 原始 type 相似度矩阵；
    - 合并后 plugin_type 相似度矩阵；
    - 可用颜色块标出合并/削弱/unknown 的区域。
  - `Fig3b_type_prior_spatial.png`：
    - baseline vs plus 的 spot×type 先验空间热图；
    - 展示 Unknown 区域如何集中到预测不确定区域。
- 表格内容：
  - `Table3_type_plugin_summary.csv`：
    - 每个类型的支持度、合并情况、Unknown 比例变化；
    - 若对类型插件参数也有 sweep，可以用 config_id 区分（但这不是本项目的重点）。
- 文本摘要：
  - 强调：
    - 类型插件帮我们在“sc/ST 类型不完全对齐”时，把高风险类型合并 / 降权；
    - Unknown 的使用是可解释的：集中在 SVG 拟合差 / marker–type 不一致区域。

#### 4.4 Block 4：模拟数据验证（阶段 5 的结果，Fig4 + Table4）

重点：在 **有真值** 的世界里，展示插件（尤其是带 SVG + 类型先验的 Cytospace-plus）在各项指标上的提升，**并将这些结果与不同配置 / seed / λ 绑定起来，为 Step3 仲裁打基础**。

- 图像内容：
  - `Fig4a_sim_S0_spot_type_L1_boxplot.png`：
    - S0 场景，baseline vs plus；
    - spot×type L1 / JS 的分布箱线图；
    - 若有多配置，可以在一张图中分组展示（例如不同 λ_svg 的 plus）。
  - `Fig4b_sim_M1_missing_to_unknown_bar.png`：
    - M1 场景，missing→Unknown vs →wrong type；
    - baseline、plus 对比；
    - 类型插件的价值在这里最明显。
  - `Fig4c_sim_multi_run_score_vs_config.png`（新加，Step3 的核心一部分）
    - 场景 S0/M1/M2 的多个指标综合成一个 `overall_sim_score`；
    - 每个 config（seed / λ 组合）一个点或一条柱；
    - 标出 canonical 配置；
    - 说明：
      - canonical 不是某个“离群点”，而是在多个配置中取得 **兼顾 S0 准确、M1/M2 稳定** 的高分解。
- 表格内容：
  - `Table4_simulation_eval.csv`：
    - 每场景 × 每 backend × 每配置的主要真值指标；
    - 自带列：`config_id`、`overall_sim_score`、`is_canonical`。
- 文本摘要：
  - 让读者看到：
    - 在有真值的前提下，我们的方法可以：
      - S0 中提升整体 mapping 精度；
      - M1/M2 中显著减少“错标成错误类型”的情况；
    - 多配置对比表明：我们的 canonical 配置不是“随便挑的”，而是**在一系列候选里总体表现最好的那个**。

#### 4.5 Block 5：真实数据验证 + 稳定性分析（阶段 6 的结果，Fig5 + Table5）

这里是 **Step3 的另一半关键证据**：在真实 BRCA 数据上，即便没有真值，也要说明：

- 插件在表达重构 / 类型一致性上有提升；
- 在多 seed / 多 λ 下，指标稳定，而且 canonical run 不是“偶然好的那次”。
- 图像内容：
  - `Fig5a_real_ST_svg_reconstruction_corr.png`：
    - baseline vs plus 的 ST 重构相关性（全基因 / SVG 子集）；
    - 若有多配置，也可以画成多个点序列。
  - `Fig5b_real_ST_marker_type_concordance.png`：
    - marker 基因 vs 类型占比的一致性；
    - 展示类型插件的生物学合理性。
  - `Fig5c_real_multi_run_stability.png`（新加）
    - 若有多 seed / 多 λ：
      - 对每条 run 计算若干关键指标（例如：SVG 重构相关性、marker–type 一致性得分、Unknown 合理性得分）；
      - 用箱线图或小提琴图展示“不同 seed / λ 下的指标分布”；
      - 用颜色或标记突出 canonical run（既要好，也不能太离谱）。
- 表格内容：
  - `Table5_real_eval_and_stability.csv`：
    - 每条 run 一行，包含：
      - `config_id / seed / lambda_svg / lambda_type / svg_recon_R2 / marker_type_corr / unknown_score / overall_real_score / is_canonical`。
- 文本摘要：
  - 对读者解释：
    - 插件在真实数据上确实提升了 SVG 拟合、marker–type 一致性等；
    - 更重要的是：**换 seed / λ，这些指标不会大幅“翻车”**；
    - canonical run 是在“整体表现较好且稳定”的区域里选出来的，而不是 cherry-pick。

#### 4.6 Block 6：整体总结图（可选 Fig6 或 Supplement）

- 若篇幅允许，可以设计一张“雷达图 / 蜘蛛网图 / 条形图”：

  - 轴包括：
    - S0 mapping 精度；
    - M1/M2 mismatch 表现；
    - 真实数据 SVG 重构；
    - marker–type 一致性；
    - Unknown 合理性；
  - 比较 baseline vs plus（canonical 配置）；

- 文本上给出一句“总评”：

  > 在模拟世界和真实 BRCA 数据上，
  >  我们的 SVG+类型插件在多项指标上**不劣于 baseline，且在关键维度上显著优于 baseline**，
  >  多配置 / 多 seed / 多 λ 的稳定性分析 + SVG 空间重构仲裁，
  >  说明这种改动不是偶然，而是稳健、可复用的改进。

------

### 五、阶段七的脚本与执行逻辑

> 目的：让 `run_stage7_report` 既能在“单次运行”的简单情形下工作，也能在“多配置 / 多 seed / 多 λ”的复杂情形下自动采集所有 run，并围绕 canonical run 生成图表和报告。

示意代码结构可以保持原思路，只补充“多 run / canonical”的处理：

```python
# src/stages/stage7_report.py
from src.config import ProjectConfig

def run_stage7_report(cfg: ProjectConfig, sample_id: str, backends: list[str]):
    """
    读入 Stage2/3/5/6 的指标和图省略信息，生成:
      - figures/
      - tables/
      - summary_json/
      - report_markdown/Stage7_summary_CN.md

    sample_id 通常为 "real_brca"，但也可以为 "sim_S0" 等场景，
    便于你输出局部报告。
    """
    # 1) 读取 Stage2/3/5/6 的 metrics JSON
    #    - 支持以下两种结构：
    #      a) 单 run：顶层就是一组指标；
    #      b) 多 run：有 "runs" + "canonical_run_id" 两级结构。
    # 2) 把多 run 的信息整理成 DataFrame：
    #      columns = [config_id, seed, lambda_svg, ..., svg_recon_R2, ...]
    # 3) 绘制 Fig1~Fig5：
    #      - Fig2c / Fig4c / Fig5c 专门展示多配置 / 多 seed / 多 λ 情况；
    #      - 在图中标注 canonical run。
    # 4) 生成 Table1~5：
    #      - 对多 run 情况，在表格中附上 config_id、is_canonical 等列。
    # 5) 生成 summary_json/overall_summary.json：
    #      - 写入 runs + canonical_run_id 信息。
    # 6) 生成 Stage7_summary_CN.md：
    #      - 按 Block1~6 的结构写段落，
    #      - 每个 Block 引用对应图表并给出结论性语句。
```

------

### 六、本阶段完成标准

阶段七的“完成”不仅仅是“画出几张图”，而是：

1. **文件结构完整**
   - `result/real_brca/stage7_report/` 存在，并包含：
     - `figures/`：至少 Fig1–Fig5 系列图片 + 关键的多 run 稳定性图（如 Fig2c / Fig4c / Fig5c）；
     - `tables/`：至少 Table1–5；
     - `summary_json/overall_summary.json`；
     - `report_markdown/Stage7_summary_CN.md`。
2. **图像完整性**
   - 每个 Block 都有对应的主图：
     - Block 1：流程图 + 数据说明图；
     - Block 2：SVG 插件图 + SVG λ_sweep 曲线（Fig2c）；
     - Block 3：type 插件图；
     - Block 4：模拟场景 baseline vs plus + 多配置综合得分图（Fig4c）；
     - Block 5：真实数据 baseline vs plus + 多 run 稳定性图（Fig5c）；
     - Block 6：整体总结图（可选）。
   - 所有标注中，都能清楚看到：
     - baseline vs plus 的对比；
     - canonical 配置 / run 的位置。
3. **报告内容完整**
   - `Stage7_summary_CN.md` 至少包含：
     - 简短的方法概述；
     - 每个 Block 的结论性文字 + 指向对应图表；
     - 清晰说明：
       - 哪些结果说明模块一（SVG 插件）有效；
       - 哪些结果说明模块二（类型插件）有效；
       - 多配置 / 多 seed / 多 λ 分析如何证明结果**稳定**；
       - canonical run 是如何通过 SVG 空间重构 + 指标综合得分被选出来的。
4. **论文可用性**
   - 你可以直接根据 Stage7 的内容，勾出论文结构：
     - Fig1 → Method Overview（包括 Step1–3）
     - Fig2 → SVG 插件 + SVG λ_sweep
     - Fig3 → 类型插件
     - Fig4 → Simulation（有真值 + 多配置得分）
     - Fig5 → Real Data（无真值 + 稳定性分析）
   - Table1–5 可以直接作为 Main Table 或 Supplement table 使用；
   - `overall_summary.json` 可以被其它脚本调用，用来生成更多分析或检查。

------

这样一来：

- 原来阶段七里所有“该有的东西”（Fig1–5、Block1–5、Stage7_summary_CN.md 等）都还在；
- 同时，我们把你最担心的那一块——
   **“多配置 / 多 seed / 多 λ 运行 + SVG 空间指标仲裁出的 canonical run”**
   正式嵌入到了 Stage7 的图表与叙述结构里。

你后面在改论文时，只需要对照这个版本的 Stage7，把：

- Fig2c / Fig4c / Fig5c 的具体实现细节；
- `overall_summary.json` 实际字段；

一步一步填上去，就可以把“我们没有被 SVG 欺骗，也没有 cherry-pick”的故事讲得很完整。

------









------

# main.py脚本设计

------

## 1. main.py 的角色和整体原则

**main.py 的职责：**

- 统一入口：用一个命令行入口跑整条流水线，而不是到处 `python xxx.py`；

- 负责 **调度阶段 0–7 + SimGen**，但不在里面堆具体算法；

- 处理几类关键维度：

  - **sample**：真实样本（`real_brca`） vs 模拟场景（`S0_matched` / `M1_*` 等）；
  - **stage**：0–7 + SimGen；
  - **backend**：目前以 **CytoSPACE** 为主，将来可扩展 tangram 等；
  - **mode**：`baseline` vs `plus`（带 SVG + 类型插件）。

  同时，对我们这次最关心的 **“SVG 真正起作用 + 多配置 / 多 seed / 多 λ”**，main.py 的定位是：

  - **只负责触发 plus 模式**，并把 Stage2 / Stage3 产出的插件结果路径、config 里的 seed/λ 网格传递给 Stage4/5/6；
  - **真正的细节**——
    - SVG 加权写入 cost（Step1），
    - plus 里对 Cytospace 做 SVG 空间重构 + 邻域 refine（Step2），
    - 在多 seed / 多 λ 上形成一组候选 config，并在 Stage5/6 里做 SVG 指标仲裁（Step3），
       都在 `stage4_mapping.py`、`stage5_eval_sim.py`、`stage6_eval_real.py` 内部实现。

你可以理解成：

> **main.py = “指定要跑什么 sample / 场景，用什么 backend，跑到哪个阶段，并触发 baseline vs plus 全链路”**，
>  具体每一阶段、尤其是 Cytospace backend 如何利用 SVG / 类型插件、如何多配置跑多次并记录候选解，
>  都在 `src/stages/*.py` & `src/simgen/*.py` 里实现。

------

## 2. 推荐的项目结构（和 main 配套）

一个建议的结构（你可以按需调整）：

```text
project_root/
  src/
    main.py
    config.py             # 读 YAML / 定义 ProjectConfig
    stages/
      stage0_setup.py
      stage1_preprocess.py    # 内部调用 R 脚本
      stage2_svg_plugin.py
      stage3_type_plugin.py
      stage4_mapping.py
      stage5_eval_sim.py
      stage6_eval_real.py
      stage7_report.py
    simgen/
      make_scenario.py        # 生成模拟 ST 场景
  config/
    project_config.yaml       # 主配置文件（包含 seed/λ 网格、路径等）
  data/
    brca/...
  result/
    real_brca/...
    sim_S0_matched/...
```

这里的关键点是：

- **配置集中在 `project_config.yaml` + `ProjectConfig`**，包括：
  - 数据路径 / 输出路径；
  - SimGen 场景列表；
  - Cytospace 相关参数（包括 **SVG cost 权重、SVG 空间重构 λ 网格、运行 seed 列表** 等）；
- **main.py 不硬编码任何阈值或文件路径**，所有具体信息都从 `ProjectConfig` 读。

------

## 3. main.py 顶层结构设计（参数）

这里先写一个比较 **完整版且可扩展** 的 main 结构示意：

```python
# src/main.py

import argparse
from src.config import load_project_config, ProjectConfig
from src.stages import (
    stage0_setup,
    stage1_preprocess,
    stage2_svg_plugin,
    stage3_type_plugin,
    stage4_mapping,
    stage5_eval_sim,
    stage6_eval_real,
    stage7_report,
)
from src.simgen import make_scenario


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SVTuner 主流水线入口（SVG 插件 + 类型插件 + 稳定性评估）"
    )

    # 1) 全局配置
    parser.add_argument(
        "--config", type=str, required=True, help="YAML 配置文件路径"
    )

    # 2) 要跑哪些 sample / 场景
    parser.add_argument(
        "--samples",
        type=str,
        default="real_brca",
        help="要跑的样本 / 场景列表，逗号分隔，如 real_brca,S0_matched,M1_missing",
    )

    # 3) 要跑到哪些阶段
    parser.add_argument(
        "--stages",
        type=str,
        default="1-7",
        help="要运行的阶段，例如 1-7 或 1,2,3,4,SimGen,5",
    )

    # 4) 使用哪些 backend（目前以 Cytospace 为主）
    parser.add_argument(
        "--backends",
        type=str,
        default="cytospace",
        help="要使用的 backend 列表，逗号分隔，例如 cytospace,tangram",
    )

    # 5) 特别为 SimGen 提供一个方便入口
    parser.add_argument(
        "--simgen_scenarios",
        type=str,
        default="",
        help="只生成模拟场景，而不跑完整 pipeline，如 S0_matched,M1_missing",
    )

    # （可选扩展位）若将来想通过命令行 override 部分 seed/λ 网格，
    # 可以在这里增加参数，但当前设计里 seed/λ 主要写在 YAML 里，
    # main.py 只负责把 cfg 传给 Stage4/5/6 即可。

    return parser


def parse_stage_list(stage_str: str) -> list[str]:
    """
    支持两种写法:
      "1-7" -> ["1","2","3","4","5","6","7"]
      "1,2,3,4,SimGen,5" -> ["1","2","3","4","SimGen","5"]
    """
    stage_str = stage_str.strip()
    if "-" in stage_str and "," not in stage_str:
        start, end = stage_str.split("-")
        return [str(i) for i in range(int(start), int(end) + 1)]
    else:
        return [s.strip() for s in stage_str.split(",") if s.strip()]


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = load_project_config(args.config)  # 返回 ProjectConfig 对象

    stages = parse_stage_list(args.stages)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    samples = [s.strip() for s in args.samples.split(",") if s.strip()]

    # 先单独处理 SimGen（只生成模拟场景，不跑整个 pipeline）
    if args.simgen_scenarios:
        scenario_ids = [s.strip() for s in args.simgen_scenarios.split(",") if s.strip()]
        for sid in scenario_ids:
            make_scenario.run_simgen_for_scenario(cfg, scenario_id=sid)
        # 仅生成模拟数据就结束
        return

    # 正式跑各个样本 / 场景的流水线
    for sample_id in samples:
        run_pipeline_for_sample(
            cfg=cfg,
            sample_id=sample_id,
            stages=stages,
            backends=backends,
        )


if __name__ == "__main__":
    main()
```

> 说明：
>
> - `ProjectConfig` 负责提供数据路径、结果路径、Rscript 路径、过滤参数、以及 **Cytospace backend 的 seed/λ 网格、SVG cost 权重** 等；
> - `run_pipeline_for_sample` 是真正的“调度核心”，下面单独讲。

------

## 4. run_pipeline_for_sample：按 sample 调度 0–7 阶段

我们要区分：

- `sample_id="real_brca"` → 真实数据，适用阶段 0,1,2,3,4,6,7；
- `sample_id="S0_matched"` 等 → 模拟场景，适用阶段 1,2,3,4,5,7（不需要 0 和 6）。

可以在 config 里有一个简单的约定：

```python
def is_simulation_sample(sample_id: str) -> bool:
    return not sample_id.startswith("real_")
```

然后：

```python
# src/main.py (继续)

def run_pipeline_for_sample(
    cfg: ProjectConfig,
    sample_id: str,
    stages: list[str],
    backends: list[str],
):
    """
    根据 sample_id 和 stages 列表，调用相应阶段的 runner。
    """
    is_sim = not sample_id.startswith("real_")

    print(f"[main] 开始运行 sample: {sample_id}, is_sim={is_sim}, stages={stages}")

    # 阶段 0：真实数据只做一次，其实更适合作为全局准备，这里给个接口
    if "0" in stages and not is_sim:
        stage0_setup.run_stage0_setup(cfg, sample_id)

    # 阶段 1：预处理（R+Python）
    # 对真实 sample 和模拟场景都适用，只是数据源路径不同
    if "1" in stages:
        stage1_preprocess.run_stage1_preprocess(cfg, sample_id, is_sim=is_sim)

    # 阶段 2：SVG 插件（Python）
    # 这里产出的 gene_weights / plugin_genes 是后续 SVG cost & SVG 空间重构的唯一入口
    if "2" in stages:
        stage2_svg_plugin.run_stage2_svg(cfg, sample_id, is_sim=is_sim)

    # 阶段 3：类型匹配 / Unknown 插件
    # 这里产出的 type_prior / type_adjustment 会影响 Cytospace 的候选池与约束
    if "3" in stages:
        stage3_type_plugin.run_stage3_type(cfg, sample_id, is_sim=is_sim)

    # 阶段 4：映射（多 backend，baseline vs plus）
    if "4" in stages:
        for backend in backends:
            # baseline：严格按照原始方法配置来跑一遍
            stage4_mapping.run_stage4_mapping(
                cfg, sample_id, backend=backend, mode="baseline", is_sim=is_sim
            )

            # plus：带 SVG + 类型插件
            # 对 CytospaceBackend 而言，plus 模式内部会完成三件事：
            #   1) 使用阶段 2 产出的 gene_weights，把 SVG 信息写入 cost（SVG+HVG 动态加权）；
            #   2) 按 cfg 中的 seed / λ_svg_cost / λ_svg_recon 网格，多次跑映射 +
            #      SVG 空间重构 loss + 邻域 refine，生成一组候选 config_id；
            #   3) 把这些候选解统一写成标准输出（细胞×spot、spot×type、SVG 重构指标等），
            #      供 Stage5/6 直接读取并做仲裁。
            #
            # 对其他 backend（如 TangramBackend），plus 模式可以用类似的接口实现自己的“插件版本”。
            stage4_mapping.run_stage4_mapping(
                cfg, sample_id, backend=backend, mode="plus_svg_type", is_sim=is_sim
            )

    # 阶段 5：模拟数据验证（只对模拟样本）
    if "5" in stages and is_sim:
        # 这里只做“场景内评估”，各 backend 的 generic + cyto 专用
        # Stage5 内部会：
        #   - 读取 Stage4 baseline + plus (多 config_id) 的映射结果；
        #   - 在有真值 (SimGen) 的前提下，对每个 config 计算真值指标；
        #   - 用 SVG 空间重构 + 真值指标一起筛选出“推荐 config”（后续 Stage6/7 也会引用）。
        stage5_eval_sim.run_stage5_for_scenario(cfg, sample_id, backends)

    # 阶段 6：真实数据验证（只对 real_brca）
    if "6" in stages and not is_sim:
        # Stage6 会读取 Stage4 中 baseline + plus 候选解，
        # 在无真值条件下结合 SVG 指标 / 表达一致性 / 类型覆盖等做“真实线”评估和仲裁。
        stage6_eval_real.run_stage6_real(cfg, backends)

    # 阶段 7：可视化与报告（既可针对单个样本，也可全局汇总）
    if "7" in stages:
        # 可以约定 sample_id="real_brca" 时生成总报告，如果你想也可以按场景分开
        stage7_report.run_stage7_report(cfg, sample_id, backends)
```

> 这样 main.py 的职责就非常清晰：
>  根据 `sample_id` 判断是不是模拟场景，然后按阶段调用对应 runner；
>  各 runner 内部负责读取该 sample 对应的数据，并在 Cytospace backend 内部实现 SVG cost + SVG 空间重构 + 多配置仲裁。

------

## 5. 关键点 1：Stage1 如何在 main 中调 R 预处理脚本？

我们之前说过：**阶段一主要用 R/Seurat 做预处理**，
 但 main.py 是 Python，所以需要一个“桥接”的 `stage1_preprocess.py`。

示意：

```python
# src/stages/stage1_preprocess.py
import subprocess
from pathlib import Path
from src.config import ProjectConfig

def run_stage1_preprocess(cfg: ProjectConfig, sample_id: str, is_sim: bool):
    """
    调用 R 脚本完成预处理，然后（可选）在 Python 这边做一些收尾。
    """
    # 1) 确定输入/输出路径
    if is_sim:
        sc_expr = cfg.simulation_sc_expr_path(sample_id)   # e.g. data/sim_S0/sc_expr.csv
        st_expr = cfg.simulation_st_expr_path(sample_id)
        st_coord = cfg.simulation_st_coord_path(sample_id)
    else:
        sc_expr = cfg.real_sc_expr_path()
        st_expr = cfg.real_st_expr_path()
        st_coord = cfg.real_st_coord_path()

    out_dir = cfg.stage1_output_dir(sample_id)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 2) 组装 Rscript 命令
    cmd = [
        "Rscript",
        cfg.r_stage1_script,   # e.g. scripts/preprocess_stage1.R
        "--sc_expr", sc_expr,
        "--st_expr", st_expr,
        "--st_coord", st_coord,
        "--out_dir", out_dir,
        "--filter_mode", cfg.filter_mode,
    ]

    # 3) 运行并收集日志
    result = subprocess.run(cmd, capture_output=True, text=True)

    (Path(out_dir) / "stage1_preprocess.log").write_text(
        result.stdout + "\n=== STDERR ===\n" + result.stderr,
        encoding="utf-8"
    )

    if result.returncode != 0:
        raise RuntimeError(f"Stage 1 preprocess failed for {sample_id}, see log.")
```

> 关键思想：
>
> - main.py 不关心具体的 Seurat 步骤，只负责**把路径和模式（filter_mode）传给 R**；
> - 阶段 1 输出统一命名的 CSV/TXT 文件（包括 `sc_hvg_genes.txt`、`st_all_genes.txt`、`st_filtered_genes.txt` 等），
>    后续 Stage2 会用这些文件做 **SVG/HVG 敏感性分析**，为 SVG cost 设计埋点。

------

## 6. 关键点 2：Stage2 和 Stage3 的 main 接口（为 SVG / 类型插件留好钩子）

在 SVG 插件和类型插件这两步，我们最关心的是：

- **Stage2**：真正算出 SVG，并给出 **“基因加权后的表达矩阵 + gene_weights”**，
   将 SVG 信息编码进 Cytospace 的 cost；
- **Stage3**：对细胞类型不匹配 / unknown 问题给出 **type_prior / type_adjustment**，
   让 Cytospace 在候选池和 spot×type 约束上尊重这些信号。

main.py 层面的接口非常简单：

```python
# src/stages/stage2_svg_plugin.py

from src.config import ProjectConfig

def run_stage2_svg(cfg: ProjectConfig, sample_id: str, is_sim: bool):
    """
    1) 读取 Stage1 输出的 sc/ST 矩阵和 SVG 相关埋点文件；
    2) 计算 SVG（如 SPARK-X / Moran's I 等）；
    3) 基于 SVG/HVG 生成 gene_weights，并输出：
       - plugin_genes.txt
       - gene_weights.csv
       - 改写后的 ST 表达矩阵 (可选，如 st_expr_svg_weighted.csv)
       - 一个 svg_summary.json 记录统计信息。
    """
    ...
# src/stages/stage3_type_plugin.py

from src.config import ProjectConfig

def run_stage3_type(cfg: ProjectConfig, sample_id: str, is_sim: bool):
    """
    1) 读取 Stage1 的 sc_metadata / celltype 信息；
    2) 结合先验 marker / Leiden 聚类 / manual mapping 等，
       对“类型不匹配 / unknown”进行诊断；
    3) 输出：
       - type_prior.csv        # 每个 spot 对不同类型的先验比重
       - type_adjustment.csv   # 对部分类型做 down-weight / up-weight
       - type_summary.json     # 记录类型覆盖、unknown 比例等统计。
    """
    ...
```

在 main.py 里，只需要：

- 在 Stage2、Stage3 调用各自的 `run_stage2_svg` / `run_stage3_type`；
- Stage4 中的 `CytoSPACEBackend` 会**显式依赖这些输出文件**来构造：
  - **SVG 加权 cost matrix**（Step1）；
  - **类型约束 / 候选池过滤**；
  - plus 模式里 SVG 空间重构 loss 的目标表达向量。

这样，我们可以保证：**只要 Stage2/3 跑过，Cytospace 在 plus 模式下一定会“看到” SVG / 类型信息，而不会把它们当成普通 HVG 随手用掉。**

------

## 7. 关键点 3：Stage4–7 的 main 接口小结（与 SVG 三步法的关系）

从 main.py 视角看 Stage4–7：

1. **Stage4：映射 orchestrator**
   - main.py 只做两件事：
     - 对每个 backend，分别跑 `baseline` 和 `plus_svg_type` 两条链路；
     - 把 `cfg`（含 seed/λ 网格）与 sample_id 传给 `stage4_mapping.run_stage4_mapping`。
   - 在 **Cytospace backend 内部**：
     - `baseline` → 原生 Cytospace 流程；
     - `plus_svg_type` → 我们提出的三步法：
       1. 用 Stage2 输出的 `gene_weights` 改写 cost（SVG+HVG 动态加权）；
       2. 在多 seed / 多 λ 网格上多次跑 mapping + SVG 空间重构 refine；
       3. 把所有候选 config_id 的结果写出，供 Stage5/6 仲裁。
2. **Stage5：模拟数据线评估**
   - main.py 调用 `run_stage5_for_scenario(cfg, scenario_id, backends)`；
   - Stage5 内部：
     - 按场景读取 baseline + plus（多 config_id）的映射结果；
     - 计算有真值指标 + SVG 空间重构指标；
     - 为每个场景给出“推荐 config”（config_id），并写入 metrics JSON。
3. **Stage6：真实数据线评估**
   - main.py 调用 `run_stage6_real(cfg, backends)`；
   - Stage6 内部：
     - 在无真值条件下，对 baseline + plus 候选解做表达拟合、marker–type 一致性、空间结构、类型 coverage 等评估；
     - 再结合 SVG 空间重构指标，做一次“真实线仲裁”，
        得出在真实 BRCA 上推荐的 plus 配置。
4. **Stage7：报告与图表**
   - main.py 调用 `run_stage7_report(cfg, sample_id, backends)`；
   - Stage7 内部：
     - 读取 Stage2/3/5/6 的指标与图表信息；
     - 画出 **baseline vs plus（多配置中被仲裁选出的推荐 config）** 的对比图；
     - 输出适合论文的 Fig/Tab/总结文本。

> 换句话说：
>  **main.py 只负责“跑完 baseline + plus，并保证 Stage5/6/7 能见到所有候选配置”**；
>  真正对 SVG 是否起作用、哪个 λ 配置最好做深度分析和仲裁的工作，
>  由 Stage4/5/6/7 内部完成。

------

## 8. 总结：main.py 的设计核心

可以用一句话概括你现在的 main 设计思路：

> **main.py 只负责“调度 + 配置”，不负责“算法细节”**：
>
> - 通过命令行参数指定要跑的 `samples / stages / backends`；
> - 调用 Stage0–7 + SimGen 各自的 runner 函数；
> - 在 Stage1 中桥接 R 预处理；
> - 在 Stage2/3 中触发 SVG 插件 + 类型插件（为 Cytospace 提供唯一的 SVG / 类型入口）；
> - 在 Stage4 中统一 baseline vs plus、不同 backend 的调用，**并把 seed/λ 网格交给 Cytospace backend 内部使用**；
> - 在 Stage5/6 中收集评估与仲裁结果，在 Stage7 中整合为论文级图表和故事。











------

# 主指标+合格线

## 一、整体思路：先画“红线”，再定“目标线”，再加“稳定性线”

我会分三块讲：

1. 整体思路（硬性“不要变差”的线 vs 期望“变更好”的线 vs 稳定性线）
2. **模拟数据线（Stage 5）主指标 + 合格线 + 配置选择**
3. **真实数据线（Stage 6）主指标 + 合格线 + 配置选择**

先统一几个原则：

1. **每条线都要先有“硬红线”**
   - 插件不能把情况搞得更糟太多；
   - 就算提升有限，也要证明“至少不比 baseline 坏”。
2. **在此基础上再定“期望改善目标线”**
   - 比如：JS 降低 ≥10%，相关性提升 ≥0.02 之类；
   - 这些用来撑“方法确实有效”。
3. **主指标要少而清楚**
   - 模拟线：各场景 1–2 个核心指标；
   - 真实线：每个模块对应 1–2 个核心指标；
   - 其他的当成辅助 / ablation 结果。
4. **plus 不是单一配置，而是一族配置：需要“多配置 / 多 seed / 多 λ”的稳定性 + 选解逻辑**
   - 对 plus，我们会在若干组合上跑：
     - 不同 **配置**：例如 `(svg_alpha, svg_beta, type_prior_strength, neighbor_smooth_weight, λ_svg_spatial, …)`；
     - 不同 **seed**：CytoSPACE 的随机种子、候选集抽样等；
     - 不同 **λ**：在 SVG 空间重构 loss / 邻域正则 / 类型先验之间的权重比例。
   - **Stage5 / Stage6 的主指标 + 合格线，既要对“单次运行”生效，也要对“跨 seed 的分布”生效**：
     - 单次运行要过“硬红线”；
     - 跨 seed 的波动要在合理范围（不会出现某些 seed 很好、某些 seed 崩盘）。
   - **Stage7 中对外展示的那一组 plus 结果，并不是随便挑一条，而是：**
     - 在所有候选配置 / seed 中，
     - 先筛掉违背“硬红线”的组合，
     - 再在剩下的组合里，按照主指标的综合得分（含 SVG 空间表现）选出“代表配置”。

> 换句话说：
>
> - 主指标 + 合格线负责定义“什么叫好”；
> - “多配置 / 多 seed / 多 λ”则决定**我们选哪一条 plus 作为最终方案**，
>    并且用“稳定性”去防止那种“只靠某一个 seed 侥幸好看”的情况。

下面所有“合格线”仍然以**相对变化**为主（比 baseline 好多少），
 因为绝对数值事先很难预估。

------

## 二、模拟数据线（Stage 5）：主指标 + 合格线 + 配置选择

模拟线是**有真值**的黄金标准，主要用来回答：

> plus 在理想可控的条件下，是否真比 baseline 好？

我们按场景拆 3 类：S0（完全匹配）、M1（sc 缺类型）、M2（ST 缺类型）。

### 2.1 S0 场景（完全匹配）

这里主要考察**整体映射质量是不是提高了**。

记：

- 真值 spot×type 分布：`Q_s(t)`；
- 预测分布（baseline / plus）：`P_s^base(t)`、`P_s^plus(t)`。

对每个 spot s：

- 计算 `L1_s = sum_t | P_s(t) - Q_s(t) |`；
- 或 JS 散度 `JS_s(P_s, Q_s)`。

**主指标 1：spot×type JS_mean（或 L1_mean）**

- 可以写成：
  - `JS_mean = mean_s JS_s`；
  - 或 `L1_mean = mean_s L1_s`。
- 方向：**越低越好**（更接近真值）。

**红线：**

- plus 不允许明显变差：
  - `JS_mean^plus ≤ JS_mean^base + ε`（可以设 ε = 0.005–0.01）；
  - 或 `L1_mean^plus ≤ L1_mean^base + δ`（δ 设得更小）。

**期望目标线：**

- 期望有明显改善，例如：
  - `JS_mean^plus ≤ JS_mean^base × (1 - 0.10)` → 至少 10% 相对下降；
  - 如果 L1 也明显下降（>5–10%），是加分项。

> 总结：
>
> - S0 场景更强调“整体质量不能变坏，最好有显著改善”。

------

### 2.2 M1 场景（sc 缺类型）

这里是**模块二（类型插件）**的主战场。

设：

- 真值中某一类 `T_missing` 在 ST 中存在；
- sc 参考中故意删掉这个类型。

定义：

- `missing_to_wrongtype_rate`：
   真值为 `T_missing` 的质量，被预测成其他具体类型（非 Unknown）的比例；
- `missing_to_unknown_rate`：
   真值为 `T_missing` 的质量，被预测成 Unknown 的比例。

**主指标 1：missing_to_wrongtype_rate**

- 方向：越低越好。

**硬合格线：**

- plus 的误标率不能比 baseline 高：
  $$
  [
   r^{plus}*{wrong} \le r^{base}*{wrong}
   ]
  $$
  

**期望目标线：**

- 至少相对降低 30%：
  $$
  [
   \frac{r^{base}*{wrong} - r^{plus}*{wrong}}{r^{base}_{wrong}} \ge 0.30
   ]
  $$
  

- 如果能做到 50% 以上降低，那是非常漂亮的结果。

------

**主指标 2：missing_to_unknown_rate（辅助）**

- 方向：plus 应该把更多 mass 诚实地标为 Unknown。

**软目标：**

- plus 的 `missing_to_unknown_rate` 至少比 baseline 高 20 个百分点：
  $$
  [
   r^{plus}*{unk} \ge r^{base}*{unk} + 0.20
   ]
  $$
  

- 同时保证 **总 Unknown 占比** 不会大面积炸裂（可以设上限，比如全局 Unknown mass < 40%）。

------

### 2.3 M2 场景（ST 缺类型）

这里关注“**rare type 虚假扩散**”。

记 `T_rare` 为在 ST 中被稀释/去掉的类型。

**主指标 1：false_positive_spots（虚假阳性 spot 数）**

- 对所有 spot：
  - 真值 `Q_s(T_rare) < eps`，但预测 `P_s(T_rare) > threshold` 的 spot 个数；
  - eps 可设很小，例如 0.01；threshold 可设 0.1 或 0.05。
- 方向：越少越好。

**红线：**

- plus 不允许 “虚假扩散” 比 baseline 更严重：
  $$
  [
   FP^{plus} \le FP^{base}
   ]
  $$
  

**期望目标线：**

- 希望 false_positive_spots 至少相对下降 30%：
  $$
  [
   \frac{FP^{base} - FP^{plus}}{FP^{base}} \ge 0.30
   ]
  $$
  

------

### 2.4 CytoSPACE 专用安全红线（模拟线）

对于 CytoSPACE，我们还关心它自己的 cost / 约束是否被破坏：

1. **mean_cost_per_cell**

   - 红线：
     $$
     [
      \frac{cost^{plus}}{cost^{base}} \le 1.10
      ]
     $$
     即不多于 10% 的增长。

2. **spot_total_count_rmse（spot 总细胞数 RMSE）**

   - 红线：
     $$
     [
      RMSE^{plus} \le RMSE^{base} + \delta
      ]
     $$
     其中 δ 可以设 0.5–1。

   - 如果 plus 还能降低 RMSE（改善约束满足度）则是 bonus。

> 总结一句：**模拟线的主指标**就是：
>
> - S0：JS_mean（+L1_mean 辅助）；
> - M1：missing_to_wrongtype_rate；
> - M2：false_positive_spots；
> - CytoSPACE cost / 约束：作为“不能变坏太多”的安全红线。

------

### 2.5 模拟数据上的“多配置 / 多 seed / 多 λ”稳定性与仲裁逻辑

**为什么要在模拟线就考虑配置选择？**

- 模拟线有真值，是最安全的地方：
  - 我们可以大胆试多组 `(配置, seed, λ)`；
  - 然后用 S0/M1/M2 的主指标来做**系统的 config selection**，
     而不只在真实 BRCA 上“拍脑袋选个参数”。

**具体做法（建议）：**

1. **定义一批候选配置**

   - 配置 c 包含：
     - `svg_alpha, svg_beta`（HVG / SVG 权重）
     - `lambda_svg_spatial`（SVG 空间重构 loss 权重）
     - `type_prior_strength`（类型先验在 cost / 约束中的权重）
     - `neighbor_smooth_weight`（邻域微调强度）
     - …（其它与 plus 相关的 λ）。
   - 对每个配置 c，跑若干 seeds（例如 3–5 个）。

2. **对每个 (配置 c, seed) 计算 Stage5 主指标**

   - 得到：
     - S0：`JS_mean` 的改善量 ΔJS；
     - M1：`missing_to_wrongtype_rate` 的改善量 Δwrong；
     - M2：`false_positive_spots` 的改善量 ΔFP；
     - Cyto cost / RMSE 的变化。

3. **硬筛选：红线 + 稳定性**

   - 对每个配置 c：
     - 要求：在大多数 seed（比如 ≥70–80%）上，
       - 不违背各场景的**红线**（不比 baseline 更坏）；
       - Cyto cost / RMSE 也在安全范围内。
     - 同时，跨 seed 的主指标改善量不能特别飘：
       - 例如：ΔJS 的 IQR / mean 不超过 30%；
       - Δwrong、ΔFP 也有类似的稳定性约束。

4. **打分 & 选出“模拟线最佳配置”**

   - 对于通过硬筛选的配置 c，定义一个综合分数，例如：

     - $$
       [
        Score(c)
        = w_{S0} \cdot \Delta JS_{S0}(c)
       
       - w_{M1} \cdot \Delta wrong_{M1}(c)
       - w_{M2} \cdot \Delta FP_{M2}(c)
          ]
       $$

       

     其中每一项都做成“越大越好”（例如 error 降低量）。

   - 在所有 c 里选出 1–2 个 score 最高且稳定性良好的配置，
      标记为 **`best_config_sim`**（可以有 primary + backup）。

> 之后：
>
> - 真实数据线（Stage6）优先用 `best_config_sim` 这一组参数；
> - Stage7 报告中，“plus” 默认就指这一配置（同时在附录里说明它是通过模拟线选出来的）。

------

## 三、真实数据线（Stage 6）：主指标 + 合格线 + 配置选择

真实线没有真值，重点变成：

> 插件在真实 BRCA 上是否**稳定、合理、不瞎搞**，
>  并且在关键点上有一些提升。

### 3.1 对模块一（SVG 插件）的主指标

**主指标 1：SVG 基因重构相关性（per-gene corr 分布）**

- 对每个 SVG 基因 g：
   计算真实 ST vs 映射重构 ST 在 spot 维度的相关性 `r_g`；
- 得到 baseline 与 plus 的 `r_g` 分布。

**合格线：**

- **红线：**
  - SVG 基因整体相关性不能显著变差：
    - `median(r_plus) ≥ median(r_base) - 0.02`；
    - 且配一个配对检验（如 Wilcoxon）p > 0.05 时不算“显著变差”。
- **期望目标线：**
  - `median(r_plus) ≥ median(r_base) + 0.02`；
  - 或在 60–70% 的 SVG gene 上 `r_plus > r_base`，并通过配对检验显著。

------

**主指标 2：marker gene（类型标记基因）的重构相关性**

- 对每个 cell type 的 marker gene 集合，做同样的 per-gene corr 分布对比。

**合格线（稍软）**

- Tumor / T cell / B cell 等关键类型的 marker gene 上：
  - plus 的中位数相关性不低于 baseline；
  - 优选：＞0.02 的提升。

> 这两类指标是**模块一在真实数据上的直接证据**，
>  也是我们在“SVG 空间重构 + 邻域微调”之后，
>  检查 SVG 空间模式有没有被真正利用的核心指标。

------

### 3.2 对模块二（类型插件）的主指标

**主指标 1：marker–type 一致性（corr_type_marker）**

- 对每个类型 t：
  - `P_s(t)` = 预测的 spot 上类型占比；
  - `M_s(t)` = 对应 marker gene 的平均表达；
  - 计算 `corr_type_marker(t) = corr(P_s(t), M_s(t))`。

**合格线：**

- 对关键类型（Tumor / T cell / B cell / Myeloid 等）：
  - plus 的 `corr_type_marker(t)` 不能比 baseline 低超过 0.03；
  - 优选：在至少一半关键类型上提升 ≥0.02。
- 整体上：
  - `mean_t corr_type_marker_plus(t) ≥ mean_t corr_type_marker_base(t)`。

------

**主指标 2：Unknown 使用 vs 误差 / 不一致性**

这里是检查 Unknown 是否“用在该用的地方”。

- 对每个 spot s：
  - Unknown 占比：`U_s = P_s(Unknown)`；
  - 重构误差：`E_s = 1 - corr_spot[s]` 或 MSE；
  - marker–type 不一致度：`D_s`（比如主要类型的 |P_s(t) - marker_norm_s(t)| 之和）。

**主指标：**

- 相关性：
  - `corr(U, E)` 和 `corr(U, D)`。

**合格线：**

- plus 要满足：
  - `corr(U, E) > 0`，`corr(U, D) > 0`；
  - 且至少有一个显著（p < 0.05）。
- 如果相关性接近 0 或负值，就要小心 Unknown 策略是否有问题。

------

### 3.3 CytoSPACE 在真实数据上的“安全红线”

和模拟类似，但不要求提升，只要求**不明显变坏**：

1. **mean_cost_per_cell：**
   $$
   [
    \frac{cost^{plus}}{cost^{base}} \le 1.10
    ]
   $$
   

2. **spot_total_count_rmse：**

   - 不高于 baseline 太多：
     - `RMSE^{plus} ≤ RMSE^{base} + 1.0`；
   - 如果降低则是 bonus，可以写入讨论。

------

### 3.4 真实数据上的配置选择与稳定性检查（与 Stage5 对齐）

在 Stage5 已经选出了一个或少数几个 **`best_config_sim`** 之后，
 真实 BRCA 的配置选择可以这样做：

1. **主配置：直接使用 `best_config_sim`**
   - 即：优先采用在模拟场景上表现最好的 `(配置, λ)` 组合；
   - 在真实数据上，仍然跑多个 seed（例如 3–5 个）。
2. **实测稳定性：跨 seed 的指标分布**
   - 对每个 seed，计算：
     - SVG 基因重构相关性指标（median r、改善比例）；
     - marker–type 一致性指标；
     - Unknown vs 误差相关性；
     - Cyto cost / RMSE。
   - 要求：
     - 在大多数 seed（≥70–80%）上，全部通过前面 3.1–3.3 的**红线**；
     - 期望：在关键指标上保持一致的改善方向（不会出现一半 seed 好、一半 seed 坏）。
3. **如有必要，进行小范围配置微调**
   - 如果在真实数据上发现：
     - 部分指标对某个 λ 非常敏感（比如 λ_svg_spatial 太大导致 cost 急剧上升），
   - 可以在 `best_config_sim` 附近做少量 λ 微调（如 ±20%），
      形成 `config_real_candidate` 集合，再用同样的稳定性 + 红线逻辑筛一次。
4. **最终报告中的“plus”定义**
   - 论文和 Stage7 图表中，默认展示：
     - `baseline`；
     - `plus(best)`：
       - 在模拟线中为 `best_config_sim`；
       - 在真实线中为“满足所有红线 + 在多 seed 下表现最稳定的一组配置”。
   - 其他配置可以放入附录（ablation），说明：
     - “如果过度加大 λ_svg_spatial，会导致 cost 明显上升，对 SVG 相关性无额外好处”等。

> 这样，整个配置选择的故事就闭环了：
>
> - Stage5 用真值保证“plus 这类配置确实有提升”；
> - Stage6 用真实数据再验证“不会乱来且稳定”；
> - Stage7 用一组清晰的指标 + 图表，把这个仲裁逻辑展示出来。

------

## 四、怎么在代码里落地这些“合格线”和“仲裁逻辑”

最后给一个实现上的建议（在原有基础上加上“配置 / seed”维度）：

1. **在 Stage5 / Stage6 的 metrics JSON 里**，
    每条记录不仅要有 `baseline` / `plus` 的数值，还要有：
   - `backend` （如 `"cytospace"`）；
   - `mode`（`"baseline"` / `"plus"`）；
   - `scenario_id`（S0 / M1 / M2 / real_brca）；
   - `config_id`（例如 `"cfg_svg1_type1_lambda0.3"`）；
   - `seed`；
   - 各主指标原始数值（JS_mean, missing_to_wrongtype_rate, …）；
   - `delta` 和 `relative_improvement` 字段（相对 baseline 的变化）；
   - 可选：预先计算好的综合得分 `score_sim` / `score_real`。
2. **在 Stage7 的 summary 中，写两个小函数：**
   - `evaluate_redlines_and_targets(metrics_df)`
     - 对每个 (config_id, seed, scenario_id, backend) 判断：
       - `"pass_redline": true/false`；
       - `"target_improved": true/false`；
       - 生成简短 `"note"`：如 `"明显改善" / "基本持平" / "略有恶化"`。
   - `select_best_config(metrics_df)`
     - 在通过红线的配置中：
       - 汇总多 seed 的平均改善量、方差；
       - 算一个综合 score；
       - 同时检查稳定性（方差不能过大）；
       - 最后给出：`best_config_sim`、真实数据上的 `best_config_real`。
3. **论文写作时**就可以非常清晰地引用：
   - “在 S0 场景中，best_config 将 JS_mean 从 0.18 降至 0.15（相对改善 16.7%，达到预设 10% 的主指标目标），且在 5 个 seed 中波动 < 3%。”
   - “在 sc 缺 B cell 场景中，missing_to_wrongtype_rate 从 0.72 降至 0.34，远高于预设 30% 的改善线；在所有 seed 上均保持这一改善方向。”
   - “在真实 BRCA 数据上，SVG 基因重构相关性的中位数提升 0.03，并且在 4/5 个 seed 上显著；Unknown 占比空间分布与重构误差正相关（p < 0.01），说明 Unknown 主要集中在拟合困难区域。”

> 这样一来，“主指标 + 合格线 + 稳定性 + 仲裁”四件事在文档里就完全打通了，
>  也正好对应你之前关心的核心问题：
>  **“CytoSPACE 真的在用 SVG 吗？我们怎么证明，而且怎么用这些指标来选出那条最可信的映射链路？”**

如果你愿意，下一步我们还可以：

- 专门为 `metrics_stage5*.json` / `metrics_stage6*.json` 设计一个字段示例，
- 或者写一段 `select_best_config` 的伪代码，直接塞进策划书的附录。

------











# 文件骨架

```
PROJECT_ROOT/
├─ README.md
├─ configs/
│   ├─ project_config.yaml        # 全局配置：路径、过滤参数、种子等
│   ├─ datasets/
│   │   └─ real_brca.yaml         # BRCA 数据集元信息
│   └─ simgen/
│       ├─ S0_matched.yaml
│       ├─ M1_sc_missing_Bcell.yaml
│       └─ M2_st_missing_Tcell.yaml
│
├─ data/
│   ├─ raw/
│   │   └─ real_brca/             # 你现在 cytospace-main 里的那套 BRCA 数据
│   ├─ processed/                 # 可选，用于持久化 Stage1 后的数据
│   └─ sim/                       # SimGen 生成的模拟数据（sc/ST + truth）
│       └─ S0_matched/
│           ├─ sim_sc_expression.csv
│           ├─ sim_sc_metadata.csv
│           ├─ sim_st_expression.csv
│           ├─ sim_st_coordinates.csv
│           ├─ sim_truth_cell_spot.csv
│           └─ sim_truth_spot_type_fraction.csv
│
├─ result/
│   ├─ real_brca/
│   │   ├─ stage0_setup/
│   │   │   └─ stage0_summary.json
│   │   ├─ stage1_preprocess/
│   │   │   ├─ sc_processed.h5ad
│   │   │   ├─ st_processed.h5ad
│   │   │   └─ stage1_summary.json
│   │   ├─ stage2_svg_plugin/
│   │   │   ├─ svg_gene_list.csv
│   │   │   ├─ gene_weights.csv
│   │   │   ├─ stage2_metrics.json
│   │   │   └─ figures/
│   │   ├─ stage3_type_plugin/
│   │   │   ├─ type_similarity_matrix.csv
│   │   │   ├─ type_mapping_table.csv
│   │   │   ├─ type_prior_matrix.csv
│   │   │   ├─ stage3_metrics.json
│   │   │   └─ figures/
│   │   ├─ stage4_mapping/
│   │   │   ├─ cytospace/         # backend 1
│   │   │   │   ├─ baseline/
│   │   │   │   │   ├─ cell_assignment.csv
│   │   │   │   │   ├─ cell_spot_matrix.npz
│   │   │   │   │   └─ spot_type_fraction.csv
│   │   │   │   └─ plus/          # 使用 SVG + 类型插件的主线 plus 映射
│   │   │   │       ├─ cell_assignment.csv
│   │   │   │       ├─ cell_spot_matrix.npz
│   │   │   │       └─ spot_type_fraction.csv
│   │   │   └─ tangram/           # backend 2（将来接入时再用）
│   │   │       ├─ baseline/
│   │   │       └─ plus/
│   │   ├─ stage6_eval/
│   │   │   ├─ generic/
│   │   │   │   └─ metrics_stage6_generic.json
│   │   │   └─ cytospace_specific/
│   │   │       └─ metrics_stage6_cytospace_specific.json
│   │   ├─ stage7_report/
│   │   │   ├─ figures/
│   │   │   ├─ tables/
│   │   │   ├─ summary_json/
│   │   │   └─ report_markdown/
│   │   └─ experiments/           # 所有“附加实验”集中在这里
│   │       └─ ablation/          # 消融实验（保底用，后期再跑）
│   │           ├─ cytospace/
│   │           │   ├─ svg_only/
│   │           │   └─ type_only/
│   │           └─ tangram/       # 将来需要再加
│   │
│   └─ sim/
│       └─ S0_matched/
│           ├─ simgen/            # SimGen 的日志 + 元信息
│           ├─ stage1_preprocess/
│           ├─ stage2_svg_plugin/
│           ├─ stage3_type_plugin/
│           ├─ stage4_mapping/
│           │   └─ cytospace/
│           │       ├─ baseline/
│           │       └─ plus/
│           ├─ stage5_eval/
│           │   ├─ generic/
│           │   └─ cytospace_specific/
│           └─ experiments/       # 如果将来在模拟场景也做 ablation，就放这里
│               └─ ablation/
│
├─ logs/
│   └─ run_20251208_real_brca.log
│
├─ r_scripts/
│   └─ stage1_preprocess.R
│
└─ src/
    ├─ main.py
    ├─ config.py
    ├─ utils/
    │   ├─ paths.py
    │   ├─ io.py
    │   ├─ metrics.py
    │   └─ viz.py
    ├─ stages/
    │   ├─ stage0_setup.py
    │   ├─ stage1_preprocess.py
    │   ├─ stage2_svg_plugin.py
    │   ├─ stage3_type_plugin.py
    │   ├─ stage4_mapping.py
    │   ├─ stage5_eval_sim.py
    │   ├─ stage6_eval_real.py
    │   ├─ stage7_report.py
    │   └─ backends/
    │       ├─ cytospace_backend.py
    │       └─ tangram_backend.py
    └─ simgen/
        ├─ make_scenario.py
        ├─ world_ref_split.py
        ├─ layout_spots.py
        └─ aggregate_truth.py

```

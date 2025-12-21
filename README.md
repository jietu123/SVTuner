# SVTuner 项目说明（2025）

SVTuner：面向空间变异性（SVG）与类型感知的细胞–空间位点映射调谐框架。当前仓库已接入 GitHub，Stage0 占位，Stage1 预处理已实现并可运行，后续阶段按 SVTuner.md 规划逐步落地。

## 目录结构（关键路径）
- `configs/`：全局 & 数据集配置
  - `project_config.yaml`：全局默认
  - `datasets/<sample>.yaml`：样本级配置（推荐在此覆盖 QC/路径）
  - `env_archive/`：历史环境 yml 备份
- `data/`
  - `raw/<sample>/`：原始输入（scRNA/ST）
  - `processed/<sample>/stage1_preprocess/`：Stage1 输出（sc/st processed、common_genes、HVG）
- `result/<sample>/stage1_preprocess/`：Stage1 摘要与 QC 图
- `r_scripts/`：Stage1 预处理（Seurat）
- `src/`
  - `main.py`：管线中控（当前实现 Stage0 占位、Stage1）
  - `config.py`：默认参数与路径
  - `stages/`：各阶段入口（目前 Stage0/Stage1 占位/空）
  - `stages/backends/`：后端占位（cytospace 等）
- `logs/`：运行日志（Stage0 envcheck 占位）
- `external/`：第三方源码（如 cytospace 副本）

## 环境要求
- Python 3.10（已在 `cytospace_v1.1.0_py310` 环境验证）
- R 4.5.x + Seurat 5.3.x（推荐通过 conda 安装，见下）
- Git LFS（已用于跟踪 data/raw 大文件）

### R/Seurat 安装（推荐）
```pwsh
conda activate cytospace_v1.1.0_py310
conda install -c conda-forge r-base r-seurat r-data.table r-matrix r-jsonlite -y
```

## 数据输入（real_brca 示例）
放置于 `data/raw/real_brca/`，默认文件名（可在 YAML 配置中覆盖）：
- `real_brca_scRNA_GEP.txt`：scRNA 表达，首列 Gene，后续列为细胞
- `real_brca_scRNA_celllabels.txt`：两列 `SpotID / CellType`
- `real_brca_STdata_GEP.txt`：ST 表达，首列 Gene，后续列为 spot
- `real_brca_STdata_coordinates.txt`：三列 `SpotID / row / col`

## 配置说明
- `configs/project_config.yaml`：全局默认（可为空）
- `configs/datasets/<sample>.yaml`：推荐在此覆盖 QC/路径，例如：
```yaml
paths:
  sc_expr: real_brca_scRNA_GEP.txt
  sc_meta: real_brca_scRNA_celllabels.txt
  st_expr: real_brca_STdata_GEP.txt
  st_meta: real_brca_STdata_coordinates.txt
  svg_marker_whitelist: configs/svg_whitelist.txt  # 可选
qc:
  sc_min_genes: 200
  sc_max_genes: 6000
  sc_max_mt: 10
  st_min_genes: 100
  st_max_genes: Inf
  st_max_mt: 20
  hvg_nfeatures: 2000
  mt_pattern: "^MT-"
gene_filter:
  min_cells_sc: 0
  min_cells_st: 0
```
注：未指定时，脚本会尝试用 `<sample>_...` 前缀自动匹配；否则报错。

## 运行管线（当前实现 Stage0 占位 + Stage1）
默认仅跑 Stage1：
```pwsh
cd D:\Experiment\SVTuner_Project
python src/main.py --sample real_brca
```
如需包含占位 Stage0（仅打印“环境配置成功”）：
```pwsh
python src/main.py --sample real_brca --stages 0,1
```
如需自定义 R 命令：
```pwsh
python src/main.py --sample real_brca --r-cmd "Rscript" --stages 1
```
当前主程序仅调度 Stage0（占位）与 Stage1（R 预处理），其余阶段会提示“未实现”。

## Stage1 预处理要点（r_scripts/stage1_preprocess.R）
- 读取配置：`configs/project_config.yaml`、`configs/datasets/<sample>.yaml`，支持覆盖 QC/路径。
- 温和过滤（可调）：`qc` 与 `gene_filter` 参数，支持 `mt_pattern`，支持 SVG/marker 白名单强制保留。
- 输出：
  - `data/processed/<sample>/stage1_preprocess/`：
    - `common_genes.txt`, `hvg_genes.txt`
    - 若加 `--export_csv`：`exported/` 下生成 `sc_expression_normalized.csv`、`st_expression_normalized.csv`、`sc_metadata.csv`、`st_coordinates.csv`（Stage2 直接复用）
  - `result/<sample>/stage1_preprocess/`：
    - `stage1_summary.json`
- Summary 包含：过滤前后细胞/基因数、whitelist 保留数、公共基因数与路径、HVG 请求/实际数、mt% 统计等。
- 运行（独立调用示例，避免 MinGW 冲突，先激活 env 再清理 PATH）：  
```pwsh
$env:PATH="E:\ANACONDA\envs\cytospace_v1.1.0_py310\Library\bin;E:\ANACONDA\envs\cytospace_v1.1.0_py310\bin;E:\ANACONDA\envs\cytospace_v1.1.0_py310\Scripts;C:\Windows\System32"
& "E:\ANACONDA\envs\cytospace_v1.1.0_py310\Scripts\Rscript.exe" r_scripts/stage1_preprocess.R --sample real_brca --project_root D:/Experiment/SVTuner_Project --export_csv
```
  也可用 `conda run -n cytospace_v1.1.0_py310 Rscript ...`（若无 DLL 冲突）。

## Stage2 SVG+HVG 动态加权（src/stages/stage2_svg_plugin.py）
- 配置来源：`configs/project_config.yaml` + `configs/datasets/<sample>.yaml` 的 `stage2` 段（支持 `alpha`、`beta`、`k_neighbors`、`svg_min_detect`、`svg_min_var`、`svg_topk`、`weight_norm`、`n_jobs`、`raw_st_expr` 等）；CLI 仅作覆盖。
- 输入：
  - Stage1 导出的 CSV：`data/processed/<sample>/stage1_preprocess/exported/` 下的 `sc_expression_normalized.csv`、`st_expression_normalized.csv`、`sc_metadata.csv`（可选）、`st_coordinates.csv`
  - 基因对齐/标签：`data/processed/<sample>/stage1_preprocess/common_genes.txt`、`hvg_genes.txt`
  - 原始 ST 表达矩阵（敏感性分析用）：在 `configs/datasets/<sample>.yaml` 里显式设置 `stage2.raw_st_expr`（示例：`data/raw/real_brca/brca_STdata_GEP.txt`）
- 输出：
  - `data/processed/<sample>/stage2_svg_plugin/`：`gene_weights.csv`、`plugin_genes.txt`
  - `result/<sample>/stage2_svg_plugin/`：`stage2_summary.json`（含 top 基因及类别标签）、`stage2_params.json`、`svg_filter_sensitivity.json`
- 运行示例（已激活 env，复用导出的 CSV）：
```pwsh
cd D:\Experiment\SVTuner_Project
python src/stages/stage2_svg_plugin.py --sample real_brca --skip_export
# 如需覆盖参数：加 --alpha/--beta/--k_neighbors/--svg_topk 等
```

## Stage3 类型不匹配 / Unknown-aware 插件（src/stages/stage3_type_plugin.py）
- 配置/参数（CLI 可覆盖）：`strong_th`、`weak_th`、`st_cluster_k`、`unknown_floor`、`min_cells_rare_type`、`eps`，相似度度量固定为 `weighted_cosine`；support_score 定义固定为 top-3 相似度均值。
- 输入：
  - Stage1 导出：`data/processed/<sample>/stage1_preprocess/exported/` 下的 `sc_expression_normalized.csv`、`st_expression_normalized.csv`、`sc_metadata.csv`（类型列为 `cell_type` 或 `celltype`）、`st_coordinates.csv`
  - Stage2 导出：`data/processed/<sample>/stage2_svg_plugin/plugin_genes.txt`、`gene_weights.csv`
- 输出：
  - `data/processed/<sample>/stage3_typematch/`：`type_support.csv`、`cell_type_relabel.csv`（Unknown 统一为 `Unknown_sc_only`）、`type_prior_matrix.csv`
  - `result/<sample>/stage3_typematch/`：`stage3_summary.json`（记录参数、支持度分布、Unknown 占比、稀有类型、plugin_type 列表）
- 运行示例（已激活 env，复用导出的 CSV）：
```pwsh
cd D:\Experiment\SVTuner_Project
python src/stages/stage3_type_plugin.py --sample real_brca
# 可调阈值/聚类等：--strong_th --weak_th --st_cluster_k --unknown_floor --min_cells_rare_type
```


## SimGen ???????src/simgen/make_scenario.py?
- ?????????????????????????? Stage4 ?????? Stage1 ?????
- ???`configs/simgen/<scenario_id>.yaml`????????? Stage1 ???`data/processed/<input_sample>/stage1_preprocess/exported/`??
- ???????
  - `data/sim/<scenario_id>/`??? sc/st ???????????cell?spot?spot?type??
  - `data/processed/<scenario_id>/stage1_preprocess/exported/`?????? CSV?sc/st ???sc_metadata?st_coordinates??
- ???SimGen ???????????? Stage4??????? Stage4?

## Stage4 ???src/stages/stage4_mapping.py + src/stages/backends/cytospace_backend.py?
- ?????? sample???? SimGen ??????????????????
- ???
  - Stage1 ???`data/processed/<sample>/stage1_preprocess/exported/` ??? CSV
  - plus ??????`data/processed/<sample>/stage2_svg_plugin/` ? `data/processed/<sample>/stage3_typematch/`
- ???
  - `result/<sample>/stage4_mapping/<backend>/baseline/`?cell_assignment / cell_spot_matrix / spot_type_fraction
  - `result/<sample>/stage4_mapping/<backend>/plus_svg_type/`??? plus ??
  - `result/<sample>/stage4_run_manifest.json|csv`?Stage4 ?? run ???
- ?????
```pwsh
python src/stages/stage4_mapping.py --sample S0_matched --backends cytospace --enable_plus
python src/stages/stage4_mapping.py --sample M1_sc_missing_Bcell --backends cytospace --enable_plus
python src/stages/stage4_mapping.py --sample M2_st_missing_Tcell --backends cytospace --enable_plus
```


## SimGen + Stage4 run examples (CLI)
```pwsh
python src/simgen/make_scenario.py --scenario_id S0_matched --config configs/simgen/S0_matched.yaml
python src/simgen/make_scenario.py --scenario_id M1_sc_missing_Bcell --config configs/simgen/M1_sc_missing_Bcell.yaml
python src/simgen/make_scenario.py --scenario_id M2_st_missing_Tcell --config configs/simgen/M2_st_missing_Tcell.yaml
python src/stages/stage4_mapping.py --sample S0_matched --backends cytospace --enable_plus
python src/stages/stage4_mapping.py --sample M1_sc_missing_Bcell --backends cytospace --enable_plus
python src/stages/stage4_mapping.py --sample M2_st_missing_Tcell --backends cytospace --enable_plus
```

## Git/LFS 注意
- 大文件（raw 数据）已由 Git LFS 跟踪；请确保本地安装并启用 Git LFS。
- 默认 `.gitignore` 忽略 `result/`，`data/processed/` 目前未忽略，如需忽略可自行调整。

## 后续阶段（占位说明）
- Stage0：当前为占位提示，未来可接入真实环境检查。
- Stage5/6/7?? `dosc/SVTuner.md` ?????????????

## 常见问题
- Rscript 找不到：确认 `--r-cmd` 指向有效 Rscript（建议 conda run）。
- DLL 相关报错（Windows）：使用默认 `--r-cmd "conda run -n cytospace_v1.1.0_py310 Rscript"`，或手动设置 `CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1`。
- 输入文件名不匹配：在 `configs/datasets/<sample>.yaml` 显式配置 `paths`。
## S0/M1 当前固定配置（主线封板）
- S0（`configs/datasets/S0_matched_dataset.yaml`）：Part2 + Part3 v1+quota 固定开启
  - `svg_post_enabled: true`
  - `type_posterior_enabled: true`
  - `type_post_mode: local_marker_swap`
  - `type_post_max_changed_cells: 12`
  - `type_post_high_hurt_threshold: 0.1`
  - `type_post_max_high_hurt_actions: 2`
- M1（`configs/datasets/M1_sc_missing_Bcell.yaml`）：baseline-compatible 安全档
  - `strict_config: true`
  - `svg_post_enabled: false`
  - `type_posterior_enabled: false`
  - `spot_weight_mode: none`
- 推荐复现实验（可选 `config_id`）
  - S0：`--config_id S0_freeze_best`
  - M1：`--config_id M1_freeze_safe`


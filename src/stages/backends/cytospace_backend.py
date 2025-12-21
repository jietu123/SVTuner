"""CytoSPACE backend: baseline vs plus (SVG+type plugins) with optional refine."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.stats import rankdata

# Stage4-plus（CytoSPACEBackend.run_plus）关键路径唯一入口（防回归）：
# - soft matrix：_parse_assigned_locations -> _assignments_to_matrix
# - SVG refine：_refine_spot_cell_matrix_svg
# - type prior refine：_type_prior_refine
# - harden（容量可行化）：_harden_assignment_quota_matching
# - outputs：_build_outputs

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from stages.backends.base_backend import MappingBackend
try:
    from cytospace.cytospace import main_cytospace
except ModuleNotFoundError:
    # Fallback to local editable source (external/cytospace) if not installed in current env.
    ext = ROOT / "external" / "cytospace"
    if ext.exists() and str(ext) not in sys.path:
        sys.path.append(str(ext))
    from cytospace.cytospace import main_cytospace


def _assert_unique_index(df: pd.DataFrame, name: str):
    if not df.index.is_unique:
        dup = df.index[df.index.duplicated()].unique()[:5]
        raise ValueError(f"{name} index 存在重复: {list(dup)}")


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _module_fingerprint() -> Dict[str, Optional[str]]:
    p = Path(__file__).resolve()
    try:
        sha1 = hashlib.sha1(p.read_bytes()).hexdigest().lower()
    except Exception:
        sha1 = None
    return {"module_file": str(p), "module_sha1": sha1}


def _sha1_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest().lower()
    except Exception:
        return None


def _sha1_assignments(assignments: List[Tuple[int, int, float]]) -> Optional[str]:
    try:
        rows = sorted(assignments, key=lambda x: (int(x[0]), int(x[1])))
        buf = "\n".join(f"{int(c)},{int(s)},{float(score):.8g}" for c, s, score in rows)
        return hashlib.sha1(buf.encode("utf-8")).hexdigest().lower()
    except Exception:
        return None


def _sha1_cost_inputs(
    sc_vals: np.ndarray,
    st_vals: np.ndarray,
    capacity: np.ndarray,
    *,
    distance_metric: str,
    solver_method: str,
) -> Optional[str]:
    try:
        h = hashlib.sha1()
        h.update(str(distance_metric).encode("utf-8"))
        h.update(b"|")
        h.update(str(solver_method).encode("utf-8"))
        h.update(b"|")
        h.update(np.asarray(capacity, dtype=np.int64).tobytes())
        h.update(np.asarray(sc_vals.shape, dtype=np.int64).tobytes())
        h.update(np.asarray(st_vals.shape, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(sc_vals, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(st_vals, dtype=np.float64).tobytes())
        return h.hexdigest().lower()
    except Exception:
        return None


_DEPRECATED_KEYS: Dict[str, str] = {
    "capacity_normalize_mode": "废弃/从未完整实现：请改用 cells_per_spot_* 族配置（cells_per_spot_source/clip/rounding）",
    "capacity_normalize_factor": "废弃：请改用 umi_to_cell_norm 或 default_cells_per_spot（配合 cells_per_spot_source）",
    "capacity_norm_mode": "废弃：同 capacity_normalize_mode",
}

_TRACE_KEYS: set[str] = {
    "sample",
    "backend",
    "config_id",
    "project_root",
    "project_config_path",
    "dataset_config_path",
    "runner_file",
    "runner_sha1",
}

_USED_KEYS: set[str] = {
    # run identity / outputs
    "mode",
    "run_id",
    "variant",
    # cyto/cost/refine
    "seed",
    "solver_method",
    "distance_metric",
    "eps",
    "svg_refine_lambda",
    "svg_refine_k",
    "svg_refine_batch_size",
    "lambda_prior",
    "prior_candidate_topk",
    "prior_candidate_weight",
    "abstain_unknown_sc_only",
    "type_prior_apply_refine",
    "type_prior_apply_harden",
    "type_posterior_enabled",
    "type_post_mode",
    "type_post_marker_source",
    "type_post_marker_list",
    "type_post_markers_per_type",
    "type_post_gene_weighting",
    "type_post_require_gainA_positive",
    "type_post_rounds",
    "type_post_neighbor_k",
    "type_post_max_changed_cells",
    "type_post_selection_mode",
    "type_post_action_topm",
    "type_post_allow_swap",
    "type_post_delta_min",
    "type_post_base_margin_tau",
    "type_post_swap_target_topk",
    "type_post_swap_scope",
    "type_post_swap_base_mode",
    "type_post_base_pair_budget_eps",
    "type_post_max_swaps_per_round",
    "type_post_partner_hurt_max",
    "type_post_max_high_hurt_actions",
    "type_post_high_hurt_threshold",
    "type_post_lambda_partner_hurt",
    "type_post_lambda_partner",
    "type_post_lambda_base",
    "knn_metric",
    "knn_block_size",
    "knn_max_dense_n",
    "harden_topk",
    "prior_ablation_enabled",
    "min_gene_overlap_ratio",
    "max_cells_missing_type_prior_ratio",
    "min_prior_row_nonzero_ratio",
    "post_assign_adjust",
    "svg_post_enabled",
    "svg_post_rounds",
    "svg_post_neighbor_k",
    "svg_post_alpha",
    "svg_post_max_move_frac",
    "svg_post_max_changed_cells",
    "svg_post_selection_mode",
    "svg_post_action_topm",
    "svg_post_sort_key",
    "svg_post_lambda_partner_hurt",
    "svg_post_lambda_base",
    "svg_post_lambda_type_shift",
    "svg_post_lambda_type_penalty",
    "svg_post_lambda_type_penalty",
    "svg_post_allow_swap",
    "svg_post_delta_min",
    "svg_post_weight_clip_max",
    "svg_post_base_margin_tau",
    "svg_post_require_delta_base_nonpositive",
    "svg_post_uncertain_margin_tau",
    "svg_post_swap_delta_min",
    "svg_post_swap_require_delta_base_nonpositive",
    "svg_post_swap_depth",
    "svg_post_swap_only_within_tieset",
    "svg_post_max_swaps_per_round",
    "svg_post_swap_require_delta_base_nonpositive_for_both",
    "svg_post_swap_base_mode",
    "svg_post_base_pair_budget_eps",
    "svg_post_swap_scope",
    "svg_post_swap_partner_topm",
    "svg_post_swap_target_topk",
    "svg_post_swap_target_expand_mode",
    "svg_post_swap_target_expand_k",
    "svg_post_swap_target_expand_mode",
    "svg_post_swap_target_expand_k",
    "cost_expr_norm",
    "cost_expr_norm_eps",
    "cost_scale_source",
    "cost_scale_metric",
    "cost_scale_eps",
    "cost_scale_floor",
    "cost_scale_clip",
    "cost_scale_report_quantiles",
    "cost_scale_beta",
    "cost_scale_clip_ratio_soft",
    "cost_scale_clip_ratio_max",
    "cost_scale_downweight_only",
    "spot_weight_mode",
    "spot_weight_topk",
    "spot_weight_k",
    "spot_weight_kappa",
    "spot_weight_weight_max",
    "spot_weight_scale",
    "spot_weight_only_up",
    "spot_weight_truth_filter",
    "spot_specific_basin",
    "spot_specific_threshold_kappa",
    "default_config_id",
    # cells_per_spot / capacity
    "cells_per_spot_source",
    "cells_per_spot_clip_min",
    "cells_per_spot_clip_max",
    "cells_per_spot_rounding",
    "umi_to_cell_norm",
    "default_cells_per_spot",
}

_CONFIG_EFFECTIVE_KEYS: list[str] = [
    "mode",
    "run_id",
    "variant",
    "seed",
    "solver_method",
    "distance_metric",
    "eps",
    "svg_refine_lambda",
    "svg_refine_k",
    "svg_refine_batch_size",
    "lambda_prior",
    "prior_candidate_topk",
    "prior_candidate_weight",
    "abstain_unknown_sc_only",
    "type_prior_apply_refine",
    "type_prior_apply_harden",
    "type_posterior_enabled",
    "type_post_mode",
    "type_post_marker_source",
    "type_post_markers_per_type",
    "type_post_gene_weighting",
    "type_post_require_gainA_positive",
    "type_post_rounds",
    "type_post_neighbor_k",
    "type_post_max_changed_cells",
    "type_post_selection_mode",
    "type_post_action_topm",
    "type_post_allow_swap",
    "type_post_delta_min",
    "type_post_base_margin_tau",
    "type_post_swap_target_topk",
    "type_post_swap_scope",
    "type_post_swap_base_mode",
    "type_post_base_pair_budget_eps",
    "type_post_max_swaps_per_round",
    "type_post_partner_hurt_max",
    "type_post_max_high_hurt_actions",
    "type_post_high_hurt_threshold",
    "type_post_lambda_partner",
    "type_post_lambda_base",
    "prior_ablation_enabled",
    "knn_metric",
    "knn_block_size",
    "knn_max_dense_n",
    "harden_topk",
    "min_gene_overlap_ratio",
    "max_cells_missing_type_prior_ratio",
    "min_prior_row_nonzero_ratio",
    "post_assign_adjust",
    "svg_post_enabled",
    "svg_post_rounds",
    "svg_post_neighbor_k",
    "svg_post_alpha",
    "svg_post_max_move_frac",
    "svg_post_max_changed_cells",
    "svg_post_selection_mode",
    "svg_post_action_topm",
    "svg_post_sort_key",
    "svg_post_lambda_partner_hurt",
    "svg_post_lambda_base",
    "svg_post_lambda_type_shift",
    "svg_post_allow_swap",
    "svg_post_delta_min",
    "svg_post_weight_clip_max",
    "svg_post_base_margin_tau",
    "svg_post_require_delta_base_nonpositive",
    "svg_post_uncertain_margin_tau",
    "svg_post_swap_delta_min",
    "svg_post_swap_require_delta_base_nonpositive",
    "svg_post_swap_depth",
    "svg_post_swap_only_within_tieset",
    "svg_post_max_swaps_per_round",
    "svg_post_swap_require_delta_base_nonpositive_for_both",
    "svg_post_swap_base_mode",
    "svg_post_base_pair_budget_eps",
    "svg_post_swap_scope",
    "svg_post_swap_partner_topm",
    "svg_post_swap_target_topk",
    "cost_expr_norm",
    "cost_expr_norm_eps",
    "cost_scale_source",
    "cost_scale_metric",
    "cost_scale_eps",
    "cost_scale_floor",
    "cost_scale_clip",
    "cost_scale_report_quantiles",
    "cost_scale_beta",
    "cost_scale_clip_ratio_soft",
    "cost_scale_clip_ratio_max",
    "cost_scale_downweight_only",
    "spot_weight_mode",
    "spot_weight_topk",
    "spot_weight_k",
    "spot_weight_kappa",
    "spot_weight_weight_max",
    "spot_weight_scale",
    "spot_weight_only_up",
    "spot_weight_truth_filter",
    "spot_specific_basin",
    "spot_specific_threshold_kappa",
    "default_config_id",
    "cells_per_spot_source",
    "umi_to_cell_norm",
    "default_cells_per_spot",
    "cells_per_spot_rounding",
    "cells_per_spot_clip_min",
    "cells_per_spot_clip_max",
    "strict_config",
]


def _validate_and_resolve_config(config: Dict[str, Any], *, context: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Risk#7: 配置严格校验/归一化，避免“配了但没生效/静默忽略”。
    返回 (cfg_effective, config_validation)。
    """
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise TypeError(f"[{context}] config must be a dict, got {type(config)!r}")

    cfg = dict(config)

    def _as_bool(v: Any, key: str) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, np.integer)):
            return bool(int(v))
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("1", "true", "yes", "y", "on"):
                return True
            if s in ("0", "false", "no", "n", "off"):
                return False
        raise ValueError(f"[{context}] invalid bool {key}={v!r}")

    def _as_int(v: Any, key: str, *, min_v: Optional[int] = None) -> int:
        if isinstance(v, bool):
            raise ValueError(f"[{context}] invalid int {key}={v!r}")
        try:
            iv = int(v)
        except Exception as e:
            raise ValueError(f"[{context}] invalid int {key}={v!r}: {e}") from e
        if min_v is not None and iv < min_v:
            raise ValueError(f"[{context}] invalid int {key}={iv} < {min_v}")
        return iv

    def _as_float(v: Any, key: str, *, min_v: Optional[float] = None, max_v: Optional[float] = None) -> float:
        if isinstance(v, bool):
            raise ValueError(f"[{context}] invalid float {key}={v!r}")
        try:
            fv = float(v)
        except Exception as e:
            raise ValueError(f"[{context}] invalid float {key}={v!r}: {e}") from e
        if min_v is not None and fv < min_v:
            raise ValueError(f"[{context}] invalid float {key}={fv} < {min_v}")
        if max_v is not None and fv > max_v:
            raise ValueError(f"[{context}] invalid float {key}={fv} > {max_v}")
        if not np.isfinite(fv):
            raise ValueError(f"[{context}] invalid float {key}={fv} (non-finite)")
        return fv

    strict = _as_bool(cfg.get("strict_config", True), "strict_config")
    cfg["strict_config"] = strict

    deprecated_found = sorted(set(cfg.keys()) & set(_DEPRECATED_KEYS.keys()))
    if deprecated_found:
        details = "; ".join([f"{k}: {_DEPRECATED_KEYS[k]}" for k in deprecated_found])
        raise ValueError(f"[{context}] deprecated config keys found: {deprecated_found}. {details}")

    allowed_keys = set(_TRACE_KEYS) | set(_USED_KEYS) | {"strict_config"}
    unknown_keys = sorted([k for k in cfg.keys() if k not in allowed_keys])
    if unknown_keys and strict:
        raise ValueError(f"[{context}] unknown config keys (strict_config=True): {unknown_keys}")

    # Enums / ranges
    if "solver_method" in cfg and cfg["solver_method"] is not None:
        solver = str(cfg["solver_method"])
        allowed_solver = {"lap_CSPR", "lapjv", "lapjv_compat"}
        if solver not in allowed_solver:
            raise ValueError(f"[{context}] invalid solver_method={solver!r}; allowed={sorted(allowed_solver)}")
        cfg["solver_method"] = solver

    if "distance_metric" in cfg and cfg["distance_metric"] is not None:
        dm = str(cfg["distance_metric"])
        allowed_dm = {"Pearson_correlation", "Spearman_correlation", "Euclidean"}
        if dm not in allowed_dm:
            raise ValueError(f"[{context}] invalid distance_metric={dm!r}; allowed={sorted(allowed_dm)}")
        cfg["distance_metric"] = dm

    if "cost_expr_norm" in cfg and cfg["cost_expr_norm"] is not None:
        norm = str(cfg["cost_expr_norm"]).strip().lower()
        if norm in ("", "none"):
            norm = "none"
        allowed_norm = {"none", "st_center", "st_zscore", "gene_scale"}
        if norm not in allowed_norm:
            raise ValueError(f"[{context}] invalid cost_expr_norm={norm!r}; allowed={sorted(allowed_norm)}")
        cfg["cost_expr_norm"] = norm
    if "cost_expr_norm_eps" in cfg and cfg["cost_expr_norm_eps"] is not None:
        cfg["cost_expr_norm_eps"] = _as_float(cfg["cost_expr_norm_eps"], "cost_expr_norm_eps", min_v=1e-12)
    if "cost_scale_source" in cfg and cfg["cost_scale_source"] is not None:
        src = str(cfg["cost_scale_source"]).strip().lower()
        allowed_src = {"st", "joint"}
        if src not in allowed_src:
            raise ValueError(f"[{context}] invalid cost_scale_source={src!r}; allowed={sorted(allowed_src)}")
        cfg["cost_scale_source"] = src
    if "cost_scale_metric" in cfg and cfg["cost_scale_metric"] is not None:
        metric = str(cfg["cost_scale_metric"]).strip().lower()
        allowed_metric = {"std", "mean", "p95"}
        if metric not in allowed_metric:
            raise ValueError(f"[{context}] invalid cost_scale_metric={metric!r}; allowed={sorted(allowed_metric)}")
        cfg["cost_scale_metric"] = metric
    if "cost_scale_eps" in cfg and cfg["cost_scale_eps"] is not None:
        cfg["cost_scale_eps"] = _as_float(cfg["cost_scale_eps"], "cost_scale_eps", min_v=1e-12)
    if "cost_scale_floor" in cfg and cfg["cost_scale_floor"] is not None:
        cfg["cost_scale_floor"] = _as_float(cfg["cost_scale_floor"], "cost_scale_floor", min_v=0.0)
    if "cost_scale_clip" in cfg and cfg["cost_scale_clip"] is not None:
        clip = cfg["cost_scale_clip"]
        if isinstance(clip, str):
            parts = [p for p in clip.replace(",", " ").split() if p]
            if len(parts) != 2:
                raise ValueError(f"[{context}] cost_scale_clip must have 2 values, got {clip!r}")
            clip = [float(parts[0]), float(parts[1])]
        if not isinstance(clip, (list, tuple)) or len(clip) != 2:
            raise ValueError(f"[{context}] cost_scale_clip must be 2-item list, got {clip!r}")
        clip_low, clip_high = float(clip[0]), float(clip[1])
        if clip_low <= 0 or clip_high <= 0 or clip_low > clip_high:
            raise ValueError(f"[{context}] invalid cost_scale_clip={clip!r}")
        cfg["cost_scale_clip"] = [clip_low, clip_high]
    if "cost_scale_report_quantiles" in cfg and cfg["cost_scale_report_quantiles"] is not None:
        q = cfg["cost_scale_report_quantiles"]
        if isinstance(q, str):
            parts = [p for p in q.replace(",", " ").split() if p]
            q = [float(x) for x in parts]
        if not isinstance(q, (list, tuple)) or len(q) == 0:
            raise ValueError(f"[{context}] cost_scale_report_quantiles must be list, got {q!r}")
        cfg["cost_scale_report_quantiles"] = [float(x) for x in q]
    if "cost_scale_beta" in cfg and cfg["cost_scale_beta"] is not None:
        cfg["cost_scale_beta"] = _as_float(cfg["cost_scale_beta"], "cost_scale_beta", min_v=0.0, max_v=1.0)
    if "cost_scale_clip_ratio_soft" in cfg and cfg["cost_scale_clip_ratio_soft"] is not None:
        cfg["cost_scale_clip_ratio_soft"] = _as_float(
            cfg["cost_scale_clip_ratio_soft"], "cost_scale_clip_ratio_soft", min_v=0.0, max_v=1.0
        )
    if "cost_scale_clip_ratio_max" in cfg and cfg["cost_scale_clip_ratio_max"] is not None:
        cfg["cost_scale_clip_ratio_max"] = _as_float(
            cfg["cost_scale_clip_ratio_max"], "cost_scale_clip_ratio_max", min_v=0.0, max_v=1.0
        )
    if "cost_scale_downweight_only" in cfg and cfg["cost_scale_downweight_only"] is not None:
        cfg["cost_scale_downweight_only"] = _as_bool(cfg["cost_scale_downweight_only"], "cost_scale_downweight_only")

    if "knn_metric" in cfg and cfg["knn_metric"] is not None:
        km = str(cfg["knn_metric"]).lower()
        allowed_km = {"euclidean", "cosine"}
        if km not in allowed_km:
            raise ValueError(f"[{context}] invalid knn_metric={km!r}; allowed={sorted(allowed_km)}")
        cfg["knn_metric"] = km

    if "spot_weight_mode" in cfg and cfg["spot_weight_mode"] is not None:
        mode = str(cfg["spot_weight_mode"]).strip().lower()
        if mode in ("", "none"):
            mode = "none"
        allowed_mode = {"none", "local_zscore"}
        if mode not in allowed_mode:
            raise ValueError(f"[{context}] invalid spot_weight_mode={mode!r}; allowed={sorted(allowed_mode)}")
        cfg["spot_weight_mode"] = mode
    if "spot_weight_scale" in cfg and cfg["spot_weight_scale"] is not None:
        scale = str(cfg["spot_weight_scale"]).strip().lower()
        allowed_scale = {"linear", "exp"}
        if scale not in allowed_scale:
            raise ValueError(f"[{context}] invalid spot_weight_scale={scale!r}; allowed={sorted(allowed_scale)}")
        cfg["spot_weight_scale"] = scale

    if "cells_per_spot_source" in cfg and cfg["cells_per_spot_source"] is not None:
        src0 = str(cfg["cells_per_spot_source"]).strip()
        src_map = {
            "auto": "auto",
            "spot_cell_counts": "spot_cell_counts",
            "umi_total": "UMI_total",
            "umi_total ": "UMI_total",
            "umi_total\n": "UMI_total",
            "UMI_total": "UMI_total",
            "uniform": "uniform",
        }
        src_norm = src0 if src0 in src_map else src0.lower()
        if src_norm not in src_map:
            raise ValueError(
                f"[{context}] invalid cells_per_spot_source={src0!r}; allowed={['auto','spot_cell_counts','UMI_total','uniform']}"
            )
        cfg["cells_per_spot_source"] = src_map[src_norm]

    if "cells_per_spot_rounding" in cfg and cfg["cells_per_spot_rounding"] is not None:
        r = str(cfg["cells_per_spot_rounding"]).strip().lower()
        allowed_r = {"round", "floor", "ceil"}
        if r not in allowed_r:
            raise ValueError(f"[{context}] invalid cells_per_spot_rounding={r!r}; allowed={sorted(allowed_r)}")
        cfg["cells_per_spot_rounding"] = r

    if "seed" in cfg and cfg["seed"] is not None:
        cfg["seed"] = _as_int(cfg["seed"], "seed")
    if "eps" in cfg and cfg["eps"] is not None:
        cfg["eps"] = _as_float(cfg["eps"], "eps", min_v=1e-16)
    if "svg_refine_lambda" in cfg and cfg["svg_refine_lambda"] is not None:
        cfg["svg_refine_lambda"] = _as_float(cfg["svg_refine_lambda"], "svg_refine_lambda", min_v=0.0)
    if "svg_refine_k" in cfg and cfg["svg_refine_k"] is not None:
        cfg["svg_refine_k"] = _as_int(cfg["svg_refine_k"], "svg_refine_k", min_v=1)
    if "svg_refine_batch_size" in cfg:
        if cfg["svg_refine_batch_size"] in (None, ""):
            cfg["svg_refine_batch_size"] = None
        else:
            cfg["svg_refine_batch_size"] = _as_int(cfg["svg_refine_batch_size"], "svg_refine_batch_size", min_v=1)

    if "lambda_prior" in cfg and cfg["lambda_prior"] is not None:
        cfg["lambda_prior"] = _as_float(cfg["lambda_prior"], "lambda_prior", min_v=0.0)
    if "prior_candidate_topk" in cfg and cfg["prior_candidate_topk"] is not None:
        cfg["prior_candidate_topk"] = _as_int(cfg["prior_candidate_topk"], "prior_candidate_topk", min_v=0)
    if "prior_candidate_weight" in cfg and cfg["prior_candidate_weight"] is not None:
        cfg["prior_candidate_weight"] = _as_float(cfg["prior_candidate_weight"], "prior_candidate_weight", min_v=0.0)
    if "abstain_unknown_sc_only" in cfg and cfg["abstain_unknown_sc_only"] is not None:
        cfg["abstain_unknown_sc_only"] = _as_bool(cfg["abstain_unknown_sc_only"], "abstain_unknown_sc_only")
    if "harden_topk" in cfg and cfg["harden_topk"] is not None:
        cfg["harden_topk"] = _as_int(cfg["harden_topk"], "harden_topk", min_v=1)
    if "spot_weight_topk" in cfg and cfg["spot_weight_topk"] is not None:
        cfg["spot_weight_topk"] = _as_int(cfg["spot_weight_topk"], "spot_weight_topk", min_v=0)
    if "spot_weight_k" in cfg and cfg["spot_weight_k"] is not None:
        cfg["spot_weight_k"] = _as_int(cfg["spot_weight_k"], "spot_weight_k", min_v=1)
    if "spot_weight_kappa" in cfg and cfg["spot_weight_kappa"] is not None:
        cfg["spot_weight_kappa"] = _as_float(cfg["spot_weight_kappa"], "spot_weight_kappa", min_v=0.0)
    if "spot_weight_weight_max" in cfg and cfg["spot_weight_weight_max"] is not None:
        cfg["spot_weight_weight_max"] = _as_float(cfg["spot_weight_weight_max"], "spot_weight_weight_max", min_v=1.0)
    if "spot_weight_only_up" in cfg and cfg["spot_weight_only_up"] is not None:
        cfg["spot_weight_only_up"] = _as_bool(cfg["spot_weight_only_up"], "spot_weight_only_up")
    if "spot_weight_truth_filter" in cfg and cfg["spot_weight_truth_filter"] is not None:
        cfg["spot_weight_truth_filter"] = _as_bool(cfg["spot_weight_truth_filter"], "spot_weight_truth_filter")

    for k in ("knn_block_size", "knn_max_dense_n"):
        if k in cfg and cfg[k] is not None:
            cfg[k] = _as_int(cfg[k], k, min_v=1)

    for k in ("min_gene_overlap_ratio", "max_cells_missing_type_prior_ratio", "min_prior_row_nonzero_ratio"):
        if k in cfg and cfg[k] is not None:
            cfg[k] = _as_float(cfg[k], k, min_v=0.0, max_v=1.0)

    for k in ("type_prior_apply_refine", "type_prior_apply_harden", "prior_ablation_enabled"):
        if k in cfg and cfg[k] is not None:
            cfg[k] = _as_bool(cfg[k], k)
    if "type_posterior_enabled" in cfg and cfg["type_posterior_enabled"] is not None:
        cfg["type_posterior_enabled"] = _as_bool(cfg["type_posterior_enabled"], "type_posterior_enabled")
    if "type_post_mode" in cfg and cfg["type_post_mode"] is not None:
        mode = str(cfg["type_post_mode"]).strip().lower()
        if mode in ("", "off", "none"):
            cfg["type_post_mode"] = None
        else:
            allowed = {"local_prior_swap", "local_marker_swap"}
            if mode not in allowed:
                raise ValueError(f"[{context}] invalid type_post_mode={mode!r}; allowed={sorted(allowed)}")
            cfg["type_post_mode"] = mode
    if "type_post_marker_source" in cfg and cfg["type_post_marker_source"] is not None:
        source = str(cfg["type_post_marker_source"]).strip().lower()
        if source in ("", "off", "none"):
            cfg["type_post_marker_source"] = None
        else:
            allowed = {"sc_auto_top", "yaml_marker_list"}
            if source not in allowed:
                raise ValueError(f"[{context}] invalid type_post_marker_source={source!r}; allowed={sorted(allowed)}")
            cfg["type_post_marker_source"] = source
    if "type_post_marker_list" in cfg and cfg["type_post_marker_list"] is not None:
        marker_list = cfg["type_post_marker_list"]
        if not isinstance(marker_list, (str, dict)):
            raise ValueError(f"[{context}] invalid type_post_marker_list={type(marker_list).__name__}")
    if "type_post_markers_per_type" in cfg and cfg["type_post_markers_per_type"] is not None:
        cfg["type_post_markers_per_type"] = _as_int(
            cfg["type_post_markers_per_type"], "type_post_markers_per_type", min_v=1
        )
    if "type_post_gene_weighting" in cfg and cfg["type_post_gene_weighting"] is not None:
        mode = str(cfg["type_post_gene_weighting"]).strip().lower()
        allowed = {"spot_zscore", "none"}
        if mode not in allowed:
            raise ValueError(f"[{context}] invalid type_post_gene_weighting={mode!r}; allowed={sorted(allowed)}")
        cfg["type_post_gene_weighting"] = mode
    if "type_post_require_gainA_positive" in cfg and cfg["type_post_require_gainA_positive"] is not None:
        cfg["type_post_require_gainA_positive"] = _as_bool(
            cfg["type_post_require_gainA_positive"], "type_post_require_gainA_positive"
        )
    if "type_post_rounds" in cfg and cfg["type_post_rounds"] is not None:
        cfg["type_post_rounds"] = _as_int(cfg["type_post_rounds"], "type_post_rounds", min_v=0)
    if "type_post_neighbor_k" in cfg and cfg["type_post_neighbor_k"] is not None:
        cfg["type_post_neighbor_k"] = _as_int(cfg["type_post_neighbor_k"], "type_post_neighbor_k", min_v=0)
    if "type_post_max_changed_cells" in cfg:
        if cfg["type_post_max_changed_cells"] in (None, ""):
            cfg["type_post_max_changed_cells"] = None
        else:
            cfg["type_post_max_changed_cells"] = _as_int(
                cfg["type_post_max_changed_cells"], "type_post_max_changed_cells", min_v=0
            )
    if "type_post_selection_mode" in cfg and cfg["type_post_selection_mode"] is not None:
        mode = str(cfg["type_post_selection_mode"]).strip().lower()
        allowed = {"sequential_greedy", "global_pool_greedy"}
        if mode not in allowed:
            raise ValueError(f"[{context}] invalid type_post_selection_mode={mode!r}; allowed={sorted(allowed)}")
        cfg["type_post_selection_mode"] = mode
    if "type_post_action_topm" in cfg:
        if cfg["type_post_action_topm"] in (None, ""):
            cfg["type_post_action_topm"] = None
        else:
            cfg["type_post_action_topm"] = _as_int(cfg["type_post_action_topm"], "type_post_action_topm", min_v=1)
    if "type_post_allow_swap" in cfg and cfg["type_post_allow_swap"] is not None:
        cfg["type_post_allow_swap"] = _as_bool(cfg["type_post_allow_swap"], "type_post_allow_swap")
    if "type_post_delta_min" in cfg and cfg["type_post_delta_min"] is not None:
        cfg["type_post_delta_min"] = _as_float(cfg["type_post_delta_min"], "type_post_delta_min", min_v=0.0)
    if "type_post_base_margin_tau" in cfg:
        if cfg["type_post_base_margin_tau"] in (None, ""):
            cfg["type_post_base_margin_tau"] = None
        else:
            cfg["type_post_base_margin_tau"] = _as_float(
                cfg["type_post_base_margin_tau"], "type_post_base_margin_tau", min_v=0.0
            )
    if "type_post_swap_target_topk" in cfg:
        if cfg["type_post_swap_target_topk"] in (None, ""):
            cfg["type_post_swap_target_topk"] = None
        else:
            cfg["type_post_swap_target_topk"] = _as_int(
                cfg["type_post_swap_target_topk"], "type_post_swap_target_topk", min_v=1
            )
    if "type_post_swap_scope" in cfg and cfg["type_post_swap_scope"] is not None:
        scope = str(cfg["type_post_swap_scope"]).strip().lower()
        allowed_scope = {"a_tieset_only", "a_and_b_tieset"}
        if scope not in allowed_scope:
            raise ValueError(f"[{context}] invalid type_post_swap_scope={scope!r}; allowed={sorted(allowed_scope)}")
        cfg["type_post_swap_scope"] = scope
    if "type_post_swap_base_mode" in cfg and cfg["type_post_swap_base_mode"] is not None:
        mode = str(cfg["type_post_swap_base_mode"]).strip().lower()
        allowed_mode = {"both_nonpositive", "pair_budget"}
        if mode not in allowed_mode:
            raise ValueError(f"[{context}] invalid type_post_swap_base_mode={mode!r}; allowed={sorted(allowed_mode)}")
        cfg["type_post_swap_base_mode"] = mode
    if "type_post_base_pair_budget_eps" in cfg and cfg["type_post_base_pair_budget_eps"] is not None:
        cfg["type_post_base_pair_budget_eps"] = _as_float(
            cfg["type_post_base_pair_budget_eps"], "type_post_base_pair_budget_eps", min_v=0.0
        )
    if "type_post_max_swaps_per_round" in cfg:
        if cfg["type_post_max_swaps_per_round"] in (None, ""):
            cfg["type_post_max_swaps_per_round"] = None
        else:
            cfg["type_post_max_swaps_per_round"] = _as_int(
                cfg["type_post_max_swaps_per_round"], "type_post_max_swaps_per_round", min_v=0
            )
    if "type_post_partner_hurt_max" in cfg and cfg["type_post_partner_hurt_max"] is not None:
        cfg["type_post_partner_hurt_max"] = _as_float(
            cfg["type_post_partner_hurt_max"], "type_post_partner_hurt_max", min_v=0.0
        )
    if "type_post_max_high_hurt_actions" in cfg:
        if cfg["type_post_max_high_hurt_actions"] in (None, ""):
            cfg["type_post_max_high_hurt_actions"] = None
        else:
            cfg["type_post_max_high_hurt_actions"] = _as_int(
                cfg["type_post_max_high_hurt_actions"], "type_post_max_high_hurt_actions", min_v=0
            )
    if "type_post_high_hurt_threshold" in cfg and cfg["type_post_high_hurt_threshold"] is not None:
        cfg["type_post_high_hurt_threshold"] = _as_float(
            cfg["type_post_high_hurt_threshold"], "type_post_high_hurt_threshold", min_v=0.0
        )
    if "type_post_lambda_partner_hurt" in cfg and cfg["type_post_lambda_partner_hurt"] is not None:
        cfg["type_post_lambda_partner_hurt"] = _as_float(
            cfg["type_post_lambda_partner_hurt"], "type_post_lambda_partner_hurt", min_v=0.0
        )
    if "type_post_lambda_partner" in cfg and cfg["type_post_lambda_partner"] is not None:
        cfg["type_post_lambda_partner"] = _as_float(
            cfg["type_post_lambda_partner"], "type_post_lambda_partner", min_v=0.0
        )
    if cfg.get("type_post_lambda_partner_hurt") is not None:
        if cfg.get("type_post_lambda_partner") is None:
            cfg["type_post_lambda_partner"] = cfg["type_post_lambda_partner_hurt"]
        elif float(cfg["type_post_lambda_partner"]) != float(cfg["type_post_lambda_partner_hurt"]):
            raise ValueError(f"[{context}] type_post_lambda_partner conflicts with type_post_lambda_partner_hurt")
    if "type_post_lambda_base" in cfg and cfg["type_post_lambda_base"] is not None:
        cfg["type_post_lambda_base"] = _as_float(cfg["type_post_lambda_base"], "type_post_lambda_base", min_v=0.0)
    if "post_assign_adjust" in cfg and cfg["post_assign_adjust"] is not None:
        cfg["post_assign_adjust"] = _as_bool(cfg["post_assign_adjust"], "post_assign_adjust")
    if "svg_post_enabled" in cfg and cfg["svg_post_enabled"] is not None:
        cfg["svg_post_enabled"] = _as_bool(cfg["svg_post_enabled"], "svg_post_enabled")
    if "svg_post_rounds" in cfg and cfg["svg_post_rounds"] is not None:
        cfg["svg_post_rounds"] = _as_int(cfg["svg_post_rounds"], "svg_post_rounds", min_v=0)
    if "svg_post_neighbor_k" in cfg and cfg["svg_post_neighbor_k"] is not None:
        cfg["svg_post_neighbor_k"] = _as_int(cfg["svg_post_neighbor_k"], "svg_post_neighbor_k", min_v=0)
    if "svg_post_alpha" in cfg and cfg["svg_post_alpha"] is not None:
        cfg["svg_post_alpha"] = _as_float(cfg["svg_post_alpha"], "svg_post_alpha", min_v=0.0, max_v=1.0)
    if "svg_post_max_move_frac" in cfg and cfg["svg_post_max_move_frac"] is not None:
        cfg["svg_post_max_move_frac"] = _as_float(cfg["svg_post_max_move_frac"], "svg_post_max_move_frac", min_v=0.0, max_v=1.0)
    if "svg_post_max_changed_cells" in cfg:
        if cfg["svg_post_max_changed_cells"] in (None, ""):
            cfg["svg_post_max_changed_cells"] = None
        else:
            cfg["svg_post_max_changed_cells"] = _as_int(cfg["svg_post_max_changed_cells"], "svg_post_max_changed_cells", min_v=0)
    if "svg_post_selection_mode" in cfg and cfg["svg_post_selection_mode"] is not None:
        mode = str(cfg["svg_post_selection_mode"]).strip().lower()
        allowed = {"sequential_greedy", "global_pool_greedy"}
        if mode not in allowed:
            raise ValueError(f"[{context}] invalid svg_post_selection_mode={mode!r}; allowed={sorted(allowed)}")
        cfg["svg_post_selection_mode"] = mode
    if "svg_post_action_topm" in cfg:
        if cfg["svg_post_action_topm"] in (None, ""):
            cfg["svg_post_action_topm"] = None
        else:
            cfg["svg_post_action_topm"] = _as_int(cfg["svg_post_action_topm"], "svg_post_action_topm", min_v=1)
    if "svg_post_sort_key" in cfg and cfg["svg_post_sort_key"] is not None:
        sort_key = str(cfg["svg_post_sort_key"]).strip().lower()
        if sort_key in ("gainsum", "gain_sum"):
            sort_key = "gainsum"
        elif sort_key in ("gaina", "gain_a"):
            sort_key = "gaina"
        allowed = {"gainA_penalized", "gainsum", "gaina"}
        if sort_key not in {k.lower() for k in allowed}:
            raise ValueError(f"[{context}] invalid svg_post_sort_key={sort_key!r}; allowed={sorted(allowed)}")
        if sort_key == "gaina":
            cfg["svg_post_sort_key"] = "gainA"
        elif sort_key == "gainsum":
            cfg["svg_post_sort_key"] = "gainSum"
        else:
            cfg["svg_post_sort_key"] = "gainA_penalized"
    if "svg_post_lambda_partner_hurt" in cfg and cfg["svg_post_lambda_partner_hurt"] is not None:
        cfg["svg_post_lambda_partner_hurt"] = _as_float(
            cfg["svg_post_lambda_partner_hurt"], "svg_post_lambda_partner_hurt", min_v=0.0
        )
    if "svg_post_lambda_base" in cfg and cfg["svg_post_lambda_base"] is not None:
        cfg["svg_post_lambda_base"] = _as_float(cfg["svg_post_lambda_base"], "svg_post_lambda_base", min_v=0.0)
    if "svg_post_lambda_type_shift" in cfg and cfg["svg_post_lambda_type_shift"] is not None:
        cfg["svg_post_lambda_type_shift"] = _as_float(
            cfg["svg_post_lambda_type_shift"], "svg_post_lambda_type_shift", min_v=0.0
        )
    if "svg_post_lambda_type_penalty" in cfg and cfg["svg_post_lambda_type_penalty"] is not None:
        cfg["svg_post_lambda_type_penalty"] = _as_float(
            cfg["svg_post_lambda_type_penalty"], "svg_post_lambda_type_penalty", min_v=0.0
        )
        if cfg.get("svg_post_lambda_type_shift") is None:
            cfg["svg_post_lambda_type_shift"] = cfg["svg_post_lambda_type_penalty"]
        elif float(cfg["svg_post_lambda_type_shift"]) != float(cfg["svg_post_lambda_type_penalty"]):
            raise ValueError(
                f"[{context}] svg_post_lambda_type_shift conflicts with svg_post_lambda_type_penalty"
            )
    if "svg_post_allow_swap" in cfg and cfg["svg_post_allow_swap"] is not None:
        cfg["svg_post_allow_swap"] = _as_bool(cfg["svg_post_allow_swap"], "svg_post_allow_swap")
    if "svg_post_delta_min" in cfg and cfg["svg_post_delta_min"] is not None:
        cfg["svg_post_delta_min"] = _as_float(cfg["svg_post_delta_min"], "svg_post_delta_min", min_v=0.0)
    if "svg_post_weight_clip_max" in cfg and cfg["svg_post_weight_clip_max"] is not None:
        cfg["svg_post_weight_clip_max"] = _as_float(cfg["svg_post_weight_clip_max"], "svg_post_weight_clip_max", min_v=1.0)
    if "svg_post_base_margin_tau" in cfg:
        if cfg["svg_post_base_margin_tau"] in (None, ""):
            cfg["svg_post_base_margin_tau"] = None
        else:
            cfg["svg_post_base_margin_tau"] = _as_float(cfg["svg_post_base_margin_tau"], "svg_post_base_margin_tau", min_v=0.0)
    if "svg_post_require_delta_base_nonpositive" in cfg and cfg["svg_post_require_delta_base_nonpositive"] is not None:
        cfg["svg_post_require_delta_base_nonpositive"] = _as_bool(
            cfg["svg_post_require_delta_base_nonpositive"], "svg_post_require_delta_base_nonpositive"
        )
    if "svg_post_uncertain_margin_tau" in cfg:
        if cfg["svg_post_uncertain_margin_tau"] in (None, ""):
            cfg["svg_post_uncertain_margin_tau"] = None
        else:
            cfg["svg_post_uncertain_margin_tau"] = _as_float(
                cfg["svg_post_uncertain_margin_tau"], "svg_post_uncertain_margin_tau", min_v=0.0
            )
    if "svg_post_swap_delta_min" in cfg and cfg["svg_post_swap_delta_min"] is not None:
        cfg["svg_post_swap_delta_min"] = _as_float(cfg["svg_post_swap_delta_min"], "svg_post_swap_delta_min", min_v=0.0)
    if "svg_post_swap_require_delta_base_nonpositive" in cfg and cfg["svg_post_swap_require_delta_base_nonpositive"] is not None:
        cfg["svg_post_swap_require_delta_base_nonpositive"] = _as_bool(
            cfg["svg_post_swap_require_delta_base_nonpositive"], "svg_post_swap_require_delta_base_nonpositive"
        )
    if "svg_post_swap_depth" in cfg and cfg["svg_post_swap_depth"] is not None:
        cfg["svg_post_swap_depth"] = _as_int(cfg["svg_post_swap_depth"], "svg_post_swap_depth", min_v=1)
        if cfg["svg_post_swap_depth"] != 1:
            raise ValueError(f"[{context}] svg_post_swap_depth>1 not supported yet")
    if "svg_post_swap_only_within_tieset" in cfg and cfg["svg_post_swap_only_within_tieset"] is not None:
        cfg["svg_post_swap_only_within_tieset"] = _as_bool(cfg["svg_post_swap_only_within_tieset"], "svg_post_swap_only_within_tieset")
    if "svg_post_max_swaps_per_round" in cfg:
        if cfg["svg_post_max_swaps_per_round"] in (None, ""):
            cfg["svg_post_max_swaps_per_round"] = None
        else:
            cfg["svg_post_max_swaps_per_round"] = _as_int(cfg["svg_post_max_swaps_per_round"], "svg_post_max_swaps_per_round", min_v=0)
    if "svg_post_swap_require_delta_base_nonpositive_for_both" in cfg and cfg["svg_post_swap_require_delta_base_nonpositive_for_both"] is not None:
        cfg["svg_post_swap_require_delta_base_nonpositive_for_both"] = _as_bool(
            cfg["svg_post_swap_require_delta_base_nonpositive_for_both"],
            "svg_post_swap_require_delta_base_nonpositive_for_both",
        )
        if cfg.get("svg_post_swap_require_delta_base_nonpositive") is not None and cfg.get("svg_post_swap_require_delta_base_nonpositive") != cfg["svg_post_swap_require_delta_base_nonpositive_for_both"]:
            raise ValueError(
                f"[{context}] svg_post_swap_require_delta_base_nonpositive conflicts with svg_post_swap_require_delta_base_nonpositive_for_both"
            )
        cfg["svg_post_swap_require_delta_base_nonpositive"] = cfg["svg_post_swap_require_delta_base_nonpositive_for_both"]
    if "svg_post_swap_base_mode" in cfg and cfg["svg_post_swap_base_mode"] is not None:
        mode = str(cfg["svg_post_swap_base_mode"]).strip().lower()
        allowed_mode = {"both_nonpositive", "pair_budget"}
        if mode not in allowed_mode:
            raise ValueError(f"[{context}] invalid svg_post_swap_base_mode={mode!r}; allowed={sorted(allowed_mode)}")
        cfg["svg_post_swap_base_mode"] = mode
    if "svg_post_base_pair_budget_eps" in cfg and cfg["svg_post_base_pair_budget_eps"] is not None:
        cfg["svg_post_base_pair_budget_eps"] = _as_float(
            cfg["svg_post_base_pair_budget_eps"], "svg_post_base_pair_budget_eps", min_v=0.0
        )
    if "svg_post_swap_scope" in cfg and cfg["svg_post_swap_scope"] is not None:
        scope = str(cfg["svg_post_swap_scope"]).strip().lower()
        allowed_scope = {"a_tieset_only", "a_and_b_tieset"}
        if scope not in allowed_scope:
            raise ValueError(f"[{context}] invalid svg_post_swap_scope={scope!r}; allowed={sorted(allowed_scope)}")
        cfg["svg_post_swap_scope"] = scope
    if "svg_post_swap_partner_topm" in cfg:
        if cfg["svg_post_swap_partner_topm"] in (None, ""):
            cfg["svg_post_swap_partner_topm"] = None
        else:
            cfg["svg_post_swap_partner_topm"] = _as_int(
                cfg["svg_post_swap_partner_topm"], "svg_post_swap_partner_topm", min_v=0
            )
    if "svg_post_swap_target_topk" in cfg:
        if cfg["svg_post_swap_target_topk"] in (None, ""):
            cfg["svg_post_swap_target_topk"] = None
        else:
            cfg["svg_post_swap_target_topk"] = _as_int(
                cfg["svg_post_swap_target_topk"], "svg_post_swap_target_topk", min_v=1
            )
    if "svg_post_swap_target_expand_mode" in cfg and cfg["svg_post_swap_target_expand_mode"] is not None:
        mode = str(cfg["svg_post_swap_target_expand_mode"]).strip().lower()
        allowed_mode = {"none", "neighbor_knn"}
        if mode not in allowed_mode:
            raise ValueError(f"[{context}] invalid svg_post_swap_target_expand_mode={mode!r}; allowed={sorted(allowed_mode)}")
        cfg["svg_post_swap_target_expand_mode"] = mode
    if "svg_post_swap_target_expand_k" in cfg and cfg["svg_post_swap_target_expand_k"] is not None:
        cfg["svg_post_swap_target_expand_k"] = _as_int(cfg["svg_post_swap_target_expand_k"], "svg_post_swap_target_expand_k", min_v=0)
    if cfg.get("svg_post_swap_base_mode") == "pair_budget" and cfg.get("svg_post_swap_require_delta_base_nonpositive_for_both"):
        raise ValueError(f"[{context}] svg_post_swap_base_mode=pair_budget conflicts with swap_require_delta_base_nonpositive_for_both")

    if "cells_per_spot_clip_min" in cfg and cfg["cells_per_spot_clip_min"] is not None:
        cfg["cells_per_spot_clip_min"] = _as_int(cfg["cells_per_spot_clip_min"], "cells_per_spot_clip_min", min_v=0)
    if "cells_per_spot_clip_max" in cfg:
        if cfg["cells_per_spot_clip_max"] in (None, ""):
            cfg["cells_per_spot_clip_max"] = None
        else:
            cfg["cells_per_spot_clip_max"] = _as_int(cfg["cells_per_spot_clip_max"], "cells_per_spot_clip_max", min_v=0)
    if cfg.get("cells_per_spot_clip_max") is not None and cfg.get("cells_per_spot_clip_min") is not None:
        if int(cfg["cells_per_spot_clip_max"]) < int(cfg["cells_per_spot_clip_min"]):
            raise ValueError(
                f"[{context}] cells_per_spot_clip_max {cfg['cells_per_spot_clip_max']} < cells_per_spot_clip_min {cfg['cells_per_spot_clip_min']}"
            )

    if "default_cells_per_spot" in cfg and cfg["default_cells_per_spot"] is not None:
        cfg["default_cells_per_spot"] = _as_float(cfg["default_cells_per_spot"], "default_cells_per_spot", min_v=0.0)
    if "umi_to_cell_norm" in cfg and cfg["umi_to_cell_norm"] is not None:
        norm = cfg["umi_to_cell_norm"]
        if isinstance(norm, str) and norm.strip().lower() == "median":
            cfg["umi_to_cell_norm"] = "median"
        else:
            cfg["umi_to_cell_norm"] = _as_float(norm, "umi_to_cell_norm", min_v=1e-12)

    config_validation = {
        "schema_version": "cytospace_backend_config_v1",
        "strict_config": strict,
        "unknown_keys": unknown_keys,
        "deprecated_keys_found": deprecated_found,
    }
    return cfg, config_validation


def _get_coord_xy(df: pd.DataFrame) -> pd.DataFrame:
    if {"row", "col"}.issubset(df.columns):
        return df[["row", "col"]]
    if {"x", "y"}.issubset(df.columns):
        return df[["x", "y"]].rename(columns={"x": "row", "y": "col"})
    raise KeyError("st_coordinates 需要包含 row/col 或 x/y 坐标列")


def _load_stage1(stage1_dir: Path):
    exp_dir = stage1_dir / "exported"
    sc_expr = pd.read_csv(exp_dir / "sc_expression_normalized.csv", index_col=0)
    st_expr = pd.read_csv(exp_dir / "st_expression_normalized.csv", index_col=0)
    sc_meta = pd.read_csv(exp_dir / "sc_metadata.csv")
    st_coords = pd.read_csv(exp_dir / "st_coordinates.csv", index_col=0)
    # 确保 cell_id 对齐
    if "cell_id" in sc_meta.columns:
        sc_meta = sc_meta.set_index("cell_id")
    else:
        raise KeyError("sc_metadata.csv 需要包含 cell_id 列用于与 sc_expression_normalized 对齐")
    missing = set(sc_expr.index) - set(sc_meta.index)
    if missing:
        raise ValueError(f"sc_metadata 缺少以下 cell_id: {list(missing)[:5]} ...")
    sc_meta = sc_meta.loc[sc_expr.index]
    # 类型列兼容 + 护栏（避免 cell_type/celltype 同时存在但不一致）
    if "cell_type" in sc_meta.columns and "celltype" in sc_meta.columns:
        a = sc_meta["cell_type"].astype(str).to_numpy()
        b = sc_meta["celltype"].astype(str).to_numpy()
        if a.shape == b.shape and (a != b).any():
            raise ValueError("sc_metadata.csv 同时存在 cell_type 与 celltype，但两列不一致（禁止 silent bug）")
    if "cell_type" in sc_meta.columns:
        type_col = "cell_type"
    elif "celltype" in sc_meta.columns:
        type_col = "celltype"
    elif "type" in sc_meta.columns:
        sc_meta = sc_meta.copy()
        sc_meta["celltype"] = sc_meta["type"].astype(str)
        type_col = "celltype"
    else:
        raise KeyError("sc_metadata.csv 需要包含 cell_type/celltype/type 之一作为类型列")
    sc_meta["type_col"] = sc_meta[type_col]
    # 坐标列容错
    coords_xy = _get_coord_xy(st_coords)
    st_coords = pd.concat([st_coords.drop(columns=[c for c in ["row", "col", "x", "y"] if c in st_coords.columns]), coords_xy], axis=1)
    return sc_expr, st_expr, sc_meta, st_coords, type_col



def _load_stage2(stage2_dir: Path):
    plugin_path = stage2_dir / "plugin_genes.txt"
    plugin_genes = [g.strip() for g in plugin_path.read_text(encoding="utf-8").splitlines() if g.strip()]
    gw_path = stage2_dir / "gene_weights.csv"
    gene_weights = pd.read_csv(gw_path)
    if "gene" not in gene_weights.columns:
        raise ValueError("gene_weights.csv ??????????????? gene")
    if "cost_weight" in gene_weights.columns:
        weight_field = "cost_weight"
        weights = pd.to_numeric(gene_weights["cost_weight"], errors="coerce").fillna(1.0).astype(float)
    elif "final_weight" in gene_weights.columns:
        weight_field = "final_weight_compat"
        weights = pd.to_numeric(gene_weights["final_weight"], errors="coerce").fillna(1.0).astype(float)
        weights = np.maximum(weights, 1.0)
    else:
        raise ValueError("gene_weights.csv ???????????? cost_weight ???final_weight")
    weight_map = dict(zip(gene_weights["gene"], weights))
    return plugin_genes, weight_map, weight_field, gene_weights


def _load_stage3(stage3_dir: Path):
    relabel_path = stage3_dir / "cell_type_relabel.csv"
    type_prior_path = stage3_dir / "type_prior_matrix.csv"
    relabel = pd.read_csv(relabel_path)
    if "cell_id" not in relabel.columns or "plugin_type" not in relabel.columns:
        raise ValueError("cell_type_relabel.csv 需要包含列 cell_id 与 plugin_type")
    type_prior = pd.read_csv(type_prior_path, index_col=0)
    return relabel, type_prior


def _load_stage3_relabel(stage3_dir: Path):
    relabel_path = stage3_dir / "cell_type_relabel.csv"
    relabel = pd.read_csv(relabel_path)
    if "cell_id" not in relabel.columns or "plugin_type" not in relabel.columns:
        raise ValueError("cell_type_relabel.csv éœ€è¦åŒ…å«åˆ— cell_id ä¸?plugin_type")
    return relabel


def _compute_cells_per_spot(st_coords: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.Series, str, Optional[float]]:
    """
    Compute raw (float) cells-per-spot series using the configured source.
    Risk#7: the *final* rounding/clipping to int must be done by _resolve_cells_per_spot.
    Returns (capacity_raw, source_resolved, norm_val_if_any).
    """
    src = cfg.get("cells_per_spot_source", "auto") or "auto"
    src = str(src)

    def _has_col(name: str) -> bool:
        return name in st_coords.columns and st_coords[name].notna().any()

    if src.lower() == "auto":
        if _has_col("spot_cell_counts"):
            src = "spot_cell_counts"
        elif _has_col("UMI_total") or _has_col("nCount_RNA") or _has_col("nCount_Spatial"):
            src = "UMI_total"
        else:
            src = "uniform"

    if src == "spot_cell_counts":
        if "spot_cell_counts" not in st_coords.columns:
            raise KeyError("cells_per_spot_source=spot_cell_counts 但 st_coords 缺少 spot_cell_counts 列")
        cps = st_coords["spot_cell_counts"].fillna(0).astype(float)
        cps.index = st_coords.index
        return cps, "spot_cell_counts", None

    if src == "UMI_total":
        umi_col = None
        for c in ("UMI_total", "nCount_RNA", "nCount_Spatial"):
            if c in st_coords.columns and st_coords[c].notna().any():
                umi_col = c
                break
        if umi_col is None:
            raise KeyError("cells_per_spot_source=UMI_total 但 st_coords 缺少 UMI_total/nCount_RNA/nCount_Spatial 列")
        umi = pd.to_numeric(st_coords[umi_col], errors="coerce").fillna(0).astype(float)
        umi.index = st_coords.index
        norm = cfg.get("umi_to_cell_norm", 1000)
        if isinstance(norm, str) and norm.strip().lower() == "median":
            norm_val = float(np.median(umi[umi > 0])) if (umi > 0).any() else 1000.0
        else:
            norm_val = float(norm)
        cps = umi / max(norm_val, 1e-8)
        cps = cps.astype(float)
        cps.index = st_coords.index
        src_resolved = "UMI_total" if umi_col == "UMI_total" else f"UMI_total[{umi_col}]"
        return cps, src_resolved, norm_val

    if src == "uniform":
        default_c = float(cfg.get("default_cells_per_spot", 1.0))
        cps = pd.Series(float(default_c), index=st_coords.index, dtype=float)
        return cps, "uniform", None

    raise ValueError(f"invalid cells_per_spot_source={src!r}; expected auto/spot_cell_counts/UMI_total/uniform")


def _resolve_cells_per_spot(capacity_raw: pd.Series, cfg: Dict[str, Any], *, context: str) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Risk#7: produce final int capacity series for cells_per_spot.csv with audit.
    - rounding: round|floor|ceil
    - clip: [clip_min, clip_max]
    """
    if not isinstance(capacity_raw, pd.Series):
        raise TypeError(f"[{context}] capacity_raw must be pd.Series, got {type(capacity_raw)!r}")

    rounding = str(cfg.get("cells_per_spot_rounding", "round") or "round").strip().lower()
    if rounding not in ("round", "floor", "ceil"):
        raise ValueError(f"[{context}] invalid cells_per_spot_rounding={rounding!r}")

    clip_min_v = cfg.get("cells_per_spot_clip_min", 1)
    if clip_min_v in (None, ""):
        clip_min_v = 1
    clip_min = int(clip_min_v)
    clip_max_v = cfg.get("cells_per_spot_clip_max", None)
    clip_max = None if clip_max_v in (None, "") else int(clip_max_v)
    if clip_max is not None and clip_max < clip_min:
        raise ValueError(f"[{context}] cells_per_spot_clip_max {clip_max} < clip_min {clip_min}")

    x0 = np.asarray(capacity_raw, dtype=float)
    n_nan = int(np.isnan(x0).sum())
    x0 = np.nan_to_num(x0, nan=0.0, posinf=0.0, neginf=0.0)

    if rounding == "round":
        xr = np.rint(x0)
    elif rounding == "floor":
        xr = np.floor(x0)
    else:
        xr = np.ceil(x0)

    xr_int = xr.astype(int)
    n_nonpositive_before_clip = int((xr_int <= 0).sum())
    n_clipped_min = int((xr_int < clip_min).sum())
    n_clipped_max = int((xr_int > clip_max).sum()) if clip_max is not None else 0

    x = xr_int
    if clip_min is not None:
        x = np.maximum(x, clip_min)
    if clip_max is not None:
        x = np.minimum(x, clip_max)
    x = x.astype(int)

    cap_int = pd.Series(x, index=capacity_raw.index, dtype=int)
    audit = {
        "context": context,
        "rounding": rounding,
        "clip_min": int(clip_min),
        "clip_max": int(clip_max) if clip_max is not None else None,
        "n_spots": int(len(cap_int)),
        "n_nan_before": int(n_nan),
        "n_nonpositive_before_clip": int(n_nonpositive_before_clip),
        "n_clipped_min": int(n_clipped_min),
        "n_clipped_max": int(n_clipped_max),
        "min": int(cap_int.min()) if len(cap_int) else None,
        "mean": float(cap_int.mean()) if len(cap_int) else None,
        "max": int(cap_int.max()) if len(cap_int) else None,
        "sum": int(cap_int.sum()) if len(cap_int) else None,
    }
    return cap_int, audit


def _build_feature_mats(sc_expr: pd.DataFrame, st_expr: pd.DataFrame, genes: List[str], weights: Dict[str, float]):
    genes_use = [g for g in genes if g in sc_expr.columns and g in st_expr.columns]
    if len(genes_use) == 0:
        raise ValueError("可用基因为空，请检查插件基因与表达矩阵交集")
    w = np.array([weights.get(g, 1.0) for g in genes_use], dtype=float)
    sc_mat = sc_expr[genes_use].to_numpy(dtype=float)
    st_mat = st_expr[genes_use].to_numpy(dtype=float)
    return sc_mat, st_mat, w, genes_use


def _apply_cost_expr_norm(
    sc_expr: pd.DataFrame,
    st_expr: pd.DataFrame,
    genes_use: List[str],
    cfg: Dict[str, Any],
    eps: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    mode = str(cfg.get("cost_expr_norm") or "none").lower()
    if mode in ("", "none"):
        return sc_expr, st_expr, {"mode": "none"}

    st_vals = st_expr[genes_use].to_numpy(dtype=float)
    st_mean = st_vals.mean(axis=0)
    audit: Dict[str, Any] = {"mode": mode}

    if mode in ("st_center", "st_zscore"):
        if mode == "st_center":
            sc_vals = sc_expr[genes_use].to_numpy(dtype=float) - st_mean
            st_vals = st_vals - st_mean
        else:
            st_std = st_vals.std(axis=0, ddof=0)
            st_std = np.where(st_std > eps, st_std, 1.0)
            sc_vals = (sc_expr[genes_use].to_numpy(dtype=float) - st_mean) / st_std
            st_vals = (st_vals - st_mean) / st_std
            audit["std_min"] = float(np.min(st_std))
            audit["std_max"] = float(np.max(st_std))

        min_val = float(min(np.min(sc_vals), np.min(st_vals)))
        if min_val < 0:
            shift = -min_val
            sc_vals = sc_vals + shift
            st_vals = st_vals + shift
            audit["shift_nonneg"] = float(shift)
        else:
            audit["shift_nonneg"] = 0.0

        audit["mean_min"] = float(np.min(st_mean))
        audit["mean_max"] = float(np.max(st_mean))
        sc_out = sc_expr.copy()
        st_out = st_expr.copy()
        sc_out.loc[:, genes_use] = sc_vals
        st_out.loc[:, genes_use] = st_vals
        return sc_out, st_out, audit

    if mode != "gene_scale":
        raise ValueError(f"invalid cost_expr_norm={mode!r}")

    beta_raw = cfg.get("cost_scale_beta", None)
    scale_beta = 1.0 if beta_raw is None else float(beta_raw)
    scale_beta = float(np.clip(scale_beta, 0.0, 1.0))
    if scale_beta <= 0.0:
        q_list = cfg.get("cost_scale_report_quantiles") or [0.01, 0.05, 0.5, 0.95, 0.99]
        q_list = [float(q) for q in q_list]
        q_stats = {f"q{int(round(q * 100)):02d}": 1.0 for q in q_list}
        clip_ratio_soft = float(cfg.get("cost_scale_clip_ratio_soft") or 0.85)
        clip_ratio_max = float(cfg.get("cost_scale_clip_ratio_max") or 0.95)
        audit = {
            "mode": "gene_scale",
            "scale_source": str(cfg.get("cost_scale_source") or "st").lower(),
            "scale_metric": str(cfg.get("cost_scale_metric") or "std").lower(),
            "scale_beta": scale_beta,
            "scale_downweight_only": bool(cfg.get("cost_scale_downweight_only", False)),
            "scale_eps": float(cfg.get("cost_scale_eps") or eps),
            "scale_floor": float(cfg.get("cost_scale_floor") or 0.0),
            "scale_clip": cfg.get("cost_scale_clip"),
            "clip_ratio": 0.0,
            "clip_ratio_soft": float(clip_ratio_soft),
            "clip_ratio_max": float(clip_ratio_max),
            "scale_stats": {
                "scale_min": None,
                "scale_max": None,
                "n_floor_applied": 0,
                "n_clip_low": 0,
                "n_clip_high": 0,
                **{k: None for k in q_stats.keys()},
            },
            "multiplier_stats_planned": {
                "min": 1.0,
                "max": 1.0,
                "n_clip_low": 0,
                "n_clip_high": 0,
                **q_stats,
            },
            "multiplier_stats_applied": {
                "min": 1.0,
                "max": 1.0,
                "n_clip_low": 0,
                "n_clip_high": 0,
                **q_stats,
            },
            "genes_use_count": int(len(genes_use)),
            "norm_applied": False,
            "norm_fallback": False,
            "norm_reason": "beta_is_zero_skip",
            "norm_fallback_reason": [],
            "scale_min_threshold": 1e-4,
        }
        return sc_expr, st_expr, audit

    source = str(cfg.get("cost_scale_source") or "st").lower()
    if source == "joint":
        base = pd.concat([sc_expr[genes_use], st_expr[genes_use]], axis=0)
    else:
        source = "st"
        base = st_expr[genes_use]

    scale_metric = str(cfg.get("cost_scale_metric") or "std").lower()
    base_vals = base.to_numpy(dtype=float)
    if scale_metric == "std":
        scale = base_vals.std(axis=0, ddof=0)
    elif scale_metric == "mean":
        scale = base_vals.mean(axis=0)
    elif scale_metric == "p95":
        scale = np.quantile(base_vals, 0.95, axis=0)
    else:
        raise ValueError(f"invalid cost_scale_metric={scale_metric!r}")
    scale = np.where(np.isfinite(scale), scale, 0.0)
    scale_eps = float(cfg.get("cost_scale_eps") or eps)
    scale_floor = float(cfg.get("cost_scale_floor") or 0.0)
    scale = np.maximum(scale, scale_eps)
    scale = np.maximum(scale, scale_floor)

    mult_raw = 1.0 / scale
    mult = 1.0 + scale_beta * (mult_raw - 1.0)
    downweight_only = bool(cfg.get("cost_scale_downweight_only", False))
    if downweight_only:
        mult = np.minimum(mult, 1.0)
    clip = cfg.get("cost_scale_clip")
    clip_low = clip_high = None
    n_clip_low = n_clip_high = 0
    if clip:
        clip_low, clip_high = float(clip[0]), float(clip[1])
        n_clip_low = int((mult < clip_low).sum())
        n_clip_high = int((mult > clip_high).sum())
        mult = np.clip(mult, clip_low, clip_high)

    sc_vals = sc_expr[genes_use].to_numpy(dtype=float) * mult
    st_vals = st_expr[genes_use].to_numpy(dtype=float) * mult

    q_list = cfg.get("cost_scale_report_quantiles") or [0.01, 0.05, 0.5, 0.95, 0.99]
    q_list = [float(q) for q in q_list]
    scale_stats = {}
    mult_stats_before = {}
    mult_stats_after = {}
    for q in q_list:
        label = f"q{int(round(q * 100)):02d}"
        scale_stats[label] = float(np.quantile(scale, q))
        mult_stats_before[label] = float(np.quantile(mult_raw, q))
        mult_stats_after[label] = float(np.quantile(mult, q))

    n_floor_applied = int((scale <= scale_floor + 1e-12).sum())
    n_genes = int(len(scale))
    floor_ratio = n_floor_applied / max(n_genes, 1)
    clip_ratio = (n_clip_low + n_clip_high) / max(n_genes, 1)
    scale_min = float(np.min(scale)) if n_genes else None
    scale_q01 = scale_stats.get("q01")
    min_threshold = 1e-4
    clip_ratio_soft = float(cfg.get("cost_scale_clip_ratio_soft") or 0.85)
    clip_ratio_max = float(cfg.get("cost_scale_clip_ratio_max") or 0.95)
    if clip_ratio > clip_ratio_soft:
        audit["clip_ratio_warning"] = float(clip_ratio)

    fallback_reasons = []
    if floor_ratio > 0.3:
        fallback_reasons.append(f"floor_ratio_too_high:{floor_ratio:.3f}")
    if scale_min is not None and scale_min < min_threshold:
        fallback_reasons.append(f"scale_too_small:{scale_min:.6g}")
    if scale_q01 is not None and scale_q01 < min_threshold:
        fallback_reasons.append(f"scale_q01_too_small:{scale_q01:.6g}")
    if clip and clip_ratio > clip_ratio_max:
        fallback_reasons.append(f"clip_ratio_too_high:{clip_ratio:.3f}")

    audit.update(
        {
            "mode": "gene_scale",
            "scale_source": source,
            "scale_metric": scale_metric,
            "scale_beta": scale_beta,
            "scale_downweight_only": downweight_only,
            "scale_eps": scale_eps,
            "scale_floor": scale_floor,
            "scale_clip": clip if clip else None,
            "clip_ratio": float(clip_ratio),
            "clip_ratio_soft": float(clip_ratio_soft),
            "clip_ratio_max": float(clip_ratio_max),
            "scale_stats": {
                "scale_min": scale_min,
                "scale_max": float(np.max(scale)) if n_genes else None,
                "n_floor_applied": n_floor_applied,
                "n_clip_low": n_clip_low,
                "n_clip_high": n_clip_high,
                **scale_stats,
            },
            "multiplier_stats_before_beta": {
                "min": float(np.min(mult_raw)) if n_genes else None,
                "max": float(np.max(mult_raw)) if n_genes else None,
                **mult_stats_before,
            },
            "genes_use_count": n_genes,
            "norm_fallback": len(fallback_reasons) > 0,
            "norm_fallback_reason": fallback_reasons,
            "scale_min_threshold": min_threshold,
        }
    )

    planned_stats = {
        "min": float(np.min(mult)) if n_genes else None,
        "max": float(np.max(mult)) if n_genes else None,
        "n_clip_low": n_clip_low,
        "n_clip_high": n_clip_high,
        **mult_stats_after,
    }
    audit["multiplier_stats_planned"] = planned_stats
    if fallback_reasons:
        applied_stats = {
            "min": 1.0,
            "max": 1.0,
            "n_clip_low": 0,
            "n_clip_high": 0,
            **{k: 1.0 for k in mult_stats_after.keys()},
        }
        audit["multiplier_stats_applied"] = applied_stats
        audit["norm_applied"] = False
        return sc_expr, st_expr, audit

    audit["multiplier_stats_applied"] = planned_stats
    audit["norm_applied"] = True

    sc_out = sc_expr.copy()
    st_out = st_expr.copy()
    sc_out.loc[:, genes_use] = sc_vals
    st_out.loc[:, genes_use] = st_vals
    return sc_out, st_out, audit


def _compute_spot_weight_matrix(
    st_expr: pd.DataFrame,
    st_coords: pd.DataFrame,
    genes_use: List[str],
    cfg: Dict[str, Any],
    *,
    eps: float,
    candidate_mask: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    mode = str(cfg.get("spot_weight_mode") or "none").lower()
    if mode in ("", "none"):
        return None, {"status": "disabled", "mode": "none"}
    if mode != "local_zscore":
        raise ValueError(f"invalid spot_weight_mode={mode!r}")

    topk = int(cfg.get("spot_weight_topk") or 0)
    if topk <= 0:
        return None, {"status": "disabled", "mode": mode, "reason": "topk<=0"}

    k = int(cfg.get("spot_weight_k") or 8)
    kappa = float(cfg.get("spot_weight_kappa") or 0.0)
    weight_max = float(cfg.get("spot_weight_weight_max") or 1.0)
    scale = str(cfg.get("spot_weight_scale") or "linear").lower()
    only_up = bool(cfg.get("spot_weight_only_up", True))

    coords = _get_coord_xy(st_coords).to_numpy(dtype=float)
    n_spots = coords.shape[0]
    if n_spots == 0:
        return None, {"status": "disabled", "mode": mode, "reason": "no_spots"}
    k_use = min(max(k, 1), n_spots)
    try:
        tree = cKDTree(coords)
        _, idxs = tree.query(coords, k=k_use)
    except Exception:
        dists = cdist(coords, coords)
        idxs = np.argsort(dists, axis=1)[:, :k_use]

    if idxs.ndim == 1:
        idxs = idxs[:, None]

    st_vals = st_expr[genes_use].to_numpy(dtype=float)
    n_genes = st_vals.shape[1]
    topk = min(topk, n_genes)

    if candidate_mask is not None:
        candidate_mask = np.asarray(candidate_mask).astype(bool)
        if candidate_mask.shape[0] != n_genes:
            raise ValueError("spot_weight candidate_mask shape mismatch")
        cand_idx = np.where(candidate_mask)[0]
    else:
        cand_idx = None

    weights = np.ones((n_spots, n_genes), dtype=float)
    weighted_per_spot = []
    for i in range(n_spots):
        neigh_idx = idxs[i]
        local = st_vals[neigh_idx]
        mean = local.mean(axis=0)
        std = local.std(axis=0)
        score = np.abs(st_vals[i] - mean) / (std + eps)
        if cand_idx is not None:
            if len(cand_idx) == 0:
                continue
            score_cand = score[cand_idx]
            topk_i = min(topk, len(cand_idx))
            pick = cand_idx[np.argpartition(-score_cand, topk_i - 1)[:topk_i]]
        else:
            topk_i = topk
            pick = np.argpartition(-score, topk_i - 1)[:topk_i]
        if topk_i <= 0:
            continue
        smax = float(np.max(score[pick]))
        if smax <= 0:
            continue
        score_norm = score[pick] / (smax + eps)
        if scale == "exp":
            w = np.exp(kappa * score_norm)
        else:
            w = 1.0 + kappa * score_norm
        if only_up:
            w = np.maximum(w, 1.0)
        w = np.clip(w, 1.0, weight_max)
        weights[i, pick] = w
        weighted_per_spot.append(int(len(pick)))

    stats = {
        "status": "ok",
        "mode": mode,
        "scale": scale,
        "topk": int(topk),
        "k": int(k_use),
        "kappa": float(kappa),
        "weight_max": float(weight_max),
        "n_spots": int(n_spots),
        "n_genes": int(n_genes),
        "weighted_per_spot_mean": float(np.mean(weighted_per_spot)) if weighted_per_spot else 0.0,
        "weights_min": float(np.min(weights)),
        "weights_p50": float(np.percentile(weights, 50)),
        "weights_p95": float(np.percentile(weights, 95)),
        "weights_max": float(np.max(weights)),
        "weighted_ratio": float((weights > 1.0).mean()),
        "candidate_n": int(len(cand_idx)) if cand_idx is not None else None,
    }
    return weights, stats


def _spot_fraction_from_mat(
    mat: csr_matrix,
    type_labels: List[str],
    st_index: List[str],
    type_order: List[str],
    *,
    context: str = "",
    min_type_intersection: float = 0.2,
    max_unique_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    从 spot×cell 矩阵聚合得到 spot×type 计数矩阵（未归一化），并返回校验/审计信息。
    目的：防止把 cell_id 等错误输入当成 type_labels 传入，导致 prior_effect_* 指标“沉默失真”。
    """
    if not sparse.isspmatrix_csr(mat):
        mat = mat.tocsr()

    n_spots, n_cells = mat.shape
    if len(type_labels) != n_cells:
        raise ValueError(
            f"[{context}] type_labels 长度 {len(type_labels)} != mat.shape[1] {n_cells}，疑似 cell 维度未对齐"
        )

    type_order = list(type_order)
    if len(type_order) == 0:
        raise ValueError(f"[{context}] type_order 为空，无法计算 spot×type")

    uniq = pd.Series(type_labels).astype(str).unique().tolist()
    inter = len(set(uniq) & set(type_order))
    inter_ratio = inter / max(len(set(uniq)), 1)
    unique_ratio = len(set(type_labels)) / max(n_cells, 1)

    if inter == 0:
        raise ValueError(
            f"[{context}] type_labels 与 type_order 无交集（inter=0），高度疑似把 cell_id 当 cell_types 传入"
        )
    if inter_ratio < float(min_type_intersection):
        raise ValueError(
            f"[{context}] type_labels 与 type_order 交集比例过低 inter_ratio={inter_ratio:.3f}，疑似类型体系不一致/传参错误"
        )
    if unique_ratio > float(max_unique_ratio):
        raise ValueError(
            f"[{context}] type_labels 唯一值比例过高 unique_ratio={unique_ratio:.3f}，高度疑似传入近似唯一的 cell_id 列表"
        )

    df = pd.DataFrame(0.0, index=st_index, columns=type_order)
    type_arr = np.array([str(x) for x in type_labels], dtype=str)

    missing_mask = ~np.isin(type_arr, np.array(type_order, dtype=str))
    n_missing = int(missing_mask.sum())
    missing_examples = sorted(set(type_arr[missing_mask]))[:5] if n_missing > 0 else []

    for t in type_order:
        mask = type_arr == t
        if mask.any():
            df[t] = mat[:, mask].sum(axis=1).A1

    audit = {
        "context": context,
        "n_spots": int(n_spots),
        "n_cells": int(n_cells),
        "n_types_unique": int(len(set(type_labels))),
        "unique_ratio": float(unique_ratio),
        "type_intersection": int(inter),
        "type_intersection_ratio": float(inter_ratio),
        "n_missing_type_labels": int(n_missing),
        "missing_type_label_examples": missing_examples,
        "min_type_intersection": float(min_type_intersection),
        "max_unique_ratio": float(max_unique_ratio),
    }
    return df, audit


def _align_spot_inputs(
    *,
    spot_ids: List[str],
    st_coords: pd.DataFrame,
    capacity: pd.Series,
    type_prior_raw: Optional[pd.DataFrame] = None,
    context: str = "",
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Stage4 Risk#5 防护：以 spot_ids 为唯一锚点，对齐所有 spot 相关输入并生成可审计信息。

    约束：
    - 不允许静默丢 spot：任何 missing/extra 都直接 raise
    - 对齐仅允许在此函数完成（统一 reindex 到 spot_ids 顺序）
    """

    def _examples(xs) -> List[str]:
        return sorted([str(x) for x in xs])[:5]

    if len(spot_ids) != len(set(spot_ids)):
        raise ValueError(f"[{context}] spot_ids 存在重复（len={len(spot_ids)}, unique={len(set(spot_ids))}）")

    if not st_coords.index.is_unique:
        dup = st_coords.index[st_coords.index.duplicated()].unique()[:5]
        raise ValueError(f"[{context}] st_coords index 存在重复: {list(dup)}")
    if not capacity.index.is_unique:
        dup = capacity.index[capacity.index.duplicated()].unique()[:5]
        raise ValueError(f"[{context}] capacity index 存在重复: {list(dup)}")
    if type_prior_raw is not None and not type_prior_raw.index.is_unique:
        dup = type_prior_raw.index[type_prior_raw.index.duplicated()].unique()[:5]
        raise ValueError(f"[{context}] type_prior_raw index 存在重复: {list(dup)}")

    spot_set = set(spot_ids)
    coords_set = set(st_coords.index)
    capacity_set = set(capacity.index)

    missing_in_coords = spot_set - coords_set
    extra_in_coords = coords_set - spot_set
    missing_in_capacity = spot_set - capacity_set
    extra_in_capacity = capacity_set - spot_set

    if missing_in_coords or extra_in_coords:
        raise ValueError(
            f"[{context}] st_coords spot_id mismatch: missing={len(missing_in_coords)}, extra={len(extra_in_coords)}, "
            f"missing_examples={_examples(missing_in_coords)}, extra_examples={_examples(extra_in_coords)}"
        )
    if missing_in_capacity or extra_in_capacity:
        raise ValueError(
            f"[{context}] capacity spot_id mismatch: missing={len(missing_in_capacity)}, extra={len(extra_in_capacity)}, "
            f"missing_examples={_examples(missing_in_capacity)}, extra_examples={_examples(extra_in_capacity)}"
        )

    type_prior_set = None
    missing_in_type_prior: set = set()
    extra_in_type_prior: set = set()
    if type_prior_raw is not None:
        type_prior_set = set(type_prior_raw.index)
        missing_in_type_prior = spot_set - type_prior_set
        extra_in_type_prior = type_prior_set - spot_set
        if missing_in_type_prior or extra_in_type_prior:
            raise ValueError(
                f"[{context}] type_prior_raw spot_id mismatch: missing={len(missing_in_type_prior)}, extra={len(extra_in_type_prior)}, "
                f"missing_examples={_examples(missing_in_type_prior)}, extra_examples={_examples(extra_in_type_prior)}"
            )

    anchor = pd.Index(spot_ids)
    order_coords = bool(st_coords.index.equals(anchor))
    order_capacity = bool(capacity.index.equals(anchor))
    order_type_prior = bool(type_prior_raw.index.equals(anchor)) if type_prior_raw is not None else None
    order_all = order_coords and order_capacity and (order_type_prior if order_type_prior is not None else True)

    coords_cols_used = "unknown"
    if {"row", "col"}.issubset(st_coords.columns):
        coords_cols_used = "row/col"
    elif {"x", "y"}.issubset(st_coords.columns):
        coords_cols_used = "x/y"

    spot_ids_sha1 = hashlib.sha1(",".join([str(x) for x in spot_ids]).encode("utf-8")).hexdigest().lower()

    audit = {
        "context": context,
        "spot_ids_sha1": spot_ids_sha1,
        "order_equal_before": bool(order_all),
        "order_equal_before_coords": bool(order_coords),
        "order_equal_before_capacity": bool(order_capacity),
        "order_equal_before_type_prior": order_type_prior,
        "n_spots_expr": int(len(spot_ids)),
        "n_spots_coords": int(len(st_coords.index)),
        "n_spots_capacity": int(len(capacity.index)),
        "n_spots_type_prior": int(len(type_prior_raw.index)) if type_prior_raw is not None else None,
        "n_missing_in_coords": int(len(missing_in_coords)),
        "example_missing_in_coords": _examples(missing_in_coords),
        "n_missing_in_capacity": int(len(missing_in_capacity)),
        "example_missing_in_capacity": _examples(missing_in_capacity),
        "n_missing_in_type_prior": int(len(missing_in_type_prior)) if type_prior_raw is not None else None,
        "example_missing_in_type_prior": _examples(missing_in_type_prior) if type_prior_raw is not None else None,
        "coords_cols_used": coords_cols_used,
        "head_spot_ids": [str(x) for x in spot_ids[:5]],
        "tail_spot_ids": [str(x) for x in spot_ids[-5:]],
    }

    st_coords_aligned = st_coords.reindex(spot_ids)
    capacity_aligned = capacity.reindex(spot_ids)
    if capacity_aligned.isna().any():
        raise ValueError(f"[{context}] capacity reindex 后存在 NA（疑似 spot_ids 锚点不一致）")
    type_prior_aligned = type_prior_raw.reindex(spot_ids) if type_prior_raw is not None else None
    return st_coords_aligned, capacity_aligned, type_prior_aligned, audit


def _write_cytospace_inputs(
    sc_expr: pd.DataFrame,
    sc_meta: pd.DataFrame,
    st_expr: pd.DataFrame,
    st_coords: pd.DataFrame,
    cell_types: List[str],
    genes_use: List[str],
    capacity: pd.Series,
    spot_ids: List[str],
    tmp_dir: Path,
    prefix: str,
):
    sc_sub = sc_expr[genes_use]
    st_sub = st_expr[genes_use]

    anchor = pd.Index(spot_ids)
    if not st_expr.index.equals(anchor):
        raise ValueError(f"[{prefix}] st_expr.index 与 spot_ids 锚点不一致（可能存在隐形错位）")
    if not st_coords.index.equals(anchor):
        raise ValueError(f"[{prefix}] st_coords.index 与 spot_ids 锚点不一致（可能存在隐形错位）")
    if not capacity.index.equals(anchor):
        raise ValueError(f"[{prefix}] capacity.index 与 spot_ids 锚点不一致（可能存在隐形错位）")

    coords_xy = _get_coord_xy(st_coords)

    sc_path = tmp_dir / f"{prefix}scRNA.csv"
    st_path = tmp_dir / f"{prefix}st.csv"
    ct_path = tmp_dir / f"{prefix}cell_type.csv"
    coord_path = tmp_dir / f"{prefix}coords.csv"
    cps_path = tmp_dir / f"{prefix}cells_per_spot.csv"

    sc_sub.T.to_csv(sc_path)
    st_sub.T.to_csv(st_path)
    # CytoSPACE 的 read_file 约定：CSV 第一列为 rownames，且必须有 header
    ct_df = pd.DataFrame({"cell_type": cell_types}, index=sc_expr.index)
    ct_df.index.name = "cell_id"
    ct_df.to_csv(ct_path)

    coord_df = pd.DataFrame(
        {"row": coords_xy.iloc[:, 0].values, "col": coords_xy.iloc[:, 1].values},
        index=anchor,
    )
    coord_df.index.name = "spot_id"
    coord_df.to_csv(coord_path)

    cps_df = pd.DataFrame({"cells": np.asarray(capacity, dtype=int)}, index=anchor)
    cps_df.index.name = "spot_id"
    cps_df.to_csv(cps_path)
    return sc_path, ct_path, st_path, coord_path, cps_path


def _parse_assigned_locations(
    assigned: pd.DataFrame,
    st_ids: List[str],
    require_original_cid: bool = False,
) -> Tuple[List[str], List[str], Optional[List[str]], List[Tuple[int, int, float]]]:
    """
    解析 CytoSPACE 的 assigned_locations.csv：
    - 使用 UniqueCID 作为 cell_id（每个“cell instance”唯一）
    - 使用 CellType 作为 type（baseline 为 orig_type；plus 为 plugin_type）
    - SpotID 必须能在 Stage1 的 spot_id 集合中找到
    """
    required_cols = {"UniqueCID", "SpotID", "CellType"}
    missing = sorted(required_cols - set(assigned.columns))
    if missing:
        raise ValueError(f"assigned_locations.csv 缺少必需列: {missing}")
    has_original = "OriginalCID" in assigned.columns
    if require_original_cid and not has_original:
        raise ValueError("assigned_locations.csv 缺少列 OriginalCID（用于从 Stage1 sc_expr 构建 cell pool 表达）")

    spot_pos = {sid: i for i, sid in enumerate(st_ids)}
    cell_ids = assigned["UniqueCID"].astype(str).tolist()
    cell_types = assigned["CellType"].astype(str).tolist()
    orig_cell_ids = assigned["OriginalCID"].astype(str).tolist() if has_original else None

    assignments: List[Tuple[int, int, float]] = []
    bad_spots: List[str] = []
    for cell_idx, sid in enumerate(assigned["SpotID"].astype(str).tolist()):
        if sid not in spot_pos:
            if len(bad_spots) < 5:
                bad_spots.append(sid)
            continue
        assignments.append((cell_idx, spot_pos[sid], 1.0))
    if len(assignments) != len(cell_ids):
        raise ValueError(f"assigned_locations 中有 spot_id 未匹配 Stage1：{len(cell_ids)-len(assignments)} 行，示例: {bad_spots}")
    return cell_ids, cell_types, orig_cell_ids, assignments


def _assignments_to_matrix(assignments: List[Tuple[int, int, float]], n_spots: int, n_cells: int) -> sparse.csr_matrix:
    if len(assignments) == 0:
        return sparse.csr_matrix((n_spots, n_cells), dtype=float)
    data = np.array([score for _, _, score in assignments], dtype=float)
    row_ind = [spot_idx for _, spot_idx, _ in assignments]
    col_ind = [cell_idx for cell_idx, _, _ in assignments]
    return sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n_spots, n_cells))


def _build_sparse_knn_safe(
    coords: pd.DataFrame,
    k: int,
    metric: str = "euclidean",
    block_size: int = 1024,
    max_dense_n: int = 5000,
) -> Tuple[csr_matrix, str]:
    """
    构建稀疏KNN 邻接矩阵（避免大规模 SxS dense）：
    - S <= max_dense_n: dense 计算后转 CSR，mode="dense"
    - S  大: 分块 brute-force，内存占用约 block_size x S，mode="block"
    返回 (W, mode)
    """
    xy = coords[["row", "col"]].to_numpy(dtype=float)
    n = xy.shape[0]
    if n <= max_dense_n:
        dist = cdist(xy, xy, metric=metric)
        np.fill_diagonal(dist, np.inf)
        idx_knn = np.argpartition(dist, kth=k, axis=1)[:, :k]
        rows = np.repeat(np.arange(n), k)
        cols = idx_knn.reshape(-1)
        data = np.ones_like(cols, dtype=float)
        W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        row_sum = np.asarray(W.sum(axis=1)).ravel()
        row_sum[row_sum == 0] = 1.0
        W = W.multiply(1.0 / row_sum[:, None])
        return W, "dense"

    rows_list, cols_list, data_list = [], [], []
    all_xy = xy.astype(np.float32, copy=False)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block_xy = all_xy[start:end]
        dist_block = cdist(block_xy, all_xy, metric=metric)
        for i in range(end - start):
            dist_block[i, start + i] = np.inf
        idx_knn = np.argpartition(dist_block, kth=k, axis=1)[:, :k]
        block_rows = np.repeat(np.arange(start, end), k)
        block_cols = idx_knn.reshape(-1)
        block_data = np.ones_like(block_cols, dtype=float)
        rows_list.append(block_rows)
        cols_list.append(block_cols)
        data_list.append(block_data)
    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    data = np.concatenate(data_list)
    W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    row_sum = np.asarray(W.sum(axis=1)).ravel()
    row_sum[row_sum == 0] = 1.0
    W = W.multiply(1.0 / row_sum[:, None])
    return W, "block"


def _row_norm(x: np.ndarray, eps: float) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def _refine_spot_cell_matrix_svg(
    mat: csr_matrix,
    st_coords: pd.DataFrame,
    st_svg: np.ndarray,
    sc_svg: np.ndarray,
    lambda_base: float,
    k: int,
    eps: float = 1e-8,
    batch_size: Optional[int] = None,
    knn_metric: str = "euclidean",
    knn_block_size: int = 1024,
    knn_max_dense_n: int = 5000,
) -> Tuple[csr_matrix, str]:
    """
    基于 SVG 重构误差的加权平滑：
    - 使用当前分配 mat 预测 ST 表达（plugin_genes）
    - 计算观测 vs 预测的余弦误差 per-spot
    - 将误差归一化到 [0,1] 并乘 lambda_base 得到每个 spot 的局部平滑系数
    - 使用稀疏 KNN 邻接矩阵在 spot 维度做加权平滑，保持稀疏
    """
    if lambda_base <= 0:
        return mat if sparse.isspmatrix_csr(mat) else mat.tocsr(), "skipped_lambda0"
    if not sparse.isspmatrix_csr(mat):
        mat = mat.tocsr()

    if st_svg.size == 0 or sc_svg.size == 0:
        return mat, "skipped_empty_svg"
    if st_svg.shape[0] != mat.shape[0]:
        raise ValueError(f"st_svg 行数 {st_svg.shape[0]} 与 mat spot 数 {mat.shape[0]} 不一致")
    if sc_svg.shape[0] != mat.shape[1]:
        raise ValueError(f"sc_svg 行数 {sc_svg.shape[0]} 与 mat cell 数 {mat.shape[1]} 不一致")

    # 预测表达：SxC · CxG -> SxG
    pred_svg = mat.dot(sc_svg)
    pred_n = _row_norm(pred_svg, eps)
    obs_n = _row_norm(st_svg, eps)
    cos = np.sum(pred_n * obs_n, axis=1)
    cos = np.clip(cos, -1.0, 1.0)
    err = 1.0 - cos
    err = np.nan_to_num(err, nan=0.0, posinf=1.0, neginf=0.0)
    err = err - err.min()
    if err.max() > 0:
        err = err / err.max()
    lambda_vec = lambda_base * err  # (S,)

    W, knn_mode = _build_sparse_knn_safe(
        st_coords,
        k,
        metric=knn_metric,
        block_size=knn_block_size,
        max_dense_n=knn_max_dense_n,
    )
    n_spots, n_cells = mat.shape
    if batch_size is None:
        smoothed = W.dot(mat)
    else:
        blocks = []
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            smoothed_block = W.dot(mat[:, start:end])
            blocks.append(smoothed_block)
        smoothed = sparse.hstack(blocks).tocsr()

    lambda_vec = lambda_vec.astype(float)
    part1 = mat.multiply((1.0 - lambda_vec)[:, None])
    part2 = smoothed.multiply(lambda_vec[:, None])
    mat_ref = (part1 + part2).tocsr()
    return mat_ref, knn_mode


def _build_spot_neighbor_sets(st_coords: pd.DataFrame, k: int) -> List[List[int]]:
    xy = st_coords[["row", "col"]].to_numpy(dtype=float)
    n = xy.shape[0]
    if n == 0:
        return []
    k_eff = min(int(k), max(n - 1, 0))
    if k_eff <= 0:
        return [[] for _ in range(n)]
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=k_eff + 1)
    if idx.ndim == 1:
        idx = idx.reshape(-1, 1)
    neighbors = [set(row[1:]) for row in idx]
    for i, nbrs in enumerate(neighbors):
        for j in list(nbrs):
            neighbors[j].add(i)
    return [sorted(list(nbrs)) for nbrs in neighbors]


def _svg_posterior_local_refine(
    assignments: List[Tuple[int, int, float]],
    *,
    distance_metric: str,
    sc_cost: pd.DataFrame,
    st_cost: pd.DataFrame,
    st_coords: pd.DataFrame,
    pool_orig_ids: List[str],
    cell_types: List[str],
    spot_ids: List[str],
    plugin_genes: List[str],
    genes_use: List[str],
    w_full: np.ndarray,
    spot_weight_matrix: Optional[np.ndarray],
    capacity: np.ndarray,
    neighbor_k: int,
    rounds: int,
    alpha: float,
    max_move_frac: float,
    max_changed_cells: Optional[int],
    selection_mode: str,
    action_topm: Optional[int],
    sort_key: str,
    lambda_partner_hurt: float,
    lambda_base: float,
    lambda_type_shift: float,
    allow_swap: bool,
    delta_min: float,
    weight_clip_max: float,
    base_margin_tau: Optional[float],
    require_delta_base_nonpositive: bool,
    uncertain_margin_tau: Optional[float],
    swap_delta_min: float,
    swap_require_delta_base_nonpositive: bool,
    swap_depth: int,
    swap_only_within_tieset: bool,
    max_swaps_per_round: Optional[int],
    swap_base_mode: str,
    base_pair_budget_eps: float,
    swap_scope: str,
    swap_partner_topm: Optional[int],
    swap_target_topk: Optional[int],
    swap_target_expand_mode: str,
    swap_target_expand_k: int,
) -> Tuple[List[Tuple[int, int, float]], csr_matrix, Dict[str, Any]]:
    meta = {
        "status": "skipped",
        "skip_reason": None,
        "rounds_requested": int(rounds),
        "rounds_executed": 0,
        "neighbor_k": int(neighbor_k),
        "alpha": float(alpha),
        "max_move_frac": float(max_move_frac),
        "max_changed_cells": int(max_changed_cells) if max_changed_cells is not None else None,
        "selection_mode": str(selection_mode),
        "action_topm": int(action_topm) if action_topm is not None else None,
        "sort_key": str(sort_key),
        "lambda_partner_hurt": float(lambda_partner_hurt),
        "lambda_base": float(lambda_base),
        "lambda_type_shift": float(lambda_type_shift),
        "allow_swap": bool(allow_swap),
        "delta_min": float(delta_min),
        "weight_clip_max": float(weight_clip_max),
        "base_distance_metric": str(distance_metric),
        "base_margin_tau": float(base_margin_tau) if base_margin_tau is not None else None,
        "require_delta_base_nonpositive": bool(require_delta_base_nonpositive),
        "uncertain_margin_tau": float(uncertain_margin_tau) if uncertain_margin_tau is not None else None,
        "swap_delta_min": float(swap_delta_min),
        "swap_require_delta_base_nonpositive": bool(swap_require_delta_base_nonpositive),
        "swap_require_delta_base_nonpositive_for_both": bool(swap_require_delta_base_nonpositive),
        "swap_depth": int(swap_depth),
        "swap_only_within_tieset": bool(swap_only_within_tieset),
        "max_swaps_per_round": int(max_swaps_per_round) if max_swaps_per_round is not None else None,
        "swap_base_mode": str(swap_base_mode),
        "base_pair_budget_eps": float(base_pair_budget_eps),
        "swap_scope": str(swap_scope),
        "swap_partner_topm": int(swap_partner_topm) if swap_partner_topm is not None else None,
        "swap_target_topk": int(swap_target_topk) if swap_target_topk is not None else None,
        "swap_target_expand_mode": str(swap_target_expand_mode),
        "swap_target_expand_k": int(swap_target_expand_k),
        "n_cells": int(len(pool_orig_ids)),
        "n_spots": int(len(spot_ids)),
        "n_svg_genes": None,
        "n_candidates_total": 0,
        "n_proposals_total": 0,
        "n_moves_total": 0,
        "n_swaps_total": 0,
        "moves_per_round": [],
        "n_cells_skipped_base_margin": 0,
        "n_cells_skipped_uncertain_margin": 0,
        "n_cells_skipped_base_delta": 0,
        "n_cells_eligible_tiebreak": 0,
        "n_cells_blocked_delta_base": 0,
        "n_moves_due_svg": 0,
        "n_cells_tieset_size_ge2": 0,
        "n_cells_has_target_with_free_capacity": 0,
        "n_cells_blocked_capacity": 0,
        "n_cells_blocked_swap_no_partner": 0,
        "n_cells_blocked_swap_no_candidate": 0,
        "n_cells_blocked_swap_svg_gain": 0,
        "n_cells_blocked_swap_conflict": 0,
        "n_cells_svg_prefers_other_in_tieset": 0,
        "n_cells_swap_attempted": 0,
        "n_cells_blocked_swap_base_budget": 0,
        "n_cells_pass_base_budget": 0,
        "n_cells_pass_all_gates_pre_conflict": 0,
        "best_pair_delta_base_sum_min": None,
        "best_pair_delta_base_sum_p10": None,
        "best_pair_delta_base_sum_p50": None,
        "swap_target_spots_per_cell_mean": None,
        "swap_target_spots_per_cell_p50": None,
        "partner_pool_size_mean": None,
        "partner_pool_size_p50": None,
        "partner_pool_size_min": None,
        "partner_pool_ge_20_count": 0,
        "partner_pool_ge_50_count": 0,
        "n_cells_target_spot_empty_partner_pool": 0,
        "n_swap_candidates_scanned_total": 0,
        "n_cells_blocked_swap_delta_min": 0,
        "n_actions_generated_total": 0,
        "n_actions_after_base_budget": 0,
        "n_actions_after_gainA": 0,
        "selected_actions_gainA_sum": 0.0,
        "selected_target_spot_unique_count": 0,
        "svg_gain_a_best_p50": None,
        "svg_gain_a_best_p90": None,
        "svg_gain_b_best_p50": None,
        "svg_gain_b_best_p90": None,
        "svg_gain_sum_best_p50": None,
        "svg_gain_sum_best_p90": None,
        "max_moves_per_round": int(np.floor(max_move_frac * max(1, len(pool_orig_ids)))),
        "capacity_violation": False,
        "capacity_check": None,
    }
    if rounds <= 0:
        meta["skip_reason"] = "rounds=0"
        return assignments, _assignments_to_matrix(assignments, len(spot_ids), len(pool_orig_ids)).tocsr(), meta
    if alpha <= 0:
        meta["skip_reason"] = "alpha=0"
        return assignments, _assignments_to_matrix(assignments, len(spot_ids), len(pool_orig_ids)).tocsr(), meta
    if max_move_frac <= 0:
        meta["skip_reason"] = "max_move_frac=0"
        return assignments, _assignments_to_matrix(assignments, len(spot_ids), len(pool_orig_ids)).tocsr(), meta
    if max_changed_cells is not None and int(max_changed_cells) <= 0:
        meta["skip_reason"] = "max_changed_cells=0"
        return assignments, _assignments_to_matrix(assignments, len(spot_ids), len(pool_orig_ids)).tocsr(), meta
    if not plugin_genes:
        meta["skip_reason"] = "empty_svg_genes"
        return assignments, _assignments_to_matrix(assignments, len(spot_ids), len(pool_orig_ids)).tocsr(), meta

    genes_set = set(genes_use)
    missing_genes = sorted([g for g in plugin_genes if g not in genes_set])
    if missing_genes:
        raise ValueError(f"svg_post: plugin_genes missing in sc/st: {missing_genes[:5]}")

    gene_to_idx = {g: i for i, g in enumerate(genes_use)}
    svg_gene_idx = [gene_to_idx[g] for g in plugin_genes]
    meta["n_svg_genes"] = int(len(svg_gene_idx))
    if len(svg_gene_idx) == 0:
        meta["skip_reason"] = "svg_genes_empty_after_align"
        return assignments, _assignments_to_matrix(assignments, len(spot_ids), len(pool_orig_ids)).tocsr(), meta

    weights = w_full[np.array(svg_gene_idx, dtype=int)]
    weights = np.minimum(weights, float(weight_clip_max))

    sc_svg = sc_cost.loc[pool_orig_ids, plugin_genes].to_numpy(dtype=float, copy=False)
    st_svg = st_cost.loc[spot_ids, plugin_genes].to_numpy(dtype=float, copy=False)
    sc_svg = sc_svg * weights
    st_svg = st_svg * weights
    if spot_weight_matrix is not None:
        if spot_weight_matrix.shape[1] != len(genes_use):
            raise ValueError("svg_post: spot_weight_matrix gene dimension mismatch")
        st_svg = st_svg * spot_weight_matrix[:, svg_gene_idx]

    if not np.isfinite(sc_svg).all():
        raise ValueError("svg_post: non-finite values in sc_svg")
    if not np.isfinite(st_svg).all():
        raise ValueError("svg_post: non-finite values in st_svg")

    dm = str(distance_metric or "Pearson_correlation")
    if dm not in {"Pearson_correlation", "Spearman_correlation", "Euclidean"}:
        raise ValueError(f"svg_post: unsupported distance_metric={dm!r}")
    meta["base_distance_metric"] = dm
    base_margin_tau = float(base_margin_tau or 0.0)
    swap_delta_min = float(swap_delta_min or 0.0)
    require_delta_base_nonpositive = bool(require_delta_base_nonpositive)
    swap_require_delta_base_nonpositive = bool(swap_require_delta_base_nonpositive)
    swap_depth = int(swap_depth or 1)
    if swap_depth != 1:
        raise ValueError("svg_post: swap_depth>1 not supported")
    swap_only_within_tieset = bool(swap_only_within_tieset)
    max_swaps_per_round = int(max_swaps_per_round) if max_swaps_per_round is not None else None
    swap_base_mode = str(swap_base_mode or "both_nonpositive").strip().lower()
    if swap_base_mode not in {"both_nonpositive", "pair_budget"}:
        raise ValueError(f"svg_post: invalid swap_base_mode={swap_base_mode!r}")
    base_pair_budget_eps = float(base_pair_budget_eps or 0.0)
    if max_changed_cells is not None:
        max_changed_cells = int(max_changed_cells)
    selection_mode = str(selection_mode or "sequential_greedy").strip().lower()
    if selection_mode not in {"sequential_greedy", "global_pool_greedy"}:
        raise ValueError(f"svg_post: invalid selection_mode={selection_mode!r}")
    if action_topm is not None:
        action_topm = int(action_topm)
    sort_key = str(sort_key or "gainA_penalized").strip()
    sort_key_norm = sort_key.strip().lower()
    if sort_key_norm in ("gainsum", "gain_sum"):
        sort_key = "gainSum"
    elif sort_key_norm in ("gaina", "gain_a"):
        sort_key = "gainA"
    elif sort_key_norm in ("gaina_penalized", "gainA_penalized".lower()):
        sort_key = "gainA_penalized"
    else:
        raise ValueError(f"svg_post: invalid sort_key={sort_key!r}")
    lambda_partner_hurt = float(lambda_partner_hurt or 0.0)
    lambda_base = float(lambda_base or 0.0)
    lambda_type_shift = float(lambda_type_shift or 0.0)
    if lambda_partner_hurt < 0 or lambda_base < 0 or lambda_type_shift < 0:
        raise ValueError("svg_post: lambda_* must be >= 0")
    swap_scope = str(swap_scope or "a_and_b_tieset").strip().lower()
    if swap_scope not in {"a_tieset_only", "a_and_b_tieset"}:
        raise ValueError(f"svg_post: invalid swap_scope={swap_scope!r}")
    if swap_partner_topm is not None:
        swap_partner_topm = int(swap_partner_topm)
    if swap_target_topk is not None:
        swap_target_topk = int(swap_target_topk)
    swap_target_expand_mode = str(swap_target_expand_mode or "none").strip().lower()
    if swap_target_expand_mode in ("", "none"):
        swap_target_expand_mode = "none"
    if swap_target_expand_mode not in {"none", "neighbor_knn"}:
        raise ValueError(f"svg_post: invalid swap_target_expand_mode={swap_target_expand_mode!r}")
    swap_target_expand_k = int(swap_target_expand_k or 0)
    if uncertain_margin_tau is not None:
        uncertain_margin_tau = float(uncertain_margin_tau)
    meta["base_margin_tau"] = float(base_margin_tau)
    meta["uncertain_margin_tau"] = float(uncertain_margin_tau) if uncertain_margin_tau is not None else None
    meta["swap_delta_min"] = float(swap_delta_min)
    meta["require_delta_base_nonpositive"] = bool(require_delta_base_nonpositive)
    meta["swap_require_delta_base_nonpositive"] = bool(swap_require_delta_base_nonpositive)

    sc_base = sc_cost.loc[pool_orig_ids, genes_use].to_numpy(dtype=float, copy=False)
    st_base = st_cost.loc[spot_ids, genes_use].to_numpy(dtype=float, copy=False)
    sc_base = sc_base * w_full
    st_base = st_base * w_full
    if spot_weight_matrix is not None:
        if spot_weight_matrix.shape[1] != len(genes_use):
            raise ValueError("svg_post: spot_weight_matrix gene dimension mismatch")
        st_base = st_base * spot_weight_matrix

    if not np.isfinite(sc_base).all():
        raise ValueError("svg_post: non-finite values in sc_base")
    if not np.isfinite(st_base).all():
        raise ValueError("svg_post: non-finite values in st_base")

    if dm == "Spearman_correlation":
        sc_base_use = np.apply_along_axis(rankdata, 1, sc_base)
        st_base_use = np.apply_along_axis(rankdata, 1, st_base)
        base_metric = "correlation"
    elif dm == "Pearson_correlation":
        sc_base_use = sc_base
        st_base_use = st_base
        base_metric = "correlation"
    else:
        sc_base_use = sc_base
        st_base_use = st_base
        base_metric = "euclidean"

    n_cells = len(pool_orig_ids)
    n_spots = len(spot_ids)
    if n_cells == 0 or n_spots == 0:
        meta["skip_reason"] = "empty_cells_or_spots"
        return assignments, _assignments_to_matrix(assignments, n_spots, n_cells).tocsr(), meta
    if len(cell_types) != n_cells:
        raise ValueError(f"svg_post: cell_types length {len(cell_types)} != n_cells {n_cells}")

    current_spot = np.full(n_cells, -1, dtype=int)
    for cell_idx, spot_idx, _ in assignments:
        current_spot[cell_idx] = int(spot_idx)
    if (current_spot < 0).any():
        raise ValueError("svg_post: missing assignments for some cells")

    spot_neighbors = _build_spot_neighbor_sets(st_coords, neighbor_k)
    if len(spot_neighbors) != n_spots:
        raise ValueError("svg_post: neighbor list size mismatch")

    cells_by_spot: List[List[int]] = [[] for _ in range(n_spots)]
    for c, s in enumerate(current_spot):
        cells_by_spot[int(s)].append(c)
    occupancy = np.array([len(x) for x in cells_by_spot], dtype=int)
    cap = np.asarray(capacity, dtype=int)
    if cap.shape[0] != n_spots:
        raise ValueError("svg_post: capacity length mismatch")

    n_genes_svg = sc_svg.shape[1]

    def _score_cell_spots_svg(cell_idx: int, spots: List[int]) -> np.ndarray:
        vec = sc_svg[cell_idx]
        diffs = np.abs(st_svg[np.array(spots, dtype=int)] - vec)
        return diffs.mean(axis=1) if n_genes_svg > 0 else np.full(len(spots), 0.0)

    def _score_cell_spots_base(cell_idx: int, spots: List[int]) -> np.ndarray:
        vec = sc_base_use[cell_idx][None, :]
        cand = st_base_use[np.array(spots, dtype=int)]
        if base_metric == "correlation":
            scores = cdist(vec, cand, metric="correlation")[0]
        else:
            scores = cdist(vec, cand, metric="euclidean")[0]
        if not np.isfinite(scores).all():
            raise ValueError("svg_post: non-finite values in base_cost")
        return scores

    moves_total = 0
    swaps_total = 0
    moves_per_round: List[int] = []
    n_candidates_total = 0
    n_proposals_total = 0
    skipped_base_margin = 0
    skipped_uncertain = 0
    skipped_base_delta = 0
    moves_due_svg = 0
    eligible_tiebreak = np.zeros(n_cells, dtype=bool)
    blocked_delta_base = np.zeros(n_cells, dtype=bool)
    tieset_size_ge2 = np.zeros(n_cells, dtype=bool)
    has_target_free_capacity = np.zeros(n_cells, dtype=bool)
    blocked_capacity = np.zeros(n_cells, dtype=bool)
    blocked_swap_no_partner = np.zeros(n_cells, dtype=bool)
    blocked_swap_no_candidate = np.zeros(n_cells, dtype=bool)
    blocked_swap_svg_gain = np.zeros(n_cells, dtype=bool)
    blocked_swap_conflict = np.zeros(n_cells, dtype=bool)
    pass_base_budget = np.zeros(n_cells, dtype=bool)
    pass_all_gates_pre_conflict = np.zeros(n_cells, dtype=bool)
    svg_prefers_other = np.zeros(n_cells, dtype=bool)
    swap_attempted = np.zeros(n_cells, dtype=bool)
    blocked_swap_base_budget = np.zeros(n_cells, dtype=bool)
    best_pair_delta_base_sum_min = None
    best_pair_delta_base_sum_cells: List[float] = []
    swap_gain_a_best_cells: List[float] = []
    swap_gain_b_best_cells: List[float] = []
    swap_gain_sum_best_cells: List[float] = []
    type_penalty_candidate_total = 0
    type_penalty_candidate_nonzero = 0
    type_penalty_selected_total = 0
    type_penalty_selected_nonzero = 0
    actions_generated_total = 0
    actions_after_base_budget = 0
    actions_after_gainA = 0
    selected_actions_gain_sum = 0.0
    selected_target_spots: set[int] = set()
    selected_actions: List[Dict[str, Any]] = []
    target_spot_empty_partner_pool = np.zeros(n_cells, dtype=bool)
    blocked_swap_delta_min = np.zeros(n_cells, dtype=bool)
    n_swap_candidates_scanned_total = 0
    swap_target_counts: List[int] = []
    swap_target_set_sizes: List[int] = []
    swap_target_after_topk_sizes: List[int] = []
    partner_pool_sizes: List[int] = []
    partner_pool_ge_20 = 0
    partner_pool_ge_50 = 0

    def _scaled_tau(tau: float, best: float) -> float:
        return float(tau) * max(1.0, abs(float(best)))

    def _record_action(
        *,
        action_type: str,
        c: int,
        s: int,
        t: int,
        gain_a: float,
        gain_b: float,
        gain_sum: float,
        partner_hurt: float,
        type_shift_penalty: float,
        base_pair_sum: float,
        c2: Optional[int] = None,
    ) -> None:
        nonlocal type_penalty_selected_total, type_penalty_selected_nonzero
        cell_type = str(cell_types[c])
        partner_cell_type = str(cell_types[c2]) if c2 is not None else None
        row = {
            "action_type": str(action_type),
            "cell_id": str(pool_orig_ids[c]),
            "cell_idx": int(c),
            "cell_type": cell_type,
            "from_spot": str(spot_ids[s]),
            "to_spot": str(spot_ids[t]),
            "spot_from_idx": int(s),
            "spot_to_idx": int(t),
            "gainA": float(gain_a),
            "gainB": float(gain_b),
            "gainSum": float(gain_sum),
            "partner_hurt": float(partner_hurt),
            "type_shift_penalty": float(type_shift_penalty),
            "base_pair_sum": float(base_pair_sum),
            "partner_cell_id": str(pool_orig_ids[c2]) if c2 is not None else None,
            "partner_cell_idx": int(c2) if c2 is not None else None,
            "partner_cell_type": partner_cell_type,
        }
        if action_type == "swap":
            type_penalty_selected_total += 1
            if type_shift_penalty > 0:
                type_penalty_selected_nonzero += 1
        selected_actions.append(row)

    base_spot = current_spot.copy()
    for r in range(int(rounds)):
        proposals: List[Tuple[float, int, int, int, float, float, float, float, bool, List[int]]] = []
        swaps_round = 0
        for c in range(n_cells):
            s = int(current_spot[c])
            cand = [s] + spot_neighbors[s]
            if len(cand) == 1:
                continue
            base_scores = _score_cell_spots_base(c, cand)
            n_candidates_total += len(cand)
            best_idx = int(np.argmin(base_scores))
            best_base = float(base_scores[best_idx])
            base_margin = _scaled_tau(base_margin_tau, best_base)
            candidate_idx = np.where(base_scores <= best_base + base_margin)[0]
            if 0 not in candidate_idx:
                skipped_base_margin += 1
                continue
            eligible_tiebreak[c] = True
            if len(candidate_idx) >= 2:
                tieset_size_ge2[c] = True
                tie_spots = [cand[i] for i in candidate_idx]
                for spot in tie_spots:
                    if spot != s and occupancy[spot] < cap[spot]:
                        has_target_free_capacity[c] = True
                        break
            if len(candidate_idx) <= 1:
                continue
            tie_idx = [int(i) for i in candidate_idx if int(i) != 0]
            cand_sel = [cand[i] for i in candidate_idx]
            svg_scores = _score_cell_spots_svg(c, cand_sel)
            current_rel_idx = int(np.where(candidate_idx == 0)[0][0])
            score_current_svg = float(svg_scores[current_rel_idx])
            best_svg_rel_idx = int(np.argmin(svg_scores))
            best_spot = int(cand_sel[best_svg_rel_idx])
            if best_spot != s:
                svg_prefers_other[c] = True
            if uncertain_margin_tau is not None:
                if len(base_scores) < 2:
                    skipped_uncertain += 1
                    continue
                base_sorted = np.partition(base_scores, 1)
                margin = float(base_sorted[1] - base_sorted[0])
                if margin > _scaled_tau(uncertain_margin_tau, best_base):
                    skipped_uncertain += 1
                    continue
            if best_spot == s:
                continue
            score_target_svg = float(svg_scores[best_svg_rel_idx])
            gain = float((score_current_svg - score_target_svg) * alpha)
            if gain <= delta_min:
                continue
            base_current = float(base_scores[0])
            base_target = float(base_scores[candidate_idx[best_svg_rel_idx]])
            delta_base = base_target - base_current
            if require_delta_base_nonpositive and delta_base > 0:
                skipped_base_delta += 1
                blocked_delta_base[c] = True
                continue
            base_best_spot = int(cand[best_idx])
            due_svg = best_spot != base_best_spot
            expand_idx: List[int] = []
            if swap_target_expand_mode == "neighbor_knn" and swap_target_expand_k > 0:
                if len(cand) > 1:
                    neighbor_idx = np.argsort(base_scores[1:])[:swap_target_expand_k] + 1
                    expand_idx = [int(i) for i in neighbor_idx if int(i) != 0]
            target_idx = sorted(set(tie_idx) | set(expand_idx))
            swap_target_set_sizes.append(len(target_idx))
            if target_idx:
                target_spots = [cand[i] for i in target_idx]
                target_scores = _score_cell_spots_svg(c, target_spots)
                svg_pairs = [(int(spot), float(score)) for spot, score in zip(target_spots, target_scores)]
                svg_pairs.sort(key=lambda x: x[1])
                if swap_target_topk is not None and swap_target_topk > 0:
                    svg_pairs = svg_pairs[:swap_target_topk]
                swap_targets = [spot for spot, _ in svg_pairs]
            else:
                swap_targets = []
            swap_target_after_topk_sizes.append(len(swap_targets))
            proposals.append((gain, c, s, best_spot, score_current_svg, score_target_svg, base_current, base_target, due_svg, swap_targets))

        if not proposals:
            break

        proposals.sort(key=lambda x: x[0], reverse=True)
        n_proposals_total += len(proposals)
        max_moves = int(np.floor(max_move_frac * n_cells))
        if max_changed_cells is not None:
            remaining_allowed = max_changed_cells - moves_total
            if remaining_allowed <= 0:
                break
            if max_moves <= 0:
                break
            if max_moves > remaining_allowed:
                max_moves = remaining_allowed
        if max_moves <= 0:
            break

        moves_round = 0
        if selection_mode == "global_pool_greedy":
            actions_by_cell: Dict[int, List[Tuple[str, float, float, float, float, float, float, int, int, int, Optional[int], bool]]] = {}
            for gain, c, s, t, score_s, score_t, base_s, base_t, due_svg, swap_targets in proposals:
                if int(current_spot[c]) != s:
                    continue
                if occupancy[t] < cap[t]:
                    base_delta = float(base_t - base_s)
                    gain_a = float(gain)
                    gain_b = 0.0
                    gain_sum = gain_a
                    type_penalty = 0.0
                    if sort_key == "gainSum":
                        select_score = gain_sum
                    elif sort_key == "gainA":
                        select_score = gain_a
                    else:
                        select_score = gain_a - (lambda_base * base_delta)
                    actions_by_cell.setdefault(c, []).append(
                        ("move", select_score, gain_a, gain_b, gain_sum, base_delta, type_penalty, c, s, t, None, due_svg)
                    )
                    continue
                if not allow_swap:
                    blocked_capacity[c] = True
                    continue
                swap_attempted[c] = True
                best_swap_gain = None
                best_swap_gain_b = None
                best_swap_gain_sum = None
                best_swap_gain_any = None
                best_swap_gain_any_b = None
                best_swap_gain_any_sum = None
                best_swap_base_sum = None
                best_c2 = None
                best_target = None
                pass_base_budget_cell = False
                base_budget_rejected = 0
                candidate_scanned = 0
                best_pair_delta_base_sum_cell = None
                swap_target_counts.append(len(swap_targets))
                any_target_pool = False
                for target in swap_targets:
                    partner_pool = [c2 for c2 in cells_by_spot[target] if c2 != c]
                    if swap_partner_topm is not None and swap_partner_topm > 0 and len(partner_pool) > swap_partner_topm:
                        scored = []
                        for c2 in partner_pool:
                            base_c2_t = float(_score_cell_spots_base(c2, [target])[0])
                            scored.append((base_c2_t, c2))
                        scored.sort(reverse=True)
                        partner_pool = [c2 for _, c2 in scored[:swap_partner_topm]]
                    pool_size = len(partner_pool)
                    partner_pool_sizes.append(pool_size)
                    if pool_size >= 20:
                        partner_pool_ge_20 += 1
                    if pool_size >= 50:
                        partner_pool_ge_50 += 1
                    if pool_size == 0:
                        continue
                    any_target_pool = True
                    for c2 in partner_pool:
                        if c2 == c:
                            continue
                        if swap_scope == "a_and_b_tieset":
                            cand2 = [target] + spot_neighbors[target]
                            if s not in cand2:
                                continue
                            base_scores2 = _score_cell_spots_base(c2, cand2)
                            best_base2 = float(np.min(base_scores2))
                            margin2 = _scaled_tau(base_margin_tau, best_base2)
                            tie_idx2 = np.where(base_scores2 <= best_base2 + margin2)[0]
                            if 0 not in tie_idx2:
                                continue
                            idx_s2 = cand2.index(s)
                            if idx_s2 not in tie_idx2:
                                continue
                        candidate_scanned += 1
                        n_swap_candidates_scanned_total += 1
                        type_penalty_candidate_total += 1
                        if cell_types[c] != cell_types[c2]:
                            type_penalty_candidate_nonzero += 1
                        score_c_t = float(_score_cell_spots_svg(c, [target])[0])
                        score_c2_t = float(_score_cell_spots_svg(c2, [target])[0])
                        score_c2_s = float(_score_cell_spots_svg(c2, [s])[0])
                        base_c2 = _score_cell_spots_base(c2, [target, s])
                        base_c_t = float(_score_cell_spots_base(c, [target])[0])
                        delta_base_c = base_c_t - base_s
                        delta_base_c2 = float(base_c2[1] - base_c2[0])
                        delta_base_sum = float(delta_base_c + delta_base_c2)
                        if best_pair_delta_base_sum_min is None or delta_base_sum < best_pair_delta_base_sum_min:
                            best_pair_delta_base_sum_min = delta_base_sum
                        if swap_base_mode == "both_nonpositive":
                            if delta_base_c > 0 or delta_base_c2 > 0:
                                base_budget_rejected += 1
                                continue
                        elif swap_base_mode == "pair_budget":
                            if delta_base_c > 0:
                                base_budget_rejected += 1
                                continue
                            if delta_base_sum > base_pair_budget_eps:
                                base_budget_rejected += 1
                                continue
                        pass_base_budget_cell = True
                        gain_a = float((score_s - score_c_t) * alpha)
                        gain_b = float((score_c2_t - score_c2_s) * alpha)
                        gain_sum = float(gain_a + gain_b)
                        if best_swap_gain_any is None or gain_a > best_swap_gain_any:
                            best_swap_gain_any = gain_a
                            best_swap_gain_any_b = gain_b
                            best_swap_gain_any_sum = gain_sum
                        if gain_a <= 0:
                            continue
                        if best_pair_delta_base_sum_cell is None or delta_base_sum < best_pair_delta_base_sum_cell:
                            best_pair_delta_base_sum_cell = delta_base_sum
                        if best_swap_gain is None or gain_a > best_swap_gain:
                            best_swap_gain = gain_a
                            best_swap_gain_b = gain_b
                            best_swap_gain_sum = gain_sum
                            best_c2 = c2
                            best_target = target
                            best_swap_base_sum = delta_base_sum
                if not any_target_pool:
                    target_spot_empty_partner_pool[c] = True
                if candidate_scanned > 0 and best_pair_delta_base_sum_cell is not None:
                    best_pair_delta_base_sum_cells.append(float(best_pair_delta_base_sum_cell))
                if candidate_scanned > 0 and best_swap_gain_any is not None:
                    swap_gain_a_best_cells.append(float(best_swap_gain_any))
                    swap_gain_b_best_cells.append(float(best_swap_gain_any_b or 0.0))
                    swap_gain_sum_best_cells.append(float(best_swap_gain_any_sum or 0.0))
                if candidate_scanned == 0:
                    blocked_swap_no_candidate[c] = True
                    blocked_swap_no_partner[c] = True
                elif not pass_base_budget_cell:
                    blocked_swap_base_budget[c] = True
                elif best_swap_gain_any is None or best_swap_gain_any <= 0:
                    blocked_swap_svg_gain[c] = True
                elif best_swap_gain is None or best_swap_gain <= swap_delta_min:
                    blocked_swap_delta_min[c] = True
                else:
                    pass_base_budget[c] = True
                    pass_all_gates_pre_conflict[c] = True
                    gain_a = float(best_swap_gain or 0.0)
                    gain_b = float(best_swap_gain_b or 0.0)
                    gain_sum = float(best_swap_gain_sum or (gain_a + gain_b))
                    base_delta_sum = float(best_swap_base_sum or 0.0)
                    partner_hurt = max(0.0, -gain_b)
                    type_penalty = 1.0 if cell_types[c] != cell_types[int(best_c2)] else 0.0
                    if sort_key == "gainSum":
                        select_score = gain_sum
                    elif sort_key == "gainA":
                        select_score = gain_a
                    else:
                        select_score = (
                            gain_a
                            - (lambda_partner_hurt * partner_hurt)
                            - (lambda_base * base_delta_sum)
                            - (lambda_type_shift * type_penalty)
                        )
                    actions_by_cell.setdefault(c, []).append(
                        ("swap", select_score, gain_a, gain_b, gain_sum, base_delta_sum, type_penalty, c, s, int(best_target), int(best_c2), due_svg)
                    )
            action_pool: List[Tuple[str, float, float, float, float, float, float, int, int, int, Optional[int], bool]] = []
            for c, acts in actions_by_cell.items():
                acts.sort(key=lambda x: x[1], reverse=True)
                if action_topm is not None and action_topm > 0:
                    acts = acts[:action_topm]
                action_pool.extend(acts)
            actions_generated_total += len(action_pool)
            actions_after_base_budget += len(action_pool)
            actions_after_gainA += len(action_pool)
            action_pool.sort(key=lambda x: (-x[1], x[5]))
            used_cells: set[int] = set()
            for action_type, select_score, gain_a, gain_b, gain_sum, base_delta_sum, type_penalty, c, s, t, c2, due_svg in action_pool:
                if moves_round >= max_moves:
                    break
                if c in used_cells:
                    if action_type == "swap":
                        blocked_swap_conflict[c] = True
                    continue
                if action_type == "swap" and c2 is not None and c2 in used_cells:
                    blocked_swap_conflict[c] = True
                    continue
                if action_type == "move":
                    if int(current_spot[c]) != s:
                        continue
                    if occupancy[t] >= cap[t]:
                        blocked_capacity[c] = True
                        continue
                    current_spot[c] = t
                    occupancy[s] -= 1
                    occupancy[t] += 1
                    cells_by_spot[s].remove(c)
                    cells_by_spot[t].append(c)
                    moves_round += 1
                    moves_total += 1
                    used_cells.add(c)
                    selected_target_spots.add(t)
                    selected_actions_gain_sum += float(gain_a)
                    _record_action(
                        action_type="move",
                        c=c,
                        s=s,
                        t=t,
                        gain_a=float(gain_a),
                        gain_b=0.0,
                        gain_sum=float(gain_a),
                        partner_hurt=0.0,
                        type_shift_penalty=float(type_penalty),
                        base_pair_sum=float(base_delta_sum),
                        c2=None,
                    )
                    if due_svg:
                        moves_due_svg += 1
                else:
                    if max_swaps_per_round is not None and swaps_round >= max_swaps_per_round:
                        blocked_swap_conflict[c] = True
                        continue
                    if c2 is None:
                        blocked_swap_conflict[c] = True
                        continue
                    if int(current_spot[c]) != s or int(current_spot[c2]) != t:
                        blocked_swap_conflict[c] = True
                        continue
                    if moves_round + 2 > max_moves:
                        blocked_swap_conflict[c] = True
                        continue
                    current_spot[c] = t
                    current_spot[c2] = s
                    cells_by_spot[s].remove(c)
                    cells_by_spot[t].append(c)
                    cells_by_spot[t].remove(c2)
                    cells_by_spot[s].append(c2)
                    moves_round += 2
                    moves_total += 2
                    swaps_total += 1
                    swaps_round += 1
                    used_cells.add(c)
                    used_cells.add(c2)
                    selected_target_spots.add(t)
                    selected_actions_gain_sum += float(gain_a)
                    _record_action(
                        action_type="swap",
                        c=c,
                        s=s,
                        t=t,
                        gain_a=float(gain_a),
                        gain_b=float(gain_b),
                        gain_sum=float(gain_sum),
                        partner_hurt=float(max(0.0, -gain_b)),
                        type_shift_penalty=float(type_penalty),
                        base_pair_sum=float(base_delta_sum),
                        c2=int(c2),
                    )
                    if due_svg:
                        moves_due_svg += 1
        else:
            for gain, c, s, t, score_s, score_t, base_s, base_t, due_svg, swap_targets in proposals:
                if moves_round >= max_moves:
                    break
                if int(current_spot[c]) != s:
                    continue
                if occupancy[t] < cap[t]:
                    current_spot[c] = t
                    occupancy[s] -= 1
                    occupancy[t] += 1
                    cells_by_spot[s].remove(c)
                    cells_by_spot[t].append(c)
                    moves_round += 1
                    moves_total += 1
                    selected_target_spots.add(t)
                    selected_actions_gain_sum += float(gain)
                    _record_action(
                        action_type="move",
                        c=c,
                        s=s,
                        t=t,
                        gain_a=float(gain),
                        gain_b=0.0,
                        gain_sum=float(gain),
                        partner_hurt=0.0,
                        type_shift_penalty=0.0,
                        base_pair_sum=float(base_t - base_s),
                        c2=None,
                    )
                    if due_svg:
                        moves_due_svg += 1
                    continue
                if not allow_swap:
                    blocked_capacity[c] = True
                    continue
                if moves_round + 2 > max_moves:
                    blocked_swap_conflict[c] = True
                    continue
                if max_swaps_per_round is not None and swaps_round >= max_swaps_per_round:
                    continue
                swap_attempted[c] = True
                best_swap_gain = None
                best_swap_gain_b = None
                best_swap_gain_sum = None
                best_swap_gain_any = None
                best_swap_gain_any_b = None
                best_swap_gain_any_sum = None
                best_swap_base_sum = None
                best_c2 = None
                best_target = None
                pass_base_budget_cell = False
                base_budget_rejected = 0
                candidate_scanned = 0
                swap_executed = False
                best_pair_delta_base_sum_cell = None
                swap_target_counts.append(len(swap_targets))
                any_target_pool = False
                for target in swap_targets:
                    partner_pool = [c2 for c2 in cells_by_spot[target] if c2 != c]
                    if swap_partner_topm is not None and swap_partner_topm > 0 and len(partner_pool) > swap_partner_topm:
                        scored = []
                        for c2 in partner_pool:
                            base_c2_t = float(_score_cell_spots_base(c2, [target])[0])
                            scored.append((base_c2_t, c2))
                        scored.sort(reverse=True)
                        partner_pool = [c2 for _, c2 in scored[:swap_partner_topm]]
                    pool_size = len(partner_pool)
                    partner_pool_sizes.append(pool_size)
                    if pool_size >= 20:
                        partner_pool_ge_20 += 1
                    if pool_size >= 50:
                        partner_pool_ge_50 += 1
                    if pool_size == 0:
                        continue
                    any_target_pool = True
                    for c2 in partner_pool:
                        if c2 == c:
                            continue
                        if swap_scope == "a_and_b_tieset":
                            cand2 = [target] + spot_neighbors[target]
                            if s not in cand2:
                                continue
                            base_scores2 = _score_cell_spots_base(c2, cand2)
                            best_base2 = float(np.min(base_scores2))
                            margin2 = _scaled_tau(base_margin_tau, best_base2)
                            tie_idx2 = np.where(base_scores2 <= best_base2 + margin2)[0]
                            if 0 not in tie_idx2:
                                continue
                            idx_s2 = cand2.index(s)
                            if idx_s2 not in tie_idx2:
                                continue
                        candidate_scanned += 1
                        n_swap_candidates_scanned_total += 1
                        type_penalty_candidate_total += 1
                        if cell_types[c] != cell_types[c2]:
                            type_penalty_candidate_nonzero += 1
                        score_c_t = float(_score_cell_spots_svg(c, [target])[0])
                        score_c2_t = float(_score_cell_spots_svg(c2, [target])[0])
                        score_c2_s = float(_score_cell_spots_svg(c2, [s])[0])
                        base_c2 = _score_cell_spots_base(c2, [target, s])
                        base_c_t = float(_score_cell_spots_base(c, [target])[0])
                        delta_base_c = base_c_t - base_s
                        delta_base_c2 = float(base_c2[1] - base_c2[0])
                        delta_base_sum = float(delta_base_c + delta_base_c2)
                        if best_pair_delta_base_sum_min is None or delta_base_sum < best_pair_delta_base_sum_min:
                            best_pair_delta_base_sum_min = delta_base_sum
                        if swap_base_mode == "both_nonpositive":
                            if delta_base_c > 0 or delta_base_c2 > 0:
                                base_budget_rejected += 1
                                continue
                        elif swap_base_mode == "pair_budget":
                            if delta_base_c > 0:
                                base_budget_rejected += 1
                                continue
                            if delta_base_sum > base_pair_budget_eps:
                                base_budget_rejected += 1
                                continue
                        pass_base_budget_cell = True
                        gain_a = float((score_s - score_c_t) * alpha)
                        gain_b = float((score_c2_t - score_c2_s) * alpha)
                        gain_sum = float(gain_a + gain_b)
                        if best_swap_gain_any is None or gain_a > best_swap_gain_any:
                            best_swap_gain_any = gain_a
                            best_swap_gain_any_b = gain_b
                            best_swap_gain_any_sum = gain_sum
                        if gain_a <= 0:
                            continue
                        if best_pair_delta_base_sum_cell is None or delta_base_sum < best_pair_delta_base_sum_cell:
                            best_pair_delta_base_sum_cell = delta_base_sum
                        if best_swap_gain is None or gain_a > best_swap_gain:
                            best_swap_gain = gain_a
                            best_swap_gain_b = gain_b
                            best_swap_gain_sum = gain_sum
                            best_c2 = c2
                            best_target = target
                            best_swap_base_sum = delta_base_sum
                if not any_target_pool:
                    target_spot_empty_partner_pool[c] = True
                if candidate_scanned > 0 and best_pair_delta_base_sum_cell is not None:
                    best_pair_delta_base_sum_cells.append(float(best_pair_delta_base_sum_cell))
                if candidate_scanned > 0 and best_swap_gain_any is not None:
                    swap_gain_a_best_cells.append(float(best_swap_gain_any))
                    swap_gain_b_best_cells.append(float(best_swap_gain_any_b or 0.0))
                    swap_gain_sum_best_cells.append(float(best_swap_gain_any_sum or 0.0))
                if candidate_scanned == 0:
                    blocked_swap_no_candidate[c] = True
                    blocked_swap_no_partner[c] = True
                elif not pass_base_budget_cell:
                    blocked_swap_base_budget[c] = True
                elif best_swap_gain_any is None or best_swap_gain_any <= 0:
                    blocked_swap_svg_gain[c] = True
                elif best_swap_gain is None or best_swap_gain <= swap_delta_min:
                    blocked_swap_delta_min[c] = True
                else:
                    pass_base_budget[c] = True
                    pass_all_gates_pre_conflict[c] = True
                    if max_swaps_per_round is not None and swaps_round >= max_swaps_per_round:
                        blocked_swap_conflict[c] = True
                    elif best_c2 is None or best_target is None:
                        blocked_swap_conflict[c] = True
                    else:
                        current_spot[c] = best_target
                        current_spot[best_c2] = s
                        cells_by_spot[s].remove(c)
                        cells_by_spot[best_target].append(c)
                        cells_by_spot[best_target].remove(best_c2)
                        cells_by_spot[s].append(best_c2)
                        moves_round += 2
                        moves_total += 2
                        swaps_total += 1
                        swaps_round += 1
                        swap_executed = True
                        selected_target_spots.add(best_target)
                        gain_a = float(best_swap_gain or 0.0)
                        gain_b = float(best_swap_gain_b or 0.0)
                        gain_sum = float(best_swap_gain_sum or (gain_a + gain_b))
                        selected_actions_gain_sum += float(gain_a)
                        _record_action(
                            action_type="swap",
                            c=c,
                            s=s,
                            t=int(best_target),
                            gain_a=float(gain_a),
                            gain_b=float(gain_b),
                            gain_sum=float(gain_sum),
                            partner_hurt=float(max(0.0, -gain_b)),
                            type_shift_penalty=float(1.0 if cell_types[c] != cell_types[int(best_c2)] else 0.0),
                            base_pair_sum=float(best_swap_base_sum or 0.0),
                            c2=int(best_c2),
                        )
                        if due_svg:
                            moves_due_svg += 1
                if pass_base_budget_cell:
                    pass_base_budget[c] = True
                if candidate_scanned > 0 and base_budget_rejected == candidate_scanned:
                    blocked_swap_base_budget[c] = True
                if not swap_executed:
                    blocked_capacity[c] = True

        moves_per_round.append(moves_round)
        meta["rounds_executed"] = r + 1
        if moves_round == 0:
            break
        if (occupancy > cap).any():
            current_spot = base_spot.copy()
            meta["status"] = "capacity_violation_reverted"
            meta["capacity_violation"] = True
            break

    counts = np.bincount(current_spot, minlength=n_spots)
    cap_diff = counts - cap
    meta["capacity_check"] = {
        "max_overflow": int(np.max(cap_diff)) if len(cap_diff) else 0,
        "max_underfill": int(np.max(-cap_diff)) if len(cap_diff) else 0,
        "n_spots_overflow": int(np.sum(cap_diff > 0)),
        "n_spots_underfill": int(np.sum(cap_diff < 0)),
    }

    assignments_out = [(int(c), int(s), 1.0) for c, s in enumerate(current_spot)]
    hard_mat = _assignments_to_matrix(assignments_out, n_spots, n_cells).tocsr()

    changed = int(np.sum(current_spot != base_spot))
    if selection_mode == "sequential_greedy" and actions_generated_total == 0:
        actions_generated_total = len(selected_actions)
        actions_after_base_budget = len(selected_actions)
        actions_after_gainA = len(selected_actions)
    if selection_mode == "sequential_greedy" and meta["n_actions_generated_total"] == 0:
        meta["n_actions_generated_total"] = int(len(selected_actions))
    meta.update(
        {
            "status": "ok" if not meta["capacity_violation"] else meta["status"],
            "n_candidates_total": int(n_candidates_total),
            "n_proposals_total": int(n_proposals_total),
            "n_moves_total": int(moves_total),
            "n_swaps_total": int(swaps_total),
            "moves_per_round": moves_per_round,
            "n_cells_skipped_base_margin": int(skipped_base_margin),
            "n_cells_skipped_uncertain_margin": int(skipped_uncertain),
            "n_cells_skipped_base_delta": int(skipped_base_delta),
            "n_cells_eligible_tiebreak": int(np.sum(eligible_tiebreak)),
            "n_cells_blocked_delta_base": int(np.sum(blocked_delta_base)),
            "n_moves_due_svg": int(moves_due_svg),
            "n_cells_tieset_size_ge2": int(np.sum(tieset_size_ge2)),
            "n_cells_has_target_with_free_capacity": int(np.sum(has_target_free_capacity)),
            "n_cells_blocked_capacity": int(np.sum(blocked_capacity)),
            "n_cells_blocked_swap_no_partner": int(np.sum(blocked_swap_no_partner)),
            "n_cells_blocked_swap_no_candidate": int(np.sum(blocked_swap_no_candidate)),
            "n_cells_blocked_swap_svg_gain": int(np.sum(blocked_swap_svg_gain)),
            "n_cells_blocked_swap_conflict": int(np.sum(blocked_swap_conflict)),
            "n_cells_svg_prefers_other_in_tieset": int(np.sum(svg_prefers_other)),
            "n_cells_swap_attempted": int(np.sum(swap_attempted)),
            "n_cells_blocked_swap_base_budget": int(np.sum(blocked_swap_base_budget)),
            "n_cells_pass_base_budget": int(np.sum(pass_base_budget)),
            "n_cells_pass_all_gates_pre_conflict": int(np.sum(pass_all_gates_pre_conflict)),
            "best_pair_delta_base_sum_min": best_pair_delta_base_sum_min,
            "n_cells_target_spot_empty_partner_pool": int(np.sum(target_spot_empty_partner_pool)),
            "n_swap_candidates_scanned_total": int(n_swap_candidates_scanned_total),
            "n_cells_blocked_swap_delta_min": int(np.sum(blocked_swap_delta_min)),
            "n_actions_generated_total": int(actions_generated_total),
            "n_actions_after_base_budget": int(actions_after_base_budget),
            "n_actions_after_gainA": int(actions_after_gainA),
            "selected_actions_gainA_sum": float(selected_actions_gain_sum),
            "selected_target_spot_unique_count": int(len(selected_target_spots)),
            "selected_actions": selected_actions,
            "type_penalty_candidate_total": int(type_penalty_candidate_total),
            "type_penalty_candidate_nonzero": int(type_penalty_candidate_nonzero),
            "type_penalty_candidate_nonzero_rate": float(type_penalty_candidate_nonzero / type_penalty_candidate_total)
            if type_penalty_candidate_total
            else None,
            "type_penalty_selected_total": int(type_penalty_selected_total),
            "type_penalty_selected_nonzero": int(type_penalty_selected_nonzero),
            "type_penalty_selected_nonzero_rate": float(type_penalty_selected_nonzero / type_penalty_selected_total)
            if type_penalty_selected_total
            else None,
            "svg_gain_a_best_p50": float(np.percentile(swap_gain_a_best_cells, 50)) if swap_gain_a_best_cells else None,
            "svg_gain_a_best_p90": float(np.percentile(swap_gain_a_best_cells, 90)) if swap_gain_a_best_cells else None,
            "svg_gain_b_best_p50": float(np.percentile(swap_gain_b_best_cells, 50)) if swap_gain_b_best_cells else None,
            "svg_gain_b_best_p90": float(np.percentile(swap_gain_b_best_cells, 90)) if swap_gain_b_best_cells else None,
            "svg_gain_sum_best_p50": float(np.percentile(swap_gain_sum_best_cells, 50)) if swap_gain_sum_best_cells else None,
            "svg_gain_sum_best_p90": float(np.percentile(swap_gain_sum_best_cells, 90)) if swap_gain_sum_best_cells else None,
            "best_pair_delta_base_sum_p10": float(np.percentile(best_pair_delta_base_sum_cells, 10)) if best_pair_delta_base_sum_cells else None,
            "best_pair_delta_base_sum_p50": float(np.percentile(best_pair_delta_base_sum_cells, 50)) if best_pair_delta_base_sum_cells else None,
            "swap_target_spots_per_cell_mean": float(np.mean(swap_target_counts)) if swap_target_counts else None,
            "swap_target_spots_per_cell_p50": float(np.percentile(swap_target_counts, 50)) if swap_target_counts else None,
            "swap_target_set_size_mean": float(np.mean(swap_target_set_sizes)) if swap_target_set_sizes else None,
            "swap_target_set_size_p50": float(np.percentile(swap_target_set_sizes, 50)) if swap_target_set_sizes else None,
            "swap_target_set_size_after_topk_mean": float(np.mean(swap_target_after_topk_sizes)) if swap_target_after_topk_sizes else None,
            "swap_target_set_size_after_topk_p50": float(np.percentile(swap_target_after_topk_sizes, 50)) if swap_target_after_topk_sizes else None,
            "partner_pool_size_mean": float(np.mean(partner_pool_sizes)) if partner_pool_sizes else None,
            "partner_pool_size_p50": float(np.percentile(partner_pool_sizes, 50)) if partner_pool_sizes else None,
            "partner_pool_size_min": int(np.min(partner_pool_sizes)) if partner_pool_sizes else None,
            "partner_pool_ge_20_count": int(partner_pool_ge_20),
            "partner_pool_ge_50_count": int(partner_pool_ge_50),
            "n_changed_cells": int(changed),
            "changed_rate": float(changed / max(1, n_cells)),
        }
    )
    return assignments_out, hard_mat, meta


def _type_posterior_local_refine(
    assignments: List[Tuple[int, int, float]],
    *,
    distance_metric: str,
    sc_cost: pd.DataFrame,
    st_cost: pd.DataFrame,
    st_coords: pd.DataFrame,
    sc_meta: Optional[pd.DataFrame],
    sc_type_col: Optional[str],
    pool_orig_ids: List[str],
    cell_types: List[str],
    spot_ids: List[str],
    genes_use: List[str],
    w_full: np.ndarray,
    spot_weight_matrix: Optional[np.ndarray],
    capacity: np.ndarray,
    neighbor_k: int,
    rounds: int,
    max_changed_cells: Optional[int],
    selection_mode: str,
    action_topm: Optional[int],
    delta_min: float,
    base_margin_tau: Optional[float],
    require_gainA_positive: bool,
    swap_target_topk: Optional[int],
    allow_swap: bool,
    swap_scope: str,
    swap_base_mode: str,
    base_pair_budget_eps: float,
    max_swaps_per_round: Optional[int],
    partner_hurt_max: Optional[float],
    max_high_hurt_actions: Optional[int],
    high_hurt_threshold: Optional[float],
    lambda_partner: float,
    lambda_base: float,
    mode: Optional[str] = None,
    marker_source: Optional[str] = None,
    marker_list: Optional[Any] = None,
    markers_per_type: Optional[int] = None,
    gene_weighting: Optional[str] = None,
    type_order: Optional[List[str]] = None,
) -> Tuple[List[Tuple[int, int, float]], csr_matrix, Dict[str, Any]]:
    meta = {
        "status": "skipped",
        "skip_reason": None,
        "mode": str(mode) if mode is not None else None,
        "marker_source": str(marker_source) if marker_source is not None else None,
        "markers_per_type": int(markers_per_type) if markers_per_type is not None else None,
        "gene_weighting": str(gene_weighting) if gene_weighting is not None else None,
        "require_gainA_positive": bool(require_gainA_positive),
        "rounds_requested": int(rounds),
        "rounds_executed": 0,
        "neighbor_k": int(neighbor_k),
        "max_changed_cells": int(max_changed_cells) if max_changed_cells is not None else None,
        "selection_mode": str(selection_mode),
        "action_topm": int(action_topm) if action_topm is not None else None,
        "delta_min": float(delta_min),
        "base_margin_tau": float(base_margin_tau) if base_margin_tau is not None else None,
        "swap_target_topk": int(swap_target_topk) if swap_target_topk is not None else None,
        "allow_swap": bool(allow_swap),
        "swap_scope": str(swap_scope),
        "swap_base_mode": str(swap_base_mode),
        "base_pair_budget_eps": float(base_pair_budget_eps),
        "max_swaps_per_round": int(max_swaps_per_round) if max_swaps_per_round is not None else None,
        "partner_hurt_max": float(partner_hurt_max) if partner_hurt_max is not None else None,
        "max_high_hurt_actions": int(max_high_hurt_actions) if max_high_hurt_actions is not None else None,
        "high_hurt_threshold": float(high_hurt_threshold) if high_hurt_threshold is not None else None,
        "lambda_partner": float(lambda_partner),
        "lambda_base": float(lambda_base),
        "n_cells": int(len(pool_orig_ids)),
        "n_spots": int(len(spot_ids)),
        "n_cells_missing_type_label": 0,
        "n_cells_missing_type_score": 0,
        "n_types_with_markers": None,
        "n_markers_used_total": None,
        "n_markers_missing_genes": None,
        "type_score_gene_intersection_ok": None,
        "type_score_stats": None,
        "type_score_source": None,
        "n_cells_considered": 0,
        "n_actions_generated_total": 0,
        "n_moves_total": 0,
        "n_swaps_total": 0,
        "moves_per_round": [],
        "n_cells_blocked_base_margin": 0,
        "n_cells_blocked_type_gain": 0,
        "n_cells_blocked_base_budget": 0,
        "n_cells_blocked_partner_hurt": 0,
        "n_actions_blocked_high_hurt_quota": 0,
        "n_high_hurt_selected": 0,
        "n_cells_blocked_swap_no_candidate": 0,
        "n_cells_blocked_swap_base_budget": 0,
        "n_cells_blocked_swap_conflict": 0,
        "n_cells_pass_base_budget": 0,
        "n_cells_pass_all_gates_pre_conflict": 0,
        "capacity_violation": False,
        "capacity_check": None,
    }
    if rounds <= 0:
        meta["skip_reason"] = "rounds=0"
        return assignments, _assignments_to_matrix(assignments, len(spot_ids), len(pool_orig_ids)).tocsr(), meta
    if max_changed_cells is not None and int(max_changed_cells) <= 0:
        meta["skip_reason"] = "max_changed_cells=0"
        return assignments, _assignments_to_matrix(assignments, len(spot_ids), len(pool_orig_ids)).tocsr(), meta

    n_cells = len(pool_orig_ids)
    n_spots = len(spot_ids)
    if n_cells == 0 or n_spots == 0:
        meta["skip_reason"] = "empty_cells_or_spots"
        return assignments, _assignments_to_matrix(assignments, n_spots, n_cells).tocsr(), meta

    if type_order is None:
        type_order = sorted(list(dict.fromkeys([str(x) for x in cell_types])))
    if not type_order:
        meta["skip_reason"] = "empty_type_order"
        return assignments, _assignments_to_matrix(assignments, n_spots, n_cells).tocsr(), meta

    mode = str(mode or "local_prior_swap").strip().lower()
    if mode in ("", "none", "off"):
        mode = "local_prior_swap"
    if mode not in {"local_prior_swap", "local_marker_swap"}:
        raise ValueError(f"type_post: invalid mode={mode!r}")
    meta["mode"] = mode
    if require_gainA_positive is None:
        require_gainA_positive = True
    require_gainA_positive = bool(require_gainA_positive)
    gain_thresh = float(delta_min or 0.0)
    if not require_gainA_positive:
        gain_thresh = -gain_thresh

    if swap_target_topk is not None:
        swap_target_topk = int(swap_target_topk)
        if swap_target_topk <= 0:
            swap_target_topk = None

    dm = str(distance_metric or "Pearson_correlation")
    if dm not in {"Pearson_correlation", "Spearman_correlation", "Euclidean"}:
        raise ValueError(f"type_post: unsupported distance_metric={dm!r}")

    base_margin_tau = float(base_margin_tau or 0.0)
    delta_min = float(delta_min or 0.0)
    swap_scope = str(swap_scope or "a_tieset_only").strip().lower()
    if swap_scope not in {"a_tieset_only", "a_and_b_tieset"}:
        raise ValueError(f"type_post: invalid swap_scope={swap_scope!r}")
    swap_base_mode = str(swap_base_mode or "pair_budget").strip().lower()
    if swap_base_mode not in {"both_nonpositive", "pair_budget"}:
        raise ValueError(f"type_post: invalid swap_base_mode={swap_base_mode!r}")
    selection_mode = str(selection_mode or "global_pool_greedy").strip().lower()
    if selection_mode not in {"sequential_greedy", "global_pool_greedy"}:
        raise ValueError(f"type_post: invalid selection_mode={selection_mode!r}")
    if action_topm is not None:
        action_topm = int(action_topm)
    max_swaps_per_round = int(max_swaps_per_round) if max_swaps_per_round is not None else None
    if lambda_partner < 0 or lambda_base < 0:
        raise ValueError("type_post: lambda_* must be >= 0")
    if partner_hurt_max is not None and float(partner_hurt_max) < 0:
        raise ValueError("type_post: partner_hurt_max must be >= 0")
    if partner_hurt_max is not None:
        partner_hurt_max = float(partner_hurt_max)
    if max_high_hurt_actions is not None:
        max_high_hurt_actions = int(max_high_hurt_actions)
        if max_high_hurt_actions < 0:
            raise ValueError("type_post: max_high_hurt_actions must be >= 0")
    if high_hurt_threshold is not None:
        high_hurt_threshold = float(high_hurt_threshold)
        if high_hurt_threshold < 0:
            raise ValueError("type_post: high_hurt_threshold must be >= 0")
    if max_high_hurt_actions is not None and high_hurt_threshold is None:
        raise ValueError("type_post: high_hurt_threshold required when max_high_hurt_actions is set")

    sc_base = sc_cost.loc[pool_orig_ids, genes_use].to_numpy(dtype=float, copy=False)
    st_base = st_cost.loc[spot_ids, genes_use].to_numpy(dtype=float, copy=False)
    sc_base = sc_base * w_full
    st_base = st_base * w_full
    if spot_weight_matrix is not None:
        if spot_weight_matrix.shape[1] != len(genes_use):
            raise ValueError("type_post: spot_weight_matrix gene dimension mismatch")
        st_base = st_base * spot_weight_matrix
    if not np.isfinite(sc_base).all():
        raise ValueError("type_post: non-finite values in sc_base")
    if not np.isfinite(st_base).all():
        raise ValueError("type_post: non-finite values in st_base")

    if dm == "Spearman_correlation":
        sc_base_use = np.apply_along_axis(rankdata, 1, sc_base)
        st_base_use = np.apply_along_axis(rankdata, 1, st_base)
        base_metric = "correlation"
    elif dm == "Pearson_correlation":
        sc_base_use = sc_base
        st_base_use = st_base
        base_metric = "correlation"
    else:
        sc_base_use = sc_base
        st_base_use = st_base
        base_metric = "euclidean"

    type_to_idx = {t: i for i, t in enumerate(type_order)}
    type_idx_by_cell = np.array([type_to_idx.get(str(t), -1) for t in cell_types], dtype=int)
    missing_type_mask = type_idx_by_cell < 0
    meta["n_cells_missing_type_label"] = int(missing_type_mask.sum())

    type_score_matrix: Optional[np.ndarray] = None
    type_has_score: Optional[np.ndarray] = None
    if mode == "local_marker_swap":
        marker_source = str(marker_source or "sc_auto_top").strip().lower()
        gene_weighting = str(gene_weighting or "spot_zscore").strip().lower()
        markers_per_type = int(markers_per_type or 30)
        if marker_source not in {"sc_auto_top", "yaml_marker_list"}:
            raise ValueError(f"type_post: invalid marker_source={marker_source!r}")
        if gene_weighting not in {"spot_zscore", "none"}:
            raise ValueError(f"type_post: invalid gene_weighting={gene_weighting!r}")
        if markers_per_type <= 0:
            raise ValueError("type_post: markers_per_type must be > 0")

        gene_to_idx = {g: i for i, g in enumerate(genes_use)}
        markers_by_type: Dict[str, List[str]] = {}
        marker_missing_genes = 0
        marker_total_genes = 0

        if marker_source == "yaml_marker_list":
            if marker_list is None:
                raise ValueError("type_post: marker_list required when marker_source=yaml_marker_list")
            if isinstance(marker_list, str):
                marker_path = Path(marker_list)
                if not marker_path.exists():
                    raise ValueError(f"type_post: marker_list file not found: {marker_list}")
                try:
                    import yaml  # type: ignore
                except Exception as e:
                    raise ValueError("type_post: PyYAML required for marker_list path") from e
                marker_map = yaml.safe_load(marker_path.read_text(encoding="utf-8")) or {}
            elif isinstance(marker_list, dict):
                marker_map = marker_list
            else:
                raise ValueError("type_post: marker_list must be dict or path string")
            for t in type_order:
                genes = marker_map.get(t, marker_map.get(str(t), [])) or []
                if not isinstance(genes, (list, tuple)):
                    raise ValueError(f"type_post: marker_list[{t!r}] must be list")
                marker_total_genes += len(genes)
                genes_use_filtered = [g for g in genes if g in gene_to_idx]
                marker_missing_genes += len(genes) - len(genes_use_filtered)
                if genes_use_filtered:
                    markers_by_type[t] = list(dict.fromkeys(genes_use_filtered))
        else:
            if sc_meta is None or sc_type_col is None:
                raise ValueError("type_post: sc_meta/type_col required for marker_source=sc_auto_top")
            if sc_type_col not in sc_meta.columns:
                raise ValueError(f"type_post: sc_meta missing type_col={sc_type_col!r}")
            sc_type_series = sc_meta.reindex(sc_cost.index)[sc_type_col]
            sc_type_labels = sc_type_series.astype(str).to_numpy()
            sc_vals = sc_cost.loc[:, genes_use].to_numpy(dtype=float, copy=False)
            gene_names = list(genes_use)
            for t in type_order:
                mask = sc_type_labels == str(t)
                if not np.any(mask):
                    continue
                mean_expr = np.mean(sc_vals[mask], axis=0)
                topk = min(markers_per_type, mean_expr.shape[0])
                idx = np.argpartition(-mean_expr, topk - 1)[:topk]
                idx = idx[np.argsort(-mean_expr[idx])]
                genes_sel = [gene_names[i] for i in idx]
                if genes_sel:
                    markers_by_type[t] = genes_sel
                marker_total_genes += len(genes_sel)

        n_types_with_markers = len(markers_by_type)
        n_markers_used_total = int(sum(len(v) for v in markers_by_type.values()))
        meta["marker_source"] = marker_source
        meta["markers_per_type"] = int(markers_per_type)
        meta["gene_weighting"] = gene_weighting
        meta["n_types_with_markers"] = int(n_types_with_markers)
        meta["n_markers_total"] = int(marker_total_genes)
        meta["n_markers_used_total"] = int(n_markers_used_total)
        meta["n_markers_missing_genes"] = int(marker_missing_genes)
        meta["type_score_gene_intersection_ok"] = bool(marker_missing_genes == 0)
        meta["type_score_source"] = "marker_score"

        if n_types_with_markers <= 0 or n_markers_used_total <= 0:
            meta["skip_reason"] = "no_markers_available"
            return assignments, _assignments_to_matrix(assignments, n_spots, n_cells).tocsr(), meta

        st_vals = st_cost.loc[spot_ids, genes_use].to_numpy(dtype=float, copy=False)
        if not np.isfinite(st_vals).all():
            raise ValueError("type_post: non-finite values in st_cost for marker scoring")
        if gene_weighting == "spot_zscore":
            mean = st_vals.mean(axis=0)
            std = st_vals.std(axis=0)
            std[std == 0] = 1.0
            st_vals = (st_vals - mean) / std

        type_score_matrix = np.full((n_spots, len(type_order)), np.nan, dtype=float)
        type_has_score = np.zeros(len(type_order), dtype=bool)
        for t, idx in type_to_idx.items():
            genes = markers_by_type.get(t)
            if not genes:
                continue
            gene_idx = [gene_to_idx[g] for g in genes if g in gene_to_idx]
            if not gene_idx:
                continue
            type_score_matrix[:, idx] = st_vals[:, gene_idx].mean(axis=1)
            type_has_score[idx] = True

        finite_vals = type_score_matrix[np.isfinite(type_score_matrix)]
        if finite_vals.size == 0:
            meta["skip_reason"] = "type_score_all_nan"
            return assignments, _assignments_to_matrix(assignments, n_spots, n_cells).tocsr(), meta
        if not np.isfinite(finite_vals).all():
            raise ValueError("type_post: non-finite values in type_score_matrix")
        meta["type_score_stats"] = {
            "min": float(np.min(finite_vals)),
            "p50": float(np.percentile(finite_vals, 50)),
            "max": float(np.max(finite_vals)),
        }
    else:
        meta["type_score_source"] = "spot_type_fraction"

    current_spot = np.full(n_cells, -1, dtype=int)
    for cell_idx, spot_idx, _ in assignments:
        current_spot[cell_idx] = int(spot_idx)
    if (current_spot < 0).any():
        raise ValueError("type_post: missing assignments for some cells")

    spot_neighbors = _build_spot_neighbor_sets(st_coords, neighbor_k)
    if len(spot_neighbors) != n_spots:
        raise ValueError("type_post: neighbor list size mismatch")

    cells_by_spot: List[List[int]] = [[] for _ in range(n_spots)]
    for c, s in enumerate(current_spot):
        cells_by_spot[int(s)].append(c)
    occupancy = np.array([len(x) for x in cells_by_spot], dtype=int)
    cap = np.asarray(capacity, dtype=int)
    if cap.shape[0] != n_spots:
        raise ValueError("type_post: capacity length mismatch")

    def _scaled_tau(tau: float, best: float) -> float:
        return float(tau) * max(1.0, abs(float(best)))

    def _score_cell_spots_base(cell_idx: int, spots: List[int]) -> np.ndarray:
        vec = sc_base_use[cell_idx][None, :]
        cand = st_base_use[np.array(spots, dtype=int)]
        if base_metric == "correlation":
            scores = cdist(vec, cand, metric="correlation")[0]
        else:
            scores = cdist(vec, cand, metric="euclidean")[0]
        if not np.isfinite(scores).all():
            raise ValueError("type_post: non-finite values in base_cost")
        return scores

    moves_total = 0
    swaps_total = 0
    moves_per_round: List[int] = []
    blocked_base_margin = np.zeros(n_cells, dtype=bool)
    blocked_type_gain = np.zeros(n_cells, dtype=bool)
    blocked_base_budget = np.zeros(n_cells, dtype=bool)
    blocked_partner_hurt = np.zeros(n_cells, dtype=bool)
    blocked_swap_no_candidate = np.zeros(n_cells, dtype=bool)
    blocked_swap_base_budget = np.zeros(n_cells, dtype=bool)
    blocked_swap_conflict = np.zeros(n_cells, dtype=bool)
    pass_base_budget = np.zeros(n_cells, dtype=bool)
    pass_all_gates_pre_conflict = np.zeros(n_cells, dtype=bool)
    selected_actions: List[Dict[str, Any]] = []
    high_hurt_selected = 0

    def _record_action(
        *,
        action_type: str,
        c: int,
        s: int,
        t: int,
        gain_a: float,
        gain_b: float,
        gain_sum: float,
        partner_hurt: float,
        base_pair_sum: float,
        c2: Optional[int] = None,
    ) -> None:
        row = {
            "action_type": str(action_type),
            "cell_id": str(pool_orig_ids[c]),
            "cell_idx": int(c),
            "cell_type": str(cell_types[c]),
            "from_spot": str(spot_ids[s]),
            "to_spot": str(spot_ids[t]),
            "spot_from_idx": int(s),
            "spot_to_idx": int(t),
            "gainA": float(gain_a),
            "gainB": float(gain_b),
            "gainSum": float(gain_sum),
            "partner_hurt": float(partner_hurt),
            "base_pair_sum": float(base_pair_sum),
            "partner_cell_id": str(pool_orig_ids[c2]) if c2 is not None else None,
            "partner_cell_idx": int(c2) if c2 is not None else None,
            "partner_cell_type": str(cell_types[c2]) if c2 is not None else None,
        }
        selected_actions.append(row)

    base_spot = current_spot.copy()
    n_types = len(type_order)
    for r in range(int(rounds)):
        if mode == "local_marker_swap":
            spot_type_frac = type_score_matrix
        else:
            counts = np.zeros((n_spots, n_types), dtype=float)
            valid = ~missing_type_mask
            np.add.at(counts, (current_spot[valid], type_idx_by_cell[valid]), 1)
            row_sum = counts.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            spot_type_frac = counts / row_sum

        proposals: List[Tuple[float, int, int, List[int], List[float], Dict[int, float]]] = []
        moves_round = 0
        swaps_round = 0
        n_cells_considered = 0
        for c in range(n_cells):
            t_idx = int(type_idx_by_cell[c])
            if t_idx < 0:
                continue
            s = int(current_spot[c])
            cand = [s] + spot_neighbors[s]
            if len(cand) <= 1:
                continue
            base_scores = _score_cell_spots_base(c, cand)
            best_idx = int(np.argmin(base_scores))
            best_base = float(base_scores[best_idx])
            margin = _scaled_tau(base_margin_tau, best_base)
            tie_idx = np.where(base_scores <= best_base + margin)[0]
            if 0 not in tie_idx:
                blocked_base_margin[c] = True
                continue
            if len(tie_idx) <= 1:
                continue
            n_cells_considered += 1
            tie_spots = [cand[i] for i in tie_idx if int(i) != 0]
            if mode == "local_marker_swap":
                if type_has_score is not None and not type_has_score[t_idx]:
                    meta["n_cells_missing_type_score"] += 1
                    continue
                cand_targets = [spot for spot in cand if spot != s]
                if not cand_targets:
                    continue
                score_s = float(spot_type_frac[s, t_idx])
                if not np.isfinite(score_s):
                    meta["n_cells_missing_type_score"] += 1
                    continue
                target_scores_all = spot_type_frac[np.array(cand_targets, dtype=int), t_idx]
                if target_scores_all.size == 0:
                    continue
                valid_mask = np.isfinite(target_scores_all)
                if not np.any(valid_mask):
                    meta["n_cells_missing_type_score"] += 1
                    continue
                cand_targets = [spot for spot, ok in zip(cand_targets, valid_mask) if ok]
                target_scores_all = target_scores_all[valid_mask]
                if swap_target_topk is not None and len(cand_targets) > swap_target_topk:
                    idx = np.argpartition(-target_scores_all, swap_target_topk - 1)[:swap_target_topk]
                    idx = idx[np.argsort(-target_scores_all[idx])]
                    cand_targets = [cand_targets[i] for i in idx]
                    target_scores_all = target_scores_all[idx]
                tie_spots = cand_targets
                target_scores = target_scores_all
            else:
                if not tie_spots:
                    continue
                score_s = float(spot_type_frac[s, t_idx])
                target_scores = spot_type_frac[np.array(tie_spots, dtype=int), t_idx]
            if not tie_spots:
                continue
            gains = target_scores - score_s
            base_score_map = {int(spot): float(score) for spot, score in zip(cand, base_scores)}
            proposals.append((score_s, c, s, tie_spots, gains.tolist(), base_score_map))

        if not proposals:
            break

        max_moves = n_cells
        if max_changed_cells is not None:
            remaining_allowed = max_changed_cells - moves_total
            if remaining_allowed <= 0:
                break
            max_moves = min(max_moves, remaining_allowed)

        if selection_mode == "global_pool_greedy":
            actions_by_cell: Dict[int, List[Tuple[str, float, float, float, float, float, float, bool, int, int, int, Optional[int]]]] = {}
            for score_s, c, s, tie_spots, gains, base_score_map in proposals:
                best_move = None
                best_swap = None
                has_gain_candidate = False
                pass_base_budget_cell = False
                base_s = float(base_score_map.get(int(s), 0.0))
                for spot, gain_a in zip(tie_spots, gains):
                    if gain_a <= gain_thresh:
                        continue
                    has_gain_candidate = True
                    base_target = float(base_score_map.get(int(spot), base_s))
                    delta_base_a = base_target - base_s
                    if delta_base_a > 0:
                        continue
                    pass_base_budget_cell = True
                    if occupancy[spot] < cap[spot]:
                        select_score = float(gain_a - (lambda_base * delta_base_a))
                        if best_move is None or select_score > best_move[1]:
                            best_move = (
                                "move",
                                select_score,
                                float(gain_a),
                                0.0,
                                float(gain_a),
                                float(delta_base_a),
                                0.0,
                                False,
                                s,
                                spot,
                                c,
                                None,
                            )
                        continue
                    if not allow_swap:
                        continue
                    partner_pool = [c2 for c2 in cells_by_spot[spot] if c2 != c]
                    if not partner_pool:
                        blocked_swap_no_candidate[c] = True
                        continue
                    best_swap_for_target = None
                    for c2 in partner_pool:
                        if swap_scope == "a_and_b_tieset":
                            cand2 = [spot] + spot_neighbors[spot]
                            if s not in cand2:
                                continue
                            base_scores2 = _score_cell_spots_base(c2, cand2)
                            best_base2 = float(np.min(base_scores2))
                            margin2 = _scaled_tau(base_margin_tau, best_base2)
                            tie_idx2 = np.where(base_scores2 <= best_base2 + margin2)[0]
                            if 0 not in tie_idx2:
                                continue
                            idx_s2 = cand2.index(s)
                            if idx_s2 not in tie_idx2:
                                continue
                        base_c2 = _score_cell_spots_base(c2, [spot, s])
                        delta_base_c2 = float(base_c2[1] - base_c2[0])
                        delta_base_sum = float(delta_base_a + delta_base_c2)
                        if swap_base_mode == "both_nonpositive":
                            if delta_base_c2 > 0:
                                continue
                        elif swap_base_mode == "pair_budget":
                            if delta_base_sum > base_pair_budget_eps:
                                continue
                        pass_base_budget_cell = True
                        t_idx_b = int(type_idx_by_cell[c2])
                        if t_idx_b < 0:
                            continue
                        if type_has_score is not None and not type_has_score[t_idx_b]:
                            continue
                        score_c2_t = float(spot_type_frac[spot, t_idx_b])
                        score_c2_s = float(spot_type_frac[s, t_idx_b])
                        if not (np.isfinite(score_c2_t) and np.isfinite(score_c2_s)):
                            continue
                        gain_b = float(score_c2_s - score_c2_t)
                        gain_sum = float(gain_a + gain_b)
                        partner_hurt = max(0.0, -gain_b)
                        if partner_hurt_max is not None and partner_hurt > partner_hurt_max:
                            blocked_partner_hurt[c] = True
                            continue
                        is_high_hurt = bool(
                            high_hurt_threshold is not None and partner_hurt > float(high_hurt_threshold)
                        )
                        select_score = float(gain_a - (lambda_partner * partner_hurt) - (lambda_base * delta_base_sum))
                        if best_swap_for_target is None or select_score > best_swap_for_target[1]:
                            best_swap_for_target = (
                                "swap",
                                select_score,
                                float(gain_a),
                                float(gain_b),
                                float(gain_sum),
                                float(delta_base_sum),
                                float(partner_hurt),
                                is_high_hurt,
                                s,
                                spot,
                                c,
                                int(c2),
                            )
                    if best_swap_for_target is not None:
                        if best_swap is None or best_swap_for_target[1] > best_swap[1]:
                            best_swap = best_swap_for_target
                if not has_gain_candidate:
                    blocked_type_gain[c] = True
                elif not pass_base_budget_cell:
                    blocked_base_budget[c] = True
                else:
                    pass_base_budget[c] = True
                    pass_all_gates_pre_conflict[c] = True
                if best_move is not None:
                    actions_by_cell.setdefault(c, []).append(best_move)
                if best_swap is not None:
                    actions_by_cell.setdefault(c, []).append(best_swap)

            action_pool: List[Tuple[str, float, float, float, float, float, float, bool, int, int, int, Optional[int]]] = []
            for c, acts in actions_by_cell.items():
                acts.sort(key=lambda x: x[1], reverse=True)
                if action_topm is not None and action_topm > 0:
                    acts = acts[:action_topm]
                action_pool.extend(acts)
            action_pool.sort(key=lambda x: (-x[1], x[5]))
            meta["n_actions_generated_total"] += int(len(action_pool))
            used_cells: set[int] = set()
            for action in action_pool:
                if moves_round >= max_moves:
                    break
                (
                    action_type,
                    select_score,
                    gain_a,
                    gain_b,
                    gain_sum,
                    base_pair_sum,
                    partner_hurt,
                    is_high_hurt,
                    s,
                    t,
                    c,
                    c2,
                ) = action
                if c in used_cells:
                    if action_type == "swap":
                        blocked_swap_conflict[c] = True
                    continue
                if action_type == "swap" and c2 is not None and c2 in used_cells:
                    blocked_swap_conflict[c] = True
                    continue
                if action_type == "swap" and is_high_hurt and max_high_hurt_actions is not None:
                    if high_hurt_selected >= max_high_hurt_actions:
                        meta["n_actions_blocked_high_hurt_quota"] += 1
                        continue
                if action_type == "move":
                    if int(current_spot[c]) != s:
                        continue
                    if occupancy[t] >= cap[t]:
                        continue
                    current_spot[c] = t
                    occupancy[s] -= 1
                    occupancy[t] += 1
                    cells_by_spot[s].remove(c)
                    cells_by_spot[t].append(c)
                    moves_round += 1
                    moves_total += 1
                    used_cells.add(c)
                    _record_action(
                        action_type="move",
                        c=c,
                        s=s,
                        t=t,
                        gain_a=float(gain_a),
                        gain_b=0.0,
                        gain_sum=float(gain_a),
                        partner_hurt=0.0,
                        base_pair_sum=float(base_pair_sum),
                        c2=None,
                    )
                else:
                    if max_swaps_per_round is not None and swaps_round >= max_swaps_per_round:
                        blocked_swap_conflict[c] = True
                        continue
                    if c2 is None:
                        blocked_swap_conflict[c] = True
                        continue
                    if int(current_spot[c]) != s or int(current_spot[c2]) != t:
                        blocked_swap_conflict[c] = True
                        continue
                    if moves_round + 2 > max_moves:
                        blocked_swap_conflict[c] = True
                        continue
                    current_spot[c] = t
                    current_spot[c2] = s
                    cells_by_spot[s].remove(c)
                    cells_by_spot[t].append(c)
                    cells_by_spot[t].remove(c2)
                    cells_by_spot[s].append(c2)
                    moves_round += 2
                    moves_total += 2
                    swaps_total += 1
                    swaps_round += 1
                    used_cells.add(c)
                    used_cells.add(int(c2))
                    if is_high_hurt:
                        high_hurt_selected += 1
                    _record_action(
                        action_type="swap",
                        c=c,
                        s=s,
                        t=t,
                        gain_a=float(gain_a),
                        gain_b=float(gain_b),
                        gain_sum=float(gain_sum),
                        partner_hurt=float(partner_hurt),
                        base_pair_sum=float(base_pair_sum),
                        c2=int(c2),
                    )
        else:
            for score_s, c, s, tie_spots, gains, base_score_map in proposals:
                if moves_round >= max_moves:
                    break
                best_target = None
                best_gain = None
                base_s = float(base_score_map.get(int(s), 0.0))
                best_delta_base = None
                for spot, gain_a in zip(tie_spots, gains):
                    if gain_a <= gain_thresh:
                        continue
                    base_target = float(base_score_map.get(int(spot), base_s))
                    delta_base_a = base_target - base_s
                    if delta_base_a > 0:
                        continue
                    if best_gain is None or gain_a > best_gain:
                        best_gain = float(gain_a)
                        best_target = int(spot)
                        best_delta_base = float(delta_base_a)
                if best_target is None:
                    blocked_type_gain[c] = True
                    continue
                if occupancy[best_target] < cap[best_target]:
                    current_spot[c] = best_target
                    occupancy[s] -= 1
                    occupancy[best_target] += 1
                    cells_by_spot[s].remove(c)
                    cells_by_spot[best_target].append(c)
                    moves_round += 1
                    moves_total += 1
                    pass_base_budget[c] = True
                    pass_all_gates_pre_conflict[c] = True
                    _record_action(
                        action_type="move",
                        c=c,
                        s=s,
                        t=best_target,
                        gain_a=float(best_gain),
                        gain_b=0.0,
                        gain_sum=float(best_gain),
                        partner_hurt=0.0,
                        base_pair_sum=float(best_delta_base or 0.0),
                        c2=None,
                    )
                elif allow_swap and (max_swaps_per_round is None or swaps_round < max_swaps_per_round):
                    partner_pool = [c2 for c2 in cells_by_spot[best_target] if c2 != c]
                    if not partner_pool:
                        blocked_swap_no_candidate[c] = True
                        continue
                    for c2 in partner_pool:
                        if moves_round + 2 > max_moves:
                            break
                        t_idx_b = int(type_idx_by_cell[c2])
                        if t_idx_b < 0:
                            continue
                        if type_has_score is not None and not type_has_score[t_idx_b]:
                            continue
                        base_c2 = _score_cell_spots_base(c2, [best_target, s])
                        delta_base_c2 = float(base_c2[1] - base_c2[0])
                        delta_base_sum = float((best_delta_base or 0.0) + delta_base_c2)
                        if swap_base_mode == "both_nonpositive":
                            if delta_base_c2 > 0:
                                continue
                        elif swap_base_mode == "pair_budget":
                            if delta_base_sum > base_pair_budget_eps:
                                continue
                        score_c2_t = float(spot_type_frac[best_target, t_idx_b])
                        score_c2_s = float(spot_type_frac[s, t_idx_b])
                        if not (np.isfinite(score_c2_t) and np.isfinite(score_c2_s)):
                            continue
                        gain_b = float(score_c2_s - score_c2_t)
                        partner_hurt = max(0.0, -gain_b)
                        if partner_hurt_max is not None and partner_hurt > partner_hurt_max:
                            blocked_partner_hurt[c] = True
                            continue
                        is_high_hurt = bool(
                            high_hurt_threshold is not None and partner_hurt > float(high_hurt_threshold)
                        )
                        if is_high_hurt and max_high_hurt_actions is not None and high_hurt_selected >= max_high_hurt_actions:
                            meta["n_actions_blocked_high_hurt_quota"] += 1
                            continue
                        current_spot[c] = best_target
                        current_spot[c2] = s
                        cells_by_spot[s].remove(c)
                        cells_by_spot[best_target].append(c)
                        cells_by_spot[best_target].remove(c2)
                        cells_by_spot[s].append(c2)
                        moves_round += 2
                        moves_total += 2
                        swaps_total += 1
                        swaps_round += 1
                        pass_base_budget[c] = True
                        pass_all_gates_pre_conflict[c] = True
                        if is_high_hurt:
                            high_hurt_selected += 1
                        _record_action(
                            action_type="swap",
                            c=c,
                            s=s,
                            t=best_target,
                            gain_a=float(best_gain),
                            gain_b=float(gain_b),
                            gain_sum=float(best_gain + gain_b),
                            partner_hurt=float(partner_hurt),
                            base_pair_sum=float(delta_base_sum),
                            c2=int(c2),
                        )
                        break
                else:
                    blocked_base_budget[c] = True

        moves_per_round.append(moves_round)
        meta["rounds_executed"] = r + 1
        meta["n_cells_considered"] += int(n_cells_considered)
        if moves_round == 0:
            break
        if (occupancy > cap).any():
            current_spot = base_spot.copy()
            meta["status"] = "capacity_violation_reverted"
            meta["capacity_violation"] = True
            break

    counts = np.bincount(current_spot, minlength=n_spots)
    cap_diff = counts - cap
    meta["capacity_check"] = {
        "max_overflow": int(np.max(cap_diff)) if len(cap_diff) else 0,
        "max_underfill": int(np.max(-cap_diff)) if len(cap_diff) else 0,
        "n_spots_overflow": int(np.sum(cap_diff > 0)),
        "n_spots_underfill": int(np.sum(cap_diff < 0)),
    }

    assignments_out = [(int(c), int(s), 1.0) for c, s in enumerate(current_spot)]
    hard_mat = _assignments_to_matrix(assignments_out, n_spots, n_cells).tocsr()
    changed = int(np.sum(current_spot != base_spot))
    meta.update(
        {
            "status": "ok" if not meta["capacity_violation"] else meta["status"],
            "n_moves_total": int(moves_total),
            "n_swaps_total": int(swaps_total),
            "moves_per_round": moves_per_round,
            "n_cells_blocked_base_margin": int(np.sum(blocked_base_margin)),
            "n_cells_blocked_type_gain": int(np.sum(blocked_type_gain)),
            "n_cells_blocked_base_budget": int(np.sum(blocked_base_budget)),
            "n_cells_blocked_partner_hurt": int(np.sum(blocked_partner_hurt)),
            "n_cells_blocked_swap_no_candidate": int(np.sum(blocked_swap_no_candidate)),
            "n_cells_blocked_swap_base_budget": int(np.sum(blocked_swap_base_budget)),
            "n_cells_blocked_swap_conflict": int(np.sum(blocked_swap_conflict)),
            "n_cells_pass_base_budget": int(np.sum(pass_base_budget)),
            "n_cells_pass_all_gates_pre_conflict": int(np.sum(pass_all_gates_pre_conflict)),
            "n_actions_generated_total": int(meta["n_actions_generated_total"]),
            "n_actions_blocked_high_hurt_quota": int(meta["n_actions_blocked_high_hurt_quota"]),
            "n_high_hurt_selected": int(high_hurt_selected),
            "selected_actions": selected_actions,
            "n_changed_cells": int(changed),
            "changed_rate": float(changed / max(1, n_cells)),
        }
    )
    return assignments_out, hard_mat, meta


def _write_meta(out_dir: Path, meta: Dict[str, Any]):
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _type_prior_refine(mat: csr_matrix, cell_types: List[str], type_cols: List[str], type_prior_spot: pd.DataFrame, lambda_prior: float, eps: float = 1e-6) -> csr_matrix:
    """
    使用 per-spot type_prior 进行软约束微调：
    - 计算当前 spot×type 分布 Curr
    - 使用 type_prior_spot（行归一化）作为 Prior
    - 权重 w = (Prior+eps) / (Curr+eps) 的 lambda_prior 次方
    - 按 type 对应的列乘以 w，再按行保持原总量
    """
    if lambda_prior <= 0:
        return mat
    mat_csr = mat.tocsr()
    n_spots, n_cells = mat_csr.shape
    type_cols = list(type_cols)
    # map cell -> type index
    type_map = [type_cols.index(t) if t in type_cols else None for t in cell_types]
    type_map = np.array(type_map)
    # 当前分布
    row_sum = np.asarray(mat_csr.sum(axis=1)).ravel()
    curr = np.zeros((n_spots, len(type_cols)), dtype=float)
    for i, t in enumerate(type_cols):
        mask = type_map == i
        if mask.any():
            curr[:, i] = mat_csr[:, mask].sum(axis=1).A1
    curr_frac = curr / np.maximum(row_sum[:, None], eps)
    # 先验分布（行归一化）
    prior = type_prior_spot[type_cols]
    prior = prior.div(prior.sum(axis=1).replace(0, 1.0), axis=0)
    prior_np = prior.to_numpy(dtype=float)
    w = ((prior_np + eps) / (curr_frac + eps)) ** float(lambda_prior)
    # 应用权重：对每个非零条目，乘以对应 type 的权重
    coo = mat_csr.tocoo()
    w_lookup = w[coo.row, type_map[coo.col]]
    data_new = coo.data * w_lookup
    mat_new = coo_matrix((data_new, (coo.row, coo.col)), shape=mat_csr.shape).tocsr()
    # 行重新缩放保持总量
    new_row_sum = np.asarray(mat_new.sum(axis=1)).ravel()
    scale = row_sum / np.maximum(new_row_sum, eps)
    mat_new = mat_new.multiply(scale[:, None])
    return mat_new


def _compute_ablation_change_rate_checked(
    base_assign: List[Tuple[int, int, float]],
    ablate_assign: List[Tuple[int, int, float]],
    *,
    context: str = "prior_ablation",
) -> Tuple[float, Dict[str, Any]]:
    """
    计算 prior ablation 的 change_rate：同一 cell 在 base vs ablate 是否换 spot。
    - 强校验：禁止长度不一致/对齐失败时写“假值”
    - 以 cell_idx 为键做对齐，避免仅靠列表顺序导致误判
    返回 (change_rate, audit)
    """

    def build_map(assign: List[Tuple[int, int, float]]) -> Dict[int, int]:
        m: Dict[int, int] = {}
        dup = 0
        for cid, sid, _ in assign:
            if cid in m:
                dup += 1
            m[cid] = sid
        if dup > 0:
            raise ValueError(f"[{context}] assignments 内存在重复 cell_idx，dup={dup}")
        return m

    base_map = build_map(base_assign)
    abl_map = build_map(ablate_assign)

    base_keys = set(base_map.keys())
    abl_keys = set(abl_map.keys())
    common = base_keys & abl_keys
    only_base = base_keys - abl_keys
    only_abl = abl_keys - base_keys

    audit = {
        "context": context,
        "n_base": len(base_assign),
        "n_ablate": len(ablate_assign),
        "n_base_unique_cells": len(base_map),
        "n_ablate_unique_cells": len(abl_map),
        "n_common_cells": len(common),
        "n_only_in_base": len(only_base),
        "n_only_in_ablate": len(only_abl),
        "example_only_in_base": sorted(list(only_base))[:5],
        "example_only_in_ablate": sorted(list(only_abl))[:5],
    }

    if len(common) == 0:
        raise ValueError(f"[{context}] base 与 ablate 无共同 cell，无法计算 change_rate")
    if len(only_base) > 0 or len(only_abl) > 0:
        raise ValueError(
            f"[{context}] base/ablate cell 集合不一致：only_base={len(only_base)}, only_ablate={len(only_abl)}"
        )

    changed = 0
    for cid in common:
        if base_map[cid] != abl_map[cid]:
            changed += 1
    audit["n_changed"] = int(changed)
    change_rate = changed / max(len(common), 1)
    return float(change_rate), audit


def _harden_assignment_quota_matching(
    mat: csr_matrix,
    capacity: np.ndarray,
    cell_types: List[str],
    type_prior: Optional[pd.DataFrame],
    spot_ids: List[str],
    lambda_prior: float,
    eps: float = 1e-8,
    topk: int = 5,
    prior_candidate_topk: int = 0,
    prior_candidate_weight: float = 1.0,
    fallback_assignments: Optional[List[Tuple[int, int, float]]] = None,
    ablate_lambda: bool = False,
    min_prior_row_nonzero_ratio: float = 0.0,
) -> Tuple[List[Tuple[int, int, float]], csr_matrix, Dict[str, Any]]:
    """
    将 soft 矩阵投影回满足容量的 hard assignment：
    - 对每个 cell 取 topK 候选 + fallback 的原始 CytoSPACE 结果
    - 效用 U = log(score) + lambda_prior*log(type_prior)
    - 用带容量的延迟接受（deferred acceptance）匹配
    """
    if not sparse.isspmatrix_csr(mat):
        mat = mat.tocsr()
    cap = np.asarray(capacity, dtype=int)
    cap[cap < 0] = 0
    n_spots, n_cells = mat.shape
    mat_csc = mat.tocsc()
    fallback_map = {}
    if fallback_assignments:
        for cidx, sidx, _ in fallback_assignments:
            fallback_map[cidx] = sidx
    use_type_prior = type_prior is not None
    type_cols: List[str] = []
    type_to_col: Dict[str, int] = {}
    prior_row_nonzero = None
    prior_entropy_mean = None
    prior_entropy_max = None
    prior_np: Optional[np.ndarray] = None
    prior_topk_by_type: Optional[List[List[int]]] = None
    prior_candidate_topk = int(prior_candidate_topk or 0)
    prior_candidate_weight = float(1.0 if prior_candidate_weight is None else prior_candidate_weight)
    if use_type_prior:
        type_cols = list(type_prior.columns)
        type_prior_aligned = type_prior.reindex(index=spot_ids).reindex(columns=type_cols)
        if type_prior_aligned.isna().all(axis=None):
            raise ValueError('type_prior alignment failed: spot_id or columns mismatch')
        prior_row_nonzero = (~(type_prior_aligned.fillna(0.0) == 0).all(axis=1)).mean()
        prior_row_entropy = []
        prior_filled = type_prior_aligned.fillna(0.0)
        for _, row in prior_filled.iterrows():
            p = row.to_numpy(dtype=float)
            p_sum = p.sum()
            if p_sum <= 0:
                prior_row_entropy.append(0.0)
            else:
                p = p / p_sum
                prior_row_entropy.append(-float(np.sum(p * np.log(p + eps))))
        prior_entropy_mean = float(np.mean(prior_row_entropy)) if prior_row_entropy else 0.0
        prior_entropy_max = float(np.max(prior_row_entropy)) if prior_row_entropy else 0.0
        prior_np = type_prior_aligned.fillna(0.0).to_numpy(dtype=float, copy=False)
        type_to_col = {t: i for i, t in enumerate(type_cols)}
        if lambda_prior and prior_candidate_topk > 0:
            k = min(prior_candidate_topk, n_spots)
            prior_topk_by_type = []
            for t_idx in range(len(type_cols)):
                col = prior_np[:, t_idx]
                if k >= n_spots:
                    idx = np.argsort(-col)
                else:
                    idx = np.argpartition(-col, k - 1)[:k]
                    idx = idx[np.argsort(-col[idx])]
                prior_topk_by_type.append(idx.tolist())
        if prior_row_nonzero < min_prior_row_nonzero_ratio:
            raise ValueError(
                f'type_prior nonzero-row ratio too low: {prior_row_nonzero:.3f} < {min_prior_row_nonzero_ratio}'
            )
    candidates: List[List[Tuple[int, float]]] = []
    local_change = 0
    for c in range(n_cells):
        col = mat_csc.getcol(c)
        rows = col.indices
        data = col.data
        if len(rows) > 0:
            order = np.argsort(data)[::-1]
            if topk and topk > 0:
                order = order[:topk]
            rows = rows[order]
            data = data[order]
        cand_dict: Dict[int, float] = {}
        for r, v in zip(rows, data):
            cand_dict[r] = max(cand_dict.get(r, 0.0), v)
        if c in fallback_map:
            cand_dict.setdefault(fallback_map[c], 0.0)
        cand_list: List[Tuple[int, float]] = []
        type_col = type_to_col.get(cell_types[c], None) if use_type_prior else None
        best_score_only = None
        if use_type_prior and lambda_prior and prior_topk_by_type is not None and type_col is not None:
            for r in prior_topk_by_type[type_col]:
                if r not in cand_dict:
                    cand_dict[r] = prior_candidate_weight * float(prior_np[r, type_col])
        for r, v in cand_dict.items():
            prior_val = prior_np[r, type_col] if (use_type_prior and type_col is not None and prior_np is not None and r < prior_np.shape[0]) else 0.0
            u = np.log(v + eps)
            if use_type_prior and lambda_prior:
                u += float(lambda_prior) * np.log(prior_val + eps)
            cand_list.append((r, u))
            if best_score_only is None or v > best_score_only[1]:
                best_score_only = (r, v)
        # 候选为空时，尝试 fallback spot
        if not cand_list and c in fallback_map:
            r = fallback_map[c]
            prior_val = prior_np[r, type_col] if (use_type_prior and type_col is not None and prior_np is not None and r < prior_np.shape[0]) else 0.0
            u = float(lambda_prior) * np.log(prior_val + eps) if (use_type_prior and lambda_prior) else 0.0
            cand_list.append((r, u))
        cand_list.sort(key=lambda x: x[1], reverse=True)
        candidates.append(cand_list)
        if best_score_only and cand_list:
            if best_score_only[0] != cand_list[0][0]:
                local_change += 1

    assignment = [-1] * n_cells
    spot_pool: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n_spots)}
    next_pos = [0] * n_cells
    free = [i for i in range(n_cells) if candidates[i]]
    while free:
        c = free.pop()
        cand = candidates[c]
        if next_pos[c] >= len(cand):
            continue
        s, util = cand[next_pos[c]]
        next_pos[c] += 1
        spot_pool[s].append((c, util))
        if len(spot_pool[s]) > cap[s]:
            spot_pool[s].sort(key=lambda x: x[1], reverse=True)
            dropped = spot_pool[s][cap[s]:]
            spot_pool[s] = spot_pool[s][: cap[s]]
            for dc, _ in dropped:
                free.append(dc)

    for s, pool in spot_pool.items():
        for c, util in pool:
            assignment[c] = s

    data = []
    rows = []
    cols = []
    for c, s in enumerate(assignment):
        if s >= 0:
            rows.append(s)
            cols.append(c)
            data.append(1.0)
    hard_mat = sparse.csr_matrix((data, (rows, cols)), shape=mat.shape)
    assigned_per_spot = np.asarray(hard_mat.sum(axis=1)).ravel()

    fallback_used = 0
    if fallback_map:
        for c, a in enumerate(assignment):
            if a == -1 and c in fallback_map:
                s = fallback_map[c]
                if assigned_per_spot[s] < cap[s]:
                    assignment[c] = s
                    assigned_per_spot[s] += 1
                    fallback_used += 1
    # 重新生成 hard 矩阵，保证包含 fallback
    data = []
    rows = []
    cols = []
    for c, s in enumerate(assignment):
        if s >= 0:
            rows.append(s)
            cols.append(c)
            data.append(1.0)
    hard_mat = sparse.csr_matrix((data, (rows, cols)), shape=mat.shape)
    assigned_per_spot = np.asarray(hard_mat.sum(axis=1)).ravel()
    assignment_arr = np.asarray(assignment, dtype=int)
    n_unassigned_before_repair = int((assignment_arr < 0).sum())

    # 修复：在 topK 偏好过窄时，延迟接受可能留下未分配 cell；用“按剩余容量+type_prior”填满剩余槽位
    repair_used = 0
    if n_unassigned_before_repair > 0:
        rem = cap - assigned_per_spot
        unassigned_cells = np.where(assignment_arr < 0)[0]
        for c in unassigned_cells:
            avail = np.where(rem > 0)[0]
            if avail.size == 0:
                break
            type_col = type_to_col.get(cell_types[c], None) if use_type_prior else None
            if not use_type_prior or type_col is None or prior_np is None:
                rem_vals = rem[avail]
                s = int(avail[int(np.argmax(rem_vals))])
            else:
                col = prior_np[avail, type_col]
                s = int(avail[int(np.argmax(col))])
            assignment[c] = s
            rem[s] -= 1
            assigned_per_spot[s] += 1
            repair_used += 1
        # 重建 hard 矩阵（包含 repair）
        data = []
        rows = []
        cols = []
        for c, s in enumerate(assignment):
            if s >= 0:
                rows.append(s)
                cols.append(c)
                data.append(1.0)
        hard_mat = sparse.csr_matrix((data, (rows, cols)), shape=mat.shape)
        assigned_per_spot = np.asarray(hard_mat.sum(axis=1)).ravel()
        assignment_arr = np.asarray(assignment, dtype=int)

    n_unassigned = int((assignment_arr < 0).sum())
    cap_diff = assigned_per_spot - cap
    overflow = np.maximum(cap_diff, 0)
    underfill = np.maximum(-cap_diff, 0)
    meta = {
        "harden_method": "quota_matching",
        "harden_topk": topk,
        "harden_fallback_n_cells": fallback_used,
        "harden_repair_n_cells": repair_used,
        "n_unassigned_before_repair": n_unassigned_before_repair,
        "capacity_check": {
            "max_overflow": float(overflow.max()) if len(overflow) else 0.0,
            "max_underfill": float(underfill.max()) if len(underfill) else 0.0,
            "n_spots_overflow": int((overflow > 0).sum()) if len(overflow) else 0,
            "n_spots_underfill": int((underfill > 0).sum()) if len(underfill) else 0,
        },
        "n_unassigned_cells": n_unassigned,
        "n_cells_total": int(n_cells),
        "type_prior_row_index_type": str(type_prior.index.__class__.__name__) if use_type_prior else None,
        "type_prior_row_nonzero_ratio": float(prior_row_nonzero) if prior_row_nonzero is not None else None,
        "type_prior_row_entropy_mean": prior_entropy_mean if use_type_prior else None,
        "type_prior_row_entropy_max": prior_entropy_max if use_type_prior else None,
        "prior_local_top1_change_rate": float(local_change / n_cells) if n_cells > 0 else 0.0,
        "prior_candidate_topk": int(prior_candidate_topk),
        "prior_candidate_weight": float(prior_candidate_weight),
    }
    if n_unassigned > 0:
        meta["status"] = "failed_unassigned"
        meta["error_msg"] = f"harden left {n_unassigned}/{n_cells} cells unassigned"
        raise RuntimeError(meta["error_msg"])
    if overflow.max() > 0:
        meta["status"] = "failed_overflow"
        meta["error_msg"] = f"harden overflow detected max_overflow={overflow.max()}"
        raise RuntimeError(meta["error_msg"])
    if ablate_lambda:
        meta["ablation_mode"] = "lambda0"
    assignments_hard = [(c, s, 1.0) for c, s in enumerate(assignment) if s >= 0]
    return assignments_hard, hard_mat, meta


def _write_global_fraction(cell_types: List[str], type_order: List[str], out_path: Path):
    ser = pd.Series(cell_types, name="cell_type").value_counts(normalize=True)
    frac_row = ser.reindex(type_order).fillna(0).to_frame().T
    frac_row.index = ["global"]
    frac_row.to_csv(out_path)
    return out_path


def _ensure_type_prior_columns(type_prior: pd.DataFrame, type_cols: List[str]) -> pd.DataFrame:
    for col in type_cols:
        if col not in type_prior.columns:
            type_prior[col] = 0.0
    return type_prior[type_cols]


def _build_outputs(
    assignments: List[Tuple[int, int, float]],
    sc_index: List[str],
    st_index: List[str],
    cell_types: List[str],
    out_dir: Path,
    mode: str,
    type_order: Optional[List[str]] = None,
    hard_matrix: Optional[sparse.csr_matrix] = None,
    unique_cell_ids: Optional[List[str]] = None,
):
    if unique_cell_ids is not None and len(unique_cell_ids) != len(sc_index):
        raise ValueError(
            f"[outputs] unique_cell_ids length {len(unique_cell_ids)} != sc_index length {len(sc_index)}"
        )
    rows = []
    for cell_idx, spot_idx, score in assignments:
        r = {
            "cell_id": sc_index[cell_idx],
            "spot_id": st_index[spot_idx],
            "type": cell_types[cell_idx],
            "backend": "cytospace",
            "mode": mode,
            "assign_score": score,
        }
        if unique_cell_ids is not None:
            r["unique_cid"] = unique_cell_ids[cell_idx]
        rows.append(r)
    ca = pd.DataFrame(rows)
    ca.to_csv(out_dir / f"cell_assignment_{mode}.csv", index=False)

    if hard_matrix is not None:
        mat = hard_matrix if sparse.isspmatrix_csr(hard_matrix) else hard_matrix.tocsr()
    else:
        data = np.ones(len(assignments), dtype=float)
        row_ind = [spot_idx for _, spot_idx, _ in assignments]
        col_ind = [cell_idx for cell_idx, _, _ in assignments]
        mat = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(st_index), len(sc_index)))
    # API 约定：cell×spot（行=cell，列=spot）
    sparse.save_npz(out_dir / f"cell_spot_matrix_{mode}.npz", mat.T.tocsr())

    if type_order is None:
        type_order = sorted(list(dict.fromkeys(cell_types)))
    df = pd.DataFrame(0.0, index=st_index, columns=type_order)
    mat_csr = mat if sparse.isspmatrix_csr(mat) else mat.tocsr()
    cell_type_arr = np.array([cell_types[i] for i in range(len(sc_index))])
    for t in type_order:
        mask = cell_type_arr == t
        if mask.any():
            df[t] = mat_csr[:, mask].sum(axis=1).A1
    df = df.div(df.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    df.index.name = "spot_id"
    df.reset_index().to_csv(out_dir / f"spot_type_fraction_{mode}.csv", index=False)


class CytoSPACEBackend(MappingBackend):
    def __init__(self):
        super().__init__(name="cytospace")

    def run_baseline(self, stage1_dir: Path, out_dir: Path, config: Dict[str, Any]) -> None:
        t0 = time.time()
        _ensure_dir(out_dir)
        status = "success"
        error_msg = None
        type_order = None
        config_validation = None
        capacity_audit = None
        config_effective_subset = None
        try:
            config, config_validation = _validate_and_resolve_config(config, context="baseline")
            config_effective_subset = {k: config.get(k) for k in _CONFIG_EFFECTIVE_KEYS if k in config}

            sc_expr, st_expr, sc_meta, st_coords, type_col = _load_stage1(stage1_dir)
            _assert_unique_index(sc_expr, "sc_expr")
            _assert_unique_index(st_expr, "st_expr")
            _assert_unique_index(sc_meta, "sc_meta")
            _assert_unique_index(st_coords, "st_coords")

            mode = str(config.get("mode") or "")
            run_id = config.get("run_id")
            variant = config.get("variant")
            if mode not in ("baseline", "plus"):
                raise ValueError(f"[baseline] invalid mode={mode!r}; expected 'baseline' or 'plus'")
            if mode != "baseline":
                raise ValueError(f"[baseline] invalid mode={mode!r}; expected 'baseline'")
            if run_id is None:
                raise ValueError("[baseline] missing config.run_id")
            run_id = str(run_id)
            if not run_id.startswith("baseline"):
                raise ValueError(f"[baseline] invalid run_id={run_id!r}; expected startswith 'baseline'")
            if out_dir.name != run_id:
                raise ValueError(f"[baseline] out_dir.name={out_dir.name!r} != run_id={run_id!r}")

            spot_ids = list(st_expr.index)
            capacity_raw, cps_source, norm_val = _compute_cells_per_spot(st_coords, config)
            st_coords, capacity_raw, _, spot_alignment_audit = _align_spot_inputs(
                spot_ids=spot_ids,
                st_coords=st_coords,
                capacity=capacity_raw,
                type_prior_raw=None,
                context="baseline",
            )
            capacity, cap_audit = _resolve_cells_per_spot(capacity_raw, config, context="baseline")
            capacity_audit = {
                **cap_audit,
                "source_config": config.get("cells_per_spot_source"),
                "source_resolved": cps_source,
                "umi_to_cell_norm": norm_val,
                "default_cells_per_spot": config.get("default_cells_per_spot"),
            }
            genes_use = sorted(set(sc_expr.columns) & set(st_expr.columns))
            cell_types = sc_meta["type_col"].tolist()
            type_order = sorted(list(dict.fromkeys(cell_types)))

            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                os.environ.setdefault("PYTHON", sys.executable)
                os.environ.setdefault("PYTHON3", sys.executable)
                ct_frac_path = _write_global_fraction(cell_types, type_order, tmp_dir / "ct_fraction.csv")
                sc_path, ct_path, st_path, coord_path, cps_path = _write_cytospace_inputs(
                    sc_expr,
                    sc_meta,
                    st_expr,
                    st_coords,
                    cell_types,
                    genes_use,
                    capacity,
                    spot_ids,
                    tmp_dir,
                    prefix="base_",
                )
                out_tmp = tmp_dir / "cyto_out"
                out_tmp.mkdir(exist_ok=True)
                solver_method = config.get("solver_method", "lap_CSPR")
                main_cytospace(
                    scRNA_path=str(sc_path),
                    cell_type_path=str(ct_path),
                    n_cells_per_spot_path=str(cps_path),
                    st_cell_type_path=None,
                    cell_type_fraction_estimation_path=str(ct_frac_path),
                    spaceranger_path=None,
                    st_path=str(st_path),
                    coordinates_path=str(coord_path),
                    output_folder=str(out_tmp),
                    output_prefix="",
                    mean_cell_numbers=int(max(1, np.mean(capacity))),
                    downsample_off=True,
                    scRNA_max_transcripts_per_cell=1500,
                    solver_method=solver_method,
                    distance_metric="Pearson_correlation",
                    sampling_method="duplicates",
                    single_cell=False,
                    number_of_selected_spots=10000,
                    sampling_sub_spots=False,
                    number_of_selected_sub_spots=10000,
                    number_of_processors=1,
                    seed=config.get("seed", 1),
                    plot_off=True,
                    geometry="honeycomb",
                    max_num_cells_plot=50000,
                    num_column=3,
                )
                assigned_path = out_tmp / "assigned_locations.csv"
                if not assigned_path.exists():
                    raise FileNotFoundError("CytoSPACE 输出缺少 assigned_locations.csv")
                assigned = pd.read_csv(assigned_path)
                cell_ids, pool_types, pool_orig_ids, assignments = _parse_assigned_locations(
                    assigned, list(st_expr.index), require_original_cid=True
                )
                try:
                    assigned[["UniqueCID", "OriginalCID", "SpotID", "CellType"]].to_csv(
                        out_dir / "cell_pool_map_baseline.csv", index=False
                    )
                except Exception:
                    pass

                if not pool_orig_ids:
                    raise ValueError("[baseline] assigned_locations 缺少 OriginalCID，无法与 Stage1/SimGen Query 真值对齐")
                mat = _assignments_to_matrix(assignments, len(st_expr.index), len(cell_ids)).tocsr()
                _build_outputs(
                    assignments,
                    [str(x) for x in pool_orig_ids],
                    list(st_expr.index),
                    pool_types,
                    out_dir,
                    mode=mode,
                    type_order=type_order,
                    hard_matrix=mat,
                    unique_cell_ids=[str(x) for x in cell_ids],
                )
        except Exception as e:
            import traceback
            status = "failed"
            error_msg = f"{e}\n{traceback.format_exc()}"
            for fname in ["cell_assignment_baseline.csv", "cell_spot_matrix_baseline.npz", "spot_type_fraction_baseline.csv"]:
                try:
                    (out_dir / fname).unlink()
                except Exception:
                    pass
        meta = {
            "backend": "cytospace",
            "mode": config.get("mode"),
            "run_id": config.get("run_id"),
            "variant": config.get("variant"),
            "out_dir_name": out_dir.name,
            "sample": config.get("sample"),
            "seed": config.get("seed"),
            **_module_fingerprint(),
            "runner_file": config.get("runner_file"),
            "runner_sha1": (str(config.get("runner_sha1")).lower() if config.get("runner_sha1") else None),
            "svg_refine_lambda": None,
            "feature_set": "all_genes",
            "type_set": "orig_type",
            "weight_field": None,
            "stage1_dir": str(stage1_dir),
            "stage2_dir": None,
            "stage3_dir": None,
            "config_id": config.get("config_id"),
            "project_config_path": config.get("project_config_path"),
            "dataset_config_path": config.get("dataset_config_path"),
            "config_validation": locals().get("config_validation", None),
            "config_effective_subset": locals().get("config_effective_subset", None),
            "capacity_audit": locals().get("capacity_audit", None),
            "cell_id_space": "original_cid",
            "cell_instance_id_column": "unique_cid",
            "cell_assignment_sha1_pre": locals().get("cell_assignment_sha1_pre"),
            "cell_assignment_sha1_post": locals().get("cell_assignment_sha1_post"),
            "cell_assignment_sha1_post_file": locals().get("cell_assignment_sha1_post_file"),
            "selected_actions_path": str(out_dir / "selected_actions.csv") if (out_dir / "selected_actions.csv").exists() else None,
            "selected_type_actions_path": str(out_dir / "selected_type_actions.csv")
            if (out_dir / "selected_type_actions.csv").exists()
            else None,
            "cells_per_spot_source": locals().get("cps_source", None),
            "umi_to_cell_norm": locals().get("norm_val", None),
            "has_capacity_constraint": status == "success",
            "cost_cfg": {"solver_method": config.get("solver_method", "lap_CSPR")},
            "refine_cfg": {},
            "n_spots_stage1": len(st_expr.index),
            "n_spots_coords": len(st_coords.index),
            "n_spots_intersection": len(set(st_expr.index) & set(st_coords.index)),
            "n_cells_sc_expr": len(sc_expr.index),
            "n_cells_sc_meta": len(sc_meta.index),
            "n_cells_intersection": len(set(sc_expr.index) & set(sc_meta.index)),
            "n_cells_pool": len(locals().get("cell_ids", [])) if status == "success" else None,
            "type_columns": type_order,
            "spot_alignment_audit": locals().get("spot_alignment_audit", None),
            "runtime_sec": time.time() - t0,
            "status": status,
            "error_msg": error_msg,
            "resolved_mapping_config": config,
        }
        _write_meta(out_dir, meta)

    def run_plus(
        self,
        stage1_dir: Path,
        stage2_dir: Path,
        stage3_dir: Path,
        out_dir: Path,
        config: Dict[str, Any],
    ) -> None:
        t0 = time.time()
        _ensure_dir(out_dir)
        status = "success"
        error_msg = None
        config_validation = None
        capacity_audit = None
        config_effective_subset = None
        try:
            config, config_validation = _validate_and_resolve_config(config, context="plus")
            config_effective_subset = {k: config.get(k) for k in _CONFIG_EFFECTIVE_KEYS if k in config}
            type_posterior_enabled_cfg = bool(config.get("type_posterior_enabled", True))
            type_posterior_enabled = bool(type_posterior_enabled_cfg)
            type_posterior_skipped_reason = None
            if not type_posterior_enabled_cfg:
                type_posterior_skipped_reason = "disabled_by_config"
            type_post_mode = config.get("type_post_mode")
            if type_post_mode in (None, ""):
                type_post_mode = None
            else:
                type_post_mode = str(type_post_mode).strip().lower()
                if type_post_mode in ("off", "none"):
                    type_post_mode = None
            type_post_enabled = bool(type_posterior_enabled_cfg) and type_post_mode is not None
            type_prior_enabled = bool(type_posterior_enabled_cfg) and not type_post_enabled
            type_post_skipped_reason = None
            if not type_post_enabled:
                type_post_skipped_reason = "disabled_by_config" if not type_posterior_enabled_cfg else "type_post_mode=off"
            svg_post_enabled = bool(config.get("svg_post_enabled", False))
            svg_post_meta: Dict[str, Any] = {
                "status": "skipped",
                "skip_reason": "disabled_by_config" if not svg_post_enabled else None,
            }
            svg_post_used = False
            type_post_meta: Dict[str, Any] = {"status": "skipped", "skip_reason": type_post_skipped_reason}
            type_post_used = False
            type_post_effective_subset = None
            cost_mode_effective = None
            spot_weight_mode_effective = None
            cost_matrix_sha1 = None
            cost_matrix_sha1_no_spot_weight = None
            part1_knobs_no_effect = False
            type_prior = None
            type_prior_raw = None
            type_prior_norm = None
            missing_type_cols = []
            n_cells_missing_type = 0
            ratio_cells_missing_type = 0.0
            prior_intersection = None
            missing_prior_examples = []

            sc_expr, st_expr, sc_meta, st_coords, type_col = _load_stage1(stage1_dir)
            _assert_unique_index(sc_expr, "sc_expr")
            _assert_unique_index(st_expr, "st_expr")
            _assert_unique_index(sc_meta, "sc_meta")
            _assert_unique_index(st_coords, "st_coords")

            mode = str(config.get("mode") or "")
            run_id = config.get("run_id")
            variant = config.get("variant")
            if mode not in ("baseline", "plus"):
                raise ValueError(f"[plus] invalid mode={mode!r}; expected 'baseline' or 'plus'")
            if mode != "plus":
                raise ValueError(f"[plus] invalid mode={mode!r}; expected 'plus'")
            if run_id is None:
                raise ValueError("[plus] missing config.run_id")
            run_id = str(run_id)
            if not run_id.startswith("plus"):
                raise ValueError(f"[plus] invalid run_id={run_id!r}; expected startswith 'plus'")
            if out_dir.name != run_id:
                raise ValueError(f"[plus] out_dir.name={out_dir.name!r} != run_id={run_id!r}")

            spot_ids = list(st_expr.index)
            plugin_genes, weight_map, weight_field, gene_weights = _load_stage2(stage2_dir)
            if type_prior_enabled:
                relabel, type_prior = _load_stage3(stage3_dir)
            else:
                relabel = _load_stage3_relabel(stage3_dir)
            stage2_gene_weights_sha1 = _sha1_file(stage2_dir / "gene_weights.csv")
            plugin_genes_sha1 = _sha1_file(stage2_dir / "plugin_genes.txt")
            if len(plugin_genes) == 0:
                raise ValueError("plugin_genes 为空")

            relabel_map = dict(zip(relabel["cell_id"], relabel["plugin_type"]))
            eps = float(config.get("eps", 1e-8) or 1e-8)
            type_cols = sorted(list(dict.fromkeys(relabel_map.get(cid, "Unknown_sc_only") for cid in sc_expr.index)))
            cell_types = [relabel_map.get(cid, "Unknown_sc_only") for cid in sc_expr.index]
            if type_prior is not None:
                prior_cols_raw = set(type_prior.columns)
                sc_types = set(type_cols)
                missing_type_cols = sorted(sc_types - prior_cols_raw)
                n_cells_missing_type = sum(1 for t in cell_types if t in missing_type_cols)
                ratio_cells_missing_type = n_cells_missing_type / max(1, len(cell_types))
                if ratio_cells_missing_type > float(config.get("max_cells_missing_type_prior_ratio", 0.0)):
                    raise ValueError(f"{ratio_cells_missing_type:.3f} 比例的细胞类型缺失在 type_prior 列中，超过阈值")

            
            genes_use = sorted(set(sc_expr.columns) & set(st_expr.columns))
            if not genes_use:
                raise ValueError("sc/st ??????")
            gene_to_idx = {g: i for i, g in enumerate(genes_use)}
            plugin_n = max(1, len(plugin_genes))
            plugin_in_sc = [g for g in plugin_genes if g in sc_expr.columns]
            plugin_in_st = [g for g in plugin_genes if g in st_expr.columns]
            sc_cov = len(plugin_in_sc) / plugin_n
            st_cov = len(plugin_in_st) / plugin_n
            plugin_overlap_genes = [g for g in plugin_genes if g in gene_to_idx]
            plugin_overlap_ratio = len(plugin_overlap_genes) / plugin_n

            truth_pass_map = None
            if "truth_pass" in gene_weights.columns:
                truth_pass_map = dict(zip(gene_weights["gene"], gene_weights["truth_pass"]))
            spot_weight_candidate_mask = None
            if config.get("spot_weight_truth_filter", False) and truth_pass_map is not None:
                spot_weight_candidate_mask = np.array(
                    [bool(truth_pass_map.get(g, 0)) for g in genes_use],
                    dtype=bool,
                )

            w_full = np.ones(len(genes_use), dtype=float)
            weighted_genes = []
            for g in plugin_overlap_genes:
                idx = gene_to_idx[g]
                w_val = float(weight_map.get(g, 1.0))
                if not np.isfinite(w_val):
                    w_val = 1.0
                w_full[idx] = w_val
                if abs(w_val - 1.0) > 1e-12:
                    weighted_genes.append(g)

            w_min = float(np.min(w_full))
            w_p50 = float(np.percentile(w_full, 50))
            w_p95 = float(np.percentile(w_full, 95))
            w_max = float(np.max(w_full))
            weighted_ratio = len(weighted_genes) / max(len(genes_use), 1)

            svg_fallback_reasons = []
            if plugin_overlap_ratio < float(config.get("min_gene_overlap_ratio", 0.0)):
                svg_fallback_reasons.append(f"plugin_overlap_low:{plugin_overlap_ratio:.3f}")
            if weighted_ratio > 0.2:
                svg_fallback_reasons.append(f"weighted_ratio>0.2:{weighted_ratio:.3f}")
            if abs(w_p50 - 1.0) > 1e-8:
                svg_fallback_reasons.append(f"p50!=1:{w_p50:.6f}")
            if w_min < 1.0 - 1e-8:
                svg_fallback_reasons.append(f"weight_below_1:{w_min:.6f}")

            svg_fallback = len(svg_fallback_reasons) > 0
            w_stats_raw = {"min": w_min, "p50": w_p50, "p95": w_p95, "max": w_max}
            if svg_fallback:
                w_full = np.ones(len(genes_use), dtype=float)
                weighted_genes_applied = []
            else:
                weighted_genes_applied = weighted_genes
            weighted_ratio_applied = len(weighted_genes_applied) / max(len(genes_use), 1)
            w_stats = {
                "min": float(np.min(w_full)),
                "p50": float(np.percentile(w_full, 50)),
                "p95": float(np.percentile(w_full, 95)),
                "max": float(np.max(w_full)),
            }

            capacity_raw, cps_source, norm_val = _compute_cells_per_spot(st_coords, config)
            st_coords, capacity_raw, type_prior_raw, spot_alignment_audit = _align_spot_inputs(
                spot_ids=spot_ids,
                st_coords=st_coords,
                capacity=capacity_raw,
                type_prior_raw=type_prior,
                context="plus_svg_type",
            )
            capacity, cap_audit = _resolve_cells_per_spot(capacity_raw, config, context="plus")
            capacity_audit = {
                **cap_audit,
                "source_config": config.get("cells_per_spot_source"),
                "source_resolved": cps_source,
                "umi_to_cell_norm": norm_val,
                "default_cells_per_spot": config.get("default_cells_per_spot"),
            }
            if type_prior_raw is None:
                if type_prior_enabled:
                    raise ValueError("type_prior_raw 缺失（align_spot_inputs 返回 None）")
            if type_prior_raw is not None:
                type_prior_raw = _ensure_type_prior_columns(type_prior_raw, type_cols)
                prior_intersection = len(set(spot_ids) & set(type_prior_raw.index))
                missing_prior_spots = list(set(spot_ids) - set(type_prior_raw.index))
                missing_prior_examples = missing_prior_spots[:5]
            cost_expr_guard_reason = None
            norm_mode = str(config.get("cost_expr_norm") or "none").lower()
            sample_id = str(config.get("sample") or "")
            is_sim = bool((ROOT / "data" / "sim" / sample_id).exists())
            norm_cfg = config
            if is_sim and norm_mode == "st_zscore":
                norm_cfg = dict(config)
                norm_cfg["cost_expr_norm"] = "none"
                cost_expr_guard_reason = "sim_zscore_forbidden"
            sc_cost, st_cost, cost_expr_audit = _apply_cost_expr_norm(
                sc_expr,
                st_expr,
                genes_use,
                norm_cfg,
                eps=float(norm_cfg.get("cost_expr_norm_eps") or 1e-8),
            )
            if cost_expr_guard_reason:
                cost_expr_audit["guard_reason"] = cost_expr_guard_reason
            spot_weight_matrix, spot_weight_stats = _compute_spot_weight_matrix(
                st_cost,
                st_coords,
                genes_use,
                config,
                eps=float(config.get("cost_expr_norm_eps") or 1e-8),
                candidate_mask=spot_weight_candidate_mask,
            )
            spot_weight_mode_effective = (spot_weight_stats or {}).get("mode") or str(config.get("spot_weight_mode") or "none")
            if (spot_weight_stats or {}).get("status") == "disabled":
                spot_weight_mode_effective = "none"
            cost_expr_mode = str((cost_expr_audit or {}).get("mode") or "none").lower()
            cost_scale_beta = float(config.get("cost_scale_beta") or 0.0)
            spot_weight_effective_ratio = float((spot_weight_stats or {}).get("weighted_ratio") or 0.0) if spot_weight_matrix is not None else 0.0
            if (
                cost_expr_mode == "none"
                and cost_scale_beta <= 0.0
                and float(weighted_ratio_applied) == 0.0
                and spot_weight_effective_ratio == 0.0
            ):
                cost_mode_effective = "baseline_compatible"
            else:
                cost_mode_effective = "plus_weighted"
            refine_lambda = float(config.get("svg_refine_lambda", 0.0) or 0.0)
            refine_k = int(config.get("svg_refine_k", 8))
            distance_metric = config.get("distance_metric", "Pearson_correlation")
            lambda_prior_cfg = float(config.get("lambda_prior", 1.0))
            lambda_prior = 0.0 if not type_prior_enabled else lambda_prior_cfg
            effective_lambda_refine = lambda_prior if (type_prior_enabled and config.get("type_prior_apply_refine", True)) else 0.0
            effective_lambda_harden = lambda_prior if (type_prior_enabled and config.get("type_prior_apply_harden", True)) else 0.0
            type_prior_eps = eps

            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                os.environ.setdefault("PYTHON", sys.executable)
                os.environ.setdefault("PYTHON3", sys.executable)
                solver_method = config.get("solver_method", "lap_CSPR")
                sc_weighted = sc_cost.copy()
                st_weighted = st_cost.copy()
                sc_vals = sc_cost[genes_use].to_numpy(dtype=float) * w_full
                st_vals_base = st_cost[genes_use].to_numpy(dtype=float) * w_full
                st_vals = st_vals_base
                if spot_weight_matrix is not None:
                    st_vals = st_vals_base * spot_weight_matrix
                cost_matrix_sha1 = _sha1_cost_inputs(
                    sc_vals,
                    st_vals,
                    np.asarray(capacity, dtype=int),
                    distance_metric=distance_metric,
                    solver_method=solver_method,
                )
                cost_matrix_sha1_no_spot_weight = _sha1_cost_inputs(
                    sc_vals,
                    st_vals_base,
                    np.asarray(capacity, dtype=int),
                    distance_metric=distance_metric,
                    solver_method=solver_method,
                )
                spot_weight_mode_raw = str(config.get("spot_weight_mode") or "none").lower()
                spot_weight_kappa = float(config.get("spot_weight_kappa") or 0.0)
                spot_weight_topk = int(config.get("spot_weight_topk") or 0)
                knobs_active = spot_weight_mode_raw not in ("none", "") and spot_weight_kappa > 0.0 and spot_weight_topk > 0
                if knobs_active and cost_matrix_sha1 == cost_matrix_sha1_no_spot_weight:
                    part1_knobs_no_effect = True
                if bool(config.get("strict_config", True)) and part1_knobs_no_effect:
                    raise ValueError(
                        "[Part1Guard] spot_weight knobs have no effect (weights degenerate). "
                        "Refuse to continue under strict_config. "
                        f"sample={sample_id} config_id={config.get('config_id')}"
                    )
                sc_weighted.loc[:, genes_use] = sc_vals
                st_weighted.loc[:, genes_use] = st_vals
                sc_path, ct_path, st_path, coord_path, cps_path = _write_cytospace_inputs(
                    sc_weighted,
                    sc_meta,
                    st_weighted,
                    st_coords,
                    cell_types,
                    genes_use,
                    capacity,
                    spot_ids,
                    tmp_dir,
                    prefix="plus_",
                )
                # NOTE (Stage5 truth alignment):
                # CytoSPACE samples a "cell pool" by the provided cell-type fractions. If we derive those
                # fractions from ST priors, it can over-demand rare types and trigger duplicate sampling,
                # breaking one-to-one Query cell_id alignment with SimGen truth. We therefore sample by the
                # Query sc composition (plugin_type) and apply type_prior only in our own refine/harden.
                st_ct_frac_path = _write_global_fraction(cell_types, type_cols, tmp_dir / "ct_fraction.csv")
                # spot×type -> 全局 1×type（喂 CytoSPACE）
                type_prior_norm = None
                if type_prior_raw is not None:
                    row_sum = type_prior_raw.sum(axis=1).replace(0, eps)
                    type_prior_norm = type_prior_raw.div(row_sum, axis=0)

                out_tmp = tmp_dir / "cyto_out"
                out_tmp.mkdir(exist_ok=True)
                main_cytospace(
                    scRNA_path=str(sc_path),
                    cell_type_path=str(ct_path),
                    n_cells_per_spot_path=str(cps_path),
                    st_cell_type_path=None,
                    cell_type_fraction_estimation_path=str(st_ct_frac_path),
                    spaceranger_path=None,
                    st_path=str(st_path),
                    coordinates_path=str(coord_path),
                    output_folder=str(out_tmp),
                    output_prefix="",
                    mean_cell_numbers=int(max(1, np.mean(capacity))),
                    downsample_off=True,
                    scRNA_max_transcripts_per_cell=1500,
                    solver_method=solver_method,
                    distance_metric=distance_metric,
                    sampling_method="duplicates",
                    single_cell=False,
                    number_of_selected_spots=10000,
                    sampling_sub_spots=False,
                    number_of_selected_sub_spots=10000,
                    number_of_processors=1,
                    seed=config.get("seed", 1),
                    plot_off=True,
                    geometry="honeycomb",
                    max_num_cells_plot=50000,
                    num_column=3,
                )
                assigned_path = out_tmp / "assigned_locations.csv"
                if not assigned_path.exists():
                    raise FileNotFoundError("CytoSPACE 输出缺少 assigned_locations.csv")
                assigned = pd.read_csv(assigned_path)
                cell_ids, pool_types, pool_orig_ids, assignments = _parse_assigned_locations(
                    assigned, list(st_expr.index), require_original_cid=True
                )
                if pool_orig_ids is None:
                    raise ValueError("pool_orig_ids 缺失")
                missing_type = [cid for cid in pool_orig_ids if cid not in sc_meta.index]
                if missing_type:
                    raise ValueError(f"sc_metadata 缺少 pool_orig_ids: {missing_type[:5]}")
                pool_cell_types = sc_meta.loc[pool_orig_ids, "type_col"].astype(str).tolist()
                cell_assignment_sha1_pre = _sha1_assignments(assignments)
                mat0 = _assignments_to_matrix(assignments, len(st_expr.index), len(cell_ids)).tocsr()
                mat = mat0
                capacity_arr = np.asarray(capacity, dtype=int)
                cap_sum = int(capacity_arr.sum())
                if cap_sum != len(cell_ids):
                    raise ValueError(f"cells_per_spot 总和 {cap_sum} 与 CytoSPACE cell pool 大小 {len(cell_ids)} 不一致")
                post_adjust = bool(config.get("post_assign_adjust", True))
                knn_mode = None
                refine_used = False
                if svg_post_enabled and not post_adjust:
                    svg_post_meta["skip_reason"] = "post_assign_adjust=false"
                svg_post_rounds = int(config.get("svg_post_rounds", 3) or 0)
                svg_post_neighbor_k = int(config.get("svg_post_neighbor_k", 10) or 0)
                svg_post_alpha = float(config.get("svg_post_alpha", 0.2) or 0.0)
                svg_post_max_move_frac = float(config.get("svg_post_max_move_frac", 0.02) or 0.0)
                svg_post_max_changed_cells = config.get("svg_post_max_changed_cells")
                if svg_post_max_changed_cells in (None, ""):
                    svg_post_max_changed_cells = None
                else:
                    svg_post_max_changed_cells = int(svg_post_max_changed_cells)
                svg_post_selection_mode = config.get("svg_post_selection_mode")
                if svg_post_selection_mode in (None, ""):
                    svg_post_selection_mode = "sequential_greedy"
                svg_post_action_topm = config.get("svg_post_action_topm")
                if svg_post_action_topm in (None, ""):
                    svg_post_action_topm = None
                else:
                    svg_post_action_topm = int(svg_post_action_topm)
                svg_post_sort_key = config.get("svg_post_sort_key")
                if svg_post_sort_key in (None, ""):
                    svg_post_sort_key = "gainA_penalized"
                else:
                    svg_post_sort_key = str(svg_post_sort_key)
                svg_post_lambda_partner_hurt = config.get("svg_post_lambda_partner_hurt")
                if svg_post_lambda_partner_hurt in (None, ""):
                    svg_post_lambda_partner_hurt = 0.0
                else:
                    svg_post_lambda_partner_hurt = float(svg_post_lambda_partner_hurt)
                svg_post_lambda_base = config.get("svg_post_lambda_base")
                if svg_post_lambda_base in (None, ""):
                    svg_post_lambda_base = 0.0
                else:
                    svg_post_lambda_base = float(svg_post_lambda_base)
                svg_post_lambda_type_shift = config.get("svg_post_lambda_type_shift")
                if svg_post_lambda_type_shift in (None, ""):
                    svg_post_lambda_type_shift = None
                else:
                    svg_post_lambda_type_shift = float(svg_post_lambda_type_shift)
                svg_post_lambda_type_penalty = config.get("svg_post_lambda_type_penalty")
                if svg_post_lambda_type_penalty in (None, ""):
                    svg_post_lambda_type_penalty = None
                else:
                    svg_post_lambda_type_penalty = float(svg_post_lambda_type_penalty)
                if svg_post_lambda_type_shift is None and svg_post_lambda_type_penalty is not None:
                    svg_post_lambda_type_shift = svg_post_lambda_type_penalty
                if svg_post_lambda_type_shift is None:
                    svg_post_lambda_type_shift = 0.0
                svg_post_allow_swap = bool(config.get("svg_post_allow_swap", True))
                svg_post_delta_min = float(config.get("svg_post_delta_min", 1e-6) or 0.0)
                svg_post_weight_clip_max = float(config.get("svg_post_weight_clip_max", 1.2) or 1.2)
                svg_post_base_margin_tau = config.get("svg_post_base_margin_tau")
                if svg_post_base_margin_tau in (None, ""):
                    svg_post_base_margin_tau = 0.0
                else:
                    svg_post_base_margin_tau = float(svg_post_base_margin_tau)
                svg_post_require_delta_base_nonpositive = bool(config.get("svg_post_require_delta_base_nonpositive", False))
                svg_post_uncertain_margin_tau = config.get("svg_post_uncertain_margin_tau")
                if svg_post_uncertain_margin_tau in (None, ""):
                    svg_post_uncertain_margin_tau = None
                else:
                    svg_post_uncertain_margin_tau = float(svg_post_uncertain_margin_tau)
                svg_post_swap_delta_min = config.get("svg_post_swap_delta_min")
                if svg_post_swap_delta_min in (None, ""):
                    svg_post_swap_delta_min = svg_post_delta_min
                else:
                    svg_post_swap_delta_min = float(svg_post_swap_delta_min)
                svg_post_swap_require_delta_base_nonpositive = config.get("svg_post_swap_require_delta_base_nonpositive_for_both")
                if svg_post_swap_require_delta_base_nonpositive in (None, ""):
                    svg_post_swap_require_delta_base_nonpositive = bool(
                        config.get("svg_post_swap_require_delta_base_nonpositive", False)
                    )
                else:
                    svg_post_swap_require_delta_base_nonpositive = bool(svg_post_swap_require_delta_base_nonpositive)
                svg_post_swap_depth = int(config.get("svg_post_swap_depth", 1) or 1)
                svg_post_swap_only_within_tieset = bool(config.get("svg_post_swap_only_within_tieset", False))
                svg_post_max_swaps_per_round = config.get("svg_post_max_swaps_per_round")
                if svg_post_max_swaps_per_round in (None, ""):
                    svg_post_max_swaps_per_round = None
                else:
                    svg_post_max_swaps_per_round = int(svg_post_max_swaps_per_round)
                svg_post_swap_base_mode = config.get("svg_post_swap_base_mode")
                if svg_post_swap_base_mode in (None, ""):
                    svg_post_swap_base_mode = "both_nonpositive"
                svg_post_base_pair_budget_eps = config.get("svg_post_base_pair_budget_eps")
                if svg_post_base_pair_budget_eps in (None, ""):
                    svg_post_base_pair_budget_eps = 0.0
                else:
                    svg_post_base_pair_budget_eps = float(svg_post_base_pair_budget_eps)
                svg_post_swap_scope = config.get("svg_post_swap_scope")
                if svg_post_swap_scope in (None, ""):
                    svg_post_swap_scope = "a_and_b_tieset" if svg_post_swap_only_within_tieset else "a_tieset_only"
                svg_post_swap_partner_topm = config.get("svg_post_swap_partner_topm")
                if svg_post_swap_partner_topm in (None, ""):
                    svg_post_swap_partner_topm = None
                else:
                    svg_post_swap_partner_topm = int(svg_post_swap_partner_topm)
                svg_post_swap_target_topk = config.get("svg_post_swap_target_topk")
                if svg_post_swap_target_topk in (None, ""):
                    svg_post_swap_target_topk = None
                else:
                    svg_post_swap_target_topk = int(svg_post_swap_target_topk)
                svg_post_swap_target_expand_mode = config.get("svg_post_swap_target_expand_mode")
                if svg_post_swap_target_expand_mode in (None, ""):
                    svg_post_swap_target_expand_mode = "none"
                svg_post_swap_target_expand_k = config.get("svg_post_swap_target_expand_k")
                if svg_post_swap_target_expand_k in (None, ""):
                    svg_post_swap_target_expand_k = 0
                else:
                    svg_post_swap_target_expand_k = int(svg_post_swap_target_expand_k)
                if type_post_enabled and not post_adjust:
                    type_post_meta["skip_reason"] = "post_assign_adjust=false"
                type_post_marker_source = config.get("type_post_marker_source")
                if type_post_marker_source in (None, ""):
                    type_post_marker_source = None
                else:
                    type_post_marker_source = str(type_post_marker_source).strip().lower()
                type_post_marker_list = config.get("type_post_marker_list")
                type_post_markers_per_type = config.get("type_post_markers_per_type")
                if type_post_markers_per_type in (None, ""):
                    type_post_markers_per_type = None
                else:
                    type_post_markers_per_type = int(type_post_markers_per_type)
                type_post_gene_weighting = config.get("type_post_gene_weighting")
                if type_post_gene_weighting in (None, ""):
                    type_post_gene_weighting = None
                else:
                    type_post_gene_weighting = str(type_post_gene_weighting).strip().lower()
                type_post_require_gainA_positive = bool(config.get("type_post_require_gainA_positive", True))
                type_post_rounds = int(config.get("type_post_rounds", 1) or 0)
                type_post_neighbor_k = int(config.get("type_post_neighbor_k", 10) or 0)
                type_post_max_changed_cells = config.get("type_post_max_changed_cells")
                if type_post_max_changed_cells in (None, ""):
                    type_post_max_changed_cells = None
                else:
                    type_post_max_changed_cells = int(type_post_max_changed_cells)
                type_post_selection_mode = config.get("type_post_selection_mode")
                if type_post_selection_mode in (None, ""):
                    type_post_selection_mode = "global_pool_greedy"
                type_post_action_topm = config.get("type_post_action_topm")
                if type_post_action_topm in (None, ""):
                    type_post_action_topm = None
                else:
                    type_post_action_topm = int(type_post_action_topm)
                type_post_allow_swap = bool(config.get("type_post_allow_swap", True))
                type_post_delta_min = float(config.get("type_post_delta_min", 1e-6) or 0.0)
                type_post_base_margin_tau = config.get("type_post_base_margin_tau")
                if type_post_base_margin_tau in (None, ""):
                    type_post_base_margin_tau = 0.0
                else:
                    type_post_base_margin_tau = float(type_post_base_margin_tau)
                type_post_swap_target_topk = config.get("type_post_swap_target_topk")
                if type_post_swap_target_topk in (None, ""):
                    type_post_swap_target_topk = None
                else:
                    type_post_swap_target_topk = int(type_post_swap_target_topk)
                if type_post_mode == "local_marker_swap":
                    if type_post_marker_source is None:
                        type_post_marker_source = "sc_auto_top"
                    if type_post_markers_per_type is None:
                        type_post_markers_per_type = 30
                    if type_post_gene_weighting is None:
                        type_post_gene_weighting = "spot_zscore"
                    if type_post_swap_target_topk is None:
                        type_post_swap_target_topk = 5
                type_post_swap_scope = config.get("type_post_swap_scope")
                if type_post_swap_scope in (None, ""):
                    type_post_swap_scope = "a_tieset_only"
                type_post_swap_base_mode = config.get("type_post_swap_base_mode")
                if type_post_swap_base_mode in (None, ""):
                    type_post_swap_base_mode = "pair_budget"
                type_post_base_pair_budget_eps = config.get("type_post_base_pair_budget_eps")
                if type_post_base_pair_budget_eps in (None, ""):
                    type_post_base_pair_budget_eps = 0.0
                else:
                    type_post_base_pair_budget_eps = float(type_post_base_pair_budget_eps)
                type_post_max_swaps_per_round = config.get("type_post_max_swaps_per_round")
                if type_post_max_swaps_per_round in (None, ""):
                    type_post_max_swaps_per_round = None
                else:
                    type_post_max_swaps_per_round = int(type_post_max_swaps_per_round)
                type_post_partner_hurt_max = config.get("type_post_partner_hurt_max")
                if type_post_partner_hurt_max in (None, ""):
                    type_post_partner_hurt_max = None
                else:
                    type_post_partner_hurt_max = float(type_post_partner_hurt_max)
                type_post_max_high_hurt_actions = config.get("type_post_max_high_hurt_actions")
                if type_post_max_high_hurt_actions in (None, ""):
                    type_post_max_high_hurt_actions = None
                else:
                    type_post_max_high_hurt_actions = int(type_post_max_high_hurt_actions)
                type_post_high_hurt_threshold = config.get("type_post_high_hurt_threshold")
                if type_post_high_hurt_threshold in (None, ""):
                    type_post_high_hurt_threshold = None
                else:
                    type_post_high_hurt_threshold = float(type_post_high_hurt_threshold)
                type_post_lambda_partner = config.get("type_post_lambda_partner")
                if type_post_lambda_partner in (None, ""):
                    type_post_lambda_partner = config.get("type_post_lambda_partner_hurt")
                if type_post_lambda_partner in (None, ""):
                    type_post_lambda_partner = 0.0
                else:
                    type_post_lambda_partner = float(type_post_lambda_partner)
                type_post_lambda_base = config.get("type_post_lambda_base")
                if type_post_lambda_base in (None, ""):
                    type_post_lambda_base = 0.0
                else:
                    type_post_lambda_base = float(type_post_lambda_base)
                if type_post_enabled:
                    type_post_effective_subset = {
                        "type_post_mode": type_post_mode,
                        "type_post_marker_source": type_post_marker_source,
                        "type_post_markers_per_type": type_post_markers_per_type,
                        "type_post_gene_weighting": type_post_gene_weighting,
                        "type_post_require_gainA_positive": type_post_require_gainA_positive,
                        "type_post_rounds": type_post_rounds,
                        "type_post_neighbor_k": type_post_neighbor_k,
                        "type_post_max_changed_cells": type_post_max_changed_cells,
                        "type_post_selection_mode": type_post_selection_mode,
                        "type_post_action_topm": type_post_action_topm,
                        "type_post_allow_swap": type_post_allow_swap,
                        "type_post_delta_min": type_post_delta_min,
                        "type_post_base_margin_tau": type_post_base_margin_tau,
                        "type_post_swap_target_topk": type_post_swap_target_topk,
                        "type_post_swap_scope": type_post_swap_scope,
                        "type_post_swap_base_mode": type_post_swap_base_mode,
                        "type_post_base_pair_budget_eps": type_post_base_pair_budget_eps,
                        "type_post_max_swaps_per_round": type_post_max_swaps_per_round,
                        "type_post_partner_hurt_max": type_post_partner_hurt_max,
                        "type_post_max_high_hurt_actions": type_post_max_high_hurt_actions,
                        "type_post_high_hurt_threshold": type_post_high_hurt_threshold,
                        "type_post_lambda_partner": type_post_lambda_partner,
                        "type_post_lambda_base": type_post_lambda_base,
                    }
                if post_adjust and svg_post_enabled:
                    assignments_hard, hard_mat, svg_post_meta = _svg_posterior_local_refine(
                        assignments,
                        distance_metric=distance_metric,
                        sc_cost=sc_cost,
                        st_cost=st_cost,
                        st_coords=st_coords,
                        pool_orig_ids=pool_orig_ids,
                        cell_types=pool_cell_types,
                        spot_ids=spot_ids,
                        plugin_genes=plugin_genes,
                        genes_use=genes_use,
                        w_full=w_full,
                        spot_weight_matrix=spot_weight_matrix,
                        capacity=capacity_arr,
                        neighbor_k=svg_post_neighbor_k,
                        rounds=svg_post_rounds,
                        alpha=svg_post_alpha,
                        max_move_frac=svg_post_max_move_frac,
                        max_changed_cells=svg_post_max_changed_cells,
                        selection_mode=svg_post_selection_mode,
                        action_topm=svg_post_action_topm,
                        sort_key=svg_post_sort_key,
                        lambda_partner_hurt=svg_post_lambda_partner_hurt,
                        lambda_base=svg_post_lambda_base,
                        lambda_type_shift=svg_post_lambda_type_shift,
                        allow_swap=svg_post_allow_swap,
                        delta_min=svg_post_delta_min,
                        weight_clip_max=svg_post_weight_clip_max,
                        base_margin_tau=svg_post_base_margin_tau,
                        require_delta_base_nonpositive=svg_post_require_delta_base_nonpositive,
                        uncertain_margin_tau=svg_post_uncertain_margin_tau,
                        swap_delta_min=svg_post_swap_delta_min,
                        swap_require_delta_base_nonpositive=svg_post_swap_require_delta_base_nonpositive,
                        swap_depth=svg_post_swap_depth,
                        swap_only_within_tieset=svg_post_swap_only_within_tieset,
                        max_swaps_per_round=svg_post_max_swaps_per_round,
                        swap_base_mode=svg_post_swap_base_mode,
                        base_pair_budget_eps=svg_post_base_pair_budget_eps,
                        swap_scope=svg_post_swap_scope,
                        swap_partner_topm=svg_post_swap_partner_topm,
                        swap_target_topk=svg_post_swap_target_topk,
                        swap_target_expand_mode=svg_post_swap_target_expand_mode,
                        swap_target_expand_k=svg_post_swap_target_expand_k,
                    )
                    svg_post_used = True
                    hard_meta = {
                        "harden_method": "svg_post_local_swap",
                        "harden_topk": None,
                        "capacity_check": (svg_post_meta or {}).get("capacity_check"),
                        "harden_fallback_n_cells": None,
                        "harden_repair_n_cells": None,
                        "n_unassigned_before_repair": None,
                        "n_unassigned_cells": None,
                        "type_prior_row_nonzero_ratio": None,
                        "type_prior_row_entropy_mean": None,
                        "type_prior_row_entropy_max": None,
                        "prior_effect_fraction_delta": 0.0,
                        "prior_effect_fraction_delta_mean_per_spot": 0.0,
                        "prior_effect_fraction_delta_audit_before": None,
                        "prior_effect_fraction_delta_audit_after": None,
                    }
                else:
                    if post_adjust and refine_lambda > 0:
                        missing_pool = [cid for cid in pool_orig_ids if cid not in sc_expr.index]
                        if missing_pool:
                            raise ValueError(f"CytoSPACE cell pool ?????????sc_expr ???????????? OriginalCID????????? {missing_pool[:5]}")
                        sc_svg = sc_expr.loc[pool_orig_ids, genes_use].to_numpy(dtype=float) * w_full
                        st_svg = st_expr[genes_use].to_numpy(dtype=float) * w_full
                        mat, knn_mode = _refine_spot_cell_matrix_svg(
                            mat=mat,
                            st_coords=st_coords,
                            st_svg=st_svg,
                            sc_svg=sc_svg,
                            lambda_base=refine_lambda,
                            k=refine_k,
                            eps=eps,
                            batch_size=config.get("svg_refine_batch_size"),
                            knn_metric=config.get("knn_metric", "euclidean"),
                            knn_block_size=int(config.get("knn_block_size", 1024) or 1024),
                            knn_max_dense_n=int(config.get("knn_max_dense_n", 5000) or 5000),
                        )
                        refine_used = True
                    else:
                        refine_used = False
                    # ???????????? refine???per-spot soft adjust???
                    if type_prior_enabled and post_adjust and effective_lambda_refine > 0 and config.get("type_prior_apply_refine", True):
                        missing_pool_types = sorted(set(pool_types) - set(type_cols))
                        if missing_pool_types:
                            raise ValueError(f"cell pool ?????? type_cols ??????????????????: {missing_pool_types[:5]}")
                        mat = _type_prior_refine(mat, pool_types, type_cols, type_prior_norm, effective_lambda_refine, eps=type_prior_eps)
                    hard_topk = int(config.get("harden_topk", 5) or 5)
                    prior_candidate_topk = int(config.get("prior_candidate_topk", 0) or 0)
                    prior_candidate_weight = float(config.get("prior_candidate_weight", 1.0) or 1.0)
                    if not type_prior_enabled:
                        prior_candidate_topk = 0
                        prior_candidate_weight = 0.0
                    if post_adjust:
                        assignments_hard, hard_mat, hard_meta = _harden_assignment_quota_matching(
                            mat=mat,
                            capacity=capacity_arr,
                            cell_types=pool_types,
                            type_prior=type_prior_norm if type_prior_enabled else None,
                            spot_ids=list(st_expr.index),
                            lambda_prior=effective_lambda_harden,
                            eps=eps,
                            topk=hard_topk,
                            prior_candidate_topk=prior_candidate_topk,
                            prior_candidate_weight=prior_candidate_weight,
                            fallback_assignments=assignments,
                            ablate_lambda=False,
                            min_prior_row_nonzero_ratio=float(config.get("min_prior_row_nonzero_ratio", 0.0)),
                        )
                    else:
                        assignments_hard = assignments
                        hard_mat = mat0
                        hard_meta = {
                            "harden_method": "skipped_post_adjust",
                            "harden_topk": None,
                            "capacity_check": None,
                            "harden_fallback_n_cells": None,
                            "harden_repair_n_cells": None,
                            "n_unassigned_before_repair": None,
                            "n_unassigned_cells": None,
                            "type_prior_row_nonzero_ratio": None,
                            "type_prior_row_entropy_mean": None,
                            "type_prior_row_entropy_max": None,
                        }
                    if post_adjust and type_prior_enabled and config.get("prior_ablation_enabled", False):
                        # ablation 仅用于证据链；即使失败也不应影响 plus 主输出
                        hard_meta["prior_ablation_enabled"] = True
                        try:
                            ablate_assign, ablate_mat, ablate_meta = _harden_assignment_quota_matching(
                                mat=mat,
                                capacity=capacity_arr,
                                cell_types=pool_types,
                                type_prior=type_prior_norm if type_prior_enabled else None,
                                spot_ids=list(st_expr.index),
                                lambda_prior=0.0,
                                eps=eps,
                                topk=hard_topk,
                                prior_candidate_topk=prior_candidate_topk,
                                prior_candidate_weight=prior_candidate_weight,
                                fallback_assignments=assignments,
                                ablate_lambda=True,
                                min_prior_row_nonzero_ratio=float(config.get("min_prior_row_nonzero_ratio", 0.0)),
                            )
                            hard_meta["prior_ablation_meta"] = ablate_meta
                            try:
                                change_rate, audit = _compute_ablation_change_rate_checked(
                                    assignments_hard,
                                    ablate_assign,
                                    context="prior_ablation_change_rate",
                                )
                                hard_meta["prior_ablation_status"] = "ok"
                                hard_meta["prior_ablation_change_rate"] = change_rate
                                hard_meta["prior_ablation_audit"] = audit
                            except Exception as e:
                                hard_meta["prior_ablation_status"] = "invalid"
                                hard_meta["prior_ablation_change_rate"] = None
                                hard_meta["prior_ablation_error_reason"] = str(e)
                                hard_meta["prior_ablation_audit"] = {
                                    "context": "prior_ablation_change_rate",
                                    "n_base": len(assignments_hard),
                                    "n_ablate": len(ablate_assign),
                                }
                        except Exception as e:
                            hard_meta["prior_ablation_status"] = "invalid"
                            hard_meta["prior_ablation_change_rate"] = None
                            hard_meta["prior_ablation_error_reason"] = f"ablation_harden_failed: {e}"
                            hard_meta["prior_ablation_audit"] = {
                                "context": "prior_ablation_change_rate",
                                "n_base": len(assignments_hard),
                                "n_ablate": None,
                            }
                    if type_prior_enabled:
                        frac_before, audit_b = _spot_fraction_from_mat(
                            mat0,
                            pool_types,
                            list(st_expr.index),
                            type_cols,
                            context="prior_effect_fraction_delta/before",
                        )
                        frac_after, audit_a = _spot_fraction_from_mat(
                            hard_mat,
                            pool_types,
                            list(st_expr.index),
                            type_cols,
                            context="prior_effect_fraction_delta/after",
                        )
                        frac_before_norm = frac_before.div(frac_before.sum(axis=1).replace(0, 1.0), axis=0)
                        frac_after_norm = frac_after.div(frac_after.sum(axis=1).replace(0, 1.0), axis=0)
                        per_spot_l1 = np.abs(frac_before_norm.values - frac_after_norm.values).sum(axis=1)
                        hard_meta["prior_effect_fraction_delta"] = float(per_spot_l1.sum())
                        hard_meta["prior_effect_fraction_delta_mean_per_spot"] = float(per_spot_l1.mean())
                        hard_meta["prior_effect_fraction_delta_audit_before"] = audit_b
                        hard_meta["prior_effect_fraction_delta_audit_after"] = audit_a
                    else:
                        hard_meta["prior_effect_fraction_delta"] = 0.0
                        hard_meta["prior_effect_fraction_delta_mean_per_spot"] = 0.0
                        hard_meta["prior_effect_fraction_delta_audit_before"] = None
                        hard_meta["prior_effect_fraction_delta_audit_after"] = None
                if post_adjust and type_post_enabled:
                    assignments_hard, hard_mat, type_post_meta = _type_posterior_local_refine(
                        assignments_hard,
                        distance_metric=distance_metric,
                        sc_cost=sc_cost,
                        st_cost=st_cost,
                        st_coords=st_coords,
                        pool_orig_ids=pool_orig_ids,
                        cell_types=pool_types,
                        spot_ids=spot_ids,
                        genes_use=genes_use,
                        w_full=w_full,
                        spot_weight_matrix=spot_weight_matrix,
                        capacity=capacity_arr,
                        neighbor_k=type_post_neighbor_k,
                        rounds=type_post_rounds,
                        max_changed_cells=type_post_max_changed_cells,
                        selection_mode=type_post_selection_mode,
                        action_topm=type_post_action_topm,
                        delta_min=type_post_delta_min,
                        base_margin_tau=type_post_base_margin_tau,
                        swap_target_topk=type_post_swap_target_topk,
                        allow_swap=type_post_allow_swap,
                        swap_scope=type_post_swap_scope,
                        swap_base_mode=type_post_swap_base_mode,
                        base_pair_budget_eps=type_post_base_pair_budget_eps,
                        max_swaps_per_round=type_post_max_swaps_per_round,
                        partner_hurt_max=type_post_partner_hurt_max,
                        max_high_hurt_actions=type_post_max_high_hurt_actions,
                        high_hurt_threshold=type_post_high_hurt_threshold,
                        lambda_partner=type_post_lambda_partner,
                        lambda_base=type_post_lambda_base,
                        mode=type_post_mode,
                        marker_source=type_post_marker_source,
                        marker_list=type_post_marker_list,
                        markers_per_type=type_post_markers_per_type,
                        gene_weighting=type_post_gene_weighting,
                        require_gainA_positive=type_post_require_gainA_positive,
                        sc_meta=sc_meta,
                        sc_type_col=type_col,
                        type_order=type_cols,
                    )
                    type_post_used = True
                    if type_post_effective_subset is not None:
                        type_post_meta["effective_subset"] = type_post_effective_subset
                    if isinstance(hard_meta, dict):
                        hard_meta["type_post_used"] = True
                        hard_meta["type_post_status"] = type_post_meta.get("status")
                        hard_meta["type_post_capacity_check"] = type_post_meta.get("capacity_check")
                        hard_meta["capacity_check"] = type_post_meta.get("capacity_check")
                try:
                    assigned[["UniqueCID", "OriginalCID", "SpotID", "CellType"]].to_csv(
                        out_dir / "cell_pool_map_plus.csv", index=False
                    )
                except Exception:
                    pass

                if not pool_orig_ids:
                    raise ValueError("[plus] assigned_locations 缺少 OriginalCID，无法与 Stage1/SimGen Query 真值对齐")
                # ⚠️cell_id 统一用 OriginalCID；UniqueCID 作为额外列 unique_cid 供审计
                abstain_unknown = bool(config.get("abstain_unknown_sc_only", False)) if post_adjust else False
                unknown_label = "Unknown_sc_only"
                unknown_mask = [t == unknown_label for t in pool_types]
                n_unknown_pool = int(sum(unknown_mask))
                n_unknown_dropped = 0
                if abstain_unknown and n_unknown_pool:
                    keep_indices = [i for i, is_unknown in enumerate(unknown_mask) if not is_unknown]
                    n_unknown_dropped = int(len(pool_types) - len(keep_indices))
                    if keep_indices:
                        keep_set = set(keep_indices)
                        index_map = {old_i: new_i for new_i, old_i in enumerate(keep_indices)}
                        assignments_hard = [
                            (index_map[cell_i], spot_i, score)
                            for cell_i, spot_i, score in assignments_hard
                            if cell_i in keep_set
                        ]
                        hard_mat = hard_mat[:, keep_indices].tocsr()
                        pool_types = [pool_types[i] for i in keep_indices]
                        pool_orig_ids = [pool_orig_ids[i] for i in keep_indices]
                        cell_ids = [cell_ids[i] for i in keep_indices]
                    else:
                        assignments_hard = []
                        hard_mat = hard_mat[:, :0].tocsr()
                        pool_types = []
                        pool_orig_ids = []
                        cell_ids = []
                hard_meta["abstain_unknown_sc_only"] = abstain_unknown
                hard_meta["abstain_unknown_label"] = unknown_label
                hard_meta["abstain_unknown_pool_count"] = n_unknown_pool
                hard_meta["abstain_unknown_dropped"] = n_unknown_dropped
                hard_meta["abstain_unknown_remaining"] = len(pool_orig_ids)
                _build_outputs(
                    assignments_hard,
                    [str(x) for x in pool_orig_ids],
                    list(st_expr.index),
                    pool_types,
                    out_dir,
                    mode=mode,
                    type_order=type_cols,
                    hard_matrix=hard_mat,
                    unique_cell_ids=[str(x) for x in cell_ids],
                )
                cell_assignment_sha1_post = _sha1_assignments(assignments_hard)
                cell_assignment_sha1_post_file = _sha1_file(out_dir / "cell_assignment_plus.csv")
                selected_actions = (svg_post_meta or {}).get("selected_actions") if svg_post_used else None
                if selected_actions:
                    pd.DataFrame(selected_actions).to_csv(out_dir / "selected_actions.csv", index=False)
                selected_type_actions = (type_post_meta or {}).get("selected_actions") if type_post_used else None
                if selected_type_actions:
                    pd.DataFrame(selected_type_actions).to_csv(out_dir / "selected_type_actions.csv", index=False)
        except Exception as e:
            import traceback
            status = "failed"
            error_msg = f"{e}\n{traceback.format_exc()}"
            for fname in ["cell_assignment_plus.csv", "cell_spot_matrix_plus.npz", "spot_type_fraction_plus.csv"]:
                try:
                    (out_dir / fname).unlink()
                except Exception:
                    pass
        meta = {
            "backend": "cytospace",
            "mode": config.get("mode"),
            "run_id": config.get("run_id"),
            "variant": config.get("variant"),
            "out_dir_name": out_dir.name,
            "sample": config.get("sample"),
            "seed": config.get("seed"),
            **_module_fingerprint(),
            "runner_file": config.get("runner_file"),
            "runner_sha1": (str(config.get("runner_sha1")).lower() if config.get("runner_sha1") else None),
            "svg_refine_lambda": config.get("svg_refine_lambda"),
            "feature_set": "all_common_genes",
            "type_set": "plugin_type",
            "weight_field": locals().get("weight_field"),
            "stage1_dir": str(stage1_dir),
            "stage2_dir": str(stage2_dir),
            "stage3_dir": str(stage3_dir),
            "stage2_gene_weights_sha1": locals().get("stage2_gene_weights_sha1"),
            "plugin_genes_sha1": locals().get("plugin_genes_sha1"),
            "n_genes_use": len(genes_use) if "genes_use" in locals() else None,
            "n_weighted_genes": len(weighted_genes_applied) if "weighted_genes_applied" in locals() else None,
            "weighted_gene_examples": (locals().get("weighted_genes_applied") or [])[:10],
            "weighted_gene_ratio": locals().get("weighted_ratio_applied"),
            "w_stats": locals().get("w_stats"),
            "w_stats_raw": locals().get("w_stats_raw"),
            "svg_fallback": locals().get("svg_fallback"),
            "svg_fallback_reason": locals().get("svg_fallback_reasons"),
            "plugin_overlap_ratio": locals().get("plugin_overlap_ratio"),
            "cost_expr_norm": locals().get("cost_expr_audit", None),
            "cost_mode_effective": locals().get("cost_mode_effective"),
            "spot_weight_mode_effective": locals().get("spot_weight_mode_effective"),
            "cost_matrix_sha1": locals().get("cost_matrix_sha1"),
            "cost_matrix_sha1_no_spot_weight": locals().get("cost_matrix_sha1_no_spot_weight"),
            "part1_knobs_no_effect": locals().get("part1_knobs_no_effect"),
            "spot_weight_stats": locals().get("spot_weight_stats", None),
            "spot_specific_basin": config.get("spot_specific_basin"),
            "spot_specific_threshold_kappa": config.get("spot_specific_threshold_kappa"),
            "norm_applied": (locals().get("cost_expr_audit") or {}).get("norm_applied"),
            "norm_fallback": (locals().get("cost_expr_audit") or {}).get("norm_fallback"),
            "norm_fallback_reason": (locals().get("cost_expr_audit") or {}).get("norm_fallback_reason"),
            "norm_reason": (locals().get("cost_expr_audit") or {}).get("norm_reason"),
            "config_id": config.get("config_id"),
            "project_config_path": config.get("project_config_path"),
            "dataset_config_path": config.get("dataset_config_path"),
            "config_validation": locals().get("config_validation", None),
            "config_effective_subset": locals().get("config_effective_subset", None),
            "capacity_audit": locals().get("capacity_audit", None),
            "cell_id_space": "original_cid",
            "cell_instance_id_column": "unique_cid",
            "cell_assignment_sha1_pre": locals().get("cell_assignment_sha1_pre"),
            "cell_assignment_sha1_post": locals().get("cell_assignment_sha1_post"),
            "cell_assignment_sha1_post_file": locals().get("cell_assignment_sha1_post_file"),
            "selected_actions_path": str(out_dir / "selected_actions.csv") if (out_dir / "selected_actions.csv").exists() else None,
            "cells_per_spot_source": locals().get("cps_source", None),
            "umi_to_cell_norm": locals().get("norm_val", None),
            "has_capacity_constraint": status == "success",
            "cost_cfg": {
                "lambda_prior": config.get("lambda_prior", 1.0),
                "solver_method": config.get("solver_method", "lap_CSPR"),
                "distance_metric": locals().get("distance_metric", None),
            },
            "refine_cfg": {
                "refine_used": locals().get("refine_used", False),
                "svg_refine_lambda": locals().get("refine_lambda", 0.0),
                "svg_refine_k": locals().get("refine_k", None),
                "eps": config.get("eps", None),
                "svg_refine_mode": "error_weighted_svg_loss",
                "lambda_prior": locals().get("lambda_prior", None),
                "knn_block_size": config.get("knn_block_size", None),
                "knn_max_dense_n": config.get("knn_max_dense_n", None),
                "knn_metric": config.get("knn_metric", None),
                "knn_mode": locals().get("knn_mode", None),
                "harden_method": locals().get("hard_meta", {}).get("harden_method"),
                "harden_topk": locals().get("hard_meta", {}).get("harden_topk"),
                "type_prior_apply_refine": config.get("type_prior_apply_refine", True),
                "type_prior_apply_harden": config.get("type_prior_apply_harden", True),
                "svg_refine_batch_size": config.get("svg_refine_batch_size", None),
                "effective_lambda_prior_refine": locals().get("effective_lambda_refine", None),
                "effective_lambda_prior_harden": locals().get("effective_lambda_harden", None),
                "prior_ablation_enabled": config.get("prior_ablation_enabled", False),
                "prior_ablation_change_rate": locals().get("hard_meta", {}).get("prior_ablation_change_rate"),
                "prior_ablation_status": locals().get("hard_meta", {}).get("prior_ablation_status"),
                "prior_ablation_error_reason": locals().get("hard_meta", {}).get("prior_ablation_error_reason"),
                "svg_post_used": svg_post_used,
                "svg_post_status": (svg_post_meta or {}).get("status"),
                "type_post_used": type_post_used,
                "type_post_status": (type_post_meta or {}).get("status"),
            },
            "svg_post_meta": svg_post_meta,
            "type_post_meta": type_post_meta,
            "type_post_used": type_post_used,
            "type_post_mode": type_post_mode,
            "type_post_effective_subset": type_post_effective_subset,
            "type_post_skipped_reason": type_post_meta.get("skip_reason") if isinstance(type_post_meta, dict) else None,
            "type_posterior_enabled": type_posterior_enabled,
            "type_posterior_skipped_reason": type_posterior_skipped_reason,
            "type_prior_effect": locals().get("hard_meta", {}).get("prior_effect_fraction_delta", 0.0) if type_prior_enabled else 0.0,
            "type_prior_mode": "off" if not type_prior_enabled else "global_mean_from_spot_matrix",
            "type_prior_mode_for_cytospace": "off" if not type_prior_enabled else "global_mean_1xT",
            "type_prior_mode_for_refine": "off" if not type_prior_enabled else "per_spot_SxT",
            "type_columns": locals().get("type_cols", None),
            "spot_alignment_audit": locals().get("spot_alignment_audit", None),
            "n_spots_stage1": len(st_expr.index),
            "n_spots_coords": len(st_coords.index),
            "n_spots_type_prior": len(type_prior.index) if isinstance(type_prior, pd.DataFrame) else None,
            "n_spots_intersection": len(set(st_expr.index) & set(st_coords.index)),
            "n_spots_expr_type_prior_intersection": prior_intersection,
            "example_missing_in_type_prior": missing_prior_examples,
            "missing_type_cols": locals().get("missing_type_cols", []),
            "n_cells_missing_type_col": locals().get("n_cells_missing_type", 0),
            "ratio_cells_missing_type_col": locals().get("ratio_cells_missing_type", 0.0),
            "n_cells_sc_expr": len(sc_expr.index),
            "n_cells_sc_meta": len(sc_meta.index),
            "n_cells_intersection": len(set(sc_expr.index) & set(sc_meta.index)),
            "n_cells_pool": len(locals().get("cell_ids", [])) if status == "success" else None,
            "capacity_check": locals().get("hard_meta", {}).get("capacity_check"),
            "harden_fallback_n_cells": locals().get("hard_meta", {}).get("harden_fallback_n_cells"),
            "harden_repair_n_cells": locals().get("hard_meta", {}).get("harden_repair_n_cells"),
            "n_unassigned_before_repair": locals().get("hard_meta", {}).get("n_unassigned_before_repair"),
            "n_unassigned_cells": locals().get("hard_meta", {}).get("n_unassigned_cells"),
            "prior_effect_fraction_delta": locals().get("hard_meta", {}).get("prior_effect_fraction_delta"),
            "prior_effect_fraction_delta_mean_per_spot": locals().get("hard_meta", {}).get(
                "prior_effect_fraction_delta_mean_per_spot"
            ),
            "prior_effect_fraction_delta_audit_before": locals().get("hard_meta", {}).get(
                "prior_effect_fraction_delta_audit_before"
            ),
            "prior_effect_fraction_delta_audit_after": locals().get("hard_meta", {}).get(
                "prior_effect_fraction_delta_audit_after"
            ),
            "prior_local_top1_change_rate": locals().get("hard_meta", {}).get("prior_local_top1_change_rate"),
            "prior_ablation_status": locals().get("hard_meta", {}).get("prior_ablation_status"),
            "prior_ablation_change_rate": locals().get("hard_meta", {}).get("prior_ablation_change_rate"),
            "prior_ablation_error_reason": locals().get("hard_meta", {}).get("prior_ablation_error_reason"),
            "prior_ablation_audit": locals().get("hard_meta", {}).get("prior_ablation_audit"),
            "type_prior_row_nonzero_ratio": locals().get("hard_meta", {}).get("type_prior_row_nonzero_ratio"),
            "type_prior_row_entropy_mean": locals().get("hard_meta", {}).get("type_prior_row_entropy_mean"),
            "type_prior_row_entropy_max": locals().get("hard_meta", {}).get("type_prior_row_entropy_max"),
            "gene_usage_stats": {
                "plugin_genes_total": len(plugin_genes),
                "genes_use": len(genes_use) if "genes_use" in locals() else None,
                "genes_missing_in_sc": int(len(set(plugin_genes) - set(sc_expr.columns))),
                "genes_missing_in_st": int(len(set(plugin_genes) - set(st_expr.columns))),
                "weight_min": (locals().get("w_stats") or {}).get("min"),
                "weight_p50": (locals().get("w_stats") or {}).get("p50"),
                "weight_p95": (locals().get("w_stats") or {}).get("p95"),
                "weight_max": (locals().get("w_stats") or {}).get("max"),
                "weighted_gene_count": len(weighted_genes_applied) if "weighted_genes_applied" in locals() else None,
                "weighted_gene_ratio": locals().get("weighted_ratio_applied"),
            },
            "runtime_sec": time.time() - t0,
            "status": status,
            "error_msg": error_msg,
            "resolved_mapping_config": config,
        }
        _write_meta(out_dir, meta)

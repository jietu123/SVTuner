"""
Stage 5: Simulated-scenario evaluation (with ground truth from SimGen).

Hard constraints (v1.2.1):
- Enumerate runs ONLY via Stage4 manifest (`result/<scenario_id>/stage4_run_manifest.*`).
- Per-run pipeline order: schema_audit -> id_gate -> metrics -> run_table (id_gate_fail runs do not enter summary/recommendation).
- gate_status is a SINGLE final value with priority:
    missing_input > schema_fail > id_gate_fail > metric_fail > pass
- Global terminate when truth/manifest cannot be parsed OR run_list is empty.
- spot×type policy switches MUST be recorded in meta:
    missing_spot_policy {fill_zero|fail}, renorm_policy {renorm|fail}, type_union_policy {union|intersect}
- topK MUST record matrix semantics + tie rule:
    cell_spot_matrix rows=cells, cols=spots; ties -> prefer_lowest_spot_index

Outputs:
- Per-run evidence chain:
    result/<scenario_id>/stage5_eval/run_<run_uid>/
      stage5_meta.json, input_fingerprint.json, schema_audit.json, id_audit.json, metrics_generic.json
- Scenario-level tables:
    result/<scenario_id>/stage5_eval/generic/metrics_<backend>.json
    result/<scenario_id>/stage5_eval/generic/metrics_<backend>_runs.csv
    result/<scenario_id>/stage5_eval/generic/stage5_run_table.csv
- Cross-scenario summary:
    result/stage5_summary/stage5_run_table.csv
    result/stage5_summary/stage5_scenario_summary.json
    result/stage5_summary/stage5_config_recommendation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from utils.run_manifest import load_stage4_run_manifest


def detect_project_root() -> Path:
    here = Path(__file__).resolve()
    try:
        return here.parents[2]
    except IndexError:
        return here.parent.parent


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f) or {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


def _as_str_series(s: pd.Series) -> pd.Series:
    # Use pandas "string" dtype to preserve missing values as <NA> (avoid "nan" strings).
    return s.astype("string")


def _sha1_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest().lower()
    except Exception:
        return None


def _json_write(path: Path, payload: Any) -> None:
    def _default(o: Any):
        if o is pd.NA:
            return None
        if isinstance(o, Path):
            return o.as_posix()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_default) + "\n", encoding="utf-8")


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, float) and np.isnan(v):
            return None
        return float(v)
    except Exception:
        return None


def _sanitize_token(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "run"


def load_query_truth(sim_dir: Path) -> Tuple[pd.DataFrame, str]:
    candidates = [
        ("sim_truth_query_cell_spot.csv", "canonical"),
        ("cell_true_spot.csv", "alias"),
    ]
    for fname, _kind in candidates:
        p = sim_dir / fname
        if not p.exists():
            continue
        df = _read_csv(p)
        required = {"cell_id", "true_spot_id"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"Query truth missing columns {sorted(list(missing))}: {p}")
        df = df.copy()
        df["cell_id"] = _as_str_series(df["cell_id"])
        df["true_spot_id"] = _as_str_series(df["true_spot_id"])
        # Treat empty string as NA (SimGen uses "" for missing spot truth).
        df.loc[df["true_spot_id"].astype("string").str.len() == 0, "true_spot_id"] = pd.NA
        if "truth_status" in df.columns:
            df["truth_status"] = _as_str_series(df["truth_status"])
        if "celltype" in df.columns:
            df["celltype"] = _as_str_series(df["celltype"])
        return df, fname
    raise FileNotFoundError(f"Query truth not found under: {sim_dir} (expected sim_truth_query_cell_spot.csv or cell_true_spot.csv)")


def load_truth_spot_type_fraction(sim_dir: Path) -> Tuple[pd.DataFrame, str]:
    candidates = [
        ("sim_truth_spot_type_fraction.csv", "canonical"),
        ("spot_true_type_fraction.csv", "alias"),
    ]
    for fname, _kind in candidates:
        p = sim_dir / fname
        if not p.exists():
            continue
        df = _read_csv(p)
        if "spot_id" not in df.columns:
            # Allow index-as-spot_id CSV.
            df = pd.read_csv(p, index_col=0)
            df.index.name = "spot_id"
            df = df.reset_index()
        df = df.copy()
        df["spot_id"] = _as_str_series(df["spot_id"])
        df = df.set_index("spot_id")
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df, fname
    raise FileNotFoundError(
        f"Spot×type truth not found under: {sim_dir} (expected sim_truth_spot_type_fraction.csv or spot_true_type_fraction.csv)"
    )


def load_sim_st_coordinates(sim_dir: Path) -> pd.DataFrame:
    p = sim_dir / "sim_st_coordinates.csv"
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = _read_csv(p)
    required = {"spot_id", "row", "col"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"sim_st_coordinates missing columns {sorted(list(missing))}: {p}")
    df = df.copy()
    df["spot_id"] = _as_str_series(df["spot_id"])
    for c in ["row", "col"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["x", "y"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "in_tissue" in df.columns:
        df["in_tissue"] = pd.to_numeric(df["in_tissue"], errors="coerce").fillna(0).astype(int)
    return df.set_index("spot_id")


@dataclass(frozen=True)
class Stage4RunRef:
    scenario_id: str
    backend: str
    mode: str
    run_id: str
    run_dir: Path
    cell_assignment_path: Path
    cell_spot_matrix_path: Path
    spot_type_fraction_path: Path
    meta_path: Optional[Path]
    config_id: Optional[str]
    seed: Optional[int]
    lambda_svg: Optional[float]
    refine_enabled: Optional[bool]

    @property
    def run_key(self) -> str:
        cfg = self.config_id if self.config_id not in (None, "") else "NA"
        seed = self.seed if self.seed is not None else "NA"
        lam = self.lambda_svg if self.lambda_svg is not None else "NA"
        ref = "NA" if self.refine_enabled is None else ("1" if self.refine_enabled else "0")
        return f"{self.backend}::{self.mode}::{self.run_id}::cfg={cfg}::seed={seed}::lam={lam}::ref={ref}"


def _run_uid(run: Stage4RunRef) -> str:
    """
    Build a stable per-run identifier for Stage5 outputs.

    IMPORTANT: run_id is not guaranteed unique across multi-seed/multi-config runs, so Stage5 must
    never key outputs purely by run_id.
    """
    cfg = str(run.config_id) if run.config_id not in (None, "") else "NA"
    seed = str(run.seed) if run.seed is not None else "NA"
    lam = str(run.lambda_svg) if run.lambda_svg is not None else "NA"
    ref = "NA" if run.refine_enabled is None else ("1" if run.refine_enabled else "0")
    h = hashlib.sha1(str(run.run_dir.as_posix()).encode("utf-8")).hexdigest().lower()[:8]
    tokens = [
        run.backend,
        run.mode,
        run.run_id,
        f"cfg_{cfg}",
        f"seed_{seed}",
        f"lam_{lam}",
        f"ref_{ref}",
    ]
    base = "__".join(_sanitize_token(t) for t in tokens)
    uid = f"{base}__{h}"
    max_len = 140
    if len(uid) > max_len:
        keep = max(8, max_len - (len(h) + 2))
        uid = f"{base[:keep]}__{h}"
    return uid


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _as_abs_path(project_root: Path, path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return project_root / p


def load_stage4_runs_from_manifest(*, project_root: Path, scenario_id: str, backends: List[str]) -> List[Stage4RunRef]:
    payload = load_stage4_run_manifest(project_root=project_root, scenario_id=scenario_id)
    runs_raw = payload.get("runs") or []

    backend_set = set(backends) if backends else set()
    out: List[Stage4RunRef] = []
    for r in runs_raw:
        b = str(r.get("backend") or "")
        if backend_set and b not in backend_set:
            continue
        mode = str(r.get("mode") or "")
        run_id = str(r.get("run_id") or "")
        run_dir = _as_abs_path(project_root, str(r.get("run_dir") or ""))
        ca_path = _as_abs_path(project_root, str(r.get("cell_assignment_path") or ""))
        mat_path = _as_abs_path(project_root, str(r.get("cell_spot_matrix_path") or ""))
        stf_path = _as_abs_path(project_root, str(r.get("spot_type_fraction_path") or ""))
        meta_path_raw = r.get("meta_path")
        meta_path = _as_abs_path(project_root, str(meta_path_raw)) if meta_path_raw else None
        cfg_id = r.get("config_id")
        cfg_id = str(cfg_id) if cfg_id not in (None, "") else None
        seed = r.get("seed")
        try:
            seed = int(seed) if seed not in (None, "") else None
        except Exception:
            seed = None
        lambda_svg = r.get("lambda_svg")
        try:
            lambda_svg = float(lambda_svg) if lambda_svg not in (None, "") else None
        except Exception:
            lambda_svg = None
        refine_enabled = r.get("refine_enabled")
        if isinstance(refine_enabled, str):
            refine_enabled = refine_enabled.strip().lower() in ("1", "true", "yes", "y", "on")

        out.append(
            Stage4RunRef(
                scenario_id=scenario_id,
                backend=b,
                mode=mode,
                run_id=run_id,
                run_dir=run_dir,
                cell_assignment_path=ca_path,
                cell_spot_matrix_path=mat_path,
                spot_type_fraction_path=stf_path,
                meta_path=meta_path,
                config_id=cfg_id,
                seed=seed,
                lambda_svg=lambda_svg,
                refine_enabled=bool(refine_enabled) if refine_enabled is not None else None,
            )
        )

    return out


def _id_stats(cell_ids: pd.Series) -> Dict[str, Any]:
    s = _as_str_series(cell_ids).copy()
    na = int(s.isna().sum())
    n = int(len(s))
    uniq = int(s.dropna().nunique())
    dup = int(s.dropna().duplicated().sum())
    dup_rate = float(dup / max(1, n - na))
    return {"n_rows": n, "n_na": na, "n_unique": uniq, "n_dup": dup, "dup_rate": dup_rate}


def _alignment_stats(stage4_ids: pd.Series, truth_ids: pd.Series, *, max_examples: int = 5) -> Dict[str, Any]:
    s4 = _as_str_series(stage4_ids).dropna()
    tr = _as_str_series(truth_ids).dropna()
    s4_set = set(s4.astype(str).tolist())
    tr_set = set(tr.astype(str).tolist())
    only_stage4 = sorted(list(s4_set - tr_set))
    only_truth = sorted(list(tr_set - s4_set))
    in_truth = len(s4_set & tr_set)
    return {
        "stage4_unique": int(len(s4_set)),
        "truth_unique": int(len(tr_set)),
        "stage4_in_truth_unique": int(in_truth),
        "stage4_in_truth_fraction": float(in_truth / max(1, len(s4_set))),
        "truth_in_stage4_unique": int(len(tr_set & s4_set)),
        "truth_in_stage4_fraction": float(len(tr_set & s4_set) / max(1, len(tr_set))),
        "stage4_only_count": int(len(only_stage4)),
        "truth_only_count": int(len(only_truth)),
        "stage4_only_examples": only_stage4[:max_examples],
        "truth_only_examples": only_truth[:max_examples],
    }


def _split_last_token(s: pd.Series, seps: List[str]) -> pd.Series:
    out = _as_str_series(s).copy()
    for sep in seps:
        out = out.str.split(sep).str[-1]
    return out


def _strip_suffixes(s: pd.Series, suffixes: List[str]) -> pd.Series:
    out = _as_str_series(s).copy()
    for suf in suffixes:
        out = out.str.removesuffix(suf)
    return out


def _try_cytospace_pool_map(run: Stage4RunRef, stage4_assignment: pd.DataFrame) -> Optional[Tuple[pd.Series, Dict[str, Any]]]:
    if "unique_cid" not in stage4_assignment.columns:
        return None
    pool_map = run.run_dir / f"cell_pool_map_{run.mode}.csv"
    if not pool_map.exists():
        return None
    df = _read_csv(pool_map)
    required = {"UniqueCID", "OriginalCID"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"cell_pool_map missing columns {sorted(list(missing))}: {pool_map}")
    m = dict(zip(df["UniqueCID"].astype(str), df["OriginalCID"].astype(str)))
    unique_cid = _as_str_series(stage4_assignment["unique_cid"]).astype(str)
    mapped = unique_cid.map(m)
    audit = {
        "strategy": "cytospace_cell_pool_map",
        "pool_map_path": str(pool_map.as_posix()),
        "pool_map_rows": int(len(df)),
        "mapped_na": int(pd.isna(mapped).sum()),
    }
    return _as_str_series(mapped), audit


@dataclass(frozen=True)
class IdPolicy:
    min_stage4_in_truth_fraction: float = 1.0
    min_truth_in_stage4_fraction: float = 1.0
    max_stage4_dup_rate: float = 0.0
    max_truth_dup_rate: float = 0.0


class IdAlignmentError(RuntimeError):
    def __init__(self, message: str, *, audit: Dict[str, Any]):
        super().__init__(message)
        self.audit = audit


def id_normalize_and_validate(
    *,
    run: Stage4RunRef,
    stage4_assignment: pd.DataFrame,
    truth_query: pd.DataFrame,
    policy: IdPolicy,
    enable_id_normalize: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Enforce hard ID alignment between Stage4 outputs and SimGen Query truth.

    If alignment fails and a traceable fix exists, apply id_normalize() and emit audit.
    """
    if "cell_id" not in stage4_assignment.columns:
        raise KeyError("Stage4 cell_assignment missing column: cell_id")
    if "cell_id" not in truth_query.columns:
        raise KeyError("Truth missing column: cell_id")

    s4 = stage4_assignment.copy()
    tr = truth_query.copy()

    s4["cell_id"] = _as_str_series(s4["cell_id"]).str.strip()
    tr["cell_id"] = _as_str_series(tr["cell_id"]).str.strip()
    if "true_spot_id" in tr.columns:
        tr["true_spot_id"] = _as_str_series(tr["true_spot_id"])
        truth_evaluable_mask = ~tr["true_spot_id"].isna()
    else:
        truth_evaluable_mask = pd.Series([True] * len(tr), index=tr.index)
    tr_eval_ids = tr.loc[truth_evaluable_mask, "cell_id"]

    before = {
        "stage4": _id_stats(s4["cell_id"]),
        "truth_all": _id_stats(tr["cell_id"]),
        "truth_evaluable": _id_stats(tr_eval_ids),
        "truth_na_n": int((~truth_evaluable_mask).sum()),
        "truth_na_fraction": float((~truth_evaluable_mask).sum() / max(1, len(tr))),
        "alignment": _alignment_stats(s4["cell_id"], tr_eval_ids),
    }

    def passes(audit_block: Dict[str, Any]) -> bool:
        return (
            audit_block["alignment"]["stage4_in_truth_fraction"] >= policy.min_stage4_in_truth_fraction
            and audit_block["alignment"]["truth_in_stage4_fraction"] >= policy.min_truth_in_stage4_fraction
            and audit_block["stage4"]["dup_rate"] <= policy.max_stage4_dup_rate
            and audit_block["truth_evaluable"]["dup_rate"] <= policy.max_truth_dup_rate
            and audit_block["stage4"]["n_na"] == 0
            and audit_block["truth_all"]["n_na"] == 0
            and audit_block["truth_evaluable"]["n_na"] == 0
        )

    if passes(before):
        return s4, tr, {"applied": False, "before": before, "after": before, "strategy": "identity_strip"}

    if not enable_id_normalize:
        raise IdAlignmentError(
            (
                f"[{run.run_key}] cell_id alignment failed and id_normalize disabled. "
                f"stage4_in_truth_fraction={before['alignment']['stage4_in_truth_fraction']:.4f}, "
                f"truth_in_stage4_fraction={before['alignment']['truth_in_stage4_fraction']:.4f}, "
                f"stage4_dup_rate={before['stage4']['dup_rate']:.4f}, truth_dup_rate={before['truth_evaluable']['dup_rate']:.4f}"
            ),
            audit={"applied": False, "before": before, "after": None, "strategy": None, "candidates": []},
        )

    candidates: List[Tuple[str, pd.Series, pd.Series, Dict[str, Any]]] = []

    # Candidate 1: cyto pool-map (traceable)
    try:
        res = _try_cytospace_pool_map(run, s4)
    except Exception as e:
        res = None
        candidates.append(("cytospace_cell_pool_map_error", s4["cell_id"], tr["cell_id"], {"error": str(e)}))
    if res is not None:
        s4_mapped, audit = res
        candidates.append(("cytospace_cell_pool_map", s4_mapped, tr["cell_id"], audit))

    # Candidate 2: split last token on common separators (heuristic, only accept if it fully passes)
    candidates.append(
        (
            "split_last_token",
            _split_last_token(s4["cell_id"], seps=[":", "|", ";"]),
            tr["cell_id"],
            {"strategy": "split_last_token", "seps": [":", "|", ";"]},
        )
    )

    # Candidate 3: strip common suffixes (heuristic)
    candidates.append(
        (
            "strip_suffixes",
            _strip_suffixes(s4["cell_id"], suffixes=["-1", ".1"]),
            _strip_suffixes(tr["cell_id"], suffixes=["-1", ".1"]),
            {"strategy": "strip_suffixes", "suffixes": ["-1", ".1"]},
        )
    )

    best: Optional[Tuple[Dict[str, Any], pd.Series, pd.Series, str, Dict[str, Any]]] = None
    for name, s4_ids, tr_ids, info in candidates:
        tr_eval = tr_ids.loc[truth_evaluable_mask] if hasattr(tr_ids, "loc") else tr_ids
        blk = {
            "stage4": _id_stats(s4_ids),
            "truth_all": _id_stats(tr_ids),
            "truth_evaluable": _id_stats(tr_eval),
            "truth_na_n": int((~truth_evaluable_mask).sum()),
            "truth_na_fraction": float((~truth_evaluable_mask).sum() / max(1, len(tr_ids))),
            "alignment": _alignment_stats(s4_ids, tr_eval),
        }
        if not passes(blk):
            continue
        # Prefer higher stage4_in_truth_fraction, then higher truth_in_stage4_fraction.
        score = (blk["alignment"]["stage4_in_truth_fraction"], blk["alignment"]["truth_in_stage4_fraction"])
        if best is None:
            best = (blk, s4_ids, tr_ids, name, info)
            continue
        best_score = (best[0]["alignment"]["stage4_in_truth_fraction"], best[0]["alignment"]["truth_in_stage4_fraction"])
        if score > best_score:
            best = (blk, s4_ids, tr_ids, name, info)

    if best is None:
        cand_summaries = []
        for name, s4_ids, tr_ids, info in candidates:
            tr_eval = tr_ids.loc[truth_evaluable_mask] if hasattr(tr_ids, "loc") else tr_ids
            blk = {
                "stage4": _id_stats(s4_ids),
                "truth_all": _id_stats(tr_ids),
                "truth_evaluable": _id_stats(tr_eval),
                "truth_na_n": int((~truth_evaluable_mask).sum()),
                "truth_na_fraction": float((~truth_evaluable_mask).sum() / max(1, len(tr_ids))),
                "alignment": _alignment_stats(s4_ids, tr_eval),
                "strategy": info.get("strategy") if isinstance(info, dict) else name,
            }
            cand_summaries.append({"name": name, "stats": blk, "info": info})

        msg = (
            f"[{run.run_key}] cell_id alignment failed after id_normalize attempts. "
            f"before_stage4_in_truth_fraction={before['alignment']['stage4_in_truth_fraction']:.4f}, "
            f"before_truth_in_stage4_fraction={before['alignment']['truth_in_stage4_fraction']:.4f}, "
            f"before_stage4_dup_rate={before['stage4']['dup_rate']:.4f}, before_truth_dup_rate={before['truth_evaluable']['dup_rate']:.4f}. "
            f"Examples_stage4_only={before['alignment']['stage4_only_examples']} Examples_truth_only={before['alignment']['truth_only_examples']}"
        )
        raise IdAlignmentError(msg, audit={"applied": False, "before": before, "after": None, "strategy": None, "candidates": cand_summaries})

    after_blk, s4_final, tr_final, strategy_name, strategy_info = best
    s4["cell_id"] = _as_str_series(s4_final)
    tr["cell_id"] = _as_str_series(tr_final)
    audit = {
        "applied": True,
        "strategy": strategy_name,
        "strategy_info": strategy_info,
        "before": before,
        "after": after_blk,
    }
    return s4, tr, audit


def _spot_id_to_index(spot_ids: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i, sid in enumerate(spot_ids):
        out[str(sid)] = int(i)
    return out


def _topk_from_row_dense(values: np.ndarray, k: int) -> np.ndarray:
    # Deterministic topK: sort by (-score, spot_index).
    n = int(values.shape[0])
    k = int(max(1, min(k, n)))
    order = np.lexsort((np.arange(n, dtype=int), -values))
    return order[:k]


def evaluate_cell_level(
    *,
    stage4_assignment: pd.DataFrame,
    cell_spot_matrix: sparse.spmatrix,
    truth_query: pd.DataFrame,
    run: Stage4RunRef,
    id_policy: IdPolicy,
    enable_id_normalize: bool,
    spot_ids: Sequence[str],
    topk_list: Sequence[int],
    topk_tie_policy: str,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    required = {"cell_id", "spot_id"}
    missing = required - set(stage4_assignment.columns)
    if missing:
        raise KeyError(f"Stage4 cell_assignment missing columns {sorted(list(missing))}")

    df_raw = stage4_assignment.copy()
    df_raw["cell_id"] = _as_str_series(df_raw["cell_id"])
    df_raw["spot_id"] = _as_str_series(df_raw["spot_id"])

    eval_id_col = "unique_cid"
    eval_id_source = "stage4"
    if eval_id_col in df_raw.columns:
        df_raw[eval_id_col] = _as_str_series(df_raw[eval_id_col])
        if df_raw[eval_id_col].isna().any():
            raise ValueError("Stage4 unique_cid contains NA")
        dup = df_raw[eval_id_col].duplicated().sum()
        if dup:
            raise ValueError(f"Stage4 unique_cid is not unique (duplicated={int(dup)})")
    else:
        eval_id_source = "generated_row_id"
        df_raw = df_raw.reset_index(drop=True)
        df_raw[eval_id_col] = [f"row_{i:06d}" for i in range(len(df_raw))]

    # Hard ID alignment (with optional fixups) before any evaluation join.
    df, truth, id_audit = id_normalize_and_validate(
        run=run,
        stage4_assignment=df_raw,
        truth_query=truth_query,
        policy=id_policy,
        enable_id_normalize=enable_id_normalize,
    )

    truth_ids = set(truth["cell_id"].astype(str).tolist())
    stage4_ids = set(df["cell_id"].astype(str).tolist())
    not_in_truth = sorted(list(stage4_ids - truth_ids))
    truth_missing_in_stage4 = sorted(list(truth_ids - stage4_ids))
    truth_evaluable_mask = ~truth["true_spot_id"].isna()
    n_truth_evaluable = int(truth_evaluable_mask.sum())

    n_rows = int(len(df))
    n_unique_cell_id = int(df["cell_id"].nunique())
    duplicate_cell_id_rate = float(1.0 - (n_unique_cell_id / n_rows)) if n_rows > 0 else 0.0

    truth_key_cols = ["cell_id", "true_spot_id"]
    for opt in ["truth_status", "celltype"]:
        if opt in truth.columns:
            truth_key_cols.append(opt)
    truth_small = truth[truth_key_cols].copy()
    truth_small["cell_id"] = _as_str_series(truth_small["cell_id"])

    merged = df.merge(truth_small, on="cell_id", how="left", validate="many_to_one", indicator=True)
    merged["true_spot_id"] = _as_str_series(merged["true_spot_id"])

    n_missing_truth_row = int((merged["_merge"] == "left_only").sum())
    merged = merged.drop(columns=["_merge"])

    evaluable_mask = ~merged["true_spot_id"].isna()
    n_evaluable = int(evaluable_mask.sum())
    n_na_truth = int((~evaluable_mask).sum())
    na_truth_fraction = float(n_na_truth / n_rows) if n_rows > 0 else 0.0
    evaluable_query_cell_fraction = float(n_evaluable / n_rows) if n_rows > 0 else 0.0

    mat = cell_spot_matrix.tocsr()
    if int(mat.shape[0]) != int(len(df)):
        raise ValueError(f"[{run.run_key}] cell_spot_matrix n_cells={mat.shape[0]} != cell_assignment n_rows={len(df)}")
    if int(mat.shape[1]) != int(len(spot_ids)):
        raise ValueError(f"[{run.run_key}] cell_spot_matrix n_spots={mat.shape[1]} != truth n_spots={len(spot_ids)}")

    topk_list = sorted({int(k) for k in topk_list if int(k) > 0})
    if not topk_list:
        topk_list = [1]
    k_max = int(max(topk_list))
    if topk_tie_policy != "prefer_lowest_spot_index":
        raise ValueError(f"Unsupported topk_tie_policy={topk_tie_policy!r}")

    if n_evaluable > 0:
        pred_top1 = merged.loc[evaluable_mask, "spot_id"].astype(str)
        truth_top1 = merged.loc[evaluable_mask, "true_spot_id"].astype(str)
        correct_top1 = pred_top1 == truth_top1
        acc_top1 = float(correct_top1.mean())
        n_correct_top1 = int(correct_top1.sum())

        spot_to_idx = _spot_id_to_index([str(x) for x in spot_ids])
        eval_idx = merged.index[evaluable_mask].to_numpy(dtype=int)
        truth_spot_ids = truth_top1.tolist()
        hits_by_k = {int(k): 0 for k in topk_list}
        n_missing_truth_spot_in_vocab = 0
        for row_i, true_sid in zip(eval_idx.tolist(), truth_spot_ids):
            col_i = spot_to_idx.get(str(true_sid))
            if col_i is None:
                n_missing_truth_spot_in_vocab += 1
                continue
            row = mat.getrow(int(row_i))
            dense = row.toarray().ravel()
            topk_idx = _topk_from_row_dense(dense, k_max)
            topk_set_prefix = set()
            for idx in topk_idx.tolist():
                topk_set_prefix.add(int(idx))
                for k in topk_list:
                    if len(topk_set_prefix) == int(k) and int(col_i) in topk_set_prefix:
                        hits_by_k[int(k)] += 1
            # If k_max > nnz and dense has many zeros, the deterministic ordering still holds.
        acc_by_k = {f"acc_top{k}": (hits_by_k[int(k)] / max(1, n_evaluable)) for k in topk_list}
    else:
        acc_top1 = None
        n_correct_top1 = 0
        acc_by_k = {f"acc_top{k}": None for k in topk_list}
        n_missing_truth_spot_in_vocab = 0
    precision_mapped = (n_correct_top1 / n_evaluable) if n_evaluable else None
    coverage = (n_evaluable / n_truth_evaluable) if n_truth_evaluable else None
    recall = (n_correct_top1 / n_truth_evaluable) if n_truth_evaluable else None

    audit_after = (id_audit.get("after") or id_audit.get("before") or {}) if isinstance(id_audit, dict) else {}
    align_after = audit_after.get("alignment") or {}
    metrics = {
        "eval_unit": eval_id_col,
        "eval_unit_source": eval_id_source,
        "id_audit_applied": bool(id_audit.get("applied", False)),
        "id_audit_strategy": id_audit.get("strategy"),
        "id_stage4_in_truth_fraction": _safe_float(align_after.get("stage4_in_truth_fraction")),
        "id_truth_in_stage4_fraction": _safe_float(align_after.get("truth_in_stage4_fraction")),
        "id_stage4_dup_rate": _safe_float((audit_after.get("stage4") or {}).get("dup_rate")),
        "id_truth_dup_rate": _safe_float((audit_after.get("truth_evaluable") or {}).get("dup_rate")),
        "id_truth_na_fraction": _safe_float(audit_after.get("truth_na_fraction")),
        "n_rows": n_rows,
        "n_unique_cell_id": n_unique_cell_id,
        "duplicate_cell_id_rate": duplicate_cell_id_rate,
        "cell_id_not_in_truth_count": int(len(not_in_truth)),
        "truth_cells_not_in_stage4_count": int(len(truth_missing_in_stage4)),
        "truth_cells_not_in_stage4_fraction": (float(len(truth_missing_in_stage4) / len(truth_ids)) if len(truth_ids) else 0.0),
        "n_evaluable": n_evaluable,
        "n_truth_evaluable": n_truth_evaluable,
        "n_na_truth": n_na_truth,
        "na_truth_fraction": na_truth_fraction,
        "evaluable_query_cell_fraction": evaluable_query_cell_fraction,
        "n_correct_top1": n_correct_top1,
        "acc_top1": acc_top1,
        "precision_mapped": precision_mapped,
        "coverage": coverage,
        "recall": recall,
        **acc_by_k,
        "topk_source": "cell_spot_matrix",
        "topk_matrix_semantics": {"rows": "cells", "cols": "spots"},
        "topk_tie_policy": topk_tie_policy,
        "n_missing_truth_spot_in_vocab": int(n_missing_truth_spot_in_vocab),
    }
    merged.attrs["id_audit"] = id_audit
    return metrics, merged


def _read_spot_type_fraction_any(path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = _read_csv(path)
    audit: Dict[str, Any] = {"path": str(path.as_posix())}

    # Long format: spot_id, cell_type, fraction
    if {"spot_id", "cell_type", "fraction"}.issubset(df.columns):
        audit["format"] = "long"
        spot_col, type_col, frac_col = "spot_id", "cell_type", "fraction"
        df[spot_col] = _as_str_series(df[spot_col])
        df[type_col] = _as_str_series(df[type_col])
        df[frac_col] = pd.to_numeric(df[frac_col], errors="coerce")
        wide = df.pivot_table(index=spot_col, columns=type_col, values=frac_col, aggfunc="sum", fill_value=0.0)
        wide.index = _as_str_series(wide.index.to_series()).astype(str)
        wide.columns = [str(c) for c in wide.columns]
        return wide, audit

    # Wide format: spot_id + type columns
    if "spot_id" in df.columns:
        audit["format"] = "wide"
        df = df.copy()
        df["spot_id"] = _as_str_series(df["spot_id"])
        df = df.set_index("spot_id")
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.columns = [str(c) for c in df.columns]
        df.index = df.index.astype(str)
        return df, audit

    # Fallback: index is spot_id
    audit["format"] = "index"
    df = pd.read_csv(path, index_col=0)
    df.index = _as_str_series(df.index.to_series()).astype(str)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.columns = [str(c) for c in df.columns]
    return df, audit


def _renorm_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    x = df.copy()
    row_sum = x.sum(axis=1)
    nonzero = row_sum > 0
    audit = {
        "n_rows": int(len(x)),
        "n_nonzero_rows": int(nonzero.sum()),
        "max_row_sum": float(row_sum.max()) if len(row_sum) else None,
        "min_row_sum": float(row_sum.min()) if len(row_sum) else None,
    }
    denom = row_sum.replace(0, np.nan)
    x = x.div(denom, axis=0).fillna(0.0)
    return x, audit


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if eps and float(eps) > 0:
        p = p + float(eps)
        q = q + float(eps)
    ps = p.sum()
    qs = q.sum()
    if ps <= 0 or qs <= 0:
        return 0.0
    p = p / ps
    q = q / qs
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if eps and float(eps) > 0:
        p = p + float(eps)
        q = q + float(eps)
    ps = p.sum()
    qs = q.sum()
    if ps <= 0 or qs <= 0:
        return 0.0
    p = p / ps
    q = q / qs
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m, eps=0.0) + 0.5 * _kl_divergence(q, m, eps=0.0)


def _pearson_corr(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.ndim != 1 or q.ndim != 1 or p.shape != q.shape:
        raise ValueError("corr: p and q must be same-shape 1D arrays")
    if p.size == 0:
        return 0.0
    p0 = p - p.mean()
    q0 = q - q.mean()
    denom = float(np.linalg.norm(p0) * np.linalg.norm(q0))
    if denom <= 0:
        return 1.0 if np.allclose(p, q) else 0.0
    return float(np.dot(p0, q0) / denom)


def evaluate_spot_type_metrics(
    *,
    pred_spot_type: pd.DataFrame,
    truth_spot_type: pd.DataFrame,
    st_coords: pd.DataFrame,
    missing_spot_policy: str,
    renorm_policy: str,
    type_union_policy: str,
    epsilon: float,
    spot_subset_policy: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    truth = truth_spot_type.copy()
    pred = pred_spot_type.copy()
    truth.index = truth.index.astype(str)
    pred.index = pred.index.astype(str)

    missing_spots = sorted(list(set(truth.index) - set(pred.index)))
    extra_spots = sorted(list(set(pred.index) - set(truth.index)))
    audit: Dict[str, Any] = {
        "missing_spot_policy": missing_spot_policy,
        "renorm_policy": renorm_policy,
        "type_union_policy": type_union_policy,
        "epsilon": float(epsilon),
        "spot_subset_policy": spot_subset_policy,
        "n_truth_spots": int(len(truth)),
        "n_pred_spots": int(len(pred)),
        "n_missing_spots": int(len(missing_spots)),
        "n_extra_spots": int(len(extra_spots)),
        "missing_spot_examples": missing_spots[:5],
        "extra_spot_examples": extra_spots[:5],
    }

    if missing_spot_policy not in ("fill_zero", "fail"):
        raise ValueError(f"invalid missing_spot_policy={missing_spot_policy!r}")
    if missing_spots and missing_spot_policy == "fail":
        raise ValueError(f"pred spot_type_fraction missing {len(missing_spots)} truth spots (policy=fail)")
    if missing_spot_policy == "fill_zero":
        pred = pred.reindex(truth.index).fillna(0.0)
    else:
        pred = pred.loc[truth.index.intersection(pred.index)]
        truth = truth.loc[pred.index]

    truth_cols = [str(c) for c in truth.columns]
    pred_cols = [str(c) for c in pred.columns]
    if type_union_policy not in ("union", "intersect"):
        raise ValueError(f"invalid type_union_policy={type_union_policy!r}")
    if type_union_policy == "union":
        cols = truth_cols + [c for c in pred_cols if c not in set(truth_cols)]
    else:
        cols = [c for c in truth_cols if c in set(pred_cols)]
    truth = truth.reindex(columns=cols).fillna(0.0)
    pred = pred.reindex(columns=cols).fillna(0.0)
    audit["n_types_aligned"] = int(len(cols))

    if spot_subset_policy not in ("in_tissue_only", "all"):
        raise ValueError(f"invalid spot_subset_policy={spot_subset_policy!r}")
    if spot_subset_policy == "in_tissue_only":
        if "in_tissue" not in st_coords.columns:
            raise KeyError("sim_st_coordinates missing in_tissue but spot_subset_policy=in_tissue_only")
        keep = st_coords.index[st_coords["in_tissue"].astype(int) > 0].astype(str)
        truth = truth.loc[truth.index.intersection(keep)]
        pred = pred.loc[pred.index.intersection(keep)]

    audit["n_spots_eval"] = int(len(truth))
    if len(truth) == 0:
        return (
            {"n_spots_eval": 0, "L1_mean": None, "L1_median": None, "JS_mean": None, "KL_mean": None, "corr_mean": None},
            audit,
        )

    row_sum_truth = truth.sum(axis=1)
    row_sum_pred = pred.sum(axis=1)
    audit["truth_row_sum_nonzero_fraction"] = float((row_sum_truth > 0).mean()) if len(row_sum_truth) else None
    audit["pred_row_sum_nonzero_fraction"] = float((row_sum_pred > 0).mean()) if len(row_sum_pred) else None
    need_renorm_truth = bool(((row_sum_truth > 0) & (~np.isclose(row_sum_truth, 1.0))).any())
    need_renorm_pred = bool(((row_sum_pred > 0) & (~np.isclose(row_sum_pred, 1.0))).any())

    if renorm_policy not in ("renorm", "fail"):
        raise ValueError(f"invalid renorm_policy={renorm_policy!r}")
    if renorm_policy == "fail" and (need_renorm_truth or need_renorm_pred):
        raise ValueError("spot_type_fraction rows do not sum to 1 (policy=fail)")
    if renorm_policy == "renorm":
        truth, renorm_truth_audit = _renorm_rows(truth)
        pred, renorm_pred_audit = _renorm_rows(pred)
        audit["renorm_truth"] = renorm_truth_audit
        audit["renorm_pred"] = renorm_pred_audit

    truth_np = truth.to_numpy(dtype=float)
    pred_np = pred.to_numpy(dtype=float)
    l1 = np.sum(np.abs(pred_np - truth_np), axis=1)
    js = np.array([_js_divergence(pred_np[i], truth_np[i], eps=epsilon) for i in range(pred_np.shape[0])], dtype=float)
    kl = np.array([_kl_divergence(truth_np[i], pred_np[i], eps=epsilon) for i in range(pred_np.shape[0])], dtype=float)
    corr = np.array([_pearson_corr(pred_np[i], truth_np[i]) for i in range(pred_np.shape[0])], dtype=float)

    out = {
        "n_spots_eval": int(len(truth_np)),
        "L1_mean": float(np.mean(l1)),
        "L1_median": float(np.median(l1)),
        "JS_mean": float(np.mean(js)),
        "KL_mean": float(np.mean(kl)),
        "corr_mean": float(np.mean(corr)),
    }
    return out, audit


def _spatial_entropy(
    spots: Sequence[str],
    st_coords: pd.DataFrame,
    *,
    n_bins_row: int = 10,
    n_bins_col: int = 10,
) -> float:
    if not spots:
        return 0.0
    sub = st_coords.loc[st_coords.index.intersection([str(s) for s in spots])]
    if len(sub) <= 1:
        return 0.0
    row = sub["row"].to_numpy(dtype=float)
    col = sub["col"].to_numpy(dtype=float)
    r_min, r_max = float(np.nanmin(row)), float(np.nanmax(row))
    c_min, c_max = float(np.nanmin(col)), float(np.nanmax(col))
    if not np.isfinite([r_min, r_max, c_min, c_max]).all():
        return 0.0
    r_edges = np.linspace(r_min, r_max + 1e-6, int(max(2, n_bins_row)) + 1)
    c_edges = np.linspace(c_min, c_max + 1e-6, int(max(2, n_bins_col)) + 1)
    r_bin = np.clip(np.digitize(row, r_edges) - 1, 0, len(r_edges) - 2)
    c_bin = np.clip(np.digitize(col, c_edges) - 1, 0, len(c_edges) - 2)
    key = r_bin * (len(c_edges) - 1) + c_bin
    counts = np.bincount(key.astype(int))
    p = counts[counts > 0].astype(float)
    p = p / p.sum()
    h = float(-np.sum(p * np.log(p)))
    h_max = float(np.log(max(1, len(r_edges) - 1) * max(1, len(c_edges) - 1)))
    return float(h / h_max) if h_max > 0 else 0.0


def _build_spot_neighbors(st_coords: pd.DataFrame, k: int) -> Dict[str, List[str]]:
    spots = [str(s) for s in st_coords.index.astype(str).tolist()]
    n = len(spots)
    if n == 0:
        return {}
    k_eff = min(int(k), max(n - 1, 0))
    if k_eff <= 0:
        return {s: [] for s in spots}
    xy = st_coords[["row", "col"]].to_numpy(dtype=float)
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=k_eff + 1)
    if idx.ndim == 1:
        idx = idx.reshape(-1, 1)
    out: Dict[str, List[str]] = {}
    for i, row in enumerate(idx):
        nbrs = [spots[j] for j in row[1:] if j != i]
        out[spots[i]] = nbrs
    return out


def _expand_spot_set(spots: Sequence[str], neighbors: Dict[str, List[str]]) -> List[str]:
    out: set[str] = set()
    for s in spots:
        s = str(s)
        out.add(s)
        out.update([str(x) for x in neighbors.get(s, [])])
    return sorted(out)


def _eval_local_spot_metrics(
    *,
    pred_spot_type: pd.DataFrame,
    truth_spot_type: pd.DataFrame,
    st_coords: pd.DataFrame,
    spot_ids: Sequence[str],
    missing_spot_policy: str,
    renorm_policy: str,
    type_union_policy: str,
    epsilon: float,
    spot_subset_policy: str,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    spots = [str(s) for s in spot_ids if str(s) in set(st_coords.index.astype(str))]
    if not spots:
        return None, {"skip_reason": "no_local_spots"}
    pred_local = pred_spot_type.loc[pred_spot_type.index.intersection(spots)]
    truth_local = truth_spot_type.loc[truth_spot_type.index.intersection(spots)]
    coords_local = st_coords.loc[st_coords.index.intersection(spots)]
    metrics, audit = evaluate_spot_type_metrics(
        pred_spot_type=pred_local,
        truth_spot_type=truth_local,
        st_coords=coords_local,
        missing_spot_policy=missing_spot_policy,
        renorm_policy=renorm_policy,
        type_union_policy=type_union_policy,
        epsilon=epsilon,
        spot_subset_policy=spot_subset_policy,
    )
    return metrics, audit


def _metric_delta(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]], keys: Sequence[str]) -> Optional[Dict[str, Any]]:
    if not a or not b:
        return None
    out: Dict[str, Any] = {}
    for k in keys:
        va = a.get(k)
        vb = b.get(k)
        if va is None or vb is None:
            out[k] = None
        else:
            out[k] = float(va) - float(vb)
    return out


def _read_selected_actions(stage4_meta: Dict[str, Any], run_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    path_raw = stage4_meta.get("selected_actions_path")
    path = None
    if path_raw:
        p = Path(str(path_raw))
        path = p if p.is_absolute() else (run_dir / p)
    else:
        path = run_dir / "selected_actions.csv"
    if path and path.exists():
        return pd.read_csv(path), path
    return None, path


def _read_selected_type_actions(stage4_meta: Dict[str, Any], run_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    path_raw = stage4_meta.get("selected_type_actions_path")
    path = None
    if path_raw:
        p = Path(str(path_raw))
        path = p if p.is_absolute() else (run_dir / p)
    else:
        path = run_dir / "selected_type_actions.csv"
    if path and path.exists():
        return pd.read_csv(path), path
    return None, path


def _build_incremental_report(
    *,
    run: Stage4RunRef,
    stage4_meta: Dict[str, Any],
    selected_actions: Optional[pd.DataFrame],
    pred_spot_type: pd.DataFrame,
    truth_spot_type: pd.DataFrame,
    st_coords: pd.DataFrame,
    baseline_run: Optional[Stage4RunRef],
    missing_spot_policy: str,
    renorm_policy: str,
    type_union_policy: str,
    epsilon: float,
    spot_subset_policy: str,
) -> Dict[str, Any]:
    if run.mode != "plus":
        return {"skip_reason": "mode_not_plus"}
    if selected_actions is None or selected_actions.empty:
        return {"skip_reason": "no_selected_actions"}

    df = selected_actions.copy()
    for col in ["cell_id", "partner_cell_id", "from_spot", "to_spot", "action_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    cell_ids = set(df["cell_id"].dropna().astype(str).tolist()) if "cell_id" in df.columns else set()
    partner_ids = set(df["partner_cell_id"].dropna().astype(str).tolist()) if "partner_cell_id" in df.columns else set()
    changed_cells = cell_ids | partner_ids
    spots_from = set(df["from_spot"].dropna().astype(str).tolist()) if "from_spot" in df.columns else set()
    spots_to = set(df["to_spot"].dropna().astype(str).tolist()) if "to_spot" in df.columns else set()
    changed_spots = spots_from | spots_to

    gainA = pd.to_numeric(df.get("gainA"), errors="coerce") if "gainA" in df.columns else pd.Series([], dtype=float)
    gainA = gainA.dropna()
    gainB = pd.to_numeric(df.get("gainB"), errors="coerce") if "gainB" in df.columns else pd.Series([], dtype=float)
    gainB = gainB.dropna()
    gainSum = pd.to_numeric(df.get("gainSum"), errors="coerce") if "gainSum" in df.columns else pd.Series([], dtype=float)
    gainSum = gainSum.dropna()
    partner_hurt = gainB.copy()
    if len(partner_hurt):
        partner_hurt = partner_hurt.apply(lambda v: max(0.0, -float(v)))
    base_pair_sum = pd.to_numeric(df.get("base_pair_sum"), errors="coerce") if "base_pair_sum" in df.columns else pd.Series([], dtype=float)
    base_pair_sum = base_pair_sum.dropna()

    eps_cfg = None
    cfg_eff = stage4_meta.get("config_effective_subset") or {}
    if "svg_post_base_pair_budget_eps" in cfg_eff:
        eps_cfg = cfg_eff.get("svg_post_base_pair_budget_eps")
    if eps_cfg is None:
        eps_cfg = (stage4_meta.get("svg_post_meta") or {}).get("base_pair_budget_eps")

    report: Dict[str, Any] = {
        "selected_actions_count": int(len(df)),
        "changed_cells_count": int(len(changed_cells)),
        "changed_spots_count": int(len(changed_spots)),
        "gainA_mean": float(gainA.mean()) if len(gainA) else None,
        "gainA_median": float(gainA.median()) if len(gainA) else None,
        "gainA_pos_rate": float((gainA > 0).mean()) if len(gainA) else None,
        "gainA_n": int(len(gainA)),
        "gainB_mean": float(gainB.mean()) if len(gainB) else None,
        "gainB_median": float(gainB.median()) if len(gainB) else None,
        "gainB_neg_rate": float((gainB < 0).mean()) if len(gainB) else None,
        "gainB_n": int(len(gainB)),
        "gainSum_mean": float(gainSum.mean()) if len(gainSum) else None,
        "gainSum_median": float(gainSum.median()) if len(gainSum) else None,
        "gainSum_n": int(len(gainSum)),
        "partner_hurt_sum": float(partner_hurt.sum()) if len(partner_hurt) else None,
        "base_pair_sum_mean": float(base_pair_sum.mean()) if len(base_pair_sum) else None,
        "base_pair_sum_max": float(base_pair_sum.max()) if len(base_pair_sum) else None,
        "base_pair_sum_n": int(len(base_pair_sum)),
        "base_pair_budget_eps": float(eps_cfg) if eps_cfg is not None else None,
        "base_pair_sum_over_eps_count": int((base_pair_sum > float(eps_cfg)).sum()) if (len(base_pair_sum) and eps_cfg is not None) else None,
    }

    swap_mask = df.get("action_type", pd.Series([], dtype=str)).astype(str) == "swap"
    swap_spots = set()
    if "from_spot" in df.columns and "to_spot" in df.columns:
        swap_spots = set(df.loc[swap_mask, "from_spot"].dropna().astype(str).tolist()) | set(
            df.loc[swap_mask, "to_spot"].dropna().astype(str).tolist()
        )
    report["swap_spots_count"] = int(len(swap_spots))

    if swap_spots:
        svg_post_meta = stage4_meta.get("svg_post_meta") or {}
        neighbor_k = int(svg_post_meta.get("neighbor_k") or 10)
        neighbors = _build_spot_neighbors(st_coords, neighbor_k)
        local_spots = _expand_spot_set(sorted(swap_spots), neighbors)
        report["local_neighbor_k"] = int(neighbor_k)
        report["local_spot_set_size"] = int(len(local_spots))

        local_plus_metrics, local_plus_audit = _eval_local_spot_metrics(
            pred_spot_type=pred_spot_type,
            truth_spot_type=truth_spot_type,
            st_coords=st_coords,
            spot_ids=local_spots,
            missing_spot_policy=missing_spot_policy,
            renorm_policy=renorm_policy,
            type_union_policy=type_union_policy,
            epsilon=epsilon,
            spot_subset_policy=spot_subset_policy,
        )
        report["local_spot_metrics_plus"] = local_plus_metrics
        report["local_spot_metrics_plus_audit"] = local_plus_audit

        local_base_metrics = None
        if baseline_run and baseline_run.spot_type_fraction_path.exists():
            base_pred, _ = _read_spot_type_fraction_any(baseline_run.spot_type_fraction_path)
            local_base_metrics, _ = _eval_local_spot_metrics(
                pred_spot_type=base_pred,
                truth_spot_type=truth_spot_type,
                st_coords=st_coords,
                spot_ids=local_spots,
                missing_spot_policy=missing_spot_policy,
                renorm_policy=renorm_policy,
                type_union_policy=type_union_policy,
                epsilon=epsilon,
                spot_subset_policy=spot_subset_policy,
            )
        report["local_spot_metrics_baseline"] = local_base_metrics
        report["local_spot_metrics_delta"] = _metric_delta(
            local_plus_metrics,
            local_base_metrics,
            keys=["L1_mean", "JS_mean", "KL_mean", "corr_mean"],
        )
    else:
        report["local_spot_metrics_plus"] = None
        report["local_spot_metrics_baseline"] = None
        report["local_spot_metrics_delta"] = None
        report["local_spot_set_size"] = 0
        report["local_neighbor_k"] = None
        report["local_skip_reason"] = "no_swap_spots"

    return report


def _build_type_post_report(
    *,
    run: Stage4RunRef,
    stage4_meta: Dict[str, Any],
    selected_actions: Optional[pd.DataFrame],
    pred_spot_type: pd.DataFrame,
    truth_spot_type: pd.DataFrame,
    st_coords: pd.DataFrame,
    baseline_run: Optional[Stage4RunRef],
    missing_spot_policy: str,
    renorm_policy: str,
    type_union_policy: str,
    epsilon: float,
    spot_subset_policy: str,
) -> Dict[str, Any]:
    if run.mode != "plus":
        return {"skip_reason": "mode_not_plus"}
    if selected_actions is None or selected_actions.empty:
        return {"skip_reason": "no_selected_type_actions"}

    df = selected_actions.copy()
    for col in ["cell_id", "partner_cell_id", "from_spot", "to_spot", "action_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    cell_ids = set(df["cell_id"].dropna().astype(str).tolist()) if "cell_id" in df.columns else set()
    partner_ids = set(df["partner_cell_id"].dropna().astype(str).tolist()) if "partner_cell_id" in df.columns else set()
    changed_cells = cell_ids | partner_ids
    spots_from = set(df["from_spot"].dropna().astype(str).tolist()) if "from_spot" in df.columns else set()
    spots_to = set(df["to_spot"].dropna().astype(str).tolist()) if "to_spot" in df.columns else set()
    changed_spots = spots_from | spots_to

    gainA = pd.to_numeric(df.get("gainA"), errors="coerce") if "gainA" in df.columns else pd.Series([], dtype=float)
    gainA = gainA.dropna()
    gainB = pd.to_numeric(df.get("gainB"), errors="coerce") if "gainB" in df.columns else pd.Series([], dtype=float)
    gainB = gainB.dropna()
    gainSum = pd.to_numeric(df.get("gainSum"), errors="coerce") if "gainSum" in df.columns else pd.Series([], dtype=float)
    gainSum = gainSum.dropna()
    partner_hurt = gainB.copy()
    if len(partner_hurt):
        partner_hurt = partner_hurt.apply(lambda v: max(0.0, -float(v)))
    base_pair_sum = pd.to_numeric(df.get("base_pair_sum"), errors="coerce") if "base_pair_sum" in df.columns else pd.Series([], dtype=float)
    base_pair_sum = base_pair_sum.dropna()

    eps_cfg = None
    cfg_eff = stage4_meta.get("config_effective_subset") or {}
    if "type_post_base_pair_budget_eps" in cfg_eff:
        eps_cfg = cfg_eff.get("type_post_base_pair_budget_eps")
    if eps_cfg is None:
        eps_cfg = (stage4_meta.get("type_post_meta") or {}).get("base_pair_budget_eps")

    report: Dict[str, Any] = {
        "selected_actions_count": int(len(df)),
        "changed_cells_count": int(len(changed_cells)),
        "changed_spots_count": int(len(changed_spots)),
        "type_gainA_mean": float(gainA.mean()) if len(gainA) else None,
        "type_gainA_median": float(gainA.median()) if len(gainA) else None,
        "type_gainA_pos_rate": float((gainA > 0).mean()) if len(gainA) else None,
        "type_gainA_n": int(len(gainA)),
        "type_gainB_mean": float(gainB.mean()) if len(gainB) else None,
        "type_gainB_median": float(gainB.median()) if len(gainB) else None,
        "type_gainB_neg_rate": float((gainB < 0).mean()) if len(gainB) else None,
        "type_gainB_n": int(len(gainB)),
        "type_gainSum_mean": float(gainSum.mean()) if len(gainSum) else None,
        "type_gainSum_median": float(gainSum.median()) if len(gainSum) else None,
        "type_gainSum_n": int(len(gainSum)),
        "partner_hurt_type_sum": float(partner_hurt.sum()) if len(partner_hurt) else None,
        "type_base_pair_sum_mean": float(base_pair_sum.mean()) if len(base_pair_sum) else None,
        "type_base_pair_sum_max": float(base_pair_sum.max()) if len(base_pair_sum) else None,
        "type_base_pair_sum_n": int(len(base_pair_sum)),
        "type_base_pair_budget_eps": float(eps_cfg) if eps_cfg is not None else None,
        "type_base_pair_sum_over_eps_count": int((base_pair_sum > float(eps_cfg)).sum())
        if (len(base_pair_sum) and eps_cfg is not None)
        else None,
    }

    swap_mask = df.get("action_type", pd.Series([], dtype=str)).astype(str) == "swap"
    swap_spots = set()
    if "from_spot" in df.columns and "to_spot" in df.columns:
        swap_spots = set(df.loc[swap_mask, "from_spot"].dropna().astype(str).tolist()) | set(
            df.loc[swap_mask, "to_spot"].dropna().astype(str).tolist()
        )
    report["swap_spots_count"] = int(len(swap_spots))

    if swap_spots:
        type_post_meta = stage4_meta.get("type_post_meta") or {}
        neighbor_k = int(type_post_meta.get("neighbor_k") or cfg_eff.get("type_post_neighbor_k") or 10)
        neighbors = _build_spot_neighbors(st_coords, neighbor_k)
        local_spots = _expand_spot_set(sorted(swap_spots), neighbors)
        report["local_neighbor_k"] = int(neighbor_k)
        report["local_spot_set_size"] = int(len(local_spots))

        local_plus_metrics, local_plus_audit = _eval_local_spot_metrics(
            pred_spot_type=pred_spot_type,
            truth_spot_type=truth_spot_type,
            st_coords=st_coords,
            spot_ids=local_spots,
            missing_spot_policy=missing_spot_policy,
            renorm_policy=renorm_policy,
            type_union_policy=type_union_policy,
            epsilon=epsilon,
            spot_subset_policy=spot_subset_policy,
        )
        report["local_spot_metrics_plus"] = local_plus_metrics
        report["local_spot_metrics_plus_audit"] = local_plus_audit

        local_base_metrics = None
        if baseline_run and baseline_run.spot_type_fraction_path.exists():
            base_pred, _ = _read_spot_type_fraction_any(baseline_run.spot_type_fraction_path)
            local_base_metrics, _ = _eval_local_spot_metrics(
                pred_spot_type=base_pred,
                truth_spot_type=truth_spot_type,
                st_coords=st_coords,
                spot_ids=local_spots,
                missing_spot_policy=missing_spot_policy,
                renorm_policy=renorm_policy,
                type_union_policy=type_union_policy,
                epsilon=epsilon,
                spot_subset_policy=spot_subset_policy,
            )
        report["local_spot_metrics_baseline"] = local_base_metrics
        report["local_spot_metrics_delta"] = _metric_delta(
            local_plus_metrics,
            local_base_metrics,
            keys=["L1_mean", "JS_mean", "KL_mean", "corr_mean"],
        )
    else:
        report["local_spot_metrics_plus"] = None
        report["local_spot_metrics_baseline"] = None
        report["local_spot_metrics_delta"] = None
        report["local_spot_set_size"] = 0
        report["local_neighbor_k"] = None
        report["local_skip_reason"] = "no_swap_spots"

    return report


def evaluate_rare_diffusion(
    *,
    pred_spot_type: pd.DataFrame,
    truth_spot_type: pd.DataFrame,
    st_coords: pd.DataFrame,
    rare_type_frac_threshold: float,
    rare_spot_threshold: int,
    pred_nonzero_threshold: float,
) -> Dict[str, Any]:
    truth = truth_spot_type.copy()
    pred = pred_spot_type.copy()
    truth.index = truth.index.astype(str)
    pred.index = pred.index.astype(str)
    pred = pred.reindex(truth.index).fillna(0.0)

    n_spots = max(1, int(len(truth)))
    type_cols = [str(c) for c in truth.columns]
    global_frac = truth.sum(axis=0) / float(n_spots)
    nonzero_spots = (truth > 0).sum(axis=0)

    rare_types = [
        t
        for t in type_cols
        if float(global_frac.get(t, 0.0)) < float(rare_type_frac_threshold) or int(nonzero_spots.get(t, 0)) < int(rare_spot_threshold)
    ]

    fp_counts = []
    fp_entropies = []
    per_type = {}
    for t in rare_types:
        truth0 = truth[t] <= 0
        pred_pos = pred.get(t, 0.0) > float(pred_nonzero_threshold)
        fp_spots = truth.index[truth0 & pred_pos].astype(str).tolist()
        fp_count = int(len(fp_spots))
        ent = _spatial_entropy(fp_spots, st_coords)
        per_type[t] = {"fp_spot_count": fp_count, "fp_spatial_entropy": float(ent)}
        fp_counts.append(fp_count)
        if fp_count > 0:
            fp_entropies.append(ent)

    return {
        "rare_type_frac_threshold": float(rare_type_frac_threshold),
        "rare_spot_threshold": int(rare_spot_threshold),
        "pred_nonzero_threshold": float(pred_nonzero_threshold),
        "n_rare_types": int(len(rare_types)),
        "rare_types": rare_types,
        "rare_fp_spot_count": int(sum(fp_counts)),
        "rare_fp_spatial_entropy_mean": float(np.mean(fp_entropies)) if fp_entropies else 0.0,
        "per_type": per_type,
    }


def evaluate_unknown_behavior(
    *,
    merged_cell_table: pd.DataFrame,
    unknown_labels: Sequence[str],
) -> Dict[str, Any]:
    df = merged_cell_table.copy()
    unknown_set = {str(x) for x in unknown_labels if str(x)}
    type_col = "type" if "type" in df.columns else None
    out: Dict[str, Any] = {"unknown_labels": sorted(list(unknown_set)), "type_column": type_col}
    if type_col is None:
        out["unknown_rate"] = None
        out["missing_truth_to_unknown_rate"] = None
        out["missing_truth_to_nonunknown_rate"] = None
        return out

    df[type_col] = _as_str_series(df[type_col]).astype(str)
    is_unknown = df[type_col].isin(list(unknown_set))
    out["unknown_rate"] = float(is_unknown.mean()) if len(df) else None

    missing_truth = df["true_spot_id"].isna()
    missing_df = df.loc[missing_truth].copy()
    out["missing_truth_n"] = int(len(missing_df))
    if len(missing_df) == 0:
        out["missing_truth_to_unknown_rate"] = None
        out["missing_truth_to_nonunknown_rate"] = None
        out["missing_truth_status_counts"] = {}
        return out

    miss_unknown = missing_df[type_col].isin(list(unknown_set))
    out["missing_truth_to_unknown_rate"] = float(miss_unknown.mean())
    out["missing_truth_to_nonunknown_rate"] = float((~miss_unknown).mean())
    if "truth_status" in df.columns:
        try:
            out["missing_truth_status_counts"] = dict(df.loc[missing_truth, "truth_status"].value_counts(dropna=False))
        except Exception:
            out["missing_truth_status_counts"] = {}
    else:
        out["missing_truth_status_counts"] = {}
    return out

def run_stage5_for_scenario(
    *,
    scenario_id: str,
    project_root: Path,
    backends: List[str],
    topk_list: Sequence[int],
    id_policy: IdPolicy,
    enable_id_normalize: bool,
    missing_spot_policy: str,
    renorm_policy: str,
    type_union_policy: str,
    divergence_epsilon: float,
    spot_subset_policy: str,
    rare_type_frac_threshold: float,
    rare_spot_threshold: int,
    pred_nonzero_threshold: float,
    unknown_labels: Sequence[str],
) -> None:
    sim_dir = project_root / "data" / "sim" / scenario_id
    stage4_root = project_root / "result" / scenario_id / "stage4_mapping"
    out_root = project_root / "result" / scenario_id / "stage5_eval"
    out_generic = out_root / "generic"
    _ensure_dir(out_generic)

    # ---- Global (scenario-level) inputs
    scenario_meta = _read_json(sim_dir / "scenario_meta.json")
    truth_query, truth_query_file_used = load_query_truth(sim_dir)
    truth_spot_type, truth_spot_type_file_used = load_truth_spot_type_fraction(sim_dir)
    st_coords = load_sim_st_coordinates(sim_dir)
    spot_ids = [str(x) for x in st_coords.index.astype(str).tolist()]

    truth_total = int(len(truth_query))
    truth_na = int(truth_query["true_spot_id"].isna().sum())
    truth_evaluable = truth_total - truth_na
    truth_summary = {
        "truth_query_file_used": truth_query_file_used,
        "truth_spot_type_file_used": truth_spot_type_file_used,
        "truth_total": truth_total,
        "truth_na": truth_na,
        "truth_na_fraction": (float(truth_na / truth_total) if truth_total else 0.0),
        "truth_evaluable": truth_evaluable,
        "truth_evaluable_fraction": (float(truth_evaluable / truth_total) if truth_total else 0.0),
    }

    # ---- Stage4 manifest (唯一 run 枚举来源)
    runs_all = load_stage4_runs_from_manifest(project_root=project_root, scenario_id=scenario_id, backends=backends)
    if not runs_all:
        raise FileNotFoundError(f"[Stage5] run_list is empty (scenario={scenario_id}, backends={backends}); terminate per v1.2.1")

    baseline_by_key: Dict[Tuple[str, Optional[str], Optional[int]], Stage4RunRef] = {}
    for r in runs_all:
        if str(r.mode).lower() == "baseline":
            baseline_by_key[(r.backend, r.config_id, r.seed)] = r

    evaluator_meta = {
        "generated_at": _now_iso(),
        "evaluator_file": str(Path(__file__).resolve().as_posix()),
        "evaluator_sha1": _sha1_file(Path(__file__).resolve()),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    metric_definition = {
        "gate_status_priority": ["missing_input", "schema_fail", "id_gate_fail", "metric_fail", "pass"],
        "id_gate": {
            "min_stage4_in_truth_fraction": id_policy.min_stage4_in_truth_fraction,
            "min_truth_in_stage4_fraction": id_policy.min_truth_in_stage4_fraction,
            "max_stage4_dup_rate": id_policy.max_stage4_dup_rate,
            "max_truth_dup_rate": id_policy.max_truth_dup_rate,
            "enable_id_normalize": bool(enable_id_normalize),
        },
        "cell_level": {
            "truth_denominator": "truth_query.true_spot_id != NA",
            "topk_list": [int(k) for k in sorted({int(x) for x in topk_list if int(x) > 0})],
            "topk_source": "cell_spot_matrix",
            "matrix_semantics": {"rows": "cells", "cols": "spots"},
            "topk_tie_policy": "prefer_lowest_spot_index",
        },
        "spot_type": {
            "missing_spot_policy": missing_spot_policy,
            "renorm_policy": renorm_policy,
            "type_union_policy": type_union_policy,
            "epsilon": float(divergence_epsilon),
            "spot_subset_policy": spot_subset_policy,
        },
        "rare_diffusion": {
            "rare_type_frac_threshold": float(rare_type_frac_threshold),
            "rare_spot_threshold": int(rare_spot_threshold),
            "pred_nonzero_threshold": float(pred_nonzero_threshold),
        },
        "unknown_behavior": {"unknown_labels": [str(x) for x in unknown_labels if str(x)]},
    }

    by_backend: Dict[str, List[Stage4RunRef]] = {}
    for r in runs_all:
        by_backend.setdefault(r.backend, []).append(r)

    scenario_run_rows: List[Dict[str, Any]] = []

    for backend, runs in sorted(by_backend.items(), key=lambda kv: kv[0].lower()):
        run_rows: List[Dict[str, Any]] = []
        metrics_by_run: Dict[str, Any] = {}

        for run in runs:
            run_uid = _run_uid(run)
            run_out = out_root / f"run_{_sanitize_token(run_uid)}"
            _ensure_dir(run_out)

            stage5_meta = {
                "stage": 5,
                "scenario_id": scenario_id,
                "gate_status": None,
                "fail_reason": None,
                "input_cell_assignment_sha1": _sha1_file(run.cell_assignment_path) if run.cell_assignment_path.exists() else None,
                "run_provenance": {
                    "scenario_id": run.scenario_id,
                    "backend": run.backend,
                    "mode": run.mode,
                    "run_id": run.run_id,
                    "run_uid": run_uid,
                    "config_id": run.config_id,
                    "seed": run.seed,
                    "lambda_svg": run.lambda_svg,
                    "refine_enabled": run.refine_enabled,
                    "run_dir": str(run.run_dir.as_posix()),
                    "meta_path": str(run.meta_path.as_posix()) if run.meta_path else None,
                    "cell_assignment_path": str(run.cell_assignment_path.as_posix()),
                    "cell_spot_matrix_path": str(run.cell_spot_matrix_path.as_posix()),
                    "spot_type_fraction_path": str(run.spot_type_fraction_path.as_posix()),
                },
                "evaluator_provenance": evaluator_meta,
                "metric_definition": metric_definition,
            }

            fingerprint = {
                "inputs": [
                    {
                        "name": "cell_assignment",
                        "path": str(run.cell_assignment_path.as_posix()),
                        "exists": run.cell_assignment_path.exists(),
                        "size_bytes": (run.cell_assignment_path.stat().st_size if run.cell_assignment_path.exists() else None),
                        "sha1": _sha1_file(run.cell_assignment_path) if run.cell_assignment_path.exists() else None,
                    },
                    {
                        "name": "cell_spot_matrix",
                        "path": str(run.cell_spot_matrix_path.as_posix()),
                        "exists": run.cell_spot_matrix_path.exists(),
                        "size_bytes": (run.cell_spot_matrix_path.stat().st_size if run.cell_spot_matrix_path.exists() else None),
                        "sha1": _sha1_file(run.cell_spot_matrix_path) if run.cell_spot_matrix_path.exists() else None,
                    },
                    {
                        "name": "spot_type_fraction",
                        "path": str(run.spot_type_fraction_path.as_posix()),
                        "exists": run.spot_type_fraction_path.exists(),
                        "size_bytes": (run.spot_type_fraction_path.stat().st_size if run.spot_type_fraction_path.exists() else None),
                        "sha1": _sha1_file(run.spot_type_fraction_path) if run.spot_type_fraction_path.exists() else None,
                    },
                    {"name": "truth_query", "path": str((sim_dir / truth_query_file_used).as_posix()), "exists": True, "size_bytes": (sim_dir / truth_query_file_used).stat().st_size, "sha1": _sha1_file(sim_dir / truth_query_file_used)},
                    {"name": "truth_spot_type", "path": str((sim_dir / truth_spot_type_file_used).as_posix()), "exists": True, "size_bytes": (sim_dir / truth_spot_type_file_used).stat().st_size, "sha1": _sha1_file(sim_dir / truth_spot_type_file_used)},
                    {"name": "sim_st_coordinates", "path": str((sim_dir / "sim_st_coordinates.csv").as_posix()), "exists": True, "size_bytes": (sim_dir / "sim_st_coordinates.csv").stat().st_size, "sha1": _sha1_file(sim_dir / "sim_st_coordinates.csv")},
                ]
            }
            _json_write(run_out / "input_fingerprint.json", fingerprint)

            gate_status = "pass"
            fail_reason = None
            schema_audit: Dict[str, Any] = {
                "run_key": run.run_key,
                "generated_at": _now_iso(),
                "exists": {
                    "cell_assignment": bool(run.cell_assignment_path.exists()),
                    "cell_spot_matrix": bool(run.cell_spot_matrix_path.exists()),
                    "spot_type_fraction": bool(run.spot_type_fraction_path.exists()),
                },
            }
            id_audit: Dict[str, Any] = {}
            metrics_generic: Dict[str, Any] = {}

            missing_inputs = [k for k, v in schema_audit["exists"].items() if not v]
            if missing_inputs:
                gate_status = "missing_input"
                fail_reason = f"missing_inputs={','.join(missing_inputs)}"
                _json_write(run_out / "schema_audit.json", schema_audit)
                _json_write(run_out / "id_audit.json", id_audit)
                _json_write(run_out / "metrics_generic.json", {"gate_status": gate_status, "fail_reason": fail_reason})
                stage5_meta["gate_status"] = gate_status
                stage5_meta["fail_reason"] = fail_reason
                _json_write(run_out / "stage5_meta.json", stage5_meta)
                row = {
                    "scenario_id": scenario_id,
                    "backend": backend,
                    "mode": run.mode,
                    "run_id": run.run_id,
                    "run_uid": run_uid,
                    "config_id": run.config_id,
                    "seed": run.seed,
                    "lambda_svg": run.lambda_svg,
                    "refine_enabled": run.refine_enabled,
                    "gate_status": gate_status,
                    "fail_reason": fail_reason,
                }
                run_rows.append(row)
                scenario_run_rows.append(row)
                metrics_by_run[run_uid] = row
                continue

            try:
                ca = _read_csv(run.cell_assignment_path)
                schema_audit["cell_assignment"] = {
                    "n_rows": int(len(ca)),
                    "columns": list(ca.columns),
                    "required_columns": ["cell_id", "spot_id"],
                    "missing_required": sorted(list({"cell_id", "spot_id"} - set(ca.columns))),
                }
                if schema_audit["cell_assignment"]["missing_required"]:
                    raise KeyError(f"cell_assignment missing columns {schema_audit['cell_assignment']['missing_required']}")
                ca["cell_id"] = _as_str_series(ca["cell_id"])
                ca["spot_id"] = _as_str_series(ca["spot_id"])
                if "type" in ca.columns:
                    ca["type"] = _as_str_series(ca["type"])

                mat = sparse.load_npz(run.cell_spot_matrix_path)
                schema_audit["cell_spot_matrix"] = {"shape": [int(mat.shape[0]), int(mat.shape[1])], "nnz": int(mat.nnz)}
                if int(mat.shape[0]) != int(len(ca)):
                    raise ValueError(f"cell_spot_matrix n_cells={mat.shape[0]} != cell_assignment n_rows={len(ca)}")
                if int(mat.shape[1]) != int(len(spot_ids)):
                    raise ValueError(f"cell_spot_matrix n_spots={mat.shape[1]} != sim spots={len(spot_ids)}")

                pred_spot_type, pred_spot_type_audit = _read_spot_type_fraction_any(run.spot_type_fraction_path)
                schema_audit["spot_type_fraction"] = {
                    **pred_spot_type_audit,
                    "n_spots": int(len(pred_spot_type)),
                    "n_types": int(len(pred_spot_type.columns)),
                }

                stage4_meta = _read_json(run.meta_path) if run.meta_path and run.meta_path.exists() else {}
                selected_actions, _ = _read_selected_actions(stage4_meta, run.run_dir)
                selected_type_actions, _ = _read_selected_type_actions(stage4_meta, run.run_dir)
                baseline_run = baseline_by_key.get((backend, run.config_id, run.seed))
                incremental_report = _build_incremental_report(
                    run=run,
                    stage4_meta=stage4_meta,
                    selected_actions=selected_actions,
                    pred_spot_type=pred_spot_type,
                    truth_spot_type=truth_spot_type,
                    st_coords=st_coords,
                    baseline_run=baseline_run,
                    missing_spot_policy=missing_spot_policy,
                    renorm_policy=renorm_policy,
                    type_union_policy=type_union_policy,
                    epsilon=divergence_epsilon,
                    spot_subset_policy=spot_subset_policy,
                )
                type_post_report = _build_type_post_report(
                    run=run,
                    stage4_meta=stage4_meta,
                    selected_actions=selected_type_actions,
                    pred_spot_type=pred_spot_type,
                    truth_spot_type=truth_spot_type,
                    st_coords=st_coords,
                    baseline_run=baseline_run,
                    missing_spot_policy=missing_spot_policy,
                    renorm_policy=renorm_policy,
                    type_union_policy=type_union_policy,
                    epsilon=divergence_epsilon,
                    spot_subset_policy=spot_subset_policy,
                )
                if isinstance(incremental_report, dict):
                    incremental_report["type_post_report"] = type_post_report

                bad_spots = sorted(list(set(ca["spot_id"].dropna().astype(str)) - set(spot_ids)))
                schema_audit["cell_assignment"]["n_bad_spot_ids"] = int(len(bad_spots))
                schema_audit["cell_assignment"]["bad_spot_id_examples"] = bad_spots[:5]
                if bad_spots:
                    raise ValueError(f"cell_assignment contains spot_id not in truth spots (examples={bad_spots[:3]})")

            except Exception as e:
                gate_status = "schema_fail"
                fail_reason = f"{type(e).__name__}: {e}"
                _json_write(run_out / "schema_audit.json", schema_audit)
                _json_write(run_out / "id_audit.json", id_audit)
                _json_write(run_out / "metrics_generic.json", {"gate_status": gate_status, "fail_reason": fail_reason})
                _json_write(run_out / "error.json", {"error": str(e), "traceback": traceback.format_exc()})
                stage5_meta["gate_status"] = gate_status
                stage5_meta["fail_reason"] = fail_reason
                _json_write(run_out / "stage5_meta.json", stage5_meta)
                row = {
                    "scenario_id": scenario_id,
                    "backend": backend,
                    "mode": run.mode,
                    "run_id": run.run_id,
                    "run_uid": run_uid,
                    "config_id": run.config_id,
                    "seed": run.seed,
                    "lambda_svg": run.lambda_svg,
                    "refine_enabled": run.refine_enabled,
                    "gate_status": gate_status,
                    "fail_reason": fail_reason,
                }
                run_rows.append(row)
                scenario_run_rows.append(row)
                metrics_by_run[run_uid] = row
                continue

            try:
                cell_metrics, merged = evaluate_cell_level(
                    stage4_assignment=ca,
                    cell_spot_matrix=mat,
                    truth_query=truth_query,
                    run=run,
                    id_policy=id_policy,
                    enable_id_normalize=enable_id_normalize,
                    spot_ids=spot_ids,
                    topk_list=topk_list,
                    topk_tie_policy="prefer_lowest_spot_index",
                )
                id_audit = merged.attrs.get("id_audit", {}) or {}

                spot_type_metrics, spot_type_audit = evaluate_spot_type_metrics(
                    pred_spot_type=pred_spot_type,
                    truth_spot_type=truth_spot_type,
                    st_coords=st_coords,
                    missing_spot_policy=missing_spot_policy,
                    renorm_policy=renorm_policy,
                    type_union_policy=type_union_policy,
                    epsilon=divergence_epsilon,
                    spot_subset_policy=spot_subset_policy,
                )
                rare_metrics = evaluate_rare_diffusion(
                    pred_spot_type=pred_spot_type,
                    truth_spot_type=truth_spot_type,
                    st_coords=st_coords,
                    rare_type_frac_threshold=rare_type_frac_threshold,
                    rare_spot_threshold=rare_spot_threshold,
                    pred_nonzero_threshold=pred_nonzero_threshold,
                )
                unknown_metrics = evaluate_unknown_behavior(merged_cell_table=merged, unknown_labels=unknown_labels)

                metrics_generic = {
                    "scenario_id": scenario_id,
                    "backend": backend,
                    "mode": run.mode,
                    "run_id": run.run_id,
                    "run_uid": run_uid,
                    "config_id": run.config_id,
                    "seed": run.seed,
                    "lambda_svg": run.lambda_svg,
                    "refine_enabled": run.refine_enabled,
                    "gate_status": "pass",
                    "fail_reason": None,
                    "truth_summary": truth_summary,
                    "cell_level": cell_metrics,
                    "spot_type": spot_type_metrics,
                    "spot_type_audit": spot_type_audit,
                    "rare_diffusion": rare_metrics,
                    "unknown_behavior": unknown_metrics,
                    "incremental_report": incremental_report,
                }

            except IdAlignmentError as e:
                gate_status = "id_gate_fail"
                fail_reason = f"{type(e).__name__}: {e}"
                id_audit = {"error": str(e), **(e.audit or {})}
                metrics_generic = {"gate_status": gate_status, "fail_reason": fail_reason}
            except Exception as e:
                gate_status = "metric_fail"
                fail_reason = f"{type(e).__name__}: {e}"
                metrics_generic = {"gate_status": gate_status, "fail_reason": fail_reason}
                _json_write(run_out / "error.json", {"error": str(e), "traceback": traceback.format_exc()})

            stage5_meta["gate_status"] = gate_status
            stage5_meta["fail_reason"] = fail_reason
            _json_write(run_out / "schema_audit.json", schema_audit)
            _json_write(run_out / "id_audit.json", id_audit)
            _json_write(run_out / "metrics_generic.json", metrics_generic)
            _json_write(run_out / "stage5_meta.json", stage5_meta)

            meta = _read_json(run.meta_path) if run.meta_path and run.meta_path.exists() else {}
            row: Dict[str, Any] = {
                "scenario_id": scenario_id,
                "backend": backend,
                "mode": run.mode,
                "run_id": run.run_id,
                "run_uid": run_uid,
                "config_id": run.config_id,
                "seed": run.seed,
                "lambda_svg": run.lambda_svg,
                "refine_enabled": run.refine_enabled,
                "gate_status": gate_status,
                "fail_reason": fail_reason,
            }
            if isinstance(metrics_generic.get("cell_level"), dict):
                row.update(metrics_generic["cell_level"])
            if isinstance(metrics_generic.get("spot_type"), dict):
                row.update(metrics_generic["spot_type"])
            if isinstance(metrics_generic.get("rare_diffusion"), dict):
                row["rare_fp_spot_count"] = metrics_generic["rare_diffusion"].get("rare_fp_spot_count")
                row["rare_fp_spatial_entropy"] = metrics_generic["rare_diffusion"].get("rare_fp_spatial_entropy_mean")
            if isinstance(metrics_generic.get("unknown_behavior"), dict):
                row["unknown_rate"] = metrics_generic["unknown_behavior"].get("unknown_rate")
                row["missing_truth_to_unknown_rate"] = metrics_generic["unknown_behavior"].get("missing_truth_to_unknown_rate")
            if isinstance(metrics_generic.get("incremental_report"), dict):
                inc = metrics_generic["incremental_report"]
                row["partner_hurt_sum"] = inc.get("partner_hurt_sum")
                row["gainA_mean"] = inc.get("gainA_mean")
                type_post_report = inc.get("type_post_report") if isinstance(inc.get("type_post_report"), dict) else None
                if type_post_report:
                    row["partner_hurt_type_sum"] = type_post_report.get("partner_hurt_type_sum")
                    row["type_gainA_mean"] = type_post_report.get("type_gainA_mean")
            if meta:
                row["stage4_meta_status"] = meta.get("status")
                row["cell_id_space"] = meta.get("cell_id_space")
                row["cell_instance_id_column"] = meta.get("cell_instance_id_column")
            run_rows.append(row)
            scenario_run_rows.append(row)
            metrics_by_run[run_uid] = row

        out_json = out_generic / f"metrics_{backend}.json"
        payload = {
            "scenario_id": scenario_id,
            "backend": backend,
            "sim_dir": str(sim_dir.as_posix()),
            "stage4_root": str(stage4_root.as_posix()),
            "stage4_manifest_used": str((project_root / "result" / scenario_id / "stage4_run_manifest.json").as_posix()),
            "out_dir": str(out_root.as_posix()),
            "scenario_meta": scenario_meta,
            "truth_summary": truth_summary,
            "metric_definition": metric_definition,
            "runs": metrics_by_run,
        }
        _json_write(out_json, payload)

        out_csv = out_generic / f"metrics_{backend}_runs.csv"
        pd.DataFrame(run_rows).sort_values(
            ["backend", "mode", "run_id", "config_id", "seed", "lambda_svg", "refine_enabled", "run_uid"]
        ).to_csv(out_csv, index=False)

        print(f"[Stage5] Wrote: {out_json}")
        print(f"[Stage5] Wrote: {out_csv}")

    # Scenario-level consolidated run_table (all backends)
    if scenario_run_rows:
        out_table = out_generic / "stage5_run_table.csv"
        pd.DataFrame(scenario_run_rows).sort_values(
            ["backend", "mode", "run_id", "config_id", "seed", "lambda_svg", "refine_enabled", "run_uid"]
        ).to_csv(out_table, index=False)
        print(f"[Stage5] Wrote: {out_table}")


def _stage5_run_dir(
    project_root: Path,
    scenario_id: str,
    backend: str,
    mode: str,
    run_id: str,
    *,
    run_uid: Optional[str] = None,
) -> Path:
    if run_uid:
        return project_root / "result" / str(scenario_id) / "stage5_eval" / f"run_{_sanitize_token(run_uid)}"
    # Backward-compatible path (older Stage5 versions).
    return (
        project_root
        / "result"
        / str(scenario_id)
        / "stage5_eval"
        / f"run_{_sanitize_token(backend)}__{_sanitize_token(mode)}__{_sanitize_token(run_id)}"
    )


def _read_stage5_meta(
    project_root: Path,
    scenario_id: str,
    backend: str,
    mode: str,
    run_id: str,
    *,
    run_uid: Optional[str] = None,
) -> Dict[str, Any]:
    p = _stage5_run_dir(project_root, scenario_id, backend, mode, run_id, run_uid=run_uid) / "stage5_meta.json"
    return _read_json(p) if p.exists() else {}


def _as_abs_path_from_meta(project_root: Path, path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p
    return project_root / p


def _cell_assignment_overlap(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[Optional[float], int]:
    if "cell_id" not in a.columns or "spot_id" not in a.columns:
        return None, 0
    if "cell_id" not in b.columns or "spot_id" not in b.columns:
        return None, 0
    aa = a[["cell_id", "spot_id"]].copy()
    bb = b[["cell_id", "spot_id"]].copy()
    aa["cell_id"] = _as_str_series(aa["cell_id"]).astype(str)
    bb["cell_id"] = _as_str_series(bb["cell_id"]).astype(str)
    aa["spot_id"] = _as_str_series(aa["spot_id"]).astype(str)
    bb["spot_id"] = _as_str_series(bb["spot_id"]).astype(str)
    m = aa.merge(bb, on="cell_id", how="inner", suffixes=("_a", "_b"))
    if len(m) == 0:
        return None, 0
    return float((m["spot_id_a"] == m["spot_id_b"]).mean()), int(len(m))


def summarize_stage5(
    *,
    project_root: Path,
    scenario_ids: Sequence[str],
    backends: Sequence[str],
) -> None:
    out_summary = project_root / "result" / "stage5_summary"
    _ensure_dir(out_summary)

    # Gather run_table rows
    tables = []
    for sid in scenario_ids:
        p = project_root / "result" / str(sid) / "stage5_eval" / "generic" / "stage5_run_table.csv"
        if p.exists():
            tables.append(pd.read_csv(p))
    if not tables:
        raise FileNotFoundError("No stage5_run_table.csv found for requested scenarios; did Stage5 run successfully?")
    run_table = pd.concat(tables, ignore_index=True)
    out_table = out_summary / "stage5_run_table.csv"
    run_table.to_csv(out_table, index=False)

    # Scenario summary
    scenario_summary: Dict[str, Any] = {"generated_at": _now_iso(), "scenarios": {}}

    # Group-level stability + recommendation candidates
    rec_candidates: Dict[str, List[Dict[str, Any]]] = {b: [] for b in backends}

    for sid in sorted({str(x) for x in run_table["scenario_id"].astype(str).unique().tolist()}):
        scenario_summary["scenarios"][sid] = {"backends": {}}
        for b in backends:
            df_sb = run_table[(run_table["scenario_id"].astype(str) == sid) & (run_table["backend"].astype(str) == b)].copy()
            if df_sb.empty:
                continue

            def _agg_metrics(df: pd.DataFrame) -> Dict[str, Any]:
                return {
                    "n_runs": int(len(df)),
                    "acc_top1_mean": _safe_float(df.get("acc_top1").mean()) if "acc_top1" in df.columns else None,
                    "acc_top1_std": _safe_float(df.get("acc_top1").std(ddof=0)) if "acc_top1" in df.columns else None,
                    "JS_mean_mean": _safe_float(df.get("JS_mean").mean()) if "JS_mean" in df.columns else None,
                    "JS_mean_std": _safe_float(df.get("JS_mean").std(ddof=0)) if "JS_mean" in df.columns else None,
                    "L1_mean_mean": _safe_float(df.get("L1_mean").mean()) if "L1_mean" in df.columns else None,
                    "rare_fp_spot_count_mean": _safe_float(df.get("rare_fp_spot_count").mean()) if "rare_fp_spot_count" in df.columns else None,
                    "unknown_rate_mean": _safe_float(df.get("unknown_rate").mean()) if "unknown_rate" in df.columns else None,
                }

            base_pass = df_sb[(df_sb["mode"].astype(str) == "baseline") & (df_sb["gate_status"].astype(str) == "pass")].copy()
            plus_pass = df_sb[(df_sb["mode"].astype(str) == "plus") & (df_sb["gate_status"].astype(str) == "pass")].copy()

            base_summary = _agg_metrics(base_pass) if len(base_pass) else {"n_runs": 0}
            scenario_summary["scenarios"][sid]["backends"][b] = {
                "baseline": base_summary,
                "plus_best_group": None,
                "status": None,
            }

            if len(plus_pass) == 0:
                scenario_summary["scenarios"][sid]["backends"][b]["status"] = "no_plus_pass"
                continue
            if len(base_pass) == 0:
                scenario_summary["scenarios"][sid]["backends"][b]["status"] = "no_baseline_pass"
                continue

            # Group by config keys (seed varies inside group)
            group_cols = ["config_id", "lambda_svg", "refine_enabled"]
            plus_groups = []
            for (cfg_id, lam, ref), g in plus_pass.groupby(group_cols, dropna=False):
                gsum = _agg_metrics(g)
                gsum.update({"config_id": cfg_id, "lambda_svg": lam, "refine_enabled": ref})

                # Stability: pairwise assignment overlap (if >=2 runs)
                overlap_vals = []
                n_pairs = 0
                n_intersections = []
                use_uid = "run_uid" in g.columns
                run_ids = g["run_uid"].astype(str).tolist() if use_uid else g["run_id"].astype(str).tolist()
                for i in range(len(run_ids)):
                    for j in range(i + 1, len(run_ids)):
                        mi = _read_stage5_meta(project_root, sid, b, "plus", run_ids[i], run_uid=(run_ids[i] if use_uid else None))
                        mj = _read_stage5_meta(project_root, sid, b, "plus", run_ids[j], run_uid=(run_ids[j] if use_uid else None))
                        ca_i = _as_abs_path_from_meta(project_root, (mi.get("run_provenance") or {}).get("cell_assignment_path"))
                        ca_j = _as_abs_path_from_meta(project_root, (mj.get("run_provenance") or {}).get("cell_assignment_path"))
                        if not ca_i or not ca_j or not ca_i.exists() or not ca_j.exists():
                            continue
                        a = pd.read_csv(ca_i)
                        c = pd.read_csv(ca_j)
                        ov, n_int = _cell_assignment_overlap(a, c)
                        if ov is None:
                            continue
                        overlap_vals.append(float(ov))
                        n_intersections.append(int(n_int))
                        n_pairs += 1
                gsum["stability_pair_count"] = int(n_pairs)
                gsum["stability_overlap_mean"] = float(np.mean(overlap_vals)) if overlap_vals else None
                gsum["stability_overlap_std"] = float(np.std(overlap_vals)) if overlap_vals else None
                gsum["stability_overlap_min"] = float(np.min(overlap_vals)) if overlap_vals else None
                gsum["stability_overlap_n_intersection_mean"] = float(np.mean(n_intersections)) if n_intersections else None
                plus_groups.append(gsum)

            # Choose best group (simple lexicographic rule)
            def _rank_key(x: Dict[str, Any]) -> Tuple[float, float]:
                acc = _safe_float(x.get("acc_top1_mean")) or -1.0
                js = _safe_float(x.get("JS_mean_mean")) or 1e9
                return (acc, -js)  # maximize acc, then minimize JS

            best = sorted(plus_groups, key=_rank_key, reverse=True)[0]
            scenario_summary["scenarios"][sid]["backends"][b]["plus_best_group"] = best
            scenario_summary["scenarios"][sid]["backends"][b]["status"] = "ok"

            # Candidate for global recommendation
            rec_candidates.setdefault(b, []).append({"scenario_id": sid, **best})

    out_scenario = out_summary / "stage5_scenario_summary.json"
    _json_write(out_scenario, scenario_summary)

    # Config recommendation (rule-based; keep it auditable)
    recommendations: Dict[str, Any] = {"generated_at": _now_iso(), "backends": {}}
    for b in backends:
        cands = rec_candidates.get(b) or []
        if not cands:
            recommendations["backends"][b] = {"status": "no_candidates", "top_groups": [], "lambda_range": None}
            continue
        # Aggregate by (config_id, lambda_svg, refine_enabled)
        dfc = pd.DataFrame(cands)
        key_cols = ["config_id", "lambda_svg", "refine_enabled"]
        agg = (
            dfc.groupby(key_cols, dropna=False)
            .agg(
                n_scenarios=("scenario_id", "nunique"),
                acc_top1_mean=("acc_top1_mean", "mean"),
                JS_mean_mean=("JS_mean_mean", "mean"),
                stability_overlap_mean=("stability_overlap_mean", "mean"),
            )
            .reset_index()
        )
        agg = agg.sort_values(["acc_top1_mean", "JS_mean_mean"], ascending=[False, True])
        top_groups = agg.head(10).to_dict(orient="records")
        lam_vals = [v for v in agg["lambda_svg"].tolist() if v is not None and not (isinstance(v, float) and np.isnan(v))]
        lam_range = {"min": float(np.min(lam_vals)), "max": float(np.max(lam_vals))} if lam_vals else None
        recommendations["backends"][b] = {"status": "ok", "top_groups": top_groups, "lambda_range": lam_range}

    out_rec = out_summary / "stage5_config_recommendation.json"
    _json_write(out_rec, recommendations)

    meta = {"generated_at": _now_iso(), "scenarios": list(scenario_ids), "backends": list(backends)}
    _json_write(out_summary / "stage5_summary_meta.json", meta)

    print(f"[Stage5] Wrote: {out_table}")
    print(f"[Stage5] Wrote: {out_scenario}")
    print(f"[Stage5] Wrote: {out_rec}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage5: evaluate simulated scenarios using SimGen truth")
    p.add_argument("--sample", required=True, help="scenario_id, e.g. S0_matched / M1_sc_missing_Bcell")
    p.add_argument("--project_root", default=None)
    p.add_argument("--backends", default="cytospace", help="comma-separated backends (default: cytospace)")
    p.add_argument("--topk", default="1,3,5", help="comma-separated topK list for cell-level accuracy")
    p.add_argument("--disable_id_normalize", action="store_true", default=False, help="Disable auto id_normalize() fixups")
    p.add_argument("--min_stage4_in_truth_fraction", type=float, default=0.995)
    p.add_argument("--min_truth_in_stage4_fraction", type=float, default=0.995)
    p.add_argument("--max_stage4_dup_rate", type=float, default=0.0)
    p.add_argument("--max_truth_dup_rate", type=float, default=0.0)
    p.add_argument("--missing_spot_policy", default="fill_zero", choices=["fill_zero", "fail"])
    p.add_argument("--renorm_policy", default="renorm", choices=["renorm", "fail"])
    p.add_argument("--type_union_policy", default="union", choices=["union", "intersect"])
    p.add_argument("--divergence_epsilon", type=float, default=1e-8)
    p.add_argument("--spot_subset_policy", default="in_tissue_only", choices=["in_tissue_only", "all"])
    p.add_argument("--rare_type_frac_threshold", type=float, default=0.05)
    p.add_argument("--rare_spot_threshold", type=int, default=10)
    p.add_argument("--pred_nonzero_threshold", type=float, default=0.01)
    p.add_argument(
        "--unknown_labels",
        default="Unknown_sc_only,Unknown",
        help="comma-separated Unknown labels (binds to Stage3 plugin_type conventions)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root) if args.project_root else detect_project_root()
    backends = [b.strip() for b in str(args.backends).split(",") if b.strip()]
    topk_list = [int(x) for x in str(args.topk).split(",") if str(x).strip()]
    id_policy = IdPolicy(
        min_stage4_in_truth_fraction=float(args.min_stage4_in_truth_fraction),
        min_truth_in_stage4_fraction=float(args.min_truth_in_stage4_fraction),
        max_stage4_dup_rate=float(args.max_stage4_dup_rate),
        max_truth_dup_rate=float(args.max_truth_dup_rate),
    )
    enable_id_normalize = not bool(args.disable_id_normalize)
    unknown_labels = [x.strip() for x in str(args.unknown_labels).split(",") if x.strip()]
    sample_arg = str(args.sample)
    if sample_arg.strip().lower() == "all":
        sim_root = project_root / "data" / "sim"
        scenario_ids = sorted([p.name for p in sim_root.iterdir() if p.is_dir()]) if sim_root.exists() else []
    else:
        scenario_ids = [s.strip() for s in sample_arg.split(",") if s.strip()]
    if not scenario_ids:
        raise FileNotFoundError("Stage5: run_list is empty because scenario list is empty")

    processed: List[str] = []
    skipped: List[Dict[str, Any]] = []
    for sid in scenario_ids:
        try:
            run_stage5_for_scenario(
                scenario_id=str(sid),
                project_root=project_root,
                backends=backends or ["cytospace"],
                topk_list=topk_list,
                id_policy=id_policy,
                enable_id_normalize=enable_id_normalize,
                missing_spot_policy=str(args.missing_spot_policy),
                renorm_policy=str(args.renorm_policy),
                type_union_policy=str(args.type_union_policy),
                divergence_epsilon=float(args.divergence_epsilon),
                spot_subset_policy=str(args.spot_subset_policy),
                rare_type_frac_threshold=float(args.rare_type_frac_threshold),
                rare_spot_threshold=int(args.rare_spot_threshold),
                pred_nonzero_threshold=float(args.pred_nonzero_threshold),
                unknown_labels=unknown_labels,
            )
            processed.append(str(sid))
        except Exception as e:
            skipped.append({"scenario_id": str(sid), "error": f"{type(e).__name__}: {e}"})
            print(f"[Stage5] Skip scenario={sid}: {type(e).__name__}: {e}")

    if not processed:
        raise RuntimeError(f"Stage5: no scenarios processed successfully; skipped={skipped[:3]}")

    summarize_stage5(project_root=project_root, scenario_ids=processed, backends=backends or ["cytospace"])


if __name__ == "__main__":
    main()

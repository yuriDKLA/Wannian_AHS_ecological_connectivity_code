# -*- coding: utf-8 -*-
'''
@File    :   resistance_model.py
@Time    :   2026-07-14
@Desc    :   本脚本完整实现万年县四种重点保护陆生哺乳动物 XGBoost 生态阻力面构建流程。
             程序以五年稳定地类栅格为模板网格，读取上一阶段已经识别的生态开放情景源地二值栅格
             和源地 ID 栅格，分别为 Manis pentadactyla、Panthera pardus、Viverra zibetha、
             Viverricula indica 构造正样本、硬负样本和自然基质负样本。模型输入严格限定为六个
             阻力因子：NDVI、距道路距离、距稳定人类用地距离、距水系距离、高程、坡度，不把 LULC
             作为 XGBoost 输入变量。脚本使用空间分块交叉验证和 StratifiedGroupKFold 评估模型，
             在内层空间交叉验证中调参，并使用 Youden J 从训练集 out-of-fold 概率选择分类阈值。
             最终模型在全样本上拟合后，按块预测源地环境概率，并立即转换为 1-100 的生态阻力值，
             不输出任何概率栅格或原始阻力中间栅格。随后施加稳定 LULC 阻力约束，输出生态开放情景
             和管理封闭情景两类正式阻力 GeoTIFF；管理封闭情景仅在农业文化遗产地范围内强制设为
             100。脚本还导出训练样本、模型指标、ROC、混淆矩阵、SHAP 解释图、阻力面组合图、阻力
             统计表和质量控制表。
@Notice  :   建议在 GIS3.9 环境运行。直接运行本文件即可启动主流程。配置在同目录 config.yaml 中，
             依赖见 requirements.txt。程序不调用 ArcGIS、QGIS、MSPA、GUIDOS、Linkage Mapper 或
             其他桌面 GIS 软件；所有栅格处理、模型训练、解释和制图均通过 Python 完成。
'''

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import ft2font
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


@dataclass
class GridTemplate:
    path: Path
    crs: Any
    transform: Any
    width: int
    height: int
    nodata: float
    profile: Dict[str, Any]
    pixel_width: float
    pixel_height: float


@dataclass
class SpeciesConfig:
    latin: str
    code: str
    source_binary_open: Path
    source_id_open: Path


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).with_name("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(config_path)
    cfg["_config_hash"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return cfg


def setup_logging(cfg: Dict[str, Any]) -> logging.Logger:
    output_dir = Path(cfg["project"]["output_dir"])
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("resistance_model")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_dir / "resistance_model.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("INFO 启动 XGBoost 生态阻力面构建流程")
    logger.info("INFO 配置哈希: %s", cfg["_config_hash"])
    return logger


def import_geo_modules() -> Dict[str, Any]:
    try:
        import geopandas as gpd
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.features import rasterize
        from rasterio.mask import mask
        from rasterio.vrt import WarpedVRT
        from rasterio.windows import Window
        import shap
        from shapely.geometry import Point
    except Exception as exc:
        raise RuntimeError(
            "缺少地理空间依赖。请在 GIS3.9 环境安装 requirements.txt 后运行。"
            f" 原始错误: {exc}"
        ) from exc
    return {
        "gpd": gpd,
        "rasterio": rasterio,
        "Resampling": Resampling,
        "rasterize": rasterize,
        "mask": mask,
        "WarpedVRT": WarpedVRT,
        "Window": Window,
        "shap": shap,
        "Point": Point,
    }


def output_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    root = Path(cfg["project"]["output_dir"])
    dirs = {
        "root": root,
        "logs": root / "logs",
        "tables": root / "tables",
        "models": root / "models",
        "samples": root / "samples",
        "resistance": root / "resistance",
        "validation_fig": root / "figures" / "model_validation",
        "shap_fig": root / "figures" / "shap",
        "resistance_fig": root / "figures" / "resistance",
        "cache": root / "cache",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def species_configs(cfg: Dict[str, Any]) -> List[SpeciesConfig]:
    source_dir = Path(cfg["paths"]["source_raster_dir"])
    out: List[SpeciesConfig] = []
    for item in cfg["species"]:
        out.append(
            SpeciesConfig(
                latin=item["latin"],
                code=item["code"],
                source_binary_open=source_dir / item["source_binary_open"],
                source_id_open=source_dir / item["source_id_open"],
            )
        )
    return out


def validate_inputs(cfg: Dict[str, Any], geo: Dict[str, Any], logger: logging.Logger) -> None:
    rasterio = geo["rasterio"]
    required = [
        Path(cfg["paths"]["boundary_vector"]),
        Path(cfg["paths"]["heritage_vector"]),
        Path(cfg["paths"]["heritage_mask"]),
        Path(cfg["paths"]["stable_lulc"]),
    ]
    for f in cfg["factors"]:
        required.append(Path(f["path"]))
    for sp in species_configs(cfg):
        required.extend([sp.source_binary_open, sp.source_id_open])
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("以下输入文件不存在:\n" + "\n".join(missing))

    with rasterio.open(cfg["paths"]["stable_lulc"]) as ds:
        lulc = ds.read(1, masked=True)
        values = set(int(v) for v in np.unique(lulc.compressed()))
        illegal = sorted(values - set(range(10)))
        if illegal:
            raise ValueError(f"LULC 存在非法编码: {illegal}")
        logger.info("OK LULC 编码检查通过: %s", sorted(values))

    for sp in species_configs(cfg):
        with rasterio.open(sp.source_binary_open) as ds:
            arr = ds.read(1, masked=True)
            vals = set(int(v) for v in np.unique(arr.compressed()))
            if not vals.issubset({0, 1}):
                raise ValueError(f"{sp.latin} 源地二值栅格只能包含 0/1/NoData，实际为 {sorted(vals)}")
        logger.info("OK %s 源地二值检查通过", sp.latin)

    logger.info("OK 所有输入文件存在且基础编码检查通过")


def prepare_template_grid(cfg: Dict[str, Any], geo: Dict[str, Any], logger: logging.Logger) -> GridTemplate:
    rasterio = geo["rasterio"]
    template_path = Path(cfg["paths"]["stable_lulc"])
    with rasterio.open(template_path) as src:
        template = GridTemplate(
            path=template_path,
            crs=src.crs,
            transform=src.transform,
            width=src.width,
            height=src.height,
            nodata=cfg["project"].get("nodata", -9999),
            profile=src.profile.copy(),
            pixel_width=abs(src.transform.a),
            pixel_height=abs(src.transform.e),
        )
    logger.info("INFO 模板网格: %s x %s, CRS=%s", template.width, template.height, template.crs)
    return template


def read_aligned_array(
    path: Path,
    template: GridTemplate,
    geo: Dict[str, Any],
    continuous: bool,
    dtype: str = "float32",
) -> np.ndarray:
    rasterio = geo["rasterio"]
    Resampling = geo["Resampling"]
    WarpedVRT = geo["WarpedVRT"]
    method = Resampling.bilinear if continuous else Resampling.nearest
    with rasterio.open(path) as src:
        same_grid = (
            src.crs == template.crs
            and src.transform == template.transform
            and src.width == template.width
            and src.height == template.height
        )
        if same_grid:
            arr = src.read(1).astype(dtype)
            nodata = src.nodata
        else:
            with WarpedVRT(
                src,
                crs=template.crs,
                transform=template.transform,
                width=template.width,
                height=template.height,
                resampling=method,
            ) as vrt:
                arr = vrt.read(1).astype(dtype)
                nodata = vrt.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    if np.issubdtype(arr.dtype, np.floating):
        arr[~np.isfinite(arr)] = np.nan
    return arr


def align_continuous_raster(path: Path, template: GridTemplate, geo: Dict[str, Any]) -> np.ndarray:
    return read_aligned_array(path, template, geo, continuous=True, dtype="float32")


def align_categorical_raster(path: Path, template: GridTemplate, geo: Dict[str, Any]) -> np.ndarray:
    arr = read_aligned_array(path, template, geo, continuous=False, dtype="float32")
    arr = np.where(np.isfinite(arr), arr, float(template.nodata))
    return np.rint(arr).astype("int32")


def rasterize_vector(
    vector_path: Path,
    template: GridTemplate,
    geo: Dict[str, Any],
    all_touched: bool = True,
) -> np.ndarray:
    gpd = geo["gpd"]
    rasterize = geo["rasterize"]
    gdf = gpd.read_file(vector_path, engine="fiona")
    if gdf.empty:
        raise ValueError(f"矢量文件为空: {vector_path}")
    gdf = gdf.to_crs(template.crs)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    return rasterize(
        shapes,
        out_shape=(template.height, template.width),
        transform=template.transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    )


def calculate_source_interior_distance(source_binary: np.ndarray, template: GridTemplate) -> np.ndarray:
    source = source_binary == 1
    sampling = (template.pixel_height, template.pixel_width)
    return distance_transform_edt(source, sampling=sampling).astype("float32")


def valid_factor_mask(factor_arrays: Dict[str, np.ndarray]) -> np.ndarray:
    mask = np.ones(next(iter(factor_arrays.values())).shape, dtype=bool)
    for arr in factor_arrays.values():
        mask &= np.isfinite(arr)
    return mask


def rowcol_to_xy(rows: np.ndarray, cols: np.ndarray, template: GridTemplate) -> Tuple[np.ndarray, np.ndarray]:
    xs = template.transform.c + (cols + 0.5) * template.transform.a
    ys = template.transform.f + (rows + 0.5) * template.transform.e
    return xs.astype("float64"), ys.astype("float64")


def spatial_thin_samples(
    rows: np.ndarray,
    cols: np.ndarray,
    scores: Optional[np.ndarray],
    template: GridTemplate,
    grid_size_m: float,
    seed: int,
    max_per_group: Optional[int] = None,
) -> np.ndarray:
    if len(rows) == 0:
        return np.array([], dtype=int)
    xs, ys = rowcol_to_xy(rows, cols, template)
    gx = np.floor((xs - xs.min()) / grid_size_m).astype("int64")
    gy = np.floor((ys - ys.min()) / grid_size_m).astype("int64")
    keys = pd.Series(gx, dtype="int64").astype(str).str.cat(pd.Series(gy, dtype="int64").astype(str), sep="_")
    df = pd.DataFrame({"idx": np.arange(len(rows)), "key": keys})
    if scores is not None:
        df["score"] = scores
        keep = df.sort_values("score", ascending=False).groupby("key", sort=False).head(1)["idx"].to_numpy()
    else:
        rng = np.random.default_rng(seed)
        df["rnd"] = rng.random(len(df))
        keep = df.sort_values("rnd").groupby("key", sort=False).head(1)["idx"].to_numpy()
    if max_per_group is not None and len(keep) > max_per_group:
        rng = np.random.default_rng(seed)
        keep = rng.choice(keep, size=max_per_group, replace=False)
    return np.sort(keep)


def generate_positive_samples(
    species: SpeciesConfig,
    source_binary: np.ndarray,
    source_id: np.ndarray,
    factor_mask: np.ndarray,
    template: GridTemplate,
    cfg: Dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    params = cfg["sampling"]
    interior = calculate_source_interior_distance(source_binary, template)
    records: List[pd.DataFrame] = []
    rng_seed = int(cfg["project"]["random_seed"])
    for sid in sorted(int(v) for v in np.unique(source_id[(source_id > 0) & (source_binary == 1)])):
        sid_mask = (source_id == sid) & (source_binary == 1) & factor_mask
        if not np.any(sid_mask):
            continue
        cand = sid_mask & (interior >= float(params["positive_inner_distance_m"]))
        if not np.any(cand):
            vals = interior[sid_mask]
            threshold = np.percentile(vals, 100 - float(params["positive_fallback_top_percent"]))
            cand = sid_mask & (interior >= threshold)
        rows, cols = np.where(cand)
        scores = interior[rows, cols]
        keep = spatial_thin_samples(
            rows,
            cols,
            scores,
            template,
            float(params["sampling_grid_size_m"]),
            rng_seed + sid,
        )
        if len(keep) > int(params["max_positive_samples_per_source"]):
            keep = keep[: int(params["max_positive_samples_per_source"])]
        elif len(keep) < int(params["min_positive_samples_per_source"]) and len(rows) <= int(params["min_positive_samples_per_source"]):
            keep = np.arange(len(rows))
        x, y = rowcol_to_xy(rows[keep], cols[keep], template)
        records.append(
            pd.DataFrame(
                {
                    "species": species.latin,
                    "label": 1,
                    "sample_type": "positive",
                    "source_id": sid,
                    "row": rows[keep],
                    "col": cols[keep],
                    "x": x,
                    "y": y,
                }
            )
        )
    if not records:
        raise ValueError(f"{species.latin} 未生成有效正样本")
    out = pd.concat(records, ignore_index=True)
    logger.info("OK %s 正样本: %s", species.latin, len(out))
    return out


def sample_negative_by_lulc(
    mask: np.ndarray,
    lulc: np.ndarray,
    n_target: int,
    template: GridTemplate,
    cfg: Dict[str, Any],
    sample_type: str,
    species: SpeciesConfig,
    seed_offset: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows_all: List[np.ndarray] = []
    cols_all: List[np.ndarray] = []
    rng = np.random.default_rng(int(cfg["project"]["random_seed"]) + seed_offset)
    classes = sorted(int(v) for v in np.unique(lulc[mask]) if int(v) > 0)
    if not classes:
        return pd.DataFrame()
    per_class = max(1, math.ceil(n_target / len(classes)))
    for lc in classes:
        rows, cols = np.where(mask & (lulc == lc))
        keep = spatial_thin_samples(
            rows,
            cols,
            None,
            template,
            float(cfg["sampling"]["sampling_grid_size_m"]),
            int(cfg["project"]["random_seed"]) + seed_offset + lc,
        )
        if len(keep) > per_class:
            keep = rng.choice(keep, size=per_class, replace=False)
        rows_all.append(rows[keep])
        cols_all.append(cols[keep])
    if not rows_all:
        return pd.DataFrame()
    rows_out = np.concatenate(rows_all)
    cols_out = np.concatenate(cols_all)
    if len(rows_out) > n_target:
        idx = rng.choice(np.arange(len(rows_out)), size=n_target, replace=False)
        rows_out = rows_out[idx]
        cols_out = cols_out[idx]
    x, y = rowcol_to_xy(rows_out, cols_out, template)
    return pd.DataFrame(
        {
            "species": species.latin,
            "label": 0,
            "sample_type": sample_type,
            "source_id": 0,
            "row": rows_out,
            "col": cols_out,
            "x": x,
            "y": y,
        }
    )


def generate_hard_negative_samples(
    species: SpeciesConfig,
    source_binary: np.ndarray,
    lulc: np.ndarray,
    factor_mask: np.ndarray,
    n_target: int,
    template: GridTemplate,
    cfg: Dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    hard_classes = np.isin(lulc, [1, 5, 6, 7, 8])
    mask = hard_classes & (source_binary != 1) & factor_mask
    df = sample_negative_by_lulc(mask, lulc, n_target, template, cfg, "hard_negative", species, 1000, logger)
    logger.info("OK %s 高阻力负样本: %s / 目标 %s", species.latin, len(df), n_target)
    return df


def generate_natural_negative_samples(
    species: SpeciesConfig,
    source_binary: np.ndarray,
    lulc: np.ndarray,
    factor_mask: np.ndarray,
    n_target: int,
    template: GridTemplate,
    cfg: Dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    dist_to_source = distance_transform_edt(source_binary != 1, sampling=(template.pixel_height, template.pixel_width))
    natural_classes = np.isin(lulc, [2, 3, 4, 9])
    mask = (
        natural_classes
        & (source_binary != 1)
        & (dist_to_source >= float(cfg["sampling"]["natural_negative_min_distance_m"]))
        & factor_mask
    )
    df = sample_negative_by_lulc(mask, lulc, n_target, template, cfg, "natural_negative", species, 2000, logger)
    logger.info("OK %s 自然基质负样本: %s / 目标 %s", species.latin, len(df), n_target)
    return df


def assign_spatial_blocks(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    size = float(cfg["cross_validation"]["spatial_block_size_m"])
    xmin = df["x"].min()
    ymin = df["y"].min()
    bx = np.floor((df["x"].to_numpy() - xmin) / size).astype("int64")
    by = np.floor((df["y"].to_numpy() - ymin) / size).astype("int64")
    df = df.copy()
    df["block_id"] = pd.Series(bx, dtype="int64").astype(str).str.cat(
        pd.Series(by, dtype="int64").astype(str),
        sep="_",
    ).to_numpy()
    return df


def build_training_dataframe(
    species: SpeciesConfig,
    source_binary: np.ndarray,
    source_id: np.ndarray,
    lulc: np.ndarray,
    factor_arrays: Dict[str, np.ndarray],
    template: GridTemplate,
    cfg: Dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    factor_mask = valid_factor_mask(factor_arrays)
    positive = generate_positive_samples(species, source_binary, source_id, factor_mask, template, cfg, logger)
    n_negative = int(round(len(positive) * float(cfg["sampling"]["negative_to_positive_ratio"])))
    n_hard = int(round(n_negative * float(cfg["sampling"]["hard_negative_fraction"])))
    n_natural = n_negative - n_hard
    hard = generate_hard_negative_samples(species, source_binary, lulc, factor_mask, n_hard, template, cfg, logger)
    natural = generate_natural_negative_samples(species, source_binary, lulc, factor_mask, n_natural, template, cfg, logger)
    if len(hard) < n_hard:
        add = generate_natural_negative_samples(species, source_binary, lulc, factor_mask, n_hard - len(hard), template, cfg, logger)
        natural = pd.concat([natural, add], ignore_index=True)
    if len(natural) < n_natural:
        add = generate_hard_negative_samples(species, source_binary, lulc, factor_mask, n_natural - len(natural), template, cfg, logger)
        hard = pd.concat([hard, add], ignore_index=True)
    df = pd.concat([positive, hard, natural], ignore_index=True)
    df.insert(0, "sample_id", [f"{species.code}_{i+1:06d}" for i in range(len(df))])
    for name, arr in factor_arrays.items():
        df[name] = arr[df["row"].to_numpy(), df["col"].to_numpy()]
    df = assign_spatial_blocks(df, cfg)
    feature_names = [f["name"] for f in cfg["factors"]]
    if not np.isfinite(df[feature_names].to_numpy()).all():
        raise ValueError(f"{species.latin} 训练样本中存在 NaN 或 Inf")
    logger.info("OK %s 训练样本总数: %s", species.latin, len(df))
    return df


def build_xgb_classifier(cfg: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> XGBClassifier:
    base = dict(
        objective="binary:logistic",
        tree_method=cfg["xgboost"].get("tree_method", "hist"),
        eval_metric=cfg["xgboost"].get("eval_metric", "logloss"),
        n_jobs=int(cfg["xgboost"].get("n_jobs", -1)),
        random_state=int(cfg["project"]["random_seed"]),
    )
    if params:
        base.update(params)
    return XGBClassifier(**base)


def choose_cv_folds(y: np.ndarray, groups: np.ndarray, requested: int, logger: logging.Logger) -> int:
    for k in range(int(requested), 2, -1):
        try:
            cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=2026)
            ok = True
            for _, test_idx in cv.split(np.zeros_like(y), y, groups):
                if len(np.unique(y[test_idx])) < 2:
                    ok = False
                    break
            if ok:
                return k
        except Exception:
            continue
    logger.info("WARN 空间折数无法满足 5/4/3 折完整性，退化为 3 折尝试")
    return 3


def select_youden_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, prob)
    j = tpr - fpr
    idx = int(np.nanargmax(j))
    threshold = float(thresholds[idx])
    if not np.isfinite(threshold):
        threshold = 0.5
    return threshold


def calculate_validation_metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "ROC-AUC": roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan,
        "PR-AUC": average_precision_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan,
        "Balanced Accuracy": balanced_accuracy_score(y_true, pred),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "Specificity": specificity,
        "F1": f1_score(y_true, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred) if len(np.unique(pred)) > 1 else 0.0,
        "Cohen Kappa": cohen_kappa_score(y_true, pred),
    }


def spatial_train_validation_test_indices(
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> Dict[str, np.ndarray]:
    base = np.arange(len(y))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(outer.split(base, y, groups))
    y_train_val = y[train_val_idx]
    groups_train_val = groups[train_val_idx]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 17)
    train_rel, validation_rel = next(inner.split(np.arange(len(train_val_idx)), y_train_val, groups_train_val))
    return {
        "Train": train_val_idx[train_rel],
        "Validation": train_val_idx[validation_rel],
        "Test": test_idx,
    }


def evaluate_model_train_validation_test(
    model: XGBClassifier,
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    species: SpeciesConfig,
) -> Dict[str, pd.DataFrame]:
    feature_names = [f["name"] for f in cfg["factors"]]
    X = df[feature_names].to_numpy(dtype="float32")
    y = df["label"].to_numpy(dtype="int32")
    groups = df["block_id"].to_numpy()
    seed = int(cfg["project"]["random_seed"])
    split_indices = spatial_train_validation_test_indices(y, groups, seed)
    roc_rows: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, Any]] = []
    for split_name in ["Train", "Validation", "Test"]:
        idx = split_indices[split_name]
        prob = model.predict_proba(X[idx])[:, 1]
        fpr, tpr, thresholds = roc_curve(y[idx], prob)
        split_auc = auc(fpr, tpr)
        roc_rows.append(
            pd.DataFrame(
                {
                    "species": species.latin,
                    "split": split_name,
                    "fpr": fpr,
                    "tpr": tpr,
                    "threshold": thresholds,
                }
            )
        )
        pred = (prob >= 0.5).astype("int32")
        metrics = calculate_validation_metrics(y[idx], prob, pred)
        metric_rows.append(
            {
                "species": species.latin,
                "split": split_name,
                "n_samples": int(len(idx)),
                "n_positive": int((y[idx] == 1).sum()),
                "n_negative": int((y[idx] == 0).sum()),
                "ROC-AUC": float(split_auc),
                **{k: float(v) if np.isscalar(v) and np.isfinite(v) else v for k, v in metrics.items() if k != "ROC-AUC"},
            }
        )
    return {
        "roc_values": pd.concat(roc_rows, ignore_index=True),
        "metrics": pd.DataFrame(metric_rows),
    }


def run_nested_spatial_cv(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    species: SpeciesConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    feature_names = [f["name"] for f in cfg["factors"]]
    X = df[feature_names].to_numpy(dtype="float32")
    y = df["label"].to_numpy(dtype="int32")
    groups = df["block_id"].to_numpy()
    outer_k = choose_cv_folds(y, groups, int(cfg["cross_validation"]["outer_cv_folds"]), logger)
    outer_cv = StratifiedGroupKFold(n_splits=outer_k, shuffle=True, random_state=int(cfg["project"]["random_seed"]))
    param_dist = cfg["xgboost"]["param_distributions"]
    all_prob = np.zeros(len(df), dtype="float32")
    all_pred = np.zeros(len(df), dtype="int32")
    fold_rows: List[Dict[str, Any]] = []
    roc_rows: List[pd.DataFrame] = []
    best_params_list: List[Dict[str, Any]] = []
    thresholds: List[float] = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
        y_train = y[train_idx]
        train_groups = groups[train_idx]
        inner_k = choose_cv_folds(y_train, train_groups, int(cfg["cross_validation"]["inner_cv_folds"]), logger)
        inner_cv = StratifiedGroupKFold(n_splits=inner_k, shuffle=True, random_state=int(cfg["project"]["random_seed"]) + fold)
        sampled_params = list(
            ParameterSampler(
                param_dist,
                n_iter=int(cfg["cross_validation"]["n_iter_search"]),
                random_state=int(cfg["project"]["random_seed"]) + fold,
            )
        )
        best_params: Dict[str, Any] = {}
        best_score = -np.inf
        for params in sampled_params:
            scores = []
            for inner_train, inner_valid in inner_cv.split(X[train_idx], y_train, train_groups):
                model = build_xgb_classifier(cfg, params)
                model.fit(X[train_idx][inner_train], y_train[inner_train])
                prob_inner = model.predict_proba(X[train_idx][inner_valid])[:, 1]
                scores.append(average_precision_score(y_train[inner_valid], prob_inner))
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_params = dict(params)
        best_params_list.append(best_params)

        oof_prob = np.zeros(len(train_idx), dtype="float32")
        for inner_train, inner_valid in inner_cv.split(X[train_idx], y_train, train_groups):
            model = build_xgb_classifier(cfg, best_params)
            model.fit(X[train_idx][inner_train], y_train[inner_train])
            oof_prob[inner_valid] = model.predict_proba(X[train_idx][inner_valid])[:, 1]
        threshold = select_youden_threshold(y_train, oof_prob)
        thresholds.append(threshold)

        model = build_xgb_classifier(cfg, best_params)
        model.fit(X[train_idx], y_train)
        prob = model.predict_proba(X[test_idx])[:, 1]
        pred = (prob >= threshold).astype("int32")
        all_prob[test_idx] = prob
        all_pred[test_idx] = pred

        metrics = calculate_validation_metrics(y[test_idx], prob, pred)
        fold_rows.append({"species": species.latin, "fold": fold, "threshold": threshold, **metrics})
        fpr, tpr, _ = roc_curve(y[test_idx], prob)
        roc_rows.append(pd.DataFrame({"species": species.latin, "fold": fold, "fpr": fpr, "tpr": tpr}))
        logger.info(
            "OK %s 外层折 %s/%s: ROC-AUC=%.4f PR-AUC=%.4f threshold=%.4f",
            species.latin,
            fold,
            outer_k,
            metrics["ROC-AUC"],
            metrics["PR-AUC"],
            threshold,
        )

    metrics = calculate_validation_metrics(y, all_prob, all_pred)
    return {
        "oof_probability": all_prob,
        "oof_prediction": all_pred,
        "fold_metrics": pd.DataFrame(fold_rows),
        "roc_values": pd.concat(roc_rows, ignore_index=True),
        "overall_metrics": {"species": species.latin, **metrics},
        "best_params_list": best_params_list,
        "mean_threshold": float(np.mean(thresholds)),
        "outer_folds": outer_k,
    }


def fit_final_model(df: pd.DataFrame, cfg: Dict[str, Any], cv_result: Dict[str, Any], species: SpeciesConfig) -> XGBClassifier:
    feature_names = [f["name"] for f in cfg["factors"]]
    X = df[feature_names].to_numpy(dtype="float32")
    y = df["label"].to_numpy(dtype="int32")
    best_params = choose_final_params(cv_result["best_params_list"])
    model = build_xgb_classifier(cfg, best_params)
    model.fit(X, y)
    return model


def choose_final_params(params_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not params_list:
        return {}
    out: Dict[str, Any] = {}
    keys = params_list[0].keys()
    for k in keys:
        values = [p[k] for p in params_list]
        if isinstance(values[0], (int, float)):
            out[k] = type(values[0])(np.median(values))
        else:
            out[k] = pd.Series(values).mode().iloc[0]
    return out


def calculate_shap_values(model: XGBClassifier, df: pd.DataFrame, cfg: Dict[str, Any], geo: Dict[str, Any]) -> Dict[str, Any]:
    shap = geo["shap"]
    feature_names = [f["name"] for f in cfg["factors"]]
    max_n = int(cfg["shap"]["max_shap_samples"])
    if len(df) > max_n:
        sample = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(
                min(len(x), max(1, int(max_n * len(x) / len(df)))),
                random_state=int(cfg["project"]["random_seed"]),
            )
        )
    else:
        sample = df.copy()
    X = sample[feature_names].to_numpy(dtype="float32")
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[-1]
    return {"sample": sample, "X": X, "values": np.asarray(values)}


def export_shap_outputs(
    species: SpeciesConfig,
    shap_result: Dict[str, Any],
    cfg: Dict[str, Any],
    dirs: Dict[str, Path],
) -> pd.DataFrame:
    feature_names = [f["name"] for f in cfg["factors"]]
    display = [f["display_name"] for f in cfg["factors"]]
    values = shap_result["values"]
    imp = np.abs(values).mean(axis=0)
    std_imp = np.abs(values).std(axis=0)
    imp_df = pd.DataFrame(
        {
            "species": species.latin,
            "feature": display,
            "mean_abs_shap": imp,
            "std_abs_shap": std_imp,
        }
    )
    imp_df.sort_values("mean_abs_shap", ascending=False, inplace=True)

    out = dirs["shap_fig"] / f"{species.code}_shap_summary_dependence.png"
    plot_shap_dependence_combined(species, shap_result, cfg, out)
    return imp_df


def disable_sci_axis(ax: Any) -> None:
    try:
        ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    except Exception:
        pass
    try:
        ax.xaxis.get_offset_text().set_visible(False)
        ax.yaxis.get_offset_text().set_visible(False)
    except Exception:
        pass


def format_plain_number(value: float) -> str:
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"{value:,.0f}"
    if abs_value >= 100:
        return f"{value:.1f}"
    if abs_value >= 10:
        return f"{value:.2f}"
    if abs_value >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def save_figure_all(fig: Any, base_path_no_ext: Path, dpi: int) -> None:
    fig.savefig(str(base_path_no_ext.with_suffix(".jpg")), dpi=300, bbox_inches="tight")
    fig.savefig(str(base_path_no_ext.with_suffix(".pdf")), dpi=dpi, bbox_inches="tight")
    fig.savefig(str(base_path_no_ext.with_suffix(".png")), dpi=dpi, bbox_inches="tight")


def beeswarm_jitter(shap_col: np.ndarray, max_jitter: float = 0.4, random_state: int = 42) -> np.ndarray:
    n = len(shap_col)
    rng = np.random.default_rng(random_state)
    if n < 2 or np.nanstd(shap_col) < 1e-12:
        return rng.uniform(-max_jitter / 3, max_jitter / 3, n)
    try:
        clean = np.asarray(shap_col, dtype=float)
        finite = np.isfinite(clean)
        if finite.sum() < 2:
            return rng.uniform(-max_jitter / 3, max_jitter / 3, n)
        fit_values = clean[finite]
        if fit_values.size > 5000:
            fit_values = rng.choice(fit_values, size=5000, replace=False)
        kde = stats.gaussian_kde(fit_values)
        density = np.zeros(n, dtype=float)
        density[finite] = kde(clean[finite])
        max_density = np.nanmax(density)
        if max_density <= 0:
            return rng.uniform(-max_jitter / 3, max_jitter / 3, n)
        density_norm = np.clip(density / max_density, 0, 1)
        return rng.uniform(-1, 1, n) * (density_norm * max_jitter)
    except Exception:
        return rng.uniform(-max_jitter / 3, max_jitter / 3, n)


def smooth_dependence(feat_vals: np.ndarray, shap_vals: np.ndarray, n_points: int = 200, sigma: int = 5) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    finite = np.isfinite(feat_vals) & np.isfinite(shap_vals)
    x = np.asarray(feat_vals[finite], dtype=float)
    y = np.asarray(shap_vals[finite], dtype=float)
    if x.size < 4:
        return None
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    unique_x, unique_idx = np.unique(x_sorted, return_index=True)
    if unique_x.size < 4:
        return None
    y_unique = y_sorted[unique_idx]
    x_grid = np.linspace(unique_x.min(), unique_x.max(), n_points)
    y_mean = np.interp(x_grid, unique_x, gaussian_filter1d(y_unique, sigma=min(20, max(1, unique_x.size // 20))))
    y_mean = gaussian_filter1d(y_mean, sigma=sigma)

    center_idx = np.searchsorted(x_sorted, x_grid)
    window = max(len(x_sorted) // n_points * 3, 10)
    min_width = min(window, len(x_sorted))
    fixed_start = np.clip(center_idx - min_width // 2, 0, max(0, len(x_sorted) - min_width))
    start_idx = fixed_start
    end_idx = fixed_start + min_width
    y_prefix = np.concatenate([[0.0], np.cumsum(y_sorted, dtype=np.float64)])
    y2_prefix = np.concatenate([[0.0], np.cumsum(np.square(y_sorted, dtype=np.float64), dtype=np.float64)])
    count = np.maximum(1, end_idx - start_idx)
    win_sum = y_prefix[end_idx] - y_prefix[start_idx]
    win_sum2 = y2_prefix[end_idx] - y2_prefix[start_idx]
    mean = win_sum / count
    var = np.maximum(win_sum2 / count - np.square(mean), 0.0)
    y_std = gaussian_filter1d(np.sqrt(var), sigma=sigma)
    return x_grid, y_mean, y_mean + y_std, y_mean - y_std


def find_zero_crossings(x: np.ndarray, y: np.ndarray) -> List[float]:
    y0 = y[:-1]
    y1 = y[1:]
    sign_change_idx = np.where((y0 * y1) < 0)[0]
    if sign_change_idx.size == 0:
        return []
    x0 = x[sign_change_idx]
    x1 = x[sign_change_idx + 1]
    yy0 = y0[sign_change_idx]
    yy1 = y1[sign_change_idx]
    return (x0 - yy0 * (x1 - x0) / (yy1 - yy0)).astype(float).tolist()


def predict_probability_in_blocks(
    model: XGBClassifier,
    factor_paths: Sequence[Path],
    template: GridTemplate,
    cfg: Dict[str, Any],
    geo: Dict[str, Any],
    valid_mask: np.ndarray,
) -> Iterable[Tuple[Any, np.ndarray, np.ndarray]]:
    rasterio = geo["rasterio"]
    Resampling = geo["Resampling"]
    WarpedVRT = geo["WarpedVRT"]
    block = int(cfg["resistance"]["block_size"])
    datasets = [rasterio.open(p) for p in factor_paths]
    vrts = [
        WarpedVRT(
            ds,
            crs=template.crs,
            transform=template.transform,
            width=template.width,
            height=template.height,
            resampling=Resampling.bilinear,
        )
        for ds in datasets
    ]
    try:
        for row in tqdm(range(0, template.height, block), desc="Predict blocks"):
            for col in range(0, template.width, block):
                h = min(block, template.height - row)
                w = min(block, template.width - col)
                window = geo["Window"](col, row, w, h)
                stack = []
                local_valid = valid_mask[row : row + h, col : col + w].copy()
                for vrt in vrts:
                    arr = vrt.read(1, window=window).astype("float32")
                    if vrt.nodata is not None:
                        arr[arr == vrt.nodata] = np.nan
                    local_valid &= np.isfinite(arr)
                    stack.append(arr)
                prob = np.full((h, w), np.nan, dtype="float32")
                if np.any(local_valid):
                    X = np.stack([a[local_valid] for a in stack], axis=1)
                    prob[local_valid] = model.predict_proba(X)[:, 1].astype("float32")
                yield window, prob, local_valid
    finally:
        for vrt in vrts:
            vrt.close()
        for ds in datasets:
            ds.close()


def probability_to_resistance(prob: np.ndarray, gamma: float) -> np.ndarray:
    # R = 1 + 99 * (1 - P)^gamma，概率越高阻力越低，输出被裁剪到 1-100。
    resistance = 1.0 + 99.0 * np.power(1.0 - prob, gamma)
    return np.clip(resistance, 1.0, 100.0).astype("float32")


def apply_lulc_constraints(resistance: np.ndarray, lulc: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    out = resistance.copy()
    rules = cfg["resistance"]["lulc_constraints"]
    nodata = float(cfg["resistance"]["nodata"])
    for key, rule in rules.items():
        lc = int(key)
        mask = lulc == lc
        mode = rule.get("mode")
        if mode == "floor":
            out[mask] = np.maximum(out[mask], float(rule["value"]))
        elif mode == "fixed":
            out[mask] = float(rule["value"])
        elif mode == "nodata":
            out[mask] = nodata
    return out


def build_open_scenario(resistance_lulc: np.ndarray) -> np.ndarray:
    return resistance_lulc.copy()


def build_closed_scenario(resistance_lulc: np.ndarray, heritage_mask: np.ndarray, nodata: float) -> np.ndarray:
    out = resistance_lulc.copy()
    valid = out != nodata
    out[(heritage_mask == 1) & valid] = 100.0
    return out


def export_resistance_tif(
    path: Path,
    arr: np.ndarray,
    template: GridTemplate,
    cfg: Dict[str, Any],
    geo: Dict[str, Any],
) -> None:
    rasterio = geo["rasterio"]
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = template.profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=float(cfg["resistance"]["nodata"]),
        compress=str(cfg["resistance"]["compression"]),
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)
        try:
            dst.build_overviews([2, 4, 8, 16], geo["Resampling"].average)
            dst.update_tags(ns="rio_overview", resampling="average")
        except Exception:
            pass


def export_samples_shp(df: pd.DataFrame, species: SpeciesConfig, cfg: Dict[str, Any], dirs: Dict[str, Path], geo: Dict[str, Any], template: GridTemplate) -> None:
    gpd = geo["gpd"]
    Point = geo["Point"]
    cols = [
        "sample_id",
        "species",
        "label",
        "sample_type",
        "source_id",
        "block_id",
        "x",
        "y",
        "ndvi",
        "distance_to_road",
        "distance_to_human_land",
        "distance_to_water",
        "elevation",
        "slope",
    ]
    gdf = gpd.GeoDataFrame(
        df[cols].copy(),
        geometry=[Point(xy) for xy in zip(df["x"], df["y"])],
        crs=template.crs,
    )
    out = dirs["samples"] / f"{species.code}_training_samples.shp"
    gdf.to_file(out, encoding="utf-8", engine="fiona")


def model_weight_path(dirs: Dict[str, Path], species: SpeciesConfig) -> Path:
    return dirs["models"] / f"{species.code}_xgboost.pth"


def load_existing_model_weights(
    dirs: Dict[str, Path],
    species: SpeciesConfig,
    cfg: Dict[str, Any],
    logger: logging.Logger,
) -> Optional[XGBClassifier]:
    pth_path = model_weight_path(dirs, species)
    if pth_path.exists():
        payload = joblib.load(pth_path)
        if isinstance(payload, dict) and "model" in payload:
            logger.info("OK %s 已加载 pth 权重: %s", species.latin, pth_path)
            return payload["model"]
        if isinstance(payload, XGBClassifier):
            logger.info("OK %s 已加载 pth 权重: %s", species.latin, pth_path)
            return payload
        raise ValueError(f"pth 权重格式不支持: {pth_path}")
    return None


def save_model_metadata(
    model: XGBClassifier,
    df: pd.DataFrame,
    species: SpeciesConfig,
    cv_result: Dict[str, Any],
    cfg: Dict[str, Any],
    dirs: Dict[str, Path],
) -> None:
    pth_path = model_weight_path(dirs, species)
    joblib.dump(
        {
            "model": model,
            "species": species.latin,
            "feature_names": [f["name"] for f in cfg["factors"]],
            "feature_display_names": [f["display_name"] for f in cfg["factors"]],
            "config_hash": cfg["_config_hash"],
        },
        pth_path,
    )
    meta = {
        "species": species.latin,
        "model_weight_file": str(pth_path),
        "best_params": model.get_params(),
        "features": [f["display_name"] for f in cfg["factors"]],
        "n_samples": int(len(df)),
        "n_positive": int((df["label"] == 1).sum()),
        "n_negative": int((df["label"] == 0).sum()),
        "outer_cv_folds": int(cv_result["outer_folds"]),
        "mean_best_threshold": float(cv_result["mean_threshold"]),
        "random_seed": int(cfg["project"]["random_seed"]),
        "config_hash": cfg["_config_hash"],
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "xgboost": getattr(sys.modules.get("xgboost"), "__version__", "unknown"),
            "sklearn": getattr(sys.modules.get("sklearn"), "__version__", "unknown"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (dirs["models"] / f"{species.code}_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def vector_heatmap(ax: Any, cm: np.ndarray, labels: Sequence[str], title: str, vmax: float) -> None:
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = plt.get_cmap("Blues")
    total_by_true = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, total_by_true, out=np.zeros_like(cm, dtype=float), where=total_by_true != 0)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = cmap(norm(pct[i, j]))
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white"))
            txt_color = "white" if pct[i, j] > 0.55 else "black"
            ax.text(j + 0.5, i + 0.5, f"{cm[i, j]}\n{pct[i, j]*100:.1f}%", ha="center", va="center", color=txt_color, fontsize=8)
    ax.set_xlim(0, cm.shape[1])
    ax.set_ylim(cm.shape[0], 0)
    ax.set_xticks(np.arange(len(labels)) + 0.5, labels)
    ax.set_yticks(np.arange(len(labels)) + 0.5, labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, fontstyle="italic")
    ax.set_aspect("equal")


def plot_roc_combined(results: List[Dict[str, Any]], cfg: Dict[str, Any], dirs: Dict[str, Path]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.4), dpi=int(cfg["figures"]["dpi"]))
    split_colors = {
        "Train": "#2f6fb0",
        "Validation": "#d08b2e",
        "Test": "#b83b4b",
    }
    for ax, item in zip(axes.ravel(), results):
        roc_df = item["roc_diagnostic"]["roc_values"]
        metrics = item["roc_diagnostic"]["metrics"]
        for split_name in ["Train", "Validation", "Test"]:
            sub = roc_df[roc_df["split"] == split_name]
            auc_value = metrics.loc[metrics["split"] == split_name, "ROC-AUC"].iloc[0]
            ax.plot(
                sub["fpr"],
                sub["tpr"],
                lw=1.9,
                color=split_colors[split_name],
                label=f"{split_name} AUC={auc_value:.3f}",
            )
        ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), color="#9a9a9a", lw=1.0, label="Random")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("False positive rate", fontsize=9)
        ax.set_ylabel("True positive rate", fontsize=9)
        ax.set_title(item["species"].latin, fontstyle="italic", fontsize=10, pad=8)
        ax.tick_params(labelsize=8, direction="out", length=3)
        ax.grid(True, color="#d9d9d9", linewidth=0.45, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.legend(loc="lower right", frameon=False, fontsize=7.2, handlelength=2.2)
        disable_sci_axis(ax)
    fig.tight_layout()
    base = dirs["validation_fig"] / "roc_2x2"
    fig.savefig(base.with_suffix(".jpg"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"), dpi=int(cfg["figures"]["dpi"]))
    fig.savefig(base.with_suffix(".png"), dpi=int(cfg["figures"]["dpi"]))
    fig.savefig(base.with_suffix(".tif"), dpi=int(cfg["figures"]["dpi"]))
    plt.close(fig)


def plot_confusion_matrix_combined(results: List[Dict[str, Any]], cfg: Dict[str, Any], dirs: Dict[str, Path]) -> None:
    cms = []
    for item in results:
        y = item["df"]["label"].to_numpy()
        pred = item["cv"]["oof_prediction"]
        cms.append(confusion_matrix(y, pred, labels=[0, 1]))
    vmax = max(np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) != 0).max() for cm in cms)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=int(cfg["figures"]["dpi"]))
    for ax, item, cm in zip(axes.ravel(), results, cms):
        vector_heatmap(ax, cm, ["Non-source", "Source"], item["species"].latin, vmax)
    fig.tight_layout()
    fig.savefig(dirs["validation_fig"] / "confusion_matrix_2x2.png", dpi=int(cfg["figures"]["dpi"]))
    fig.savefig(dirs["validation_fig"] / "confusion_matrix_2x2.tif", dpi=int(cfg["figures"]["dpi"]))
    plt.close(fig)


def plot_shap_dependence_combined(species: SpeciesConfig, shap_result: Dict[str, Any], cfg: Dict[str, Any], out: Path) -> None:
    values = shap_result["values"]
    X = shap_result["X"]
    display = [f["display_name"] for f in cfg["factors"]]
    ranks = np.argsort(np.abs(values).mean(axis=0))[::-1]
    fig = plt.figure(figsize=(11, 9.6), dpi=int(cfg["figures"]["dpi"]))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    ax0 = fig.add_subplot(gs[0, :])
    plot_shap_beeswarm_axis(
        fig=fig,
        ax_scatter=ax0,
        shap_values=values,
        feature_values=X,
        feature_names=display,
        random_seed=int(cfg["project"]["random_seed"]),
        title=species.latin,
        title_italic=True,
        show_colorbar=True,
        compact=True,
    )

    cmap = plt.get_cmap("BrBG_r")
    for n, idx in enumerate(ranks):
        ax = fig.add_subplot(gs[1 + n // 3, n % 3])
        feat_vals = X[:, idx]
        shap_col = values[:, idx]
        finite = np.isfinite(feat_vals) & np.isfinite(shap_col)
        feat_vals = feat_vals[finite]
        shap_col = shap_col[finite]

        if feat_vals.size < 4 or np.std(feat_vals) < 1e-12 or np.std(shap_col) < 1e-12:
            ax.set_title(display[idx], fontsize=9)
            ax.text(0.5, 0.5, "Constant", transform=ax.transAxes, ha="center", va="center", fontsize=8)
            ax.set_xlabel("Feature value", fontsize=8)
            ax.set_ylabel("SHAP", fontsize=8)
            continue

        smooth = smooth_dependence(feat_vals, shap_col)
        if smooth is None:
            ax.scatter(feat_vals, shap_col, s=4, color="#777777", alpha=0.55, linewidths=0)
        else:
            x_grid, y_mean, y_upper, y_lower = smooth
            pos_mask = y_mean >= 0
            change_points = np.where(np.diff(pos_mask.astype(np.int8)) != 0)[0] + 1
            seg_starts = np.r_[0, change_points]
            seg_ends = np.r_[change_points - 1, len(pos_mask) - 1]
            for seg_start, seg_end, is_positive in zip(seg_starts, seg_ends, pos_mask[seg_starts]):
                ax.axvspan(
                    x_grid[seg_start],
                    x_grid[seg_end],
                    color="#e8f3f1" if is_positive else "#f2e8e8",
                    alpha=0.55,
                    zorder=0,
                )
            crossings = find_zero_crossings(x_grid, y_mean)
            if crossings:
                xc = crossings[0]
                ax.axvline(xc, color="#c44e52", linestyle="--", linewidth=0.8, zorder=4)
                ax.text(xc, 0, format_plain_number(xc), color="#c44e52", fontsize=7, ha="center", va="bottom")
            ax.fill_between(x_grid, y_lower, y_upper, alpha=0.16, color="#d6604d", zorder=2)
            ax.plot(x_grid, y_mean, color="#d6604d", linewidth=1.1, zorder=4, label="Lowess curve")

        finite_feat = feat_vals[np.isfinite(feat_vals)]
        vmin, vmax = np.nanpercentile(finite_feat, [2, 98])
        norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1.0)
        sc = ax.scatter(
            feat_vals,
            shap_col,
            c=feat_vals,
            cmap=cmap,
            norm=norm,
            s=5,
            alpha=0.68,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        ax.axhline(0, color="#999999", linestyle="--", linewidth=0.7, zorder=1)
        try:
            r2 = stats.spearmanr(feat_vals, shap_col).statistic ** 2
            ax.text(
                0.03,
                0.05,
                f"R2={r2:.3f}",
                transform=ax.transAxes,
                fontsize=6,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.85),
            )
        except Exception:
            pass
        ax.set_title(display[idx], fontsize=9)
        ax.set_xlabel("Feature value", fontsize=8)
        ax.set_ylabel("SHAP", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle=":", linewidth=0.35, alpha=0.45)
        disable_sci_axis(ax)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.015)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label("Feature value", fontsize=7)
    fig.suptitle("SHAP summary and single-feature dependence plots", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save_figure_all(fig, out, int(cfg["figures"]["dpi"]))
    plt.close(fig)


def plot_shap_beeswarm_axis(
    fig: Any,
    ax_scatter: Any,
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: Sequence[str],
    random_seed: int,
    title: Optional[str] = None,
    title_italic: bool = False,
    show_colorbar: bool = False,
    compact: bool = False,
) -> None:
    n_features = shap_values.shape[1]
    mean_abs = np.abs(shap_values).mean(axis=0).ravel()
    order = np.argsort(mean_abs)[::-1].astype(int)
    sorted_names = [feature_names[i] for i in order]
    sorted_mean = mean_abs[order]
    sorted_pct = sorted_mean / (sorted_mean.sum() + 1e-12) * 100
    y_positions = np.arange(n_features)
    cmap = plt.get_cmap("coolwarm")

    ax_bar = ax_scatter.twiny()
    ax_scatter.set_zorder(ax_bar.get_zorder() + 1)
    ax_scatter.patch.set_visible(False)
    ax_bar.barh(
        y_positions,
        sorted_mean,
        height=0.70,
        color="#e6eaf3",
        edgecolor="#c5cbe0",
        linewidth=0.5,
        zorder=1,
    )
    ax_bar.set_xlim(0, max(float(sorted_mean.max()) * 1.25, 1e-12))
    ax_bar.set_xlabel("mean(|SHAP value|)\n(average impact on magnitude)", fontsize=8 if compact else 9, labelpad=5)
    ax_bar.tick_params(axis="x", labelsize=7 if compact else 8)
    ax_bar.grid(False)
    disable_sci_axis(ax_bar)

    for yy in y_positions:
        ax_scatter.axhline(y=yy, color="#e0e0e0", linewidth=0.3, linestyle="--", zorder=0)

    for rank, feat_idx in enumerate(order):
        shap_col = shap_values[:, feat_idx]
        feat_col = feature_values[:, feat_idx]
        finite_feat = feat_col[np.isfinite(feat_col)]
        if finite_feat.size:
            vmin, vmax = np.nanpercentile(finite_feat, [2, 98])
            if vmax - vmin < 1e-12:
                vmin, vmax = float(np.nanmin(finite_feat)), float(np.nanmax(finite_feat))
        else:
            vmin, vmax = 0.0, 1.0
        norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1.0)
        colors = cmap(norm(np.clip(feat_col, vmin, vmax if vmax > vmin else vmin + 1.0)))
        jitter = beeswarm_jitter(shap_col, max_jitter=0.35, random_state=random_seed + rank)
        n_points = len(shap_col)
        if n_points > 10000:
            rng = np.random.default_rng(random_seed + rank)
            plot_idx = rng.choice(n_points, size=10000, replace=False)
            shap_plot = shap_col[plot_idx]
            jitter_plot = jitter[plot_idx]
            colors_plot = colors[plot_idx]
        else:
            shap_plot = shap_col
            jitter_plot = jitter
            colors_plot = colors
        ax_scatter.scatter(
            shap_plot,
            rank + jitter_plot,
            c=colors_plot,
            s=3 if compact else 4,
            alpha=0.80,
            linewidths=0,
            zorder=3,
            rasterized=True,
        )

    ax_scatter.axvline(x=0, color="#888888", linewidth=0.8, zorder=2)
    ax_scatter.set_xlabel("SHAP value (impact on model output)", fontsize=8 if compact else 9)
    ax_scatter.set_yticks(y_positions)
    ax_scatter.set_yticklabels(sorted_names, fontsize=7 if compact else 8)
    ax_scatter.set_ylim(n_features - 0.5, -0.5)
    ax_scatter.tick_params(axis="x", labelsize=7 if compact else 8)
    ax_scatter.grid(False)
    disable_sci_axis(ax_scatter)
    if title:
        ax_scatter.set_title(title, fontstyle="italic" if title_italic else "normal", fontsize=9 if compact else 10, pad=8)

    ax_pct = ax_scatter.twinx()
    ax_pct.set_ylim(ax_scatter.get_ylim())
    ax_pct.set_yticks(y_positions)
    ax_pct.set_yticklabels([f"{p:.1f}%" for p in sorted_pct], fontsize=7 if compact else 8)
    if compact:
        ax_pct.set_ylabel("")
    else:
        ax_pct.set_ylabel("Global feature importance %", fontsize=9, rotation=270, labelpad=12)
    ax_pct.tick_params(axis="y", length=0)

    if show_colorbar:
        sm = ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_scatter, fraction=0.030 if compact else 0.045, pad=0.065)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"])
        cbar.ax.tick_params(labelsize=7 if compact else 8)
        cbar.set_label("Feature value", fontsize=8 if compact else 9, rotation=270, labelpad=12)


def plot_shap_importance_combined(importance: pd.DataFrame, cfg: Dict[str, Any], dirs: Dict[str, Path]) -> None:
    species_order = [s["latin"] for s in cfg["species"]]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), dpi=int(cfg["figures"]["dpi"]))
    for ax, species in zip(axes.ravel(), species_order):
        result = importance.attrs.get(species)
        if result is None or result.get("values") is None or result.get("X") is None:
            ax.axis("off")
            continue
        plot_shap_beeswarm_axis(
            fig=fig,
            ax_scatter=ax,
            shap_values=result["values"],
            feature_values=result["X"],
            feature_names=[f["display_name"] for f in cfg["factors"]],
            random_seed=int(cfg["project"]["random_seed"]),
            title=species,
            title_italic=True,
            show_colorbar=False,
            compact=True,
        )
    sm = ScalarMappable(cmap=plt.get_cmap("coolwarm"), norm=Normalize(0, 1))
    sm.set_array([])
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.08, top=0.94, wspace=0.38, hspace=0.42)
    cbar_ax = fig.add_axes([0.925, 0.20, 0.016, 0.60])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value", rotation=270, labelpad=14)
    base = dirs["shap_fig"] / "shap_importance_2x2"
    save_figure_all(fig, base, int(cfg["figures"]["dpi"]))
    fig.savefig(dirs["shap_fig"] / "shap_importance_2x2.tif", dpi=int(cfg["figures"]["dpi"]), bbox_inches="tight")
    plt.close(fig)


def get_esri_north_arrow_symbol(font_path: Path, symbol_index: int) -> Tuple[str, FontProperties]:
    if not font_path.exists():
        raise FileNotFoundError(f"指北针字体不存在: {font_path}")
    font = ft2font.FT2Font(str(font_path))
    codes = sorted(code for code in font.get_charmap().keys() if code != 0x20)
    if symbol_index < 1 or symbol_index > len(codes):
        raise ValueError(f"指北针编号 #{symbol_index} 越界，字体可用符号数量为 {len(codes)}。")
    return chr(codes[symbol_index - 1]), FontProperties(fname=str(font_path))


def add_north_arrow(ax: Any, cfg: Dict[str, Any]) -> None:
    symbol, prop = get_esri_north_arrow_symbol(
        Path(cfg["figures"]["north_arrow_font_path"]),
        int(cfg["figures"]["north_arrow_symbol_index"]),
    )
    ax.text(
        0.92,
        0.88,
        symbol,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=24,
        fontproperties=prop,
        color="black",
        zorder=30,
        clip_on=False,
    )


def add_scale_bar(ax: Any, cfg: Dict[str, Any]) -> None:
    xmin, xmax = ax.get_xlim()
    width = xmax - xmin
    length_km = float(cfg["figures"].get("scale_bar_length_km", 10))
    length_frac = min((length_km * 1000.0) / width, 0.42)
    x0 = 0.08
    x1 = x0 + length_frac
    xmid = (x0 + x1) / 2.0
    y0 = -0.075
    tick_h = 0.030
    style = {"transform": ax.transAxes, "color": "black", "lw": 1.1, "clip_on": False, "zorder": 30}
    ax.plot([x0, x1], [y0, y0], **style)
    ax.plot([x0, x0], [y0, y0 + tick_h], **style)
    ax.plot([xmid, xmid], [y0, y0 + tick_h * 0.72], **style)
    ax.plot([x1, x1], [y0, y0 + tick_h], **style)
    label_y = y0 + tick_h * 1.20
    ax.text(x0, label_y, "0", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, clip_on=False)
    ax.text(xmid, label_y, f"{int(length_km / 2)}", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, clip_on=False)
    ax.text(x1, label_y, f"{int(length_km)}", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, clip_on=False)
    ax.text(x1 + 0.035, label_y, "km", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, clip_on=False)


def soften_hex_color(hex_color: str, amount: float) -> str:
    amount = min(max(float(amount), 0.0), 0.85)
    text = hex_color.strip().lstrip("#")
    rgb = np.array([int(text[i : i + 2], 16) for i in (0, 2, 4)], dtype=float)
    softened = rgb * (1.0 - amount) + 255.0 * amount
    return "#" + "".join(f"{int(round(v)):02x}" for v in np.clip(softened, 0, 255))


def smooth_resistance_for_display(arr: np.ndarray, nodata: float, sigma: float) -> np.ndarray:
    shown = np.where(arr == nodata, np.nan, arr).astype("float32")
    if sigma <= 0:
        return shown
    valid = np.isfinite(shown)
    if not np.any(valid):
        return shown
    values = np.where(valid, shown, 0.0)
    weights = valid.astype("float32")
    smooth_values = gaussian_filter(values, sigma=sigma, mode="nearest")
    smooth_weights = gaussian_filter(weights, sigma=sigma, mode="nearest")
    out = np.divide(
        smooth_values,
        smooth_weights,
        out=np.full_like(shown, np.nan, dtype="float32"),
        where=smooth_weights > 1e-6,
    )
    out[~valid] = np.nan
    return np.clip(out, 1, 100)


def resistance_colormap(cfg: Dict[str, Any]) -> LinearSegmentedColormap:
    colors = cfg["figures"].get("colormap_hex")
    soften = float(cfg["figures"].get("resistance_colormap_soften", 0.0))
    if colors:
        colors = [soften_hex_color(c, soften) for c in colors]
        if bool(cfg["figures"].get("resistance_colormap_reverse", False)):
            colors = list(reversed(colors))
        cmap = LinearSegmentedColormap.from_list(
            str(cfg["figures"].get("colormap", "Style_MPL_RdYlBu")),
            colors,
            N=256,
        )
    else:
        cmap = plt.get_cmap(str(cfg["figures"]["colormap"]))
    cmap = cmap.copy()
    cmap.set_bad((1, 1, 1, 0))
    return cmap


def raster_extent(template: GridTemplate) -> Tuple[float, float, float, float]:
    left = template.transform.c
    top = template.transform.f
    right = left + template.width * template.transform.a
    bottom = top + template.height * template.transform.e
    return (left, right, bottom, top)


def plot_resistance_surfaces(
    results: List[Dict[str, Any]],
    template: GridTemplate,
    heritage_mask: np.ndarray,
    cfg: Dict[str, Any],
    dirs: Dict[str, Path],
    geo: Dict[str, Any],
) -> None:
    gpd = geo["gpd"]
    boundary = gpd.read_file(cfg["paths"]["boundary_vector"], engine="fiona").to_crs(template.crs)
    heritage = gpd.read_file(cfg["paths"]["heritage_vector"], engine="fiona").to_crs(template.crs)
    fig, axes = plt.subplots(2, 4, figsize=(16.2, 8.0), dpi=int(cfg["figures"]["dpi"]))
    cmap = resistance_colormap(cfg)
    extent = raster_extent(template)
    letters = list("abcdefgh")
    nodata = float(cfg["resistance"]["nodata"])
    display_sigma = float(cfg["figures"].get("resistance_display_smoothing_sigma", 0.0))
    interpolation = str(cfg["figures"].get("resistance_display_interpolation", "bilinear"))
    im = None
    for col, item in enumerate(results):
        for row, scenario in enumerate(["open", "closed"]):
            ax = axes[row, col]
            arr = item[scenario]
            shown = smooth_resistance_for_display(arr, nodata, display_sigma)
            im = ax.imshow(
                shown,
                extent=extent,
                cmap=cmap,
                vmin=1,
                vmax=100,
                origin="upper",
                interpolation=interpolation,
                alpha=0.96,
            )
            boundary.boundary.plot(ax=ax, color="#202020", linewidth=0.55)
            heritage.boundary.plot(ax=ax, color="#6f1d1b", linewidth=0.55, linestyle="--")
            if row == 0:
                ax.set_title(item["species"].latin, fontstyle="italic", fontsize=10, pad=8)
            ax.text(0.02, 0.95, f"({letters[row*4+col]})", transform=ax.transAxes, ha="left", va="top", fontsize=10)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_frame_on(False)
            add_north_arrow(ax, cfg)
            add_scale_bar(ax, cfg)
    cax = fig.add_axes([0.905, 0.24, 0.014, 0.52])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Ecological resistance (1-100; higher = less movement)", rotation=270, labelpad=16)
    fig.text(0.025, 0.70, "Open scenario", rotation=90, ha="center", va="center", fontsize=11)
    fig.text(0.025, 0.31, "Closed scenario", rotation=90, ha="center", va="center", fontsize=11)
    for artist in fig.findobj():
        try:
            artist.set_clip_on(False)
        except Exception:
            pass
    fig.subplots_adjust(left=0.045, right=0.885, bottom=0.14, top=0.94, hspace=0.16, wspace=0.06)
    fig.savefig(dirs["resistance_fig"] / "resistance_surfaces_2x4.png", dpi=int(cfg["figures"]["dpi"]))
    fig.savefig(dirs["resistance_fig"] / "resistance_surfaces_2x4.jpg", dpi=300)
    fig.savefig(dirs["resistance_fig"] / "resistance_surfaces_2x4.pdf", dpi=int(cfg["figures"]["dpi"]))
    fig.savefig(dirs["resistance_fig"] / "resistance_surfaces_2x4.tif", dpi=int(cfg["figures"]["dpi"]))
    plt.close(fig)


def build_resistance_summary(
    results: List[Dict[str, Any]],
    heritage_mask: np.ndarray,
    template: GridTemplate,
    cfg: Dict[str, Any],
    dirs: Dict[str, Path],
) -> pd.DataFrame:
    nodata = float(cfg["resistance"]["nodata"])
    pixel_area_km2 = template.pixel_width * template.pixel_height / 1_000_000.0
    rows: List[Dict[str, Any]] = []
    for item in results:
        for scenario in ["open", "closed"]:
            arr = item[scenario]
            valid = arr != nodata
            vals = arr[valid]
            hvals = arr[(heritage_mask == 1) & valid]
            row = {
                "species": item["species"].latin,
                "scenario": scenario,
                "min_resistance": float(np.min(vals)),
                "max_resistance": float(np.max(vals)),
                "mean_resistance": float(np.mean(vals)),
                "median_resistance": float(np.median(vals)),
                "std_resistance": float(np.std(vals)),
                "p05": float(np.percentile(vals, 5)),
                "p25": float(np.percentile(vals, 25)),
                "p75": float(np.percentile(vals, 75)),
                "p95": float(np.percentile(vals, 95)),
                "heritage_mean_resistance": float(np.mean(hvals)) if len(hvals) else np.nan,
                "heritage_median_resistance": float(np.median(hvals)) if len(hvals) else np.nan,
            }
            bins = [(1, 20), (20, 40), (40, 60), (60, 80), (80, 100.000001)]
            names = [
                "area_resistance_1_20_km2",
                "area_resistance_20_40_km2",
                "area_resistance_40_60_km2",
                "area_resistance_60_80_km2",
                "area_resistance_80_100_km2",
            ]
            for name, (lo, hi) in zip(names, bins):
                row[name] = float(((vals >= lo) & (vals < hi)).sum() * pixel_area_km2)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(dirs["tables"] / "resistance_summary.csv", index=False, encoding="utf-8-sig")
    df.to_excel(dirs["tables"] / "resistance_summary.xlsx", index=False)
    return df


def run_quality_control(
    results: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    dirs: Dict[str, Path],
    heritage_mask: np.ndarray,
) -> pd.DataFrame:
    nodata = float(cfg["resistance"]["nodata"])
    checks: List[Dict[str, str]] = []

    def add(name: str, ok: bool, details: str) -> None:
        checks.append({"check_name": name, "status": "PASS" if ok else "FAIL", "details": details})

    for item in results:
        df = item["df"]
        add(f"{item['species'].latin} has positive and negative samples", (df["label"] == 1).any() and (df["label"] == 0).any(), f"samples={len(df)}")
        feature_names = [f["name"] for f in cfg["factors"]]
        add(f"{item['species'].latin} no NaN Inf in samples", np.isfinite(df[feature_names].to_numpy()).all(), "six factors checked")
        add(f"{item['species'].latin} six factors only", feature_names == ["ndvi", "distance_to_road", "distance_to_human_land", "distance_to_water", "elevation", "slope"], ",".join(feature_names))
        for scenario in ["open", "closed"]:
            arr = item[scenario]
            valid = arr != nodata
            ok = bool(valid.any() and np.nanmin(arr[valid]) >= 1 and np.nanmax(arr[valid]) <= 100)
            add(f"{item['species'].latin} {scenario} resistance range 1-100", ok, f"min={np.nanmin(arr[valid]):.4f}, max={np.nanmax(arr[valid]):.4f}")
        open_arr = item["open"]
        closed = item["closed"]
        valid = (open_arr != nodata) & (closed != nodata)
        add(f"{item['species'].latin} closed not lower than open", bool(np.all(closed[valid] >= open_arr[valid])), "pixelwise comparison")
        add(f"{item['species'].latin} heritage closed equals 100", bool(np.all(closed[(heritage_mask == 1) & valid] == 100)), "heritage mask")
        outside = (heritage_mask != 1) & valid
        add(f"{item['species'].latin} open closed same outside heritage", bool(np.allclose(open_arr[outside], closed[outside])), "outside heritage")
    expected_files = [
        dirs["validation_fig"] / "roc_2x2.png",
        dirs["resistance_fig"] / "resistance_surfaces_2x4.png",
        dirs["shap_fig"] / "shap_importance_2x2.png",
    ]
    add("required combined figures exist", all(p.exists() for p in expected_files), "; ".join(str(p.name) for p in expected_files))
    qc = pd.DataFrame(checks)
    qc.to_csv(dirs["tables"] / "quality_control.csv", index=False, encoding="utf-8-sig")
    if (qc["status"] == "FAIL").any():
        raise RuntimeError("质量控制存在 FAIL，请查看 output/tables/quality_control.csv")
    return qc


def save_tables(results: List[Dict[str, Any]], dirs: Dict[str, Path]) -> None:
    cv_results = [r for r in results if r.get("cv") is not None]
    if cv_results:
        metrics = pd.DataFrame([r["cv"]["overall_metrics"] for r in cv_results])
        fold = pd.concat([r["cv"]["fold_metrics"] for r in cv_results], ignore_index=True)
        roc = pd.concat([r["cv"]["roc_values"] for r in cv_results], ignore_index=True)
        cm_rows = []
        for r in cv_results:
            cm = confusion_matrix(r["df"]["label"].to_numpy(), r["cv"]["oof_prediction"], labels=[0, 1])
            for i, true_label in enumerate(["Non-source", "Source"]):
                for j, pred_label in enumerate(["Non-source", "Source"]):
                    cm_rows.append({"species": r["species"].latin, "true_label": true_label, "pred_label": pred_label, "count": int(cm[i, j])})
        metrics.to_csv(dirs["tables"] / "model_metrics.csv", index=False, encoding="utf-8-sig")
        metrics.to_excel(dirs["tables"] / "model_metrics.xlsx", index=False)
        fold.to_csv(dirs["tables"] / "fold_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(cm_rows).to_csv(dirs["tables"] / "confusion_matrix_values.csv", index=False, encoding="utf-8-sig")
        roc.to_csv(dirs["tables"] / "roc_curve_values.csv", index=False, encoding="utf-8-sig")
    roc_diag = pd.concat([r["roc_diagnostic"]["roc_values"] for r in results], ignore_index=True)
    roc_metrics = pd.concat([r["roc_diagnostic"]["metrics"] for r in results], ignore_index=True)
    roc_diag.to_csv(dirs["tables"] / "roc_train_validation_test_curve_values.csv", index=False, encoding="utf-8-sig")
    roc_metrics.to_csv(dirs["tables"] / "roc_train_validation_test_metrics.csv", index=False, encoding="utf-8-sig")
    roc_metrics.to_excel(dirs["tables"] / "roc_train_validation_test_metrics.xlsx", index=False)


def process_species(
    species: SpeciesConfig,
    cfg: Dict[str, Any],
    dirs: Dict[str, Path],
    geo: Dict[str, Any],
    template: GridTemplate,
    factor_arrays: Dict[str, np.ndarray],
    lulc: np.ndarray,
    heritage_mask: np.ndarray,
    logger: logging.Logger,
) -> Dict[str, Any]:
    logger.info("INFO 开始处理物种: %s", species.latin)
    source_binary = align_categorical_raster(species.source_binary_open, template, geo)
    source_id = align_categorical_raster(species.source_id_open, template, geo)
    df = build_training_dataframe(species, source_binary, source_id, lulc, factor_arrays, template, cfg, logger)
    export_samples_shp(df, species, cfg, dirs, geo, template)
    model = None
    cv_result: Optional[Dict[str, Any]] = None
    if bool(cfg["project"].get("use_existing_model_weights", False)):
        model = load_existing_model_weights(dirs, species, cfg, logger)
    if model is None:
        cv_result = run_nested_spatial_cv(df, cfg, species, logger)
        model = fit_final_model(df, cfg, cv_result, species)
        save_model_metadata(model, df, species, cv_result, cfg, dirs)
    roc_diagnostic = evaluate_model_train_validation_test(model, df, cfg, species)
    shap_result = calculate_shap_values(model, df, cfg, geo)
    shap_imp = export_shap_outputs(species, shap_result, cfg, dirs)

    factor_paths = [Path(f["path"]) for f in cfg["factors"]]
    valid = valid_factor_mask(factor_arrays) & np.isin(lulc, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    resistance = np.full((template.height, template.width), float(cfg["resistance"]["nodata"]), dtype="float32")
    gamma = float(cfg["resistance"]["resistance_gamma"])
    for window, prob, local_valid in predict_probability_in_blocks(model, factor_paths, template, cfg, geo, valid):
        if np.any(local_valid):
            r = probability_to_resistance(prob, gamma)
            lulc_win = lulc[int(window.row_off) : int(window.row_off + window.height), int(window.col_off) : int(window.col_off + window.width)]
            r = apply_lulc_constraints(r, lulc_win, cfg)
            sub = resistance[int(window.row_off) : int(window.row_off + window.height), int(window.col_off) : int(window.col_off + window.width)]
            sub[:, :] = np.where(local_valid, r, float(cfg["resistance"]["nodata"]))

    open_arr = build_open_scenario(resistance)
    closed_arr = build_closed_scenario(resistance, heritage_mask, float(cfg["resistance"]["nodata"]))
    sp_dir = dirs["resistance"] / species.code
    export_resistance_tif(sp_dir / "resistance_open.tif", open_arr, template, cfg, geo)
    export_resistance_tif(sp_dir / "resistance_closed.tif", closed_arr, template, cfg, geo)
    logger.info("OK %s 正式阻力栅格已输出", species.latin)
    return {
        "species": species,
        "df": df,
        "cv": cv_result,
        "roc_diagnostic": roc_diagnostic,
        "model": model,
        "open": open_arr,
        "closed": closed_arr,
        "shap_importance": shap_imp,
        "shap_importance_values": shap_result["values"],
        "shap_feature_values": shap_result["X"],
    }


def main() -> int:
    try:
        cfg = load_config()
        logger = setup_logging(cfg)
        dirs = output_paths(cfg)
        geo = import_geo_modules()
        validate_inputs(cfg, geo, logger)
        template = prepare_template_grid(cfg, geo, logger)
        factor_arrays = {
            f["name"]: align_continuous_raster(Path(f["path"]), template, geo)
            for f in tqdm(cfg["factors"], desc="Load factors")
        }
        lulc = align_categorical_raster(Path(cfg["paths"]["stable_lulc"]), template, geo)
        if Path(cfg["paths"]["heritage_mask"]).exists():
            heritage_mask = align_categorical_raster(Path(cfg["paths"]["heritage_mask"]), template, geo)
        else:
            heritage_mask = rasterize_vector(Path(cfg["paths"]["heritage_vector"]), template, geo, all_touched=True)
        results = []
        for sp in species_configs(cfg):
            results.append(process_species(sp, cfg, dirs, geo, template, factor_arrays, lulc, heritage_mask, logger))
        cv_results = [r for r in results if r.get("cv") is not None]
        if cv_results:
            plot_confusion_matrix_combined(cv_results, cfg, dirs)
        else:
            logger.info("INFO 本轮使用 pth 权重直接输出结果，跳过重训外层 CV 和混淆矩阵图")
        save_tables(results, dirs)
        plot_roc_combined(results, cfg, dirs)
        importance = pd.concat([r["shap_importance"] for r in results], ignore_index=True)
        for r in results:
            importance.attrs[r["species"].latin] = {
                "values": r["shap_importance_values"] if "shap_importance_values" in r else None,
                "X": r["shap_feature_values"] if "shap_feature_values" in r else None,
            }
        importance.to_csv(dirs["tables"] / "shap_importance.csv", index=False, encoding="utf-8-sig")
        plot_shap_importance_combined(importance, cfg, dirs)
        plot_resistance_surfaces(results, template, heritage_mask, cfg, dirs, geo)
        build_resistance_summary(results, heritage_mask, template, cfg, dirs)
        run_quality_control(results, cfg, dirs, heritage_mask)
        logger.info("DONE 全部流程完成")
        return 0
    except Exception as exc:
        try:
            logging.getLogger("resistance_model").exception("ERROR 流程失败: %s", exc)
        except Exception:
            pass
        print(f"ERROR 流程失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

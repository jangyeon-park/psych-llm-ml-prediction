#!/usr/bin/env python
"""07. SHAP-based Clustering — Patient Segmentation

UMAP dimensionality reduction + adaptive DBSCAN on SHAP values.
Cluster characterization and significance testing.

Usage:
    python scripts/07_clustering.py --config configs/default.yaml
    python scripts/07_clustering.py --config configs/default.yaml --label label_30d
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import PROJECT_ROOT, LABELS, load_experiment_config, build_data_files
from src.clustering import run_cluster_analysis, run_significance_test


def parse_args():
    p = argparse.ArgumentParser(description="SHAP-based clustering")
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--model-dir", type=str, default=None,
                   help="Directory with best_model.pkl and SHAP artifacts")
    p.add_argument("--imp-dir", type=str, default=None, help="Imputed data directory")
    p.add_argument("--out-dir", type=str, default=None, help="Clustering output directory")
    p.add_argument("--prefix", type=str, default="simple", help="File prefix")
    p.add_argument("--label", nargs="+", default=None, help="Specific labels")
    p.add_argument("--skip-significance", action="store_true", help="Skip significance tests")
    # UMAP/DBSCAN overrides
    p.add_argument("--umap-neighbors", type=int, default=None)
    p.add_argument("--umap-min-dist", type=float, default=None)
    p.add_argument("--dbscan-min-samples", type=int, default=None)
    p.add_argument("--dbscan-eps", type=float, default=None)
    p.add_argument("--max-noise-ratio", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.config:
        cfg = load_experiment_config(args.config)
    else:
        cfg = {}

    labels = args.label or cfg.get("labels", LABELS)
    clust_cfg = cfg.get("clustering", {})

    model_dir = Path(args.model_dir) if args.model_dir else (
        Path(cfg.get("paths", {}).get("results_dir", str(PROJECT_ROOT / "results/new_analysis")))
        / "modeling" / "step2_modeling"
    )
    imp_dir = Path(args.imp_dir) if args.imp_dir else (
        Path(cfg.get("paths", {}).get("data_imp_dir", str(PROJECT_ROOT / "data/processed_imp")))
        / "imputation" / "simple_imput"
    )
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(cfg.get("paths", {}).get("results_dir", str(PROJECT_ROOT / "results/new_analysis")))
        / "clustering"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    data_files = build_data_files(str(imp_dir), args.prefix, labels)

    # Clustering parameters (CLI overrides > config)
    umap_neighbors = args.umap_neighbors or clust_cfg.get("umap_n_neighbors", 45)
    umap_min_dist = args.umap_min_dist if args.umap_min_dist is not None else clust_cfg.get("umap_min_dist", 0.0)
    umap_metric = clust_cfg.get("umap_metric", "euclidean")
    umap_n_comp = clust_cfg.get("umap_n_components", 2)
    dbscan_min_samples = args.dbscan_min_samples or clust_cfg.get("dbscan_min_samples", 17)
    dbscan_eps = args.dbscan_eps or clust_cfg.get("dbscan_eps", None)
    dbscan_metric = clust_cfg.get("dbscan_metric", "euclidean")
    target_min_clusters = clust_cfg.get("target_min_clusters", 3)
    max_noise = args.max_noise_ratio or clust_cfg.get("max_noise_ratio", 0.45)
    max_trials = clust_cfg.get("max_trials", 5)

    print(f"Model dir: {model_dir}")
    print(f"Output:    {out_dir}")
    print(f"Labels:    {labels}")

    all_summaries = []

    for label in labels:
        try:
            print(f"\n{'='*20} {label} {'='*20}")
            summary_df, cluster_labels = run_cluster_analysis(
                label=label,
                model_dir=model_dir,
                data_files=data_files,
                out_dir=out_dir,
                umap_n_components=umap_n_comp,
                umap_n_neighbors=umap_neighbors,
                umap_min_dist=umap_min_dist,
                umap_metric=umap_metric,
                min_samples=dbscan_min_samples,
                eps=dbscan_eps,
                dbscan_metric=dbscan_metric,
                target_min_clusters=target_min_clusters,
                max_noise_ratio=max_noise,
                max_trials=max_trials,
                random_state=42,
            )
            summary_df.insert(0, "label", label)
            all_summaries.append(summary_df)

            # Significance tests
            if not args.skip_significance:
                try:
                    sig_dir = out_dir / "significance"
                    sig_dir.mkdir(parents=True, exist_ok=True)
                    xlsx = run_significance_test(label, out_dir, sig_dir)
                    print(f"  Significance saved: {xlsx}")
                except Exception as e:
                    print(f"  Significance test error: {e}")

        except Exception as e:
            print(f"[WARN] {label}: {e}")

    if all_summaries:
        all_df = pd.concat(all_summaries, ignore_index=True)
        all_df.to_csv(out_dir / "ALL_labels__cluster_summary.csv", index=False)
        print(f"\nSummary saved: {out_dir / 'ALL_labels__cluster_summary.csv'}")

    print("Done.")


if __name__ == "__main__":
    main()

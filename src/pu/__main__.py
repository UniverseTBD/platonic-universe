import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Platonic Universe Experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for running experiments
    parser_run = subparsers.add_parser("run", help="Run an experiment to generate embeddings.")
    parser_run.add_argument("--model", required=True, help="Model to run inference on (e.g., 'vit', 'dino', 'astropt').")
    parser_run.add_argument("--mode", required=True, help="Dataset to compare to HSC (e.g., 'jwst', 'legacysurvey', 'sdss', 'desi').")
    parser_run.add_argument("--output-dataset", help="Output HuggingFace dataset.")
    parser_run.add_argument("--batch-size", type=int, default=128, help="Batch size for processing.")
    parser_run.add_argument("--num-workers", type=int, default=0, help="Number of data loader workers.")
    parser_run.add_argument("--knn-k", type=int, default=10, help="K value for mutual KNN calculation.")
    parser_run.add_argument("--all-metrics", action="store_true", help="Compute all available metrics (not just MKNN and CKA).")
    parser_run.add_argument("--no-resize", dest="resize", action="store_false", help="Disable galaxy resizing during preprocessing (enabled by default)")
    parser_run.add_argument("--resize-mode", type=str, default="match", choices=["match", "fill"], help="Resize strategy: 'match' aligns HSC/LegacySurvey to the compared survey's framing using fixed extents; 'fill' uses adaptive per-galaxy Otsu cropping so each galaxy fills the frame. Default: match.")
    parser_run.add_argument("--test", action="store_true", help="Quick test run using only 1000 samples.")
    parser_run.add_argument("--test-10k", action="store_true", help="Test run using only 10000 samples.")

    # Subparser for running metrics comparisons
    parser_comparisons = subparsers.add_parser("compare", help="Run metrics comparisons on existing embeddings.")
    parser_comparisons.add_argument("parquet_file", help="Path to the Parquet file with embeddings.")
    parser_comparisons.add_argument("--ref", type=str, default=None, help="Path to a reference parquet file. Compares embeddings for --mode across the two files instead of comparing modes within one file.")
    parser_comparisons.add_argument("--mode", type=str, default=None, help="Mode to compare when using --ref (e.g., 'hsc'). Default: first mode in file.")
    parser_comparisons.add_argument("--metrics", nargs="+", default=["all"], help="Metrics to run. Use 'all' for all metrics, or specify: cka, mmd, procrustes, cosine_similarity, frechet, svcca, pwcca, tucker_congruence, eigenspectrum, riemannian, kl_divergence, js_divergence, mutual_information, mknn, jaccard, rsa, linear_r2.")
    parser_comparisons.add_argument("--k", type=int, default=10, help="K value for neighbor-based metrics (mknn, jaccard).")
    parser_comparisons.add_argument("--size", type=str, default=None, help="Model size to compare (e.g., 'base', 'large', 'huge'). Use 'all' to process all sizes. Default: first size in file.")

    # Subparser for calibrated comparisons
    parser_calibrate = subparsers.add_parser("calibrate", help="Run calibrated similarity on existing embeddings.")
    parser_calibrate.add_argument("parquet_file", help="Path to the Parquet file with embeddings.")
    parser_calibrate.add_argument("--ref", type=str, default=None, help="Path to a reference parquet file. Compares embeddings for --mode across the two files instead of comparing modes within one file.")
    parser_calibrate.add_argument("--mode", type=str, default=None, help="Mode to compare when using --ref (e.g., 'hsc'). Default: first mode in file.")
    parser_calibrate.add_argument("--metrics", nargs="+", default=["cka"], help="Metrics to calibrate.")
    parser_calibrate.add_argument("--k", type=int, default=10, help="K value for neighbor-based metrics (mknn, jaccard).")
    parser_calibrate.add_argument("--size", type=str, default=None, help="Model size to compare. Default: first size in file.")
    parser_calibrate.add_argument("--n-permutations", type=int, default=1000, help="Number of permutations for null distribution.")
    parser_calibrate.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    # Subparser for physics validation tests
    parser_physics = subparsers.add_parser(
        "run-physics",
        help="Test whether embeddings encode physical galaxy properties using Smith42/galaxies.",
    )
    parser_physics.add_argument(
        "--model", required=True,
        help="Model to test (e.g., 'vit', 'dino', 'convnext').",
    )
    parser_physics.add_argument(
        "--split", default="test", choices=["test", "validation", "train"],
        help="Dataset split to use (default: test).",
    )
    parser_physics.add_argument(
        "--max-samples", type=int, default=0,
        help="Max galaxies to process (default: all). Use 0 for all.",
    )
    parser_physics.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size for inference.",
    )
    parser_physics.add_argument(
        "--num-workers", type=int, default=0,
        help="Number of data loader workers.",
    )
    parser_physics.add_argument(
        "--knn-k", type=int, default=10,
        help="K for neighbour consistency metric.",
    )
    parser_physics.add_argument(
        "--cv", type=int, default=5,
        help="Cross-validation folds for linear probe.",
    )
    parser_physics.add_argument(
        "--properties", nargs="+", default=None,
        help="Physical properties to test (default: standard set). "
             "Options: stellar_mass, u_minus_r, redshift, sersic_n, "
             "smooth_fraction, spiral_arms, sfr, etc.",
    )
    parser_physics.add_argument(
        "--projection", default="pca", choices=["pca", "umap"],
        help="Dimensionality reduction for visualisation (default: pca).",
    )
    parser_physics.add_argument(
        "--from-parquet", action="store_true",
        help="Skip inference and load embeddings from saved parquet files in data/.",
    )
    parser_physics.add_argument(
        "--input-dir", default="data",
        help="Directory containing parquet files when using --from-parquet (default: data).",
    )
    parser_physics.add_argument(
        "--pca-components", type=int, default=None,
        help="Reduce embeddings to this many PCA components before linear probe "
             "(default: no PCA). PCA is fit per CV fold to avoid leakage.",
    )

    # Subparser for running physics tests across all models
    parser_physics_all = subparsers.add_parser(
        "run-physics-all",
        help="Run physics tests across all (or specified) models and produce a combined comparison.",
    )
    parser_physics_all.add_argument(
        "--split", default="test", choices=["test", "validation", "train"],
        help="Dataset split to use (default: test).",
    )
    parser_physics_all.add_argument(
        "--max-samples", type=int, default=0,
        help="Max galaxies to process (default: all). Use 0 for all.",
    )
    parser_physics_all.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size for inference.",
    )
    parser_physics_all.add_argument(
        "--models", nargs="+", default=None,
        help="Models to test (default: all in PHYSICS_MODEL_MAP).",
    )
    parser_physics_all.add_argument(
        "--from-parquet", action="store_true",
        help="Skip inference and load embeddings from saved parquet files.",
    )
    parser_physics_all.add_argument(
        "--input-dir", default="data",
        help="Directory containing parquet files when using --from-parquet (default: data).",
    )
    parser_physics_all.add_argument(
        "--pca-components", type=int, default=None,
        help="Reduce embeddings to this many PCA components before linear probe "
             "(default: no PCA). PCA is fit per CV fold to avoid leakage.",
    )

    # Subparser for computing dataset percentiles
    parser_percentiles = subparsers.add_parser("percentiles", help="Compute 1st/99th percentiles for dataset bands.")
    parser_percentiles.add_argument("--max-samples", type=int, default=10000, help="Max galaxies per dataset (default: 10000).")
    parser_percentiles.add_argument("--resize-mode", type=str, default="match", choices=["match", "fill"], help="Resize strategy (default: match).")
    parser_percentiles.add_argument("--output", type=str, default="data/percentiles.json", help="Output JSON path (default: data/percentiles.json).")

    # Subparser for pushing embeddings to HF Hub
    parser_push = subparsers.add_parser("push", help="Push parquet embeddings to a Hugging Face Hub dataset repo.")
    parser_push.add_argument("parquet_file", nargs="?", default=None, help="Path to a specific parquet file to push.")
    parser_push.add_argument("--all", action="store_true", dest="push_all", help="Push all data/*.parquet files.")
    parser_push.add_argument("--dataset", required=True, help="HF dataset repo ID (e.g., 'Smith42/my-embeddings').")
    parser_push.add_argument("--token", default=None, help="HF token (defaults to cached login).")
    # Subparser for layerwise extraction
    parser_extract = subparsers.add_parser("extract-layers", help="Extract embeddings from all layers of a model.")
    parser_extract.add_argument("--model", required=True, help="Model to extract (e.g., 'vit', 'dino').")
    parser_extract.add_argument("--mode", required=True, help="Dataset mode (e.g., 'jwst', 'desi').")
    parser_extract.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64, lower than run due to layerwise memory).")
    parser_extract.add_argument("--num-workers", type=int, default=0, help="Number of data loader workers.")
    parser_extract.add_argument("--no-resize", dest="resize", action="store_false", help="Disable galaxy resizing.")
    parser_extract.add_argument("--resize-mode", type=str, default="match", choices=["match", "fill"], help="Resize strategy (default: match).")
    parser_extract.add_argument("--test", action="store_true", help="Quick test run using only 1000 samples.")
    parser_extract.add_argument("--test-10k", action="store_true", help="Test run using only 10000 samples.")
    parser_extract.add_argument("--hf-repo", type=str, default=os.environ.get("PU_HF_REPO"), help="HuggingFace dataset repo ID for upload. Default: $PU_HF_REPO.")
    parser_extract.add_argument("--hf-token", type=str, default=None, help="HuggingFace token. Default: $HF_TOKEN env var.")
    parser_extract.add_argument("--no-upload", action="store_true", help="Disable HuggingFace upload (upload is on by default when --hf-repo is set).")
    parser_extract.add_argument("--delete-after-upload", action="store_true", help="Delete local parquet file after successful upload to HuggingFace. Saves disk space.")
    parser_extract.add_argument("--output-dir", type=str, default="data", help="Directory to write parquet files (default: data/).")
    parser_extract.add_argument("--granularity", type=str, default="blocks", choices=["blocks", "residual", "leaves", "all"], help="Extraction granularity: 'blocks' (default, ~14 for ViT-base, matches upstream PRH), 'residual' (~76, all non-leaf), 'leaves' (~137, leaf only), 'all' (~213, everything).")
    parser_extract.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")

    # Subparser for pushing parquet files to HuggingFace Hub
    parser_push = subparsers.add_parser("push", help="Upload parquet files to a HuggingFace dataset repo.")
    parser_push.add_argument("file", nargs="?", help="Path to a .parquet file to upload.")
    parser_push.add_argument("--all", action="store_true", help="Upload all .parquet files in data/.")
    parser_push.add_argument("--repo", required=True, help="HuggingFace dataset repo ID (e.g., 'org/dataset-name').")
    parser_push.add_argument("--token", type=str, default=None, help="HuggingFace token. Default: $HF_TOKEN env var.")

    # Subparser for benchmarking performance optimizations
    parser_benchmark = subparsers.add_parser("benchmark", help="Run performance benchmarks with optimization flags.")
    parser_benchmark.add_argument("--model", required=True, help="Model to benchmark (e.g., 'vit', 'dino').")
    parser_benchmark.add_argument("--mode", required=True, help="Dataset mode (e.g., 'jwst', 'legacysurvey').")
    parser_benchmark.add_argument("--size", default="base", help="Model size to benchmark (e.g., 'base', 'large'). Default: base.")
    parser_benchmark.add_argument("--batch-size", type=int, default=128, help="Batch size for processing.")
    parser_benchmark.add_argument("--num-workers", type=int, default=0, help="Number of data loader workers.")
    parser_benchmark.add_argument("--knn-k", type=int, default=10, help="K value for MKNN calculation.")
    # Optimization flags
    parser_benchmark.add_argument("--enable-amp", action="store_true", help="Enable automatic mixed precision (float16).")
    parser_benchmark.add_argument("--enable-compile", action="store_true", help="Use torch.compile for model optimization.")
    parser_benchmark.add_argument("--enable-cache", action="store_true", help="Cache embeddings to skip repeated inference.")
    parser_benchmark.add_argument("--pin-memory", action="store_true", help="Pin memory in DataLoader for faster GPU transfer.")
    parser_benchmark.add_argument("--persistent-workers", action="store_true", help="Keep DataLoader workers alive between batches.")
    parser_benchmark.add_argument("--warmup-batches", type=int, default=2, help="Number of warmup batches to exclude from timing.")
    parser_benchmark.add_argument("--max-samples", type=int, default=None, help="Limit dataset to N samples for quick testing.")
    parser_benchmark.add_argument("--no-streaming", action="store_true", help="Download dataset locally instead of streaming (faster iteration).")
    # Output
    parser_benchmark.add_argument("--output-json", type=str, default=None, help="Save benchmark results to JSON file.")
    parser_benchmark.add_argument("--compare-baseline", type=str, default=None, help="Compare results to a baseline JSON file.")

    args = parser.parse_args()

    PAIRED_MODES = {"sdss", "desi"}
    if args.command == "run":
        # Lazy import to avoid loading transformers/torchvision when using compare command
        from pu.experiments import run_experiment
        if args.mode in PAIRED_MODES and args.num_workers > 0:
            print(f"Warning: Setting num_workers=0 for paired mode '{args.mode}' because multiple workers can change draw order and break pairing.")
            args.num_workers = 0
        run_experiment(
            args.model,
            args.mode,
            args.output_dataset,
            args.batch_size,
            args.num_workers,
            args.knn_k,
            resize=args.resize,
            resize_mode=args.resize_mode,
            all_metrics=args.all_metrics,
            max_samples=1000 if args.test else 10000 if args.test_10k else None,
            plot_samples=args.test or args.test_10k,
        )
    elif args.command == "compare":
        # Lazy import to avoid loading transformers/torchvision
        from pu.metrics import compare_from_parquet, compare, load_single_embedding

        if args.ref:
            Z1, meta1 = load_single_embedding(args.parquet_file, size=args.size, mode=args.mode)
            Z2, meta2 = load_single_embedding(args.ref, mode=args.mode)
            metric_results = compare(Z1, Z2, metrics=args.metrics, mknn__k=args.k, jaccard__k=args.k)
            results = {
                "model1": meta1["model"], "size1": meta1["size"],
                "model2": meta2["model"], "size2": meta2["size"],
                "mode": args.mode,
                "metrics": metric_results,
            }
        else:
            results = compare_from_parquet(
                args.parquet_file,
                metrics=args.metrics,
                size=args.size,
                mknn__k=args.k,
                jaccard__k=args.k,
            )

        # Save the results to a JSON file under data
        output_file = f"data/{os.path.basename(args.parquet_file)}.json"
        os.makedirs("data", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)  # default=str handles numpy types

        # Print the results
        print(json.dumps(results, indent=2, default=str))
    elif args.command == "calibrate":
        from functools import partial
        from pu.metrics import calibrate, load_embeddings_from_parquet, load_single_embedding, METRICS_REGISTRY

        if args.ref:
            if args.mode is None:
                parser.error("--mode is required when using --ref")
            Z1, meta1 = load_single_embedding(args.parquet_file, size=args.size, mode=args.mode)
            Z2, meta2 = load_single_embedding(args.ref, mode=args.mode)
            metadata = {
                "model1": meta1["model"], "size1": meta1["size"],
                "model2": meta2["model"], "size2": meta2["size"],
                "mode": args.mode,
            }
        else:
            Z1, Z2, metadata = load_embeddings_from_parquet(args.parquet_file, size=args.size)

        results = {**metadata, "calibration": {}}
        for name in args.metrics:
            fn = METRICS_REGISTRY[name]
            if name in ("mknn", "jaccard"):
                fn = partial(fn, k=args.k)
            results["calibration"][name] = calibrate(
                Z1, Z2, fn,
                n_permutations=args.n_permutations,
                seed=args.seed,
            )

        output_file = f"data/{os.path.basename(args.parquet_file)}.calibrated.json"
        os.makedirs("data", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(json.dumps(results, indent=2, default=str))
    elif args.command == "run-physics":
        max_samples = args.max_samples if args.max_samples != 0 else None

        if args.from_parquet:
            from pu.physics_experiment import rerun_physics_from_parquet
            results = rerun_physics_from_parquet(
                model_alias=args.model,
                split=args.split,
                max_samples=max_samples,
                knn_k=args.knn_k,
                cv=args.cv,
                properties=args.properties,
                input_dir=args.input_dir,
                pca_components=args.pca_components,
            )
        else:
            from pu.physics_experiment import run_physics_experiment
            results = run_physics_experiment(
                model_alias=args.model,
                split=args.split,
                max_samples=max_samples,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                knn_k=args.knn_k,
                cv=args.cv,
                properties=args.properties,
                projection=args.projection,
                pca_components=args.pca_components,
            )

        # Print summary across sizes
        print(f"\n{'='*70}")
        print(f"PHYSICS TEST SUMMARY: {args.model}")
        print(f"{'='*70}")
        for size, size_data in results["sizes"].items():
            r2_mean = size_data.get("r2_mean")
            r2_se = size_data.get("r2_se")
            r2_mean_str = f"{r2_mean:.4f}" if r2_mean is not None else "N/A"
            r2_se_str = f"{r2_se:.4f}" if r2_se is not None else "N/A"
            print(f"\n  {args.model}-{size} ({size_data['n_samples']} samples, "
                  f"dim={size_data['embedding_dim']})  mean_R²={r2_mean_str} ±{r2_se_str}")
            for prop, metrics in size_data["properties"].items():
                lr2 = metrics.get("linear_probe_r2")
                lr2_str = f"{lr2:.4f}" if lr2 is not None else "N/A"
                print(f"    {prop:<25} linear_probe_r2={lr2_str}")
        print(f"{'='*70}")
    elif args.command == "run-physics-all":
        from pu.physics_experiment import PHYSICS_MODEL_MAP, run_physics_experiment, rerun_physics_from_parquet

        max_samples = args.max_samples if args.max_samples != 0 else None
        model_list = args.models or list(PHYSICS_MODEL_MAP.keys())

        combined = {"split": args.split, "models": {}}

        for model_alias in model_list:
            if model_alias not in PHYSICS_MODEL_MAP:
                print(f"Warning: '{model_alias}' not in PHYSICS_MODEL_MAP, skipping.")
                continue

            print(f"\n{'#'*70}")
            print(f"# Running physics tests for: {model_alias}")
            print(f"{'#'*70}")

            if args.from_parquet:
                results = rerun_physics_from_parquet(
                    model_alias=model_alias,
                    split=args.split,
                    max_samples=max_samples,
                    input_dir=args.input_dir,
                    pca_components=args.pca_components,
                )
            else:
                results = run_physics_experiment(
                    model_alias=model_alias,
                    split=args.split,
                    max_samples=max_samples,
                    batch_size=args.batch_size,
                    pca_components=args.pca_components,
                )

            model_entry = {}
            for size, size_data in results["sizes"].items():
                model_entry[size] = {
                    "r2_mean": size_data.get("r2_mean"),
                    "r2_se": size_data.get("r2_se"),
                    "r2_std": size_data.get("r2_std"),
                    "r2_per_property": size_data.get("r2_per_property"),
                    "n_samples": size_data["n_samples"],
                    "embedding_dim": size_data["embedding_dim"],
                }
            combined["models"][model_alias] = model_entry

        # Print comparison table
        print(f"\n{'='*70}")
        print("PHYSICS COMPARISON: mean R² across models")
        print(f"{'='*70}")
        print(f"  {'Model':<15} {'Size':<15} {'mean R² ± SE':<18} {'dim':<8} {'n':<8}")
        print(f"  {'-'*64}")
        for model_alias, sizes in combined["models"].items():
            for size, data in sizes.items():
                r2 = data.get("r2_mean")
                se = data.get("r2_se")
                if r2 is not None and se is not None:
                    r2_str = f"{r2:.4f} ±{se:.4f}"
                elif r2 is not None:
                    r2_str = f"{r2:.4f}"
                else:
                    r2_str = "N/A"
                print(f"  {model_alias:<15} {size:<15} {r2_str:<18} {data['embedding_dim']:<8} {data['n_samples']:<8}")
        print(f"{'='*70}")

        # Save combined JSON
        os.makedirs("data", exist_ok=True)
        output_path = f"data/physics_all_{args.split}.json"
        with open(output_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nCombined results saved to {output_path}")

    elif args.command == "percentiles":
        from pu.percentiles import compute_percentiles
        compute_percentiles(
            max_samples=args.max_samples,
            resize_mode=args.resize_mode,
            output_path=args.output,
        )
    elif args.command == "extract-layers":
        from pu.experiments_layerwise import extract_all_layers
        if args.mode in PAIRED_MODES and args.num_workers > 0:
            print(f"Warning: Setting num_workers=0 for paired mode '{args.mode}' because multiple workers can change draw order and break pairing.")
            args.num_workers = 0
        extract_all_layers(
            args.model,
            args.mode,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=1000 if args.test else 10000 if args.test_10k else None,
            resize=args.resize,
            resize_mode=args.resize_mode,
            output_dir=args.output_dir,
            hf_repo=args.hf_repo,
            hf_token=args.hf_token,
            upload=not args.no_upload,
            delete_after_upload=args.delete_after_upload,
            granularity=args.granularity,
            seed=args.seed,
        )
    elif args.command == "push":
        from pu.hub import push_parquet, push_all
        if args.all:
            push_all("data", args.repo, token=args.token)
        elif args.file:
            push_parquet(args.file, args.repo, token=args.token)
        else:
            parser.error("Specify a file or --all")
    elif args.command == "benchmark":
        from pu.benchmark import run_benchmark, BenchmarkConfig

        config = BenchmarkConfig(
            model_alias=args.model,
            model_size=args.size,
            mode=args.mode,
            batch_size=args.batch_size,
            knn_k=args.knn_k,
            enable_amp=args.enable_amp,
            enable_compile=args.enable_compile,
            enable_cache=args.enable_cache,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            warmup_batches=args.warmup_batches,
            max_samples=args.max_samples,
            no_streaming=args.no_streaming,
            output_json=args.output_json,
            compare_baseline=args.compare_baseline,
        )

        results = run_benchmark(config)

        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Total time: {results['total_time_seconds']:.2f}s")
        print(f"Throughput: {results['throughput']['samples_per_second']:.1f} samples/sec")
        print(f"Peak GPU memory: {results['memory']['peak_gpu_mb']:.0f} MB")
        print(f"MKNN score: {results['metrics']['mknn_k10']:.4f}")
        print(f"CKA score: {results['metrics']['cka']:.4f}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()

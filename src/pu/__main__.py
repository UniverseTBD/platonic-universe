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

    # Subparser for running metrics comparisons
    parser_comparisons = subparsers.add_parser("compare", help="Run metrics comparisons on existing embeddings.")
    parser_comparisons.add_argument("parquet_file", help="Path to the Parquet file with embeddings.")
    parser_comparisons.add_argument("--metrics", nargs="+", default=["all"], help="Metrics to run. Use 'all' for all metrics, or specify: cka, mmd, procrustes, cosine_similarity, frechet, svcca, pwcca, tucker_congruence, eigenspectrum, riemannian, kl_divergence, js_divergence, mutual_information, mknn, jaccard, rsa, linear_r2.")
    parser_comparisons.add_argument("--k", type=int, default=10, help="K value for neighbor-based metrics (mknn, jaccard).")
    parser_comparisons.add_argument("--size", type=str, default=None, help="Model size to compare (e.g., 'base', 'large', 'huge'). Use 'all' to process all sizes. Default: first size in file.")

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
        run_experiment(args.model, args.mode, args.output_dataset, args.batch_size, args.num_workers, args.knn_k)
    elif args.command == "compare":
        # Lazy import to avoid loading transformers/torchvision
        from pu.metrics import compare_from_parquet
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
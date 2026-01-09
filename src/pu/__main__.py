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
    # Physical parameter options
    parser_run.add_argument("--physical-params", nargs="+", default=None, 
                           help="Physical parameters to compute Wasserstein distances for (column names in dataset).")
    parser_run.add_argument("--n-samples", type=int, default=None,
                           help="Number of samples to use for physical parameter analysis (default: all).")
    parser_run.add_argument("--intramodal", action="store_true",
                           help="Run intra-modal comparison across model sizes instead of cross-modal.")

    # Subparser for running comparisons on existing embeddings
    parser_comparisons = subparsers.add_parser("compare", help="Run metrics comparisons on existing embeddings.")
    parser_comparisons.add_argument("parquet_file", help="Path to the Parquet file with embeddings.")
    parser_comparisons.add_argument("--metrics", nargs="+", default=["mknn", "jaccard", "cka", "rsm", "procrustes"], 
                                   help="Metrics to run (e.g., 'mknn', 'jaccard', 'cka', 'rsm', 'procrustes').")
    parser_comparisons.add_argument("--k", type=int, default=10, help="K value for mutual KNN calculation.")
    parser_comparisons.add_argument("--size", type=str, default=None, 
                                   help="Model size to compare (e.g., 'base', 'large'). Use 'all' for all sizes.")
    args = parser.parse_args()

    PAIRED_MODES = {"sdss", "desi"}
    if args.command == "run":
        # Lazy import to avoid loading transformers/torchvision when using compare command
        from pu.experiments import run_experiment
        if args.mode in PAIRED_MODES and args.num_workers > 0:
            print(f"Warning: Setting num_workers=0 for paired mode '{args.mode}' because multiple workers can change draw order and break pairing.")
            args.num_workers = 0
        results = run_experiment(
            model_alias=args.model,
            mode=args.mode,
            output_dataset=args.output_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            knn_k=args.knn_k,
            physical_params=args.physical_params,
            n_samples=args.n_samples,
            intramodal=args.intramodal,
        )
        
        # Save results
        experiment_type = "intra" if args.intramodal else "cross"
        output_file = f"data/{args.mode}_{args.model}_{experiment_type}_results.json"
        os.makedirs("data", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_file}")
        
    elif args.command == "compare":
        # Lazy import to avoid loading transformers/torchvision
        from pu.metrics import run_comparisons
        results = run_comparisons(args.parquet_file, args.metrics, args.k, size=args.size)

        # Save the results to a JSON file under data #TODO: Make this more robust
        output_file = f"data/{os.path.basename(args.parquet_file)}.json"
        os.makedirs("data", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()

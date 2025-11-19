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

    # Subparser for running mknn comparisons
    parser_comparisons = subparsers.add_parser("compare", help="Run metrics comparisons on existing embeddings.")
    parser_comparisons.add_argument("parquet_file", help="Path to the Parquet file with embeddings.")
    parser_comparisons.add_argument("--metrics", nargs="+", default=["mknn", "jaccard", "cka", "rsm", "procrustes"], help="Metrics to run (e.g., 'mknn', 'jaccard', 'cka', 'rsm', 'procrustes').")
    parser_comparisons.add_argument("--k", type=int, default=10, help="K value for mutual KNN calculation.")
    parser_comparisons.add_argument("--size", type=str, default=None, help="Model size to compare (e.g., 'base', 'large', 'huge'). Use 'all' to process all sizes. Default: first size in file.")
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
        from pu.metrics import run_comparisons
        results = run_comparisons(args.parquet_file, args.metrics, args.k, size=args.size)

        # Save the results to a JSON file under data #TODO: Make this more robust
        output_file = f"data/{os.path.basename(args.parquet_file)}.json"
        os.makedirs("data", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)  # default=str handles numpy types

        # Print the results
        print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    main()
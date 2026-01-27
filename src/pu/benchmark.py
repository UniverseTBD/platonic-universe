"""
Benchmarking infrastructure for performance optimization testing.

This module provides timing utilities and a benchmark runner for comparing
different optimization strategies on the platonic-universe pipeline.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pu.models import get_adapter
from pu.pu_datasets import get_dataset_adapter
from pu.metrics import mknn, compute_cka_mmap
from pu.utils import write_bin
import tempfile


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    # Model settings
    model_alias: str = "vit"
    model_size: str = "base"
    mode: str = "jwst"
    batch_size: int = 128
    knn_k: int = 10

    # Optimization flags
    enable_amp: bool = False
    enable_compile: bool = False
    enable_cache: bool = False

    # DataLoader settings
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2

    # Benchmark settings
    warmup_batches: int = 2
    profile_memory: bool = True
    max_samples: Optional[int] = None  # Limit samples for quick testing
    no_streaming: bool = False  # Download dataset locally instead of streaming

    # Output
    output_json: Optional[str] = None
    compare_baseline: Optional[str] = None


@dataclass
class TimingResult:
    """Stores timing results for a benchmark phase."""
    phase: str
    duration_seconds: float
    samples_processed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def throughput(self) -> float:
        """Samples per second."""
        if self.duration_seconds > 0 and self.samples_processed > 0:
            return self.samples_processed / self.duration_seconds
        return 0.0


class BenchmarkTimer:
    """Context manager for timing code blocks with GPU synchronization."""

    def __init__(self, name: str, cuda_sync: bool = True):
        self.name = name
        self.cuda_sync = cuda_sync and torch.cuda.is_available()
        self.start_time = 0.0
        self.duration = 0.0

    def __enter__(self):
        if self.cuda_sync:
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.cuda_sync:
            torch.cuda.synchronize()
        self.duration = time.perf_counter() - self.start_time


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_gpu_memory_stats():
    """Reset GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def run_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    """
    Run benchmark with specified configuration.

    Returns a dictionary with timing results and metrics.
    """
    timings: Dict[str, TimingResult] = {}
    reset_gpu_memory_stats()

    comp_mode = config.mode
    modes = ["hsc", comp_mode]
    hf_ds = f"Smith42/{comp_mode}_hsc_crossmatched"

    # Model configuration - use only specified size
    model_map = {
        "vit": {
            "base": "google/vit-base-patch16-224-in21k",
            "large": "google/vit-large-patch16-224-in21k",
            "huge": "google/vit-huge-patch14-224-in21k",
        },
        "dino": {
            "small": "facebook/dinov2-with-registers-small",
            "base": "facebook/dinov2-with-registers-base",
            "large": "facebook/dinov2-with-registers-large",
            "giant": "facebook/dinov2-with-registers-giant",
        },
    }

    if config.model_alias not in model_map:
        raise ValueError(f"Model '{config.model_alias}' not supported for benchmarking")

    if config.model_size not in model_map[config.model_alias]:
        raise ValueError(f"Size '{config.model_size}' not available for {config.model_alias}")

    model_name = model_map[config.model_alias][config.model_size]

    # Filter function for JWST
    def filterfun(idx):
        if "jwst" != comp_mode:
            return True
        im = idx["jwst_image"]["flux"][3]
        v0, v1 = np.nanpercentile(im, 5), np.nanpercentile(im, 99)
        return v0 - v1 != 0

    # === Phase 1: Model Loading ===
    print(f"\n[Benchmark] Loading model: {config.model_alias}-{config.model_size}")
    with BenchmarkTimer("model_loading") as t:
        adapter_cls = get_adapter(config.model_alias)
        adapter = adapter_cls(model_name, config.model_size, alias=config.model_alias)
        adapter.load(compile_model=config.enable_compile)

        # Enable AMP if requested
        if config.enable_amp:
            adapter.enable_amp(True)

        processor = adapter.get_preprocessor(modes)

    timings["model_loading"] = TimingResult("model_loading", t.duration)
    print(f"  Model loaded in {t.duration:.2f}s")

    # === Phase 2: Data Loading ===
    print(f"\n[Benchmark] Loading dataset: {hf_ds}")
    with BenchmarkTimer("data_loading") as t:
        if config.no_streaming:
            # Download and cache dataset locally for faster iteration
            from datasets import load_dataset
            print("  [Downloading dataset locally (no streaming)...]")
            raw_ds = load_dataset(hf_ds, split="train")

            # Apply filter
            raw_ds = raw_ds.filter(filterfun)

            # Limit samples if specified
            if config.max_samples is not None:
                raw_ds = raw_ds.select(range(min(config.max_samples, len(raw_ds))))
                print(f"  [Limited to {config.max_samples} samples]")

            # Apply preprocessing
            ds = raw_ds.map(processor, remove_columns=[f"{mode}_image" for mode in modes])
            ds.set_format("torch")
        else:
            # Use streaming mode (original behavior)
            dataset_adapter_cls = get_dataset_adapter(comp_mode)
            dataset_adapter = dataset_adapter_cls(hf_ds, comp_mode)
            dataset_adapter.load()
            ds = dataset_adapter.prepare(processor, modes, filterfun)

            # Limit samples for quick testing if specified
            if config.max_samples is not None:
                ds = ds.take(config.max_samples)
                print(f"  [Limited to {config.max_samples} samples (streaming)]")

        # Create DataLoader with optimizations
        dl_kwargs = {
            'batch_size': config.batch_size,
            'num_workers': config.num_workers,
        }
        if config.num_workers > 0:
            dl_kwargs['pin_memory'] = config.pin_memory
            dl_kwargs['persistent_workers'] = config.persistent_workers
            if config.persistent_workers:
                dl_kwargs['prefetch_factor'] = config.prefetch_factor

        dl = DataLoader(ds, **dl_kwargs)

    timings["data_loading"] = TimingResult("data_loading", t.duration)
    print(f"  Dataset loaded in {t.duration:.2f}s")

    # === Phase 3: Inference ===
    print(f"\n[Benchmark] Running inference (AMP={config.enable_amp}, compile={config.enable_compile})")

    zs = {mode: [] for mode in modes}
    batch_times = []
    n_samples = 0

    with BenchmarkTimer("inference_total") as t_total:
        with torch.no_grad():
            for batch_idx, B in enumerate(tqdm(dl, desc="Inference")):
                with BenchmarkTimer(f"batch_{batch_idx}", cuda_sync=True) as t_batch:
                    for mode in modes:
                        if mode == "sdss":
                            zs[mode].append(torch.tensor(np.array(B["embedding"])).T)
                        elif mode == "desi":
                            zs[mode].append(torch.tensor(np.array(B["embeddings"])).T)
                        else:
                            outputs = adapter.embed_for_mode(B, mode)
                            zs[mode].append(outputs)

                # Skip warmup batches for timing stats
                if batch_idx >= config.warmup_batches:
                    batch_times.append(t_batch.duration)

                n_samples += B[modes[0]].shape[0] if modes[0] in B else config.batch_size

    # Concatenate embeddings
    zs = {mode: torch.cat(embs) for mode, embs in zs.items()}

    avg_batch_time = np.mean(batch_times) if batch_times else 0.0
    timings["inference"] = TimingResult(
        "inference",
        t_total.duration,
        samples_processed=n_samples,
        extra={
            "warmup_batches": config.warmup_batches,
            "avg_batch_time": avg_batch_time,
            "batch_times_after_warmup": batch_times,
        }
    )
    print(f"  Inference completed in {t_total.duration:.2f}s ({n_samples} samples)")
    print(f"  Average batch time (post-warmup): {avg_batch_time*1000:.1f}ms")

    # === Phase 4: MKNN Computation ===
    print(f"\n[Benchmark] Computing MKNN (k={config.knn_k})")

    Z1 = zs[modes[0]].cpu().numpy()
    Z2 = zs[modes[1]].cpu().numpy()

    with BenchmarkTimer("mknn") as t:
        mknn_score = mknn(Z1, Z2, k=config.knn_k)

    timings["mknn"] = TimingResult("mknn", t.duration, samples_processed=len(Z1))
    print(f"  MKNN computed in {t.duration:.2f}s (score={mknn_score:.4f})")

    # === Phase 5: CKA Computation ===
    print(f"\n[Benchmark] Computing CKA")

    with BenchmarkTimer("cka") as t:
        temp1 = tempfile.NamedTemporaryFile(delete=False)
        temp2 = tempfile.NamedTemporaryFile(delete=False)
        temp1.close()
        temp2.close()

        k1 = Z1 @ Z1.T
        k2 = Z2 @ Z2.T

        write_bin(k1, str(temp1.name))
        write_bin(k2, str(temp2.name))

        cka_score = compute_cka_mmap(str(temp1.name), str(temp2.name), k1.shape[0], k1.shape[1])

        # Cleanup temp files
        Path(temp1.name).unlink(missing_ok=True)
        Path(temp2.name).unlink(missing_ok=True)

    timings["cka"] = TimingResult("cka", t.duration, samples_processed=len(Z1))
    print(f"  CKA computed in {t.duration:.2f}s (score={cka_score:.4f})")

    # === Compile Results ===
    total_time = sum(tr.duration_seconds for tr in timings.values())
    peak_memory = get_gpu_memory_mb()

    results = {
        "run_id": f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "config": asdict(config),
        "dataset": {
            "name": hf_ds,
            "samples": n_samples,
        },
        "timings": {
            name: {
                "duration_seconds": tr.duration_seconds,
                "samples_processed": tr.samples_processed,
                "throughput_sps": tr.throughput,
                **tr.extra,
            }
            for name, tr in timings.items()
        },
        "total_time_seconds": total_time,
        "throughput": {
            "samples_per_second": n_samples / total_time if total_time > 0 else 0,
        },
        "memory": {
            "peak_gpu_mb": peak_memory,
        },
        "metrics": {
            "mknn_k10": mknn_score,
            "cka": cka_score,
        },
    }

    # === Save Results ===
    if config.output_json:
        output_path = Path(config.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[Benchmark] Results saved to {output_path}")

    # === Compare with Baseline ===
    if config.compare_baseline:
        compare_results(results, config.compare_baseline)

    return results


def compare_results(current: Dict[str, Any], baseline_path: str):
    """Compare current results with a baseline JSON file."""
    try:
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
    except FileNotFoundError:
        print(f"\n[Warning] Baseline file not found: {baseline_path}")
        return

    print(f"\n{'='*60}")
    print("COMPARISON WITH BASELINE")
    print(f"{'='*60}")
    print(f"Baseline: {baseline.get('run_id', 'unknown')}")
    print(f"Current:  {current.get('run_id', 'unknown')}")
    print(f"{'-'*60}")

    # Compare timings
    print(f"\n{'Phase':<20} {'Baseline (s)':<15} {'Current (s)':<15} {'Speedup':<10}")
    print(f"{'-'*60}")

    for phase in current.get("timings", {}):
        curr_time = current["timings"][phase]["duration_seconds"]
        base_time = baseline.get("timings", {}).get(phase, {}).get("duration_seconds", curr_time)

        if base_time > 0:
            speedup = base_time / curr_time
            speedup_str = f"{speedup:.2f}x" if speedup >= 1 else f"{1/speedup:.2f}x slower"
        else:
            speedup_str = "N/A"

        print(f"{phase:<20} {base_time:<15.2f} {curr_time:<15.2f} {speedup_str:<10}")

    # Total
    curr_total = current.get("total_time_seconds", 0)
    base_total = baseline.get("total_time_seconds", curr_total)
    if base_total > 0:
        total_speedup = base_total / curr_total
        total_str = f"{total_speedup:.2f}x"
    else:
        total_str = "N/A"

    print(f"{'-'*60}")
    print(f"{'TOTAL':<20} {base_total:<15.2f} {curr_total:<15.2f} {total_str:<10}")

    # Verify metrics match
    print(f"\n{'Metric Verification':<40}")
    print(f"{'-'*40}")
    for metric in ["mknn_k10", "cka"]:
        curr_val = current.get("metrics", {}).get(metric, 0)
        base_val = baseline.get("metrics", {}).get(metric, 0)
        diff = abs(curr_val - base_val)
        status = "OK" if diff < 0.01 else "DIFF"
        print(f"{metric:<15} base={base_val:.4f} curr={curr_val:.4f} [{status}]")

    print(f"{'='*60}\n")

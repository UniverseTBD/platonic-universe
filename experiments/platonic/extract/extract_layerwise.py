"""Cooperative layer-wise embedding extraction worker.

For each ``(survey, family, size)`` tuple in the chosen grid this worker:

  1. Atomically claims the tuple via a ``running/<tag>.running`` marker on
     a coordination HF dataset (same pattern as ``pu_regress.py``), so any
     number of workers across any number of clusters cooperate without
     speaking to each other.
  2. Calls ``extract_layerwise_one`` to extract per-layer hidden states
     for both bands of the survey (e.g. HSC + JWST for cosmosweb), with
     the ``lm_head`` filter applied so vocab-projection columns are
     dropped before parquet write.
  3. Writes the parquet to
     ``<out-dir>/<survey>/<family>_<size>_blocks_layerwise.parquet`` and,
     if ``--upload-to`` is set, ships it to the HF dataset repo.

Idempotent and restart-safe: stale claims (>1 h) auto-released, finished
tuples skipped via the same `done/<tag>.done` marker.

Required environment
--------------------
    HF_TOKEN              token with write access to --upload-to
    PU_LW_OUT_DIR         persistent local output dir (shared across workers)
    PU_LW_LOCK_DIR        persistent local lock dir
    PU_LW_COORD_REPO      HF dataset id used for cooperation, e.g.
                          <owner>/pu-cosmosweb-layerwise

Optional environment
--------------------
    PU_LW_SURVEYS         space- or comma-sep list, default "cosmosweb"
    PU_LW_MODELS          space- or comma-sep family aliases; default = all
                          registered families (less the spectral models)
    PU_LW_BATCH_SIZE      default 32; lower for VLMs
    PU_LW_GRANULARITY     default "blocks" — "blocks"/"residual"/"leaves"/"all"
    PU_LW_STALE_S         default 3600 — auto-release stale running markers
    PU_LW_EXCLUDE_NAMES   default "lm_head" — comma-sep list of substrings
                          to drop from layer column names
    PU_LW_TARGET          single-tuple mode: "<survey>/<family>_<size>"
"""
from __future__ import annotations

import argparse
import gc
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

DEFAULT_FAMILIES = [
    "vit", "clip", "convnext", "vit-mae", "astropt", "ijepa",
    "dinov3", "vjepa",
    "llava_15", "paligemma_3b", "paligemma_10b", "paligemma_28b",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--surveys", nargs="+",
                   default=_split_env("PU_LW_SURVEYS", ["cosmosweb"]),
                   help="Survey aliases (registered in SURVEY_REGISTRY).")
    p.add_argument("--models", nargs="+",
                   default=_split_env("PU_LW_MODELS", DEFAULT_FAMILIES),
                   help="Model family aliases (registered in pu.models).")
    p.add_argument("--out-dir", type=Path,
                   default=os.environ.get("PU_LW_OUT_DIR"),
                   help="Persistent local output dir.")
    p.add_argument("--lock-dir", type=Path,
                   default=os.environ.get("PU_LW_LOCK_DIR"),
                   help="Persistent local lock dir.")
    p.add_argument("--coord-repo",
                   default=os.environ.get("PU_LW_COORD_REPO"),
                   help="HF dataset id used as the coordination & artifact repo.")
    p.add_argument("--upload-to",
                   default=os.environ.get("PU_LW_COORD_REPO"),
                   help="Where finished parquets land. Defaults to --coord-repo.")
    p.add_argument("--batch-size", type=int,
                   default=int(os.environ.get("PU_LW_BATCH_SIZE", "32")))
    p.add_argument("--granularity",
                   default=os.environ.get("PU_LW_GRANULARITY", "blocks"))
    p.add_argument("--max-samples", type=int, default=None,
                   help="If set, take only this many samples per tuple "
                        "(smoke-test knob).")
    p.add_argument("--target",
                   default=os.environ.get("PU_LW_TARGET", ""),
                   help='Single-tuple mode, e.g. "cosmosweb/vit_base".')
    p.add_argument("--no-coord", action="store_true",
                   help="Disable HF coordination; just iterate locally and "
                        "skip already-present parquets. For one-box runs.")
    return p.parse_args()


def _split_env(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    parts = [p for p in raw.replace(",", " ").split() if p]
    return parts or default


def log(*a, **kw):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}]", *a, flush=True, **kw)


# ---------------------------------------------------------------------------
# HF coordination — ported from experiments/platonic/regress/01_extract_and_probe.py
# ---------------------------------------------------------------------------
def _hf_state(api, repo: str
              ) -> tuple[set[str], dict[str, datetime]]:
    """Returns (done_tags, running_ages). Mirrors pu_regress.hf_list_state."""
    from huggingface_hub.utils import RepositoryNotFoundError
    try:
        files = api.list_repo_files(repo, repo_type="dataset")
    except RepositoryNotFoundError:
        return set(), {}
    done = {f[5:].rsplit(".done", 1)[0]
            for f in files if f.startswith("done/") and f.endswith(".done")}
    running_files = [f for f in files
                     if f.startswith("running/") and f.endswith(".running")]
    ages: dict[str, datetime] = {}
    for f in running_files:
        tag = f[8:].rsplit(".running", 1)[0]
        try:
            commits = api.list_repo_commits(repo, repo_type="dataset")
            for c in commits:
                if any(p == f for p in (c.commit_message or "").split()):
                    ages[tag] = c.created_at
                    break
            ages.setdefault(tag, datetime.now(timezone.utc))
        except Exception:
            ages[tag] = datetime.now(timezone.utc)
    return done, ages


def _try_claim(api, repo: str, tag: str, lock_dir: Path) -> bool:
    """Atomically claim a tuple. Local O_EXCL + HF marker upload."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_dir / f"{tag}.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return False
    try:
        # Upload HF marker. Tag-named so the same path is reused across
        # workers — last writer wins, but the local lock above prevents
        # two workers on the same box from trying simultaneously.
        from io import BytesIO
        api.upload_file(
            path_or_fileobj=BytesIO(b""),
            path_in_repo=f"running/{tag}.running",
            repo_id=repo, repo_type="dataset",
            commit_message=f"running {tag}",
        )
        return True
    except Exception as e:
        log(f"  HF claim upload failed for {tag}: {e}; releasing local lock")
        try: lock.unlink()
        except FileNotFoundError: pass
        return False


def _release_claim(api, repo: str, tag: str, lock_dir: Path,
                   *, mark_done: bool) -> None:
    lock = lock_dir / f"{tag}.lock"
    try: lock.unlink()
    except FileNotFoundError: pass
    try:
        api.delete_file(path_in_repo=f"running/{tag}.running",
                        repo_id=repo, repo_type="dataset",
                        commit_message=f"release {tag}")
    except Exception as e:
        log(f"  HF release failed for {tag}: {e}")
    if mark_done:
        try:
            from io import BytesIO
            api.upload_file(
                path_or_fileobj=BytesIO(b""),
                path_in_repo=f"done/{tag}.done",
                repo_id=repo, repo_type="dataset",
                commit_message=f"done {tag}",
            )
        except Exception as e:
            log(f"  HF done-marker failed for {tag}: {e}")


def _reap_stale(api, repo: str, ages: dict[str, datetime], stale_s: int) -> None:
    now = datetime.now(timezone.utc)
    for tag, claimed in list(ages.items()):
        age_s = (now - claimed).total_seconds()
        if age_s > stale_s:
            log(f"  reaping stale claim {tag} (age={age_s:.0f}s)")
            try:
                api.delete_file(path_in_repo=f"running/{tag}.running",
                                repo_id=repo, repo_type="dataset",
                                commit_message=f"reap stale {tag}")
            except Exception as e:
                log(f"    [warn] {e}")


# ---------------------------------------------------------------------------
# Work list
# ---------------------------------------------------------------------------
def build_work_list(surveys, families):
    """Return [(survey, family, size, hf_id), ...] for the requested grid."""
    from pu.experiments_layerwise import MODEL_MAP, SURVEY_REGISTRY
    work = []
    for survey in surveys:
        if survey not in SURVEY_REGISTRY:
            print(f"[fatal] unknown survey {survey!r}; known: "
                  f"{sorted(SURVEY_REGISTRY)}", file=sys.stderr)
            sys.exit(2)
        for fam in families:
            if fam not in MODEL_MAP:
                print(f"[fatal] unknown model alias {fam!r}; known: "
                      f"{sorted(MODEL_MAP)}", file=sys.stderr)
                sys.exit(2)
            sizes, hf_ids = MODEL_MAP[fam]
            for size, hf_id in zip(sizes, hf_ids):
                work.append((survey, fam, size, hf_id))
    return work


def tag_for(survey: str, family: str, size: str) -> str:
    return f"{survey}__{family}_{size}"


def output_path_for(out_dir: Path, survey: str, family: str, size: str) -> Path:
    return out_dir / survey / f"{family}_{size}_blocks_layerwise.parquet"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    if args.out_dir is None:
        print("[fatal] --out-dir or PU_LW_OUT_DIR is required", file=sys.stderr)
        return 2
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    use_coord = not args.no_coord
    if use_coord:
        if not args.coord_repo:
            print("[fatal] --coord-repo or PU_LW_COORD_REPO required "
                  "(or pass --no-coord)", file=sys.stderr)
            return 2
        if args.lock_dir is None:
            print("[fatal] --lock-dir or PU_LW_LOCK_DIR required", file=sys.stderr)
            return 2
        args.lock_dir = Path(args.lock_dir)
        args.lock_dir.mkdir(parents=True, exist_ok=True)
    # If --upload-to wasn't explicitly set, fall back to the coord repo so
    # parquets ship to the same place where the .done markers live.
    if not args.upload_to and args.coord_repo:
        args.upload_to = args.coord_repo

    excludes = tuple(s for s in os.environ.get("PU_LW_EXCLUDE_NAMES", "lm_head")
                     .split(",") if s.strip())
    stale_s = int(os.environ.get("PU_LW_STALE_S", "3600"))

    log(f"worker pid={os.getpid()} on {os.uname().nodename}")
    log(f"surveys={args.surveys}  families={args.models}")
    log(f"out_dir={args.out_dir}  coord={'on' if use_coord else 'off'}  "
        f"upload_to={args.upload_to}")
    log(f"exclude_substrings={excludes}  granularity={args.granularity}  "
        f"batch_size={args.batch_size}")

    work = build_work_list(args.surveys, args.models)
    if args.target:
        try:
            t_survey, fam_size = args.target.split("/", 1)
            t_family, t_size = fam_size.rsplit("_", 1)
        except ValueError:
            log(f"[fatal] bad --target {args.target!r}; expected "
                "'<survey>/<family>_<size>'")
            return 2
        work = [w for w in work
                if w[0] == t_survey and w[1] == t_family and w[2] == t_size]
        if not work:
            log(f"[fatal] target {args.target} not in grid")
            return 2

    rng = random.Random((os.getpid() << 16) ^ int(time.time()))
    rng.shuffle(work)
    log(f"work queue: {len(work)} (survey, family, size) tuples")

    api = None
    if use_coord or args.upload_to:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))

    done_tags: set[str] = set()
    running_ages: dict[str, datetime] = {}
    if use_coord:
        done_tags, running_ages = _hf_state(api, args.coord_repo)
        _reap_stale(api, args.coord_repo, running_ages, stale_s)
        log(f"already done on coord: {len(done_tags)}; running: "
            f"{len(running_ages)}")

    from pu.experiments_layerwise import extract_layerwise_one

    # Outer progress: shows our worker's traversal of the queue. Most ticks
    # are sub-second skips (already done / claimed); the slow ticks are the
    # tuples we actually process. Postfix shows our local outcomes plus the
    # last-known global done count refreshed every tick when coord is on.
    pbar = tqdm(work, desc="tuples", unit="tuple", dynamic_ncols=True)
    n_local_done = n_local_skipped = n_local_failed = 0

    for survey, family, size, hf_id in pbar:
        tag = tag_for(survey, family, size)
        out_path = output_path_for(args.out_dir, survey, family, size)

        if out_path.exists():
            n_local_skipped += 1
            pbar.set_postfix(done=n_local_done, skip=n_local_skipped,
                             err=n_local_failed,
                             global_=f"{len(done_tags)}/{len(work)}")
            log(f"[skip-local] {tag}: {out_path} present")
            continue
        if use_coord:
            if tag in done_tags:
                n_local_skipped += 1
                continue
            if tag in running_ages:
                n_local_skipped += 1
                continue
            if not _try_claim(api, args.coord_repo, tag, args.lock_dir):
                n_local_skipped += 1
                continue

        log(f"[claim] {tag} ({hf_id})")
        ok = False
        try:
            extract_layerwise_one(
                family=family, size=size, hf_id=hf_id, survey=survey,
                output_path=out_path,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                granularity=args.granularity,
                exclude_substrings=excludes,
            )
            ok = out_path.exists()
            if ok and args.upload_to:
                api.upload_file(
                    path_or_fileobj=str(out_path),
                    path_in_repo=f"{survey}/{out_path.name}",
                    repo_id=args.upload_to, repo_type="dataset",
                    commit_message=f"layerwise {tag}",
                )
                log(f"[upload] {tag} -> {args.upload_to}/{survey}/{out_path.name}")
            log(f"[done] {tag}")
            n_local_done += 1
            done_tags.add(tag)  # local cache for the postfix counter
        except Exception:
            log(f"[ERROR] {tag}\n{traceback.format_exc()}")
            n_local_failed += 1
        finally:
            if use_coord:
                _release_claim(api, args.coord_repo, tag, args.lock_dir,
                               mark_done=ok)
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            pbar.set_postfix(done=n_local_done, skip=n_local_skipped,
                             err=n_local_failed,
                             global_=f"{len(done_tags)}/{len(work)}")

    pbar.close()
    log(f"queue exhausted: local done={n_local_done} "
        f"skipped={n_local_skipped} failed={n_local_failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

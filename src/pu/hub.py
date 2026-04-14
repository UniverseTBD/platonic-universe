"""Upload parquet embeddings to a Hugging Face Hub dataset repo."""

from __future__ import annotations

import glob
import os
import re


def _parse_filename(parquet_path: str) -> tuple[str, str, str]:
    """Parse a parquet filename into (subdirectory, filename_in_repo, config_name).

    E.g. data/jwst_vit_base.parquet -> ("jwst", "vit_base.parquet", "jwst_vit_base")
    """
    basename = os.path.basename(parquet_path)
    stem = basename.removesuffix(".parquet")

    parts = stem.split("_", 1)
    if len(parts) < 2:
        raise ValueError(
            f"Cannot parse '{basename}': expected {{mode}}_{{model}}_{{size}}.parquet"
        )

    mode = parts[0]
    remainder = parts[1]
    return mode, f"{remainder}.parquet", stem


def _parse_readme_configs(readme_text: str) -> tuple[list[dict], str]:
    """Extract YAML configs list and body from a README.md with front matter."""
    match = re.match(r"^---\n(.*?)\n---\n?(.*)", readme_text, re.DOTALL)
    if not match:
        return [], ""

    import yaml
    front_matter = yaml.safe_load(match.group(1)) or {}
    body = match.group(2)
    return front_matter.get("configs", []), body


def _build_readme(configs: list[dict], body: str = "") -> str:
    """Build a README.md string with YAML front matter configs."""
    import yaml
    front_matter = yaml.dump(
        {"configs": configs},
        default_flow_style=False,
        sort_keys=False,
    )
    return f"---\n{front_matter}---\n{body}"


def push_parquet(
    parquet_path: str,
    repo_id: str,
    *,
    token: str | None = None,
) -> None:
    """Upload a single parquet file to an HF dataset repo.

    1. Creates the repo if it doesn't exist.
    2. Parses the filename to determine subdirectory layout.
    3. Uploads the file.
    4. Updates the README.md with a dataset config entry.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    mode, filename, config_name = _parse_filename(parquet_path)
    path_in_repo = f"{mode}/{filename}"

    api.upload_file(
        path_or_fileobj=parquet_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded {parquet_path} -> {repo_id}/{path_in_repo}")

    _update_readme_config(api, repo_id, config_name, path_in_repo)


def _update_readme_config(
    api,
    repo_id: str,
    config_name: str,
    path_in_repo: str,
) -> None:
    """Download, update, and re-upload README.md with the new config entry."""
    try:
        readme_path = api.hf_hub_download(
            repo_id, "README.md", repo_type="dataset"
        )
        with open(readme_path) as f:
            readme_text = f.read()
    except Exception:
        readme_text = ""

    configs, body = _parse_readme_configs(readme_text)

    new_entry = {
        "config_name": config_name,
        "data_files": [{"split": "train", "path": path_in_repo}],
    }

    replaced = False
    for i, cfg in enumerate(configs):
        if cfg.get("config_name") == config_name:
            configs[i] = new_entry
            replaced = True
            break
    if not replaced:
        configs.append(new_entry)

    configs.sort(key=lambda c: c["config_name"])

    readme_content = _build_readme(configs, body)

    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False
    ) as tmp:
        tmp.write(readme_content)
        tmp_path = tmp.name

    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
    finally:
        os.unlink(tmp_path)

    print(f"Updated README.md configs in {repo_id} (config: {config_name})")


def push_all(
    data_dir: str,
    repo_id: str,
    *,
    token: str | None = None,
) -> None:
    """Upload all .parquet files in data_dir to an HF dataset repo."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        print(f"No .parquet files found in {data_dir}")
        return

    print(f"Found {len(files)} parquet file(s) to upload")
    for path in files:
        push_parquet(path, repo_id, token=token)

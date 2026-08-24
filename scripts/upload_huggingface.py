from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a validated standalone Texelator model")
    parser.add_argument("directory")
    parser.add_argument("repo_id")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    directory = Path(args.directory).expanduser().resolve()
    if not (directory / "texelator.json").is_file():
        raise RuntimeError(f"not a standalone Texelator repository: {directory}")
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=directory,
        commit_message="Publish standalone Texelator AW-BC4 model",
    )
    print(f"Uploaded https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

# Publishing the repository

Run the local checks before creating the public GitHub repository:

```bash
python3 scripts/check_repository.py
python3 -m pytest -q
bash -n scripts/install.sh
```

Create the repository as `texelator`. If GitHub CLI is installed and
authenticated:

```bash
git init -b main
git add .
git commit -m "Initial public Texelator runtime"
gh repo create texelator --public --source=. --remote=origin --push
```

Add the arXiv URL to `CITATION.cff` after the identifier is assigned.

Do not attach original model weights or converted BC4 artifacts to the Git repository.
Standalone converted models belong in separate Hugging Face model repositories, never
in Git. Build and validate Qwen3.8-27B with:

```bash
bash scripts/package_qwen38_27b.sh SOURCE_CHECKPOINT COMPLETED_ARTIFACT OUTPUT_REPOSITORY
texelator benchmark OUTPUT_REPOSITORY
texelator run OUTPUT_REPOSITORY "Reply with one sentence."
python scripts/upload_huggingface.py OUTPUT_REPOSITORY OWNER/REPOSITORY
```

Keep the upstream license, model revision, exact palette hash, package report, and
hardware compatibility in the model repository. The shared weight artifact may list
sm120 as validated only after an RTX 50-series runtime test reproduces the palette hash.

## GitHub release

Build the native Windows and Linux/WSL2 source archives from a clean checkout. The
Windows archive contains the runtime, installer, documentation, and tests. The
Linux/WSL2 archive is a clean tagged source snapshot. Neither archive contains model
weights, caches, or generated profiles. Publish both archives and their SHA256 files:

```bash
gh release create v0.2.0 \
  texelator-v0.2.0-windows.zip \
  texelator-v0.2.0-windows.zip.sha256 \
  texelator-v0.2.0-linux-wsl2.zip \
  texelator-v0.2.0-linux-wsl2.zip.sha256 \
  --title "Texelator v0.2.0" \
  --notes-file docs/releases/v0.2.0.md
```

Do not upload a converted model to the GitHub Release. Published BC4 model artifacts
belong in their separately licensed Hugging Face model repository.

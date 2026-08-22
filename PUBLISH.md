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
If preconverted models are distributed later, place them in a separate Hugging Face
model repository with the source checkpoint license and exact palette hash.

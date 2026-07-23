# Data layout — see specs.md

```
data/
├── raw/asirra/          ← download labeled + unlabeled images here
├── interim/             ← manifest.csv, splits.json (generated)
└── processed/           ← train/val/test splits (generated)
    ├── train/{cat,dog}
    ├── val/{cat,dog}
    └── test/{cat,dog}
```

Quick start after downloading images to `raw/asirra/train/`:

```bash
python src/app.py --stage manifest
python src/app.py --stage split
```

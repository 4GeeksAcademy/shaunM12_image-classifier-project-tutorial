# Raw Asirra data (immutable)

Download the Asirra / Kaggle Dogs vs Cats dataset and unzip here.

## Layout

```
train/   ← ~25,000 labeled images (cat.*.jpg, dog.*.jpg)
test/    ← ~12,500 unlabeled Kaggle images (1.jpg, 2.jpg, …)
```

## Rules

- **Do not train from this folder** after the split script has run.
- **Do not modify or delete** original files.
- Run the pipeline to create leak-free splits:

```bash
python src/app.py --stage manifest
python src/app.py --stage split
```

See [specs.md](../../specs.md) for the full data policy.

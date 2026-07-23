# Processed splits — generators read ONLY these paths

Created by `python src/app.py --stage split` from `data/raw/asirra/train/`.

## Layout

```
train/cat/  train/dog/   ← ~70% — fit() + augmentation
val/cat/    val/dog/     ← ~15% — EarlyStopping, ModelCheckpoint
test/cat/   test/dog/    ← ~15% — final evaluate() once
```

## Rules

- **Never** point `ImageDataGenerator` at `data/raw/asirra/train/` after split exists.
- **Never** copy Kaggle unlabeled images from `data/raw/asirra/test/` into this tree.
- **Do not** manually move images between `train/`, `val/`, and `test/` after creation.

See [specs.md](../../specs.md) for leakage rules.

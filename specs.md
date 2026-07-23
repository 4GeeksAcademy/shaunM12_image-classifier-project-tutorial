# Image Classifier Tutorial Spec — Asirra (Cats vs Dogs)

## Resolved decisions

These choices align the tutorial with the **Step 3 VGG architecture**, preserve the **Step 1–5 workflow**, and enforce a **leak-free** train / validation / test protocol.

| # | Topic | Decision |
|---|--------|----------|
| 1 | **What “test” means** | **Labeled holdout** created from the 25k labeled `train/` download (~15%). Materialized as `data/processed/test/{cat,dog}/`. Used **once** for final metrics. Kaggle’s unlabeled `data/raw/asirra/test/` is **inference-only** (optional prediction CSV); **no accuracy** on that folder. |
| 2 | **Validation split** | **Yes.** Ratios: **70% train / 15% val / 15% test**, stratified by class. `ModelCheckpoint` and `EarlyStopping` monitor **`val_accuracy`** only. Test is **never** passed to `model.fit()` as `validation_data`. |
| 3 | **Split timing & method** | **Before** EDA and generators. Pipeline: scan raw → `manifest.csv` → assign splits (seed **42**) → `splits.json` → materialize `data/processed/{train,val,test}/{cat,dog}/`. Generators read **`processed/` only**, not the full 25k raw folder. Optional: SHA-256 dedup before split. |
| 4 | **Image size** | **224×224** everywhere (`target_size`, `input_shape`, any in-memory arrays). Step 2 mentions 200×200; we use **224** to match Step 3 `input_shape=(224,224,3)`. |
| 5 | **Output, loss, generators** | **`Dense(2, softmax)`** + **`categorical_crossentropy`** + **`class_mode='categorical'`**. All generators: **`rescale=1./255`**. **Train only:** augmentation (`rotation_range`, `horizontal_flip`, etc.). **Val & test:** rescale only. Use **`model.fit(..., validation_data=val_generator)`** (not deprecated `fit_generator`). |

### Non-negotiable leakage rules

1. **`data/raw/` is immutable** — never train directly from the full 25k folder after split exists.
2. **Callbacks and model selection use validation only.**
3. **Test evaluation runs exactly once** for final reporting (gate with `ALLOW_TEST_EVAL=true` or equivalent in code).
4. **EDA in `explore.ipynb` uses `data/processed/train/` only** after the split step.

### Deviations from original instructions (intentional)

- The assignment rubric says “ANN” in the general sense; this project implements a **CNN** (VGG-style / VGG16 transfer), which is appropriate for photo-based cat vs dog classification and matches Step 3’s `Conv2D` architecture.
- Adds **validation** split (original only mentions train/test).
- **Test** for accuracy = labeled holdout, not Kaggle unlabeled `test/`.
- Image size **224** not 200 (matches Step 3 architecture).
- **`model.fit`** instead of **`fit_generator`**.
- Adds manifest + frozen split for reproducibility and leakage prevention.

---

## 1. Overview

### Context

Binary image classifier on the **Asirra** cats-and-dogs subset (~25,000 labeled `.jpg` files). Labels come from filenames (`cat.*`, `dog.*`). The dataset was originally used as a CAPTCHA; a 2007 SVM paper (“Machine Learning Attacks against Asirra’s CAPTCHA”) achieved ~80% accuracy and showed the task was no longer suitable as a CAPTCHA.

### Goal

Teach a complete Keras workflow (load → explore → preprocess → CNN → optimize → save) while enforcing a **leak-free** train / validation / test protocol. Produce **detailed per-step reports** so every decision, result, and test outcome is documented for learning and review (see §2).

### Learning outcomes

1. Download and organize the Asirra dataset in a reproducible folder structure.
2. Visualize cat/dog images and standardize size to 224×224.
3. Build a VGG-style CNN with Keras `Sequential`.
4. Use `ModelCheckpoint` and `EarlyStopping` on **validation** metrics.
5. Evaluate once on a **held-out labeled test split**.
6. Save the model and optionally generate a Kaggle-style submission CSV.
7. Explain common data leakage paths and how this project avoids them.
8. Read and interpret **step reports**, **test summaries**, and **`PROJECT_SUMMARY.md`** to understand the full pipeline.

### Stack

- Python 3.11+
- TensorFlow / Keras
- Existing boilerplate: `src/explore.ipynb`, `src/app.py`, `src/utils.py`, `data/`

---

## 2. Reporting & observability requirements

Every pipeline stage must produce a **detailed, human-readable report** so you can understand **what happened**, **why it was done**, and **what the results mean** — without reading source code. Reports are the primary learning artifact alongside code.

### Goals

1. **Traceability** — link each action to a tutorial step and a leakage rule.
2. **Reasoning** — explain *why* a choice was made, not only *what* was run.
3. **Quick view** — one summary file shows pass/fail tests, key metrics, and stage status.
4. **Reproducibility** — record seeds, paths, counts, and timestamps.

### Report output layout

```
reports/
├── README.md                         # index + how to read reports
├── PROJECT_SUMMARY.md                # single-page quick view (auto-generated)
├── steps/
│   ├── step01_manifest.md
│   ├── step01_manifest.json
│   ├── step015_split.md
│   ├── step015_split.json
│   ├── step02_eda.md
│   ├── step02_eda.json
│   ├── step03_train_baseline.md
│   ├── step03_train_baseline.json
│   ├── step04_optimize.md
│   ├── step04_optimize.json
│   ├── step05_save.md
│   └── step05_save.json
├── tests/
│   ├── test_results.json             # machine-readable: all unittest/pytest results
│   └── test_results.md               # human-readable summary table
├── eda/
│   ├── train_cats_grid.png
│   ├── train_dogs_grid.png
│   └── class_balance_train.json
└── metrics/
    ├── training_history.json
    ├── validation_summary.json
    └── final_test.json
```

Each stage writes **both** `.md` (narrative) and `.json` (structured data for dashboards/scripts).

### Master quick-view: `reports/PROJECT_SUMMARY.md`

Regenerated after every stage. Must fit on one screen where possible and include:

| Section | Contents |
|---------|----------|
| **Pipeline status** | Table: stage name → `DONE` / `SKIPPED` / `FAILED` + timestamp |
| **Data counts** | Raw, manifest, train, val, test counts (total + per class) |
| **Leakage checks** | All tests from `tests/test_leakage.py` → PASS / FAIL / SKIP |
| **Model metrics** | Best val accuracy, epoch stopped, final test accuracy (if run) |
| **Key decisions** | Seed, ratios, image size, loss, callback monitors |
| **Warnings** | Nested folders, duplicates removed, missing Kaggle test images |
| **Next step** | One-line instruction for what to run next |

Example quick-view fragment:

```markdown
## Pipeline status
| Stage | Status | Time |
|-------|--------|------|
| manifest | DONE | 2026-07-22T19:32:01Z |
| split | DONE | 2026-07-22T19:32:45Z |
| leakage tests | 8/8 PASS | 2026-07-22T19:33:10Z |

## Data counts
| Split | Cats | Dogs | Total |
|-------|------|------|-------|
| train | 8747 | 8748 | 17495 |
| val | 1875 | 1874 | 3749 |
| test | 1875 | 1875 | 3750 |
```

### Per-step report template (required fields)

Every `reports/steps/stepXX_*.md` file **must** include these sections:

#### 1. Step header
- Tutorial step number and name
- CLI command run (copy-pasteable)
- Timestamp (UTC)
- Duration (seconds)

#### 2. What happened
- Plain-language summary (2–5 sentences)
- Inputs read (paths)
- Outputs written (paths)

#### 3. Why we did it (reasoning)
- Which **leakage rule** or **resolved decision** this step enforces
- Why this step comes **before/after** adjacent steps
- Trade-offs considered (e.g. copy vs symlink, 224 vs 200)

#### 4. Results (numbers)
- Tables of counts, ratios, shapes, hyperparameters
- Before/after comparison where relevant

#### 5. Artifacts
- Bullet list of every file created or updated

#### 6. Leakage & validation checks
- Checks run at this step and their outcome
- What would have gone wrong if skipped

#### 7. Interpretation
- What a good result looks like
- What to investigate if results look wrong

#### 8. Next step
- Exact command or notebook cell to run next

### Per-step JSON schema (minimum)

Each `reports/steps/stepXX_*.json` must include:

```json
{
  "step": "1.5_split",
  "tutorial_step": 1,
  "command": "python src/app.py --stage split",
  "timestamp_utc": "2026-07-22T19:32:45Z",
  "duration_seconds": 44.2,
  "status": "success",
  "reasoning": {
    "leakage_rules": ["split before generators", "stratified by label"],
    "decisions": ["seed=42", "70/15/15", "copy not symlink"]
  },
  "inputs": { "raw_train_dir": "data/raw/asirra/train" },
  "outputs": {
    "manifest": "data/interim/manifest.csv",
    "splits": "data/interim/splits.json",
    "processed_dirs": ["data/processed/train", "data/processed/val", "data/processed/test"]
  },
  "metrics": {
    "counts_by_split": { "train": 17495, "val": 3749, "test": 3750 }
  },
  "warnings": ["nested train/train/ folder detected"],
  "checks_passed": ["split_disjointness", "stratification"],
  "next_step": "python -m unittest tests.test_leakage -v"
}
```

### Test results reporting

After any stage that affects data splits or evaluation policy, run:

```bash
python -m unittest tests.test_leakage -v
```

Write results to:

- **`reports/tests/test_results.json`** — one object per test: name, status, message, duration
- **`reports/tests/test_results.md`** — summary table for quick view

Required columns in `test_results.md`:

| Test | Status | What it verifies | Why it matters |
|------|--------|------------------|----------------|
| `test_split_disjointness` | PASS | No image in two splits | Prevents train/test overlap |
| `test_content_hash_not_shared_across_splits` | PASS | Duplicates stay in one split | Prevents duplicate leakage |
| … | … | … | … |

Include a **totals line**: `8 passed, 0 failed, 0 skipped`.

### Step-specific report content

| Step | Report file(s) | Must explain | Must include in results |
|------|----------------|--------------|-------------------------|
| **1 — manifest** | `step01_manifest.*` | Why scan raw before split; label parsing from filename; dedup | Files found, cat/dog counts, duplicates removed, nested-folder warning |
| **1.5 — split** | `step015_split.*` | Why 70/15/15; why stratified; why processed/ not raw/ | Per-split counts, seed, ratios, copy duration |
| **2 — EDA** | `step02_eda.*` | Why train-only EDA; why 224×224 | Grid paths, class balance, sample image sizes before resize |
| **3 — baseline train** | `step03_train_baseline.*` | Why VGG-style CNN; why val not test | Epoch loss/acc curves, steps_per_epoch, param count |
| **4 — optimize** | `step04_optimize.*` | Why EarlyStopping on val; why ModelCheckpoint | Best epoch, best val accuracy, early-stop triggered Y/N |
| **4 — test eval** | section in `step04_optimize.*` | Why ALLOW_TEST_EVAL gate; why run once | Test loss/acc, timestamp of single run |
| **5 — save** | `step05_save.*` | Why two model paths | File sizes, reload sanity check on val |

### Implementation notes

- **`src/reporting.py`** (to implement): helpers `write_step_report()`, `update_project_summary()`, `write_test_results()`.
- Each `app.py --stage` call should invoke reporting at the end of the stage.
- **`explore.ipynb`**: final cell exports EDA numbers to `reports/steps/step02_eda.json`.
- Regenerate **`PROJECT_SUMMARY.md`** after every stage and after tests run.

### What “good documentation” looks like

A reader with no ML background should be able to:

1. Open `reports/PROJECT_SUMMARY.md` and know project status in **30 seconds**.
2. Open any `reports/steps/stepXX_*.md` and understand that step in **5 minutes**.
3. Open `reports/tests/test_results.md` and see **every leakage test** and why it passed.
4. Trace any number in a report back to a **file on disk** or a **spec decision**.

---

## 3. Dataset

### Source

Download the Asirra / Kaggle Dogs vs Cats dataset (linked in the tutorial).

### Raw layout (after unzip — manual step)

```
data/raw/asirra/
  train/                    # ~25,000 labeled images
    cat.0.jpg
    dog.0.jpg
    ...
  test/                     # ~12,500 UNLABELED Kaggle images
    1.jpg
    2.jpg
    ...
```

### Label rule

- Filename contains `cat` → class `cat` (label index 0)
- Filename contains `dog` → class `dog` (label index 1)
- Parse label from filename **before** any split; store in manifest.

### Important

- **`data/raw/` is immutable** — never move, delete, or overwrite originals.
- All labeled splits live under **`data/processed/`** (copies or symlinks).
- Kaggle `test/` has **no labels** → not used for accuracy; optional final inference exercise only.

---

## 4. Project structure

```
project-root/
├── data/
│   ├── raw/
│   │   └── asirra/
│   │       ├── train/              # YOU: unzip labeled images here
│   │       └── test/               # YOU: unzip unlabeled Kaggle images here
│   ├── interim/
│   │   ├── manifest.csv            # SCRIPT: filepath, label, hash, split
│   │   ├── splits.json             # SCRIPT: seed, ratios, counts
│   │   └── dedup_report.csv        # SCRIPT: optional duplicate log
│   └── processed/
│       ├── train/cat/  train/dog/  # ~70% — training + augmentation
│       ├── val/cat/    val/dog/    # ~15% — callbacks / tuning
│       └── test/cat/   test/dog/   # ~15% — final evaluate once
│
├── submissions/
│   ├── sample_submission.csv       # YOU: Kaggle template (format reference)
│   └── asirra_predictions.csv      # SCRIPT: predictions on unlabeled test
│
├── saved_models/
│   └── best_model.keras              # ModelCheckpoint on val_accuracy
│
├── reports/
│   ├── README.md                     # index — how to read all reports
│   ├── PROJECT_SUMMARY.md            # quick view: status, metrics, tests
│   ├── steps/                        # per-step .md + .json reports
│   ├── tests/
│   │   ├── test_results.json         # all leakage test outcomes
│   │   └── test_results.md           # quick-view test summary table
│   ├── eda/                          # sample grids (train only)
│   └── metrics/
│       ├── training_history.json
│       ├── validation_summary.json
│       └── final_test.json
│
├── src/
│   ├── explore.ipynb                 # Step 2 EDA (train split only)
│   ├── app.py                        # pipeline entry point
│   ├── utils.py
│   └── config.py                     # paths, seed, hyperparams
│
├── specs.md
└── requirements.txt
```

### What you create manually vs what the pipeline creates

| Path | Who creates it | Purpose | Used for training? |
|------|----------------|---------|-------------------|
| `data/raw/asirra/train/` | **You** (download/unzip) | Original labeled 25k | **No** (after split) |
| `data/raw/asirra/test/` | **You** (download/unzip) | Kaggle unlabeled 12.5k | **Never** |
| `submissions/sample_submission.csv` | **You** (from Kaggle) | Format reference | **Never** |
| `data/interim/manifest.csv` | Script | File list, labels, hashes, split | Metadata only |
| `data/interim/splits.json` | Script | Frozen split record | Metadata only |
| `data/processed/train/` | Script | Training + augmentation | **Yes** |
| `data/processed/val/` | Script | EarlyStopping, ModelCheckpoint | **Val only** |
| `data/processed/test/` | Script | Final `evaluate()` once | **Once at end** |
| `submissions/asirra_predictions.csv` | Script | Predictions on unlabeled test | **No** (inference only) |

### Recommended `.gitignore` entries

```gitignore
data/raw/asirra/train/
data/raw/asirra/test/
data/processed/
data/interim/manifest.csv
data/interim/splits.json
saved_models/
reports/
submissions/asirra_predictions.csv
```

Keep `submissions/sample_submission.csv` in the repo if small enough to serve as a format reference.

---

## 5. Data leakage policy

### Golden rule

> **Assign splits in `manifest.csv` first. Every later step respects `split`.**

### Split specification

| Split | Ratio | ~Images | Purpose | Used how often |
|-------|-------|---------|---------|----------------|
| Train | 70% | ~17,500 | Learn weights + augmentation | Every epoch |
| Val | 15% | ~3,750 | EarlyStopping, ModelCheckpoint, tuning | Many times |
| Test | 15% | ~3,750 | Final accuracy report | **Once** |

- `RANDOM_SEED = 42`
- **Stratified** by label (~50/50 cats and dogs in each split).
- Materialize `data/processed/{train,val,test}/{cat,dog}/` via copy (recommended) or symlink.

### Forbidden

- Pointing generators at `data/raw/asirra/train/` after split exists.
- Passing `processed/test/` to `model.fit()` as `validation_data`.
- EarlyStopping / ModelCheckpoint monitoring **test** loss or accuracy.
- Tuning architecture, epochs, or augmentation after seeing test results.
- EDA or normalization stats computed on val or test.
- Copying Kaggle unlabeled images into `data/processed/`.
- Shuffling all 25k images and slicing ad hoc without a frozen manifest.

### Required

- `flow_from_directory()` on **`data/processed/train`**, **`val`**, **`test`** separately.
- Augmentation only on **train** generator.
- Val and test generators: **`rescale=1./255` only** (no rotation, flip, zoom).
- `evaluate.py` raises unless `ALLOW_TEST_EVAL=true` for final run.

### Duplicate check (recommended)

Hash file bytes (SHA-256). If duplicates exist, keep one canonical row and assign all copies to the **same split** via shared `group_id`.

### Leak-free data flow

```
data/raw/asirra/train/  ──►  manifest.csv + splits.json
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            processed/train   processed/val   processed/test
                    │               │               │
                    ▼               ▼               ▼
            train_generator   val_generator   test_generator
            (+ augmentation)    (rescale only)  (rescale only)
                    │               │
                    └───────► model.fit(validation_data=val)
                                        │
                                        ▼
                              evaluate ONCE → final_test.json

data/raw/asirra/test/ (unlabeled) ──► predict only ──► submissions/asirra_predictions.csv
submissions/sample_submission.csv ──► format reference only
```

---

## 6. Configuration (`src/config.py`)

```python
# Paths
RAW_TRAIN_DIR = "data/raw/asirra/train"
RAW_KAGGLE_TEST_DIR = "data/raw/asirra/test"

PROCESSED_TRAIN_DIR = "data/processed/train"
PROCESSED_VAL_DIR = "data/processed/val"
PROCESSED_TEST_DIR = "data/processed/test"

MANIFEST_PATH = "data/interim/manifest.csv"
SPLITS_PATH = "data/interim/splits.json"
DEDUP_REPORT_PATH = "data/interim/dedup_report.csv"

SAMPLE_SUBMISSION_PATH = "submissions/sample_submission.csv"
PREDICTIONS_PATH = "submissions/asirra_predictions.csv"

SAVED_MODEL_PATH = "saved_models/best_model.keras"
FINAL_MODEL_PATH = "saved_models/asirra_cats_dogs_final.keras"
FINAL_TEST_METRICS_PATH = "reports/metrics/final_test.json"
PROJECT_SUMMARY_PATH = "reports/PROJECT_SUMMARY.md"
STEP_REPORTS_DIR = "reports/steps"
TEST_RESULTS_JSON = "reports/tests/test_results.json"
TEST_RESULTS_MD = "reports/tests/test_results.md"

# Split
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Images
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
BATCH_SIZE = 32

# Training
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
ALLOW_TEST_EVAL = False  # set True or use env var for one-time final run
```

---

## 7. Tutorial steps (implementation)

### Instruction ↔ spec mapping

| Tutorial step | Spec behavior |
|---------------|---------------|
| **Step 1** — Download | Unzip to `data/raw/asirra/`; run manifest + split (Step 1.5) |
| **Step 2** — Visualize & resize | EDA on **train split**; 3×3 cat/dog grids; **224×224**; `ImageDataGenerator` + `flow_from_directory()` on `processed/` |
| **Step 3** — Build CNN | VGG-style `Sequential` as provided; compile with `categorical_crossentropy`; `fit` uses `validation_data=val_generator` |
| **Step 4** — Optimize | `ModelCheckpoint` + `EarlyStopping` on **`val_accuracy`**; load best weights; **one-time** `evaluate()` on `processed/test` |
| **Step 5** — Save | `saved_models/best_model.keras` (+ optional final export) |

---

### Step 1 — Load the dataset

**Learner actions**

1. Download dataset; unzip labeled images into `data/raw/asirra/train/`.
2. Unzip unlabeled Kaggle images into `data/raw/asirra/test/`.
3. Copy Kaggle `sample_submission.csv` to `submissions/sample_submission.csv`.
4. Verify ~25,000 `.jpg` files in `data/raw/asirra/train/`.

**Run manifest + split**

```bash
python src/app.py --stage manifest
python src/app.py --stage split
```

**Manifest columns (`data/interim/manifest.csv`)**

| Column | Description |
|--------|-------------|
| `image_id` | Stable identifier (hash prefix or index) |
| `filepath` | Relative path from project root |
| `label` | `cat` or `dog` |
| `content_hash` | SHA-256 of file bytes |
| `group_id` | Same as `content_hash` if deduping; optional |
| `split` | `train`, `val`, or `test` |

**Split script rules**

1. Read **only** from `data/raw/asirra/train/`.
2. Never read `data/raw/asirra/test/` during split.
3. Stratified split 70/15/15 with seed 42.
4. Copy files into `data/processed/{train,val,test}/{cat,dog}/`.
5. Write `data/interim/splits.json`.

**Report deliverable (`reports/steps/step01_manifest.*`, `step015_split.*`)**

- Explain why raw data is scanned before any training.
- Document cat/dog counts, duplicates removed, nested-folder warnings.
- Table of per-split counts after materialization.
- Run leakage tests; append results to `reports/tests/test_results.md`.

---

### Step 2 — Visualize input information

**Where:** `src/explore.ipynb`

**Rules**

- Load images **only from `data/processed/train/`** for EDA (after split).
- Plot first 9 **cat** and first 9 **dog** images (3×3 grids).
- Note: varying sizes, color, aspect ratios, multi-animal edge cases.

**Deliverables**

- `reports/eda/train_cats_grid.png`
- `reports/eda/train_dogs_grid.png`
- Class balance counts on **train split only**

**Report deliverable (`reports/steps/step02_eda.*`)**

- Why EDA uses **train split only** (leakage reasoning).
- Why images are resized to 224×224 (architecture alignment).
- Table of original vs resized dimensions (sample of images).
- Class balance JSON at `reports/eda/class_balance_train.json`.

**Two loading paths (RAM branch)**

| Condition | Approach |
|-----------|----------|
| **> 12 GB RAM** | Load resized arrays per split; optional cache per split (not one tuple of all 25k) |
| **≤ 12 GB RAM** | `ImageDataGenerator` + `flow_from_directory()` on `data/processed/` |

**Generator setup**

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    horizontal_flip=True,
)

val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    "data/processed/train",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
)

val_generator = val_test_datagen.flow_from_directory(
    "data/processed/val",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
)

test_generator = val_test_datagen.flow_from_directory(
    "data/processed/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
)
```

**Steps per epoch**

```python
import math

steps_per_epoch = math.ceil(train_generator.samples / BATCH_SIZE)
validation_steps = math.ceil(val_generator.samples / BATCH_SIZE)
test_steps = math.ceil(test_generator.samples / BATCH_SIZE)
```

---

### Step 3 — Build the CNN (VGG-style)

**Where:** `src/models/vgg.py` (or inline in notebook, then migrate to `app.py`)

**Architecture** (from tutorial — input 224×224×3):

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(units=4096, activation="relu"))
model.add(Dense(units=4096, activation="relu"))
model.add(Dense(units=2, activation="softmax"))
```

**Compile**

```python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
```

**Initial training**

```python
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=validation_steps,
)
```

**Report deliverable (`reports/steps/step03_train_baseline.*`)**

- Why a CNN (not plain ANN) and why this VGG-style stack.
- Model summary: layer count, approximate parameter count.
- Per-epoch train/val loss and accuracy table.
- Why validation data is used instead of test.

---

### Step 4 — Optimize the model

**Callbacks (validation only)**

```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint(
    filepath="saved_models/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=30,
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=[checkpoint, early_stop],
)
```

**After training**

1. Load `saved_models/best_model.keras`.
2. Report validation metrics from training history.
3. Run **one-time** test evaluation:

```bash
ALLOW_TEST_EVAL=true python src/app.py --stage evaluate
```

```python
test_loss, test_acc = model.evaluate(test_generator, steps=test_steps)
# Save to reports/metrics/final_test.json — do not re-run for tuning
```

**Report deliverable (`reports/steps/step04_optimize.*`)**

- Why `ModelCheckpoint` and `EarlyStopping` monitor **`val_accuracy`** only.
- Best epoch, best val accuracy, whether early stopping fired.
- Training curves saved to `reports/metrics/training_history.json`.
- If test was run: document single-run timestamp and gate (`ALLOW_TEST_EVAL=true`).
- Leakage reasoning for not re-running test after tuning.

**Optional — Kaggle submission (no accuracy)**

```bash
python src/app.py --stage predict
```

- Load images from `data/raw/asirra/test/`.
- Write predictions to `submissions/asirra_predictions.csv` matching `sample_submission.csv` format (`id`, `label` = probability of dog).

---

### Step 5 — Save the model

**Save**

- Best checkpoint: `saved_models/best_model.keras` (via ModelCheckpoint).
- Final export: `saved_models/asirra_cats_dogs_final.keras`.

**Verify reload**

```python
import tensorflow as tf

loaded = tf.keras.models.load_model("saved_models/best_model.keras")
loaded.evaluate(val_generator, steps=validation_steps)  # sanity check on val, not test
```

**Report deliverable (`reports/steps/step05_save.*`)**

- Paths and file sizes of saved models.
- Reload sanity-check result on **validation** (not test).
- Final `PROJECT_SUMMARY.md` regenerated with full pipeline status.

---

## 8. Pipeline CLI (`src/app.py`)

```bash
python src/app.py --stage manifest     # Step 1 — scan raw, build manifest
python src/app.py --stage split        # Step 1.5 — create processed splits
python src/app.py --stage train        # Steps 3–4 — train with callbacks
ALLOW_TEST_EVAL=true python src/app.py --stage evaluate   # Step 4 — final test once
python src/app.py --stage predict      # Optional — Kaggle submission CSV
python src/app.py --stage save         # Step 5 — export final model
python src/app.py --stage all          # Full pipeline (test eval at end only)
```

---

## 9. Dependencies

Add to `requirements.txt`:

```
tensorflow>=2.15
Pillow>=10.0
matplotlib>=3.7
numpy>=1.24
scikit-learn>=1.3
tqdm
```

Existing boilerplate packages (`pandas`, `sqlalchemy`, etc.) may remain unless removed intentionally.

---

## 10. Acceptance criteria

- [ ] `manifest.csv` and `splits.json` exist; reproducible with seed 42.
- [ ] No image appears in more than one split.
- [ ] No duplicate `content_hash` across different splits (if dedup enabled).
- [ ] EDA notebook uses **train split only** (`data/processed/train/`).
- [ ] Train generator has augmentation; val and test generators do not.
- [ ] `ModelCheckpoint` and `EarlyStopping` monitor **`val_accuracy`**.
- [ ] Test accuracy computed **once** and saved to `reports/metrics/final_test.json`.
- [ ] Model saved to `saved_models/`.
- [ ] Optional predictions written to `submissions/asirra_predictions.csv`.
- [ ] README documents Asirra context, folder layout, and split policy.
- [ ] **`reports/PROJECT_SUMMARY.md` exists and reflects latest pipeline state.**
- [ ] **Each completed stage has `reports/steps/stepXX_*.md` and `.json` with all required sections (§2).**
- [ ] **`reports/tests/test_results.md` lists every leakage test with PASS/FAIL and reasoning.**
- [ ] **Every report explains *why* (reasoning), not only *what* (results).**

---

## 11. Automated leakage tests (recommended)

`tests/test_leakage.py`:

| Test | Assert |
|------|--------|
| Split disjointness | train ∩ val ∩ test = ∅ on `image_id` |
| Stratification | cat/dog ratio within ±2% across splits |
| Generator paths | train/val/test dirs are under `processed/`, not `raw/train` |
| Callback monitors | checkpoint monitors `val_*`, not `test_*` |
| Test gate | evaluate without `ALLOW_TEST_EVAL` raises |
| Kaggle test isolation | no files from `raw/asirra/test/` in manifest or processed |

All test runs must be exported to **`reports/tests/test_results.json`** and summarized in **`reports/tests/test_results.md`** (see §2).

---

## 12. Instructor notes / common student mistakes

1. **Using full 25k as training** and Kaggle unlabeled folder as labeled “test” — wrong; Kaggle test has no labels.
2. **Copying Kaggle unlabeled test into `processed/test/`** — breaks the labeled holdout protocol.
3. **Monitoring test in EarlyStopping** because val is “too small” — inflates reported performance.
4. **Augmenting val/test** — distorts evaluation distribution.
5. **Re-running test** after changing epochs — repeated peeking; reset decisions using val only.
6. **200 vs 224 mismatch** — standardize on 224 everywhere.
7. **Building one tuple of all 25k before split** — split indices must be frozen in manifest first.

---

## 13. Expected performance (honest ranges)

| Model | Val accuracy (typical) | Test (single run) |
|-------|------------------------|-------------------|
| VGG-style from scratch, ~10–30 epochs | 70–85% | similar ± noise |
| 2007 SVM paper baseline | — | ~80% (different protocol — cite cautiously) |
| Transfer learning (optional extension) | 90%+ | report once |

Do not claim the CAPTCHA is “solved” unless the test protocol above is followed. The original CAPTCHA claim was about **human** accuracy (~99.6%), not model accuracy on a held-out set.

---

## 14. Optional extensions

- Transfer learning: `VGG16(weights='imagenet', include_top=False)` + custom head.
- Confusion matrix and misclassified gallery on **val** (error analysis); test only for final number.
- Experiment logging via existing `utils.db_connect()` and PostgreSQL (v2).
- Perceptual-hash near-duplicate detection before split.

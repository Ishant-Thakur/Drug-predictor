# Attention Architecture Comparison &mdash; Two Separate Notebooks

Three notebooks, meant to be run in order:

```
notebooks/
├── 01_attention_mlp.ipynb   <- your original multi-task model, with self-attention added
├── 02_attention_cnn.ipynb   <- new architecture: 1D-CNN + attention pooling
└── 03_comparison.ipynb      <- loads both results and compares them side by side
data/
├── aqsoldb_raw.csv          <- solubility, 9,982 compounds
├── bbbp_raw.csv             <- BBB permeability, ~2,000 compounds
├── tox21_raw.csv            <- 12 toxicity assays, ~7,800 compounds
└── multitask_merged.csv     <- already merged + descriptors computed (~16,400 molecules)
```

Nothing has been executed &mdash; all outputs are cleared, ready for you to run fresh in Colab.

## Why two notebooks instead of one

You asked for the attention mechanism added to the model you already had, and a second,
separate architecture (attention + 1D-CNN), each in its own notebook, so they can be run,
inspected, and modified independently. `03_comparison.ipynb` is the only thing that ties them
together, and it does so *after the fact* by reading each notebook's saved results &mdash; it
doesn't need both notebooks open at once.

## `01_attention_mlp.ipynb`

Your original multi-task setup (solubility + BBB + toxicity, masked multi-task loss, same 12
RDKit descriptors) with the MLP trunk replaced by **self-attention**: each descriptor becomes
its own token (FT-Transformer style), a `[CLS]` token is prepended, and a Transformer encoder
lets descriptors attend to each other before the three output heads.

## `02_attention_cnn.ipynb`

A different architecture: the 12 descriptors are treated as a 1D signal. `Conv1d` layers learn
local interactions between neighbouring descriptors, then an **additive (Bahdanau-style)
attention layer** pools the result instead of average/max pooling &mdash; letting the model weight
which descriptors matter most per molecule. Also includes a short section visualizing the
learned attention weights for a few sample molecules.

## `03_comparison.ipynb`

Run this **after** both training notebooks. It just loads `../results/attention_mlp_results.json`
and `../results/attention_cnn_results.json` (each training notebook saves its own automatically)
and builds a side-by-side comparison table + bar charts (MAE/RMSE/R² for solubility,
Accuracy/ROC-AUC for BBB and toxicity, parameter counts, training time).

## Running in Colab

1. Upload the whole folder, keeping `notebooks/`, `data/`, `models/`, `results/` as siblings
   (the notebooks create `models/` and `results/` automatically if missing).
2. `%pip install rdkit` if not already available in your Colab runtime.
3. Run `01_attention_mlp.ipynb` top to bottom, then `02_attention_cnn.ipynb` top to bottom
   (both will reuse `data/multitask_merged.csv` instead of re-downloading/re-merging).
4. Run `03_comparison.ipynb` last.

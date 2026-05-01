# Question-Conditioned GraphVQA

This repository is a course-project extension of
[GraphVQA: Language-Guided Graph Neural Networks for Scene Graph Question
Answering](https://arxiv.org/abs/2104.10283). The original model answers GQA
questions by encoding a ground-truth scene graph and a natural-language
question, then decoding both a symbolic program and a short answer.

Our project keeps the GraphVQA pipeline and studies whether the scene graph
itself should stay fixed. We add question-conditioned graph construction
strategies that can reweight, add, or prune edges before message passing.

## What Is In This Repo

`GraphVQA/` is now committed as a normal folder, not as a Git submodule pointer,
so it should open normally on GitHub.

Important files:

| Path | Purpose |
| --- | --- |
| `GraphVQA/mainExplain_gat.py` | Main training and evaluation entry point for the GAT GraphVQA model. |
| `GraphVQA/pipeline_model_gat.py` | End-to-end model pipeline: question encoder, scene-graph encoder, program decoder, and answer head. |
| `GraphVQA/graph_construction.py` | Our question-conditioned graph methods: static, fully connected, reweight, augment, and prune. |
| `GraphVQA/profile_memory_methods.py` | Runtime and memory profiler for comparing graph construction methods. |
| `GraphVQA/gqa_dataset_entry.py` | GQA scene-graph and question/program dataset loader. |
| `GraphVQA/preprocess.py` | Converts raw GQA questions into the program format used by GraphVQA. |
| `GraphVQA/eval.py` | GQA official-style evaluation script. |
| `GraphVQA/eval_result/` | Small text summaries of validation results from prior runs. |
| `GraphVQA/meta_info/` | Object, relation, attribute, and answer vocabularies needed by the loader. |
| `GraphVQA/figs/` | Original GraphVQA architecture and result figures. |

Large generated files are intentionally not committed: raw GQA data, processed
question JSON files, checkpoints, logs, memory profiles, local vector caches,
and virtual environments.

## Methods

The training script exposes the graph method through `--graph_method`:

| Method | Meaning |
| --- | --- |
| `static` | Use the original scene-graph edges. This is the GraphVQA-style baseline. |
| `fully_connected` | Build a dense graph inside each scene graph. Useful as a stress test. |
| `reweight` | Learn a question-conditioned scalar weight for each existing edge. |
| `augment` | Add top-k question-conditioned candidate edges. |
| `prune` | Keep the top-k question-conditioned incoming neighbors and remove weaker edges. |

The learned methods share the implementation in
`GraphVQA/graph_construction.py` and are inserted inside the scene-graph encoder
in `GraphVQA/pipeline_model_gat.py`.

## Results Snapshot

These are the compact validation summaries checked into `GraphVQA/eval_result/`.
Accuracy is the main GQA answer accuracy; consistency is the official GQA
consistency metric.

| Model/run | Accuracy | Consistency |
| --- | ---: | ---: |
| Only scene graph baseline | 19.63 | 46.98 |
| LCGN instruction baseline | 88.43 | 93.88 |
| GCN | 90.18 | 95.44 |
| GINE | 90.39 | 94.79 |
| GAT | 94.75 | 97.73 |
| GAT best | 94.78 | 98.37 |

Memory profiling was run with `profile_memory_methods.py` over 3 batches on
`val_unbiased`. The local summaries show the expected graph-size tradeoffs:

| Batch | Method | Edges in -> out | Eval VRAM MB | Train VRAM MB | Eval s/batch | Train s/batch |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 16 | static | 1328 -> 1328 | 349.6 | 1294.5 | 0.010 | 0.027 |
| 16 | reweight | 1328 -> 1328 | 352.5 | 1323.1 | 0.010 | 0.027 |
| 16 | augment | 1328 -> 1408 | 354.0 | 1299.4 | 0.095 | 0.112 |
| 16 | prune | 1328 -> 1128 | 347.2 | 1278.9 | 0.027 | 0.048 |
| 64 | static | 5566 -> 5566 | 486.1 | 2361.4 | 0.014 | 0.044 |
| 64 | reweight | 5566 -> 5566 | 489.0 | 2436.2 | 0.014 | 0.044 |
| 64 | augment | 5566 -> 5886 | 494.8 | 2388.1 | 0.316 | 0.348 |
| 64 | prune | 5566 -> 4729 | 471.8 | 2316.8 | 0.092 | 0.123 |

## Setup

The code uses the legacy `torchtext.data.Field` API, so use an environment that
still provides legacy torchtext. The local working environment used for these
runs was:

- Python 3.10.13
- PyTorch 2.10.0 with CUDA 12 packages
- torchtext 0.5.0
- torch-geometric 2.4.0
- spaCy with `en_core_web_sm`
- NLTK with `wordnet`

One working setup pattern is:

```bash
cd GraphVQA
python3.10 -m venv .venv
source .venv/bin/activate

pip install torch torchtext==0.5.0 torch-geometric==2.4.0
pip install numpy scipy scikit-learn h5py matplotlib nltk spacy psutil tqdm
python -m spacy download en_core_web_sm
python - <<'PY'
import nltk
nltk.download("wordnet")
PY
```

If your CUDA/PyTorch combination needs custom PyTorch Geometric wheels, install
them using the wheel index that matches your PyTorch and CUDA versions.

## Data Layout

Download GQA scene graphs and questions from the GQA/Stanford data release:

- `sceneGraphs.zip`
- `questions1.2.zip`
- optional `eval.zip` if you want the official consistency/validity metrics

After unzipping, arrange files like this:

```text
GraphVQA/
  sceneGraphs/
    train_sceneGraphs.json
    val_sceneGraphs.json
  questions/
    original/
      train_balanced_questions.json
      val_balanced_questions.json
      test_balanced_questions.json
      testdev_balanced_questions.json
      val_all_questions.json        # optional, from eval.zip
```

`GraphVQA/Constants.py` now derives `ROOT_DIR` from the checkout location, so
you should not need to edit an absolute path when moving the repo.

## Quick Reproduction

Run these commands from `GraphVQA/`.

### 1. Preprocess GQA Questions

```bash
python preprocess.py
```

This creates the large processed program files under `GraphVQA/questions/`.
Those files are ignored by Git because they can be regenerated from the raw GQA
release.

### 2. Smoke-Test Data Loading

```bash
python gqa_dataset_entry.py
python pipeline_model_gat.py
```

These commands catch the most common setup issues: missing GQA files, missing
spaCy model, missing GloVe cache, or incompatible torchtext.

### 3. Train A Method

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python mainExplain_gat.py \
  --workers=4 \
  --batch-size=256 \
  --epochs=100 \
  --graph_method=augment \
  --lr_drop=90 \
  --output_dir=./outputdir/gat_augment_e100/
```

Four GPUs, matching the style used for the logged experiments:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --use_env mainExplain_gat.py \
  --workers=4 \
  --batch-size=512 \
  --epochs=100 \
  --graph_method=augment \
  --lr_drop=90 \
  --output_dir=./outputdir/gat_augment_e100/
```

Change `--graph_method` to `static`, `reweight`, `augment`, or `prune` to run
the main comparison. For the learned graph methods, `--graph-top-k` and
`--graph-hidden-dim` control the graph-construction module.

### 4. Evaluate A Checkpoint

Checkpoints are large and are not committed. Place them under `GraphVQA/outputdir/`
using the same directory names, or pass the path through `--resume`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --use_env mainExplain_gat.py \
  --workers=4 \
  --batch-size=256 \
  --evaluate \
  --resume=outputdir/gat_augment_e100/checkpoint0099.pth \
  --evaluate_sets val_unbiased \
  --graph_method=augment \
  --output_dir=./outputdir/gat_augment_e100_eval/
```

The script writes predictions to `dump_results.json` inside the evaluation
output directory.

### 5. Run Official-Style GQA Metrics

After downloading the GQA evaluation files:

```bash
python eval.py \
  --predictions=./outputdir/gat_augment_e100_eval/dump_results.json \
  --consistency
```

The small checked-in summaries under `GraphVQA/eval_result/` show the format of
the expected output.

### 6. Profile Memory And Runtime

```bash
python profile_memory_methods.py \
  --methods static reweight augment prune \
  --batch-size 64 \
  --batches 3 \
  --warmup-batches 1 \
  --split val_unbiased \
  --output-dir outputdir/memory_profile_bs64_fresh
```

By default this expects checkpoints at:

```text
outputdir/gat_static_e100/checkpoint.pth
outputdir/gat_reweight_e100/checkpoint.pth
outputdir/gat_augment_e100/checkpoint.pth
outputdir/gat_prune_e100/checkpoint.pth
```

Use `--checkpoint METHOD=PATH` to point to a different checkpoint.

## Notes For Future Work

- The repo intentionally tracks code, metadata, figures, and compact result
  summaries, while leaving large data/checkpoint artifacts out of Git.
- The original GraphVQA README is preserved at `GraphVQA/README.md`.
- If upgrading dependencies, start by replacing the legacy `torchtext.data.Field`
  usage in `GraphVQA/gqa_dataset_entry.py`.

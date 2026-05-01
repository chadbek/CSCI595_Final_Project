# Question-Conditioned GraphVQA

This project extends GraphVQA for GQA scene-graph question answering by making
the graph structure depend on the question. Instead of always passing the
original scene graph into the GNN, we test lightweight graph construction
modules that decide which relationships should matter for the current question.

Our main additions are in:

| Path | What we changed |
| --- | --- |
| `GraphVQA/graph_construction.py` | Question-conditioned edge reweighting, augmentation, and pruning modules. |
| `GraphVQA/pipeline_model_gat.py` | Integrates the graph-construction module before scene-graph message passing. |
| `GraphVQA/mainExplain_gat.py` | Adds `--graph_method`, `--graph-top-k`, and `--graph-hidden-dim` flags. |
| `GraphVQA/profile_memory_methods.py` | Profiles VRAM, RAM, edge counts, and runtime for graph methods. |

`GraphVQA/` is committed as a normal folder, so it should open directly on
GitHub. Large generated artifacts are intentionally ignored: raw GQA data,
processed program JSON files, checkpoints, logs, local GloVe caches, virtual
envs, and `outputdir/`.

## Methods

Use `--graph_method` to select the graph construction strategy:

| Method | Description |
| --- | --- |
| `static` | Original scene graph, used as the baseline. |
| `reweight` | Learns question-conditioned weights for existing edges. |
| `augment` | Adds top-k question-conditioned candidate edges. |
| `prune` | Keeps only the top-k question-conditioned incoming edges per node. |

## Key Finding

Pruning reduces the active graph size and saves VRAM compared with the static
scene graph baseline.

Memory profile on `val_unbiased`, 3 batches:

| Batch size | Method | Edges in -> out | Eval VRAM MB | Train VRAM MB |
| ---: | --- | --- | ---: | ---: |
| 16 | static | 1328 -> 1328 | 349.6 | 1294.5 |
| 16 | prune | 1328 -> 1128 | 347.2 | 1278.9 |
| 64 | static | 5566 -> 5566 | 486.1 | 2361.4 |
| 64 | prune | 5566 -> 4729 | 471.8 | 2316.8 |

At batch size 64, prune removes 837 edges from the profiled batch and lowers
peak training VRAM by about 44.6 MB.

## Setup

The code uses the legacy `torchtext.data.Field` API. The local environment used
for our runs was Python 3.10, PyTorch with CUDA, torchtext 0.5.0, PyTorch
Geometric, spaCy, NLTK, and psutil.

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

If PyTorch Geometric needs CUDA-specific wheels on your machine, install the
wheel set matching your PyTorch/CUDA version.

## Data

Download the GQA scene graphs and questions, then place them here:

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
      val_all_questions.json        # optional, for full GQA evaluation
```

`GraphVQA/Constants.py` auto-detects the checkout root, so no local absolute
path edit is needed.

## Reproduce

Run commands from `GraphVQA/`.

Preprocess the GQA questions:

```bash
python preprocess.py
```

Train a graph method:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --use_env mainExplain_gat.py \
  --workers=4 \
  --batch-size=512 \
  --epochs=100 \
  --graph_method=prune \
  --lr_drop=90 \
  --output_dir=./outputdir/gat_prune_e100/
```

Evaluate a checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --use_env mainExplain_gat.py \
  --workers=4 \
  --batch-size=256 \
  --evaluate \
  --resume=outputdir/gat_prune_e100/checkpoint0099.pth \
  --evaluate_sets val_unbiased \
  --graph_method=prune \
  --output_dir=./outputdir/gat_prune_e100_eval/
```

Profile the prune VRAM comparison:

```bash
python profile_memory_methods.py \
  --methods static prune \
  --batch-size 64 \
  --batches 3 \
  --warmup-batches 1 \
  --split val_unbiased \
  --output-dir outputdir/memory_profile_bs64_fresh
```

Expected checkpoint locations for profiling:

```text
outputdir/gat_static_e100/checkpoint.pth
outputdir/gat_prune_e100/checkpoint.pth
```

Use `--checkpoint METHOD=PATH` to point to different checkpoint files.

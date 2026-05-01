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

All three question-conditioned graph methods improve final short-answer
accuracy over the static scene graph baseline. Program prediction accuracy stays
roughly flat, while the answer head benefits from dynamic graph structure.

| Method | Program | Program Group | Program Non Empty | Short | Short Gain | Train min/epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `static` | 96.75 | 87.25 | 94.72 | 90.13 | - | 6.61 |
| `reweight` | 96.72 | 87.15 | 94.67 | 91.51 | +1.38 | 6.80 |
| `augment` | 96.65 | 87.08 | 94.56 | 93.51 | +3.38 | 38.47 |
| `prune` | 96.65 | 87.11 | 94.57 | 93.50 | +3.37 | 18.61 |

Reweighting is the cheapest accuracy gain. Augmentation gives the highest
short-answer accuracy. Pruning reaches almost the same short-answer accuracy as
augmentation with a lower training-time cost.

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

Train a graph method by setting `METHOD` to `static`, `reweight`, `augment`, or
`prune`:

```bash
METHOD=augment
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --use_env mainExplain_gat.py \
  --workers=4 \
  --batch-size=512 \
  --epochs=100 \
  --graph_method=${METHOD} \
  --lr_drop=90 \
  --output_dir=./outputdir/gat_${METHOD}_e100/
```

Evaluate a checkpoint:

```bash
METHOD=augment
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --use_env mainExplain_gat.py \
  --workers=4 \
  --batch-size=256 \
  --evaluate \
  --resume=outputdir/gat_${METHOD}_e100/checkpoint0099.pth \
  --evaluate_sets val_unbiased \
  --graph_method=${METHOD} \
  --output_dir=./outputdir/gat_${METHOD}_e100_eval/
```

Profile all graph methods:

```bash
python profile_memory_methods.py \
  --methods static reweight augment prune \
  --batch-size 64 \
  --batches 3 \
  --warmup-batches 1 \
  --split val_unbiased \
  --output-dir outputdir/memory_profile_bs64_fresh
```

Expected checkpoint locations for profiling:

```text
outputdir/gat_static_e100/checkpoint.pth
outputdir/gat_reweight_e100/checkpoint.pth
outputdir/gat_augment_e100/checkpoint.pth
outputdir/gat_prune_e100/checkpoint.pth
```

Use `--checkpoint METHOD=PATH` to point to different checkpoint files.

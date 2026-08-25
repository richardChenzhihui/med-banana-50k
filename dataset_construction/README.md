# Dataset construction

The two scripts here generated **Med-Banana-80K** from real clinical images using the Gemini API.
They are included for reproducibility — to *use* the dataset, download it from
[🤗 Med-Banana-80K](https://huggingface.co/datasets/RichardChenZH/Med-Banana-80K) instead of
regenerating it.

| Script | Direction | Output suffix |
|---|---|---|
| `add_disease.py` | normal → diseased | `<dataset>-edit/` |
| `remove_disease.py` | diseased → normal | `<dataset>-remove/` |

Both implement the same edit → verify → refine loop that the trained system later learns:

1. A reasoning model writes an editing prompt for the target pathology.
2. An image model applies it to the source image.
3. The same reasoning model acts as LLM-as-Judge over four criteria and returns a structured verdict.
4. On rejection, the judge's reason plus the full prompt history feed a prompt rewrite, and the loop
   retries — **always re-editing the original source image**, never the rejected candidate.

Accepted candidates become success trajectories; rejected ones are kept with their reason and the
refined prompt, and become the failure trajectories that supervise the verifier and refiner.

## Setup

```bash
pip install -r ../requirements.txt
export GOOGLE_API_KEY="YOUR_API_KEY"
```

## Input layout

Nothing is auto-created. Lay out your own source images as:

```text
dataset_construction/full-data/<dataset-name>/
├── normal/          # healthy source images
├── <disease-A>/     # diseased source images, one directory per pathology
└── <disease-B>/
```

`add_disease.py` builds its task list as the cross product of every image in `normal/` × every
disease directory name. `remove_disease.py` takes every image in each non-`normal/` directory.

**The `<dataset-name>` matters.** The modality description sent to the model is keyword-matched
from it (`get_dataset_type`):

| `<dataset-name>` contains | Modality described to the model |
|---|---|
| `mimic` or `chest` | chest X-ray |
| `brain` or `tumor` | brain MRI |
| `odir` or `fundus` | fundus |
| anything else | a generic "medical" description |

Name your directory accordingly, or the prompts lose their modality grounding.

## Running

```bash
python add_disease.py --dataset Chest-XRay --max-workers 20 --max-rounds 5
python remove_disease.py --dataset Chest-XRay --max-workers 20 --max-rounds 5
```

| Flag | Default | Meaning |
|---|---|---|
| `--dataset` | *(required)* | Directory name under `full-data/` |
| `--max-workers` | 20 | Concurrent worker threads |
| `--max-rounds` | 5 | Maximum edit–verify–refine rounds per image |
| `--editor-model` | `gemini-2.5-flash-image-preview` | Image-editing model (the API id of the editor released as *Nano Banana*) |
| `--verifier-model` | `gemini-2.5-pro` | Reasoning model for prompt generation, refinement and verification |

The two model defaults are the ones used to build the released corpus.

## Output layout

```text
dataset_construction/
├── <dataset>-edit/
│   ├── <disease>/<image>-<disease>-edit.jpeg        # accepted edits
│   ├── progress.json                                # per-task state; enables resume
│   ├── final_prompts.json                           # accepted prompt per task
│   ├── all_conversations.json                       # full trajectories, all rounds
│   └── api_failures.json                            # API-level failures (not edit rejections)
└── <dataset>-edit-failed/
    ├── <disease>/<image>-<disease>-edit-failed-<round>.jpeg
    └── failed_summary.json
```

**Resume is automatic.** Re-running the same command skips every task already recorded in
`progress.json`. To redo a task, delete its entry from `progress.json` and re-run.

`api_failures.json` records transport-level failures (timeouts, quota, malformed responses) as
distinct from edits the judge rejected on quality — the latter are trajectories, not errors.

Generated output directories are gitignored; nothing produced here is meant to be committed.

## Judge criteria

The verification step returns a structured verdict over three booleans plus a rationale:

- `has_disease` — a credible instance of the target pathology is present for this modality
- `structure_reasonable` — anatomy and scene hold together, no catastrophic edit damage
- `looks_realistic` — passes as a real acquisition to a trained eye

`qualified` requires all three for **addition**; for **removal** it requires `has_disease` to be
false and the other two true. The full record schema is in
[`../docs/data-format.md`](../docs/data-format.md).

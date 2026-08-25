# Med-Banana-80K record schemas

No dataset files are bundled with this repository. Download the corpus from
[🤗 RichardChenZH/Med-Banana-80K](https://huggingface.co/datasets/RichardChenZH/Med-Banana-80K)
and point the scripts at it via `--medbanana-root` / `$MEDBANANA_ROOT`.

This page documents the record shapes so you can tell what a script expects without downloading
first. Every path below is a placeholder — no real image identifiers appear here.

## Directory layout

```text
Med-Banana-80K/
├── metadata.jsonl        # one row per image record (114,093 rows)
├── success_edit.jsonl    # accepted edits only        (50,635 rows)
├── failed_edit.jsonl     # rejected attempts only     (37,822 rows)
└── data/
    ├── chest_xray/images/{real,addition,removal,failed}/
    ├── brain_mri/images/{real,addition,removal,failed}/
    └── fundus/images/{real,addition,removal,failed}/
```

- `real/` — the original clinical source images
- `addition/` — accepted **normal → diseased** edits
- `removal/` — accepted **diseased → normal** edits
- `failed/` — candidates the verifier rejected, kept with their reason

## `metadata.jsonl`

One row per image record, covering originals, accepted edits and rejected attempts.

```json
{
  "file_name": "data/<modality>/images/<addition|removal|failed|real>/<image>.jpeg",
  "image_id": "<source-dataset key>",
  "modality": "chest_xray",
  "disease": "<target pathology>",
  "edit_type": "addition",
  "status": "success",
  "instruction": "add <disease> to this image",
  "source_image": "data/<modality>/images/real/<image>.jpeg",
  "judge_verdict": { "...": "see below" }
}
```

| Field | Meaning |
|---|---|
| `file_name` | Path to this image, relative to the dataset root |
| `image_id` | Identifier inherited from the source dataset |
| `modality` | `chest_xray`, `brain_mri` or `fundus` |
| `disease` | Target pathology label |
| `edit_type` | `addition`, `removal`, or `none` for original images |
| `status` | `success`, `failed`, or `original` |
| `instruction` | The natural-language editing instruction |
| `source_image` | Path to the paired real image — this is what makes the corpus usable for paired image-to-image training, and what keeps every edited image traceable to its synthetic origin |
| `judge_verdict` | The verifier's structured decision (below) |

## `success_edit.jsonl` and `failed_edit.jsonl`

Edited rows only — `status: original` rows are excluded. These are the files the training and
inference preparation scripts read.

```json
{
  "edited_image": "data/<modality>/images/<addition|removal|failed>/<image>.jpeg",
  "source_image": "data/<modality>/images/real/<image>.jpeg",
  "prompt": "add <disease> to this image",
  "image_id": "<source-dataset key>",
  "modality": "brain_mri",
  "disease": "<target pathology>",
  "label": "<class label>",
  "edit_type": "addition",
  "status": "success",
  "judge_verdict": { "...": "see below" }
}
```

`prompt` may be `null` on some failed rows. Field semantics otherwise match `metadata.jsonl`.

## `judge_verdict`

The construction-time LLM-as-Judge decision, and the same shape the trained verifier emits at
inference (see [`prompt-flow.md`](prompt-flow.md) §7.2):

```json
{
  "qualified": false,
  "has_disease": true,
  "structure_reasonable": true,
  "looks_realistic": false,
  "reason": "one paragraph: decisive evidence for each boolean and why it passed or failed"
}
```

| Field | Meaning |
|---|---|
| `has_disease` | A credible instance of the target pathology is present for this modality |
| `structure_reasonable` | Anatomy and scene hold together; no catastrophic edit damage |
| `looks_realistic` | Passes as a real acquisition to a trained eye at normal review |
| `qualified` | **addition:** all three booleans true. **removal:** `has_disease` false and the other two true |
| `reason` | Free-text rationale — the diagnostic signal that supervises the refiner |

## Verifier / refiner SFT records

`finetune/build_vlm_sft_from_conversations.py` turns raw trajectory logs into ms-swift VLM SFT
JSON. Both datasets share this envelope:

```json
{
  "messages": [
    {"role": "system",    "content": "<verifier or refiner instruction template>"},
    {"role": "user",      "content": "<inquiry text>"},
    {"role": "assistant", "content": "<JSON answer>"}
  ],
  "images": ["<release-dir>/data/<modality>/images/<...>.jpeg"]
}
```

The assistant content differs by module:

**Verifier** — a quality decision over the edited candidate:

```json
{"qualified": false, "has_disease": true, "structure_reasonable": true,
 "looks_realistic": false, "reason": "..."}
```

**Refiner** — a rewritten prompt pair for the next attempt:

```json
{"prompt": "...", "negative_prompt": "..."}
```

`prompt` is the main editing directive the diffusion conditioner sees; `negative_prompt` feeds the
diffusion negative channel and lists at least three distinct prohibitions. Neither may contain
clinical nouns, disease names, or modality acronyms.

Validate that every `images[]` entry resolves before training:

```bash
python finetune/check_vlm_dataset_images_exist.py \
  --publish-root /path/to/raw/publish/tree \
  --prompt-json finetune/data/vlm_prompt_generation_dataset.json \
  --verify-json finetune/data/vlm_verification_dataset.json
```

## Inference task manifests

`verify-refine-loop/runtime/build_inference_set.py` emits a flat task directory that the scheduler
consumes: one image plus one `.txt` holding the initial instruction per task, and a
`manifest.json` describing the batch.

```text
inference/data/addition/
├── <modality>__<disease>__addition__<image_id>.jpg
├── <modality>__<disease>__addition__<image_id>.txt   # e.g. "add <disease> to this image"
└── manifest.json
```

```json
[
  {
    "stem": "<modality>__<disease>__addition__<image_id>",
    "task_type": "addition",
    "modality": "brain_mri",
    "disease": "<target pathology>",
    "image_id": "<source-dataset key>",
    "source_kind": "real_healthy",
    "prompt": "add <disease> to this image"
  }
]
```

The scheduler parses `task_type`, `modality` and `disease` from the stem, reads the `.txt` as the
round-1 prompt, and edits the paired image. See [`prompt-flow.md`](prompt-flow.md) for the full
round-by-round contract.

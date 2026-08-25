# Med-Banana blind human evaluation

A single-file Gradio app for the blind expert assessment reported in the paper. It scores
`edit/addition` outputs from every evaluated model side by side with their source images, without
revealing which model produced which edit.

## Usage

From this directory:

```bash
python eval_gui.py --prepare    # copy and organise source data into ./data
python eval_gui.py --check      # validate the organised data, then exit
python eval_gui.py --port 7861  # launch the scoring UI
```

Then open <http://127.0.0.1:7861>.

The app opens your default browser via Python's standard-library `webbrowser`. If port `7861` is
taken it picks the next free port and prints the real URL. To skip the browser launch:

```bash
python eval_gui.py --port 7861 --no-browser
```

Dependencies are deliberately minimal: Python and Gradio.

## Models under evaluation

By default the app organises and scores these six systems:

- `openai`
- `gemini`
- `qwen-ori`
- `firered-ori`
- `med-banana-qwen-r10`
- `med-banana-firered-r10`

Cases are presented in a random blind order (seed `82` by default, `--seed` to change). The
interface never shows a model name, but `result.json` and `result.jsonl` record the true one.

## Data layout

`python eval_gui.py --prepare` produces:

```text
data/
  manifest.json
  original/
  prompts/
  edited/
    openai/
    gemini/
    qwen-ori/
    firered-ori/
    med-banana-qwen-r10/
    med-banana-firered-r10/
```

To add a model, drop its outputs at `data/edited/<model_name>/<disease>/<edited_image>`. The file
stem must match the source image stem; a trailing `-<disease>-edit` suffix is stripped
automatically before matching.

## Results

Scores are written as you go:

- `result.json` — full records plus aggregate statistics
- `result.jsonl` — one record per line, for downstream scripts

Each record contains `model`, `stem`, `modality`, `disease`, `instruction`, `original_image`,
`edited_image`, `scores` and `average_score`. Statistics are aggregated overall and broken down by
model, by disease and by modality.

## Scoring

Four dimensions, each scored **0–10**. Click each score in the UI, or type all four into the
keyboard-entry box, for example:

```text
8 7 8 6
```

then press **Save and next**.

### Rubric

Score only the edited image on the right; the original on the left is reference. The edit should
add the target disease manifestation while preserving the non-lesion regions as much as possible.

| Band | Meaning |
|---|---|
| **8–10** | Excellent — target clear, location and medical appearance reasonable, only minor issues |
| **6–7** | Acceptable — target recognisable, but moderate flaws, incomplete supporting signs, or imperfect integration |
| **4–5** | Marginally usable — target weak, ambiguous, poorly localised, or of limited medical significance |
| **0–3** | Failure — target missing or wrong, anatomy impossible, image globally replaced, text or symbols instead of a lesion, or clearly non-medical content |

Do not default to 5. If the target lesion is clearly visible, roughly well located, and has no
obvious medical contradiction, the score generally belongs in the 6–8 range, with deductions for
specific defects.

The four dimensions are scored **independently** — do not apply the same defect as a penalty across
all four:

1. **Target lesion / instruction fulfilment.** Does the edit complete the disease-addition task the
   instruction specified? High: the target disease is clearly visible, approximately well located,
   consistent with the original image. Low: target absent, wrong disease, extremely ambiguous, or
   the edited content does not support the intended disease. Slightly fake texture or unnatural
   edges should not by itself push this dimension very low, as long as the target stays recognisable.

2. **Biological and pathological plausibility.** Is the disease appearance consistent with medical
   knowledge and pathological logic? High: lesion morphology, extent, progression pattern and
   related tissue response are basically reasonable. Low: clear biological contradictions —
   impossible lesion morphology, unreasonable disease stage, completely mismatched associated
   structures. Missing some ideal secondary signs may justify deductions, but recognisable lesions
   should not be treated as failures by default.

3. **Modality and anatomical consistency.** Does the edit preserve the imaging modality and major
   anatomical structures? High: modality features (brain MRI, fundus photography, chest X-ray) and
   lesion location and spatial relationships are reasonable. Low: key anatomical structures
   disrupted, lesion in an impossible location, or modality characteristics clearly incorrect.
   Texture, noise and blending issues belong mainly in dimension 4, not here.

4. **Imaging realism and counterfactual fidelity.** Does the edited image look like a real medical
   image, and was only the region that should change altered? High: noise, texture, edges, contrast
   and scanner style look natural, non-lesion regions basically preserved. Low: obvious pasted
   artifacts, hard edges, inconsistent noise, local smearing, repeated textures, irrelevant global
   changes, or image corruption. If the target lesion remains clear and sits within a medical image,
   mild to moderate synthetic artifacts usually merit 5–6, not automatically 0–4.

First determine whether the target disease is recognisable, then assess medical plausibility,
anatomical consistency and imaging realism. Synthetic artifacts should mainly affect the realism
and counterfactual fidelity score, unless they make the target unrecognisable or anatomically
impossible.

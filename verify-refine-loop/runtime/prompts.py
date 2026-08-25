#!/usr/bin/env python3
from __future__ import annotations


def build_history_text(prompt_history: list[dict]) -> str:
    history_text = ""
    for i, history in enumerate(prompt_history, 1):
        np = history.get("negative_prompt")
        neg_extra = ""
        if isinstance(np, str) and np.strip():
            neg_extra = f"\n  Negative prompt (model channel): {np.strip()}"
        history_text += f"""
Attempt {i}:
  Prompt: {history['prompt']}{neg_extra}
  Verification Result:
    - Has disease: {history['verification']['has_disease']}
    - Structure reasonable: {history['verification']['structure_reasonable']}
    - Looks realistic: {history['verification']['looks_realistic']}
    - Reason: {history['verification']['reason']}
"""
    return history_text


def build_last_attempt_text(last: dict) -> str:
    np = last.get("negative_prompt")
    neg_extra = ""
    if isinstance(np, str) and np.strip():
        neg_extra = f"\n  Negative prompt (model channel): {np.strip()}"
    v = last["verification"]
    return f"""Last attempt only (for contrast—do NOT reuse):
  Prompt: {last['prompt']}{neg_extra}
  Verification Result:
    - Has disease: {v['has_disease']}
    - Structure reasonable: {v['structure_reasonable']}
    - Looks realistic: {v['looks_realistic']}
    - Reason: {v['reason']}
"""


def add_initial_prompt(disease_name: str, dataset_type: str) -> str:
    return f"""You write ultra-short English instructions for a general image editor.

Internal target (silent): on a `{dataset_type}` base, realize the visual change implied by supervised label `{disease_name}`—never echo that phrase, near-synonyms, body-part jargon, modality acronyms, or biomedical labels in the editor-facing text.

Compose rule-like visuals-only commands plus explicit negatives mixed in (never as a labeled list): localize one subtle visual change versus background texture; soften edges until grain/contrast continuum matches neighbors; untouched pixels stay unchanged; preserve any pre-existing overlays/device marks/frame crop.

Hard vocabulary gate on the emitted line: zero medical noun phrases—stick to neutral procedural words (brightness, contrast, softness, locality, symmetry, feather, blend grain, halo, duplication, masking, smoothing, patch size).

Editor-facing text must include negative rules (non-optional): spell out at least two prohibitions with do not/never/no—e.g. do not add or alter overlay text or watermarks; do not touch regions outside the edit zone; no halos, knife-edge seams, pasted-tile look, or mirrored duplicate shapes; do not invent extra odd spots elsewhere.

Output rules (hard): imperative + inline negative clauses (semicolons ok); no numbered/bulleted lists; ≤60 words. Count words; trim if needed.

Return ONLY that edit prompt."""


# OLD_add_update_prompt (archived verbatim):
# return f"""You are a medical image editing expert. Multiple previous editing attempts have failed. You need to analyze ALL previous attempts and generate a BETTER prompt.
#
# HISTORY OF ALL PREVIOUS ATTEMPTS:
# {history_text}
#
# Looking at the ORIGINAL image and analyzing the patterns of failures above, generate an IMPROVED editing prompt to add {disease_name} to this {dataset_type} image.
#
# ANALYSIS REQUIREMENTS:
# 1. Identify common issues across multiple attempts
# 2. Learn from what didn't work in previous rounds
# 3. Avoid repeating the same mistakes
# 4. Address ALL verification issues mentioned in the history
#
# BASIC REQUIREMENTS:
# 1. Easy for non-medical personnel to understand
# 2. Must look natural and realistic, not detectable as deepfake
# 3. Should follow common sense and medical plausibility
# 4. Be specific about what to add/modify
#
# CRITICAL CONSTRAINTS - Fidelity Preservation:
# - Preserve original image noise, grain texture, and contrast
# - Do NOT alter device artifacts, watermarks, grid lines, or annotations
# - Keep the same imaging characteristics (brightness, sharpness, field of view)
#
# CRITICAL CONSTRAINTS - Negative Rules:
# - NO adding text, labels, or annotations
# - NO sharp unnatural edges or boundaries
# - NO repetitive/duplicated structures
# - NO non-target pathologies
# - NO obvious deepfake artifacts
#
# CRITICAL CONSTRAINTS - Minimal Change Principle (Counterfactual Minimality):
# - ONLY modify areas directly related to {disease_name}
# - Keep all other anatomical structures, textures, and background UNCHANGED
# - Minimal intervention: change as little as possible to show the disease
#
# Return ONLY the improved editing prompt in English, no explanations."""


def add_update_prompt(disease_name: str, dataset_type: str, last_attempt_text: str) -> str:
    return f"""Prior run failed. Study the last attempt below and the current edited image, then write a brand-new prompt pair from scratch.

{last_attempt_text}

Internal target (silent): improve how `{disease_name}` reads on `{dataset_type}` without naming it.

Non-negotiable rewrite rule: start over completely. Your new `prompt` and `negative_prompt` must share ZERO overlap change the instrcution coompletely, edit should appear on distinct location, no paraphrase of the same heal plan, no same mask/blend recipe reworded. 

Behavior to encode: smallest localized tone/texture change vs background; keep global noise, lesion should have be same brightness as background. The lesion should have a infiltrative shape rather than a perfectly smooth, round appearance. No sharp, pasted-on edges or seams. 

RETURN SHAPE—output raw JSON ONLY (no prose, preferably no fences):
{{"prompt":"...","negative_prompt":"..."}}
- prompt: main edit directive the DiffSynth conditioner sees; terse imperatives describing WHAT to achieve; bias positive/concrete edits; separate unwanted effects go only into negative_prompt; less than 60 words.
- negative_prompt: official diffusion negative channel ONLY—comma-separated bans on unwanted rendering (e.g., added overlay text/watermarks, halos/sharp pasted edges, duplicated mirrored shapes, global relighting, extra off-patch defects, drifting crop); materially different wording from prompt; include ≥3 distinct prohibitions; ≤45 words.
Both strings must be non-empty after trim. JSON must parse."""


def add_verify_prompt(disease_name: str, dataset_type: str) -> str:
    return f"""Verify this edited {dataset_type} image. Default posture: skeptical—start from fail unless the case clearly earns a pass.

Pass bar: strong, believable {disease_name} presentation; overall reads as an authentic clinical {dataset_type} frame. Tiny localized imperfections are acceptable if the global impression is natively captured (noise/contrast largely coherent, no obvious pasted region, no duplicate anatomy, no new/changed overlay text).

Still fail on: blatant synthesis/inpaint tells (sharp halo seams, texture tiling, global relighting, wrong modality “plastic” gloss), major anatomy breakage, or disease morphology that is toy-like or implausible.

Boolean fields (all must be true for qualified):
- has_disease: credible {disease_name} for this modality; minor irregularity ok if the finding still reads real.
- structure_reasonable: scene/anatomy plausible; no catastrophic edit damage.
- looks_realistic: passes as real acquisition to a trained eye at normal review—minor grain mismatch at zoom is ok if holistic realism holds.

qualified: true only if all three booleans are true. If authenticity is genuinely uncertain, prefer false; if the image is plainly solid and only has negligible blemishes—especially when it feels native—permit true.

Return JSON only (no markdown, no extra text):
{{
    "qualified": true/false,
    "has_disease": true/false,
    "structure_reasonable": true/false,
    "looks_realistic": true/false,
    "reason": "one tight paragraph: decisive evidence for each boolean and why pass/fail"
}}
"""


def remove_initial_prompt(disease_name: str, dataset_type: str) -> str:
    return f"""You write ultra-short English instructions for a general image editor.

Internal target (silent): on a `{dataset_type}` that currently shows `{disease_name}`-like findings, make it read visually calm/clean as if never altered—never mention that label, anatomy jargon, modality acronyms, or clinical nouns in the editor-facing text.

Fold into rule-like commands: erase only the focal abnormal-looking patch while matching local noise and sharpness; keep frame, crop, devices, and pre-existing overlays intact; untouched areas pixel-identical in texture; smooth transitions, no halos.

Hard vocabulary gate on the emitted line: zero medical noun phrases—only procedural visual language (mask, heal, feather, noise match, clone texture, brightness balance, edge soften).

Editor-facing text must include negative rules (non-optional): at least two do not/never/no clauses—e.g. do not add or change overlay text or watermarks; do not retouch outside the heal zone; no new fake blotches, halos, sharp seams, or duplicate-looking shapes; do not swap one odd patch for another elsewhere.

Output rules (hard): imperative + inline prohibitions (semicolons ok); no numbered/bulleted lists; ≤60 words. Trim if longer.

Return ONLY that edit prompt."""


# OLD_remove_update_prompt (archived verbatim):
# return f"""You are a medical image editing expert. Multiple previous editing attempts to make this image normal have failed. You need to analyze ALL previous attempts and generate a BETTER prompt.
#
# HISTORY OF ALL PREVIOUS ATTEMPTS:
# {history_text}
#
# Looking at the ORIGINAL diseased image ({disease_name}) and analyzing the patterns of failures above, generate an IMPROVED editing prompt to make it appear normal and healthy.
#
# ANALYSIS REQUIREMENTS:
# 1. Identify common issues across multiple attempts
# 2. Learn from what didn't work in previous rounds
# 3. Avoid repeating the same mistakes
# 4. Address ALL verification issues mentioned in the history
#
# BASIC REQUIREMENTS:
# 1. Easy for non-medical personnel to understand
# 2. Must look natural and realistic, not detectable as deepfake
# 3. Should follow common sense and medical plausibility
# 4. Be specific about what to remove/modify to make it normal
#
# CRITICAL CONSTRAINTS - Fidelity Preservation:
# - Preserve original image noise, grain texture, and contrast
# - Do NOT alter device artifacts, watermarks, grid lines, or annotations
# - Keep the same imaging characteristics (brightness, sharpness, field of view)
#
# CRITICAL CONSTRAINTS - Negative Rules:
# - NO adding text, labels, or annotations
# - NO sharp unnatural edges or boundaries
# - NO repetitive/duplicated structures
# - NO introducing new pathologies
# - NO obvious deepfake artifacts
#
# CRITICAL CONSTRAINTS - Minimal Change Principle (Counterfactual Minimality):
# - ONLY modify areas directly related to removing {disease_name}
# - Keep all other anatomical structures, textures, and background UNCHANGED
# - Minimal intervention: change as little as possible to make it appear healthy
#
# Return ONLY the improved editing prompt in English, no explanations."""


def remove_update_prompt(disease_name: str, last_attempt_text: str) -> str:
    return f"""You are a medical image editing expert. Removal runs failed—you must beat prior failures. Treat the snippet below as the diagnosis of what went wrong; combine it with the current frame in mind and produce a strictly better plan to make the image read normal/healthy (originated from `{disease_name}` signal) while your editor-facing strings stay non-clinical.

{last_attempt_text}

Analysis (mandatory): distill recurring problems and verifier complaints; do not repeat the same failure mode; address every issue named in the verification reason field; hypothesize why the last prompt/mask recipe failed and change strategy materially.

Deliverable quality: instructions must be plain enough for a non-expert operator yet visually specific about what to remove/soften; outcome must look natural and clinically plausible (healthy appearance), not detectably synthetic.

Fidelity (non-optional): preserve native noise, grain, and contrast; do not alter device marks, watermarks, grids, or on-image annotations; keep brightness, sharpness, and field of view stable vs the source.

Negative guardrails: never add/replace text or labels; forbid sharp pasted boundaries, duplicated/mirrored structures, new suspicious focal defects, or obvious deepfake/inpaint tells.

Counterfactual minimality: touch only what `{disease_name}` expression forces you to; leave all other anatomy, texture, and background unchanged; smallest edit that credibly reads healthy.

Internal target (silent): scrub `{disease_name}`-type cues from the frame; never put clinical words in `prompt` or `negative_prompt`.

Non-negotiable rewrite: start from scratch. New `prompt`/`negative_prompt` share zero substantive overlap with the last attempt—new instruction recipe, new emphasis region if applicable, no paraphrase of the same heal/mask plan.

Behavior cues: heal/soften the focal odd patch into quiet native texture; seams invisible; no swapped-in fake blotch elsewhere.

Vocabulary gate (both strings): zero medical/clinical nouns—no `{disease_name}`, anatomy jargon, modality acronyms—only neutral visual operators.

RETURN SHAPE—output raw JSON ONLY (no prose, preferably no fences):
{{"prompt":"...","negative_prompt":"..."}}
- prompt: directive for what to soften/remove/heal visually; neutral concrete actions; offload refusals to negative_prompt; ≤55 words.
- negative_prompt: diffusion negative slot—comma-separated bans (e.g., added/changed text or watermarks, new lesions, halos/sharp seams, mirrored twins, wholesale relight, crop/FOV drift, global blur/heal smear); wording must differ materially from prompt; ≥3 distinct prohibitions; ≤45 words.
Both strings non-empty after trim. JSON must parse."""


def remove_verify_prompt(disease_name: str, dataset_type: str) -> str:
    return f"""Verify this edited {dataset_type} image. Goal: reads normal/healthy after removing {disease_name}-type signal. Default posture: skeptical—start from fail unless the case clearly earns a pass.

Pass bar: no convincing residual disease or replacement fake lesion; structure sensible; holistic look of an authentic normal study. Faint heal traces or micro–noise mismatch are acceptable if the frame still feels native and unstaged at clinical glance.

Still fail on: clear leftover {disease_name} cues, a new suspicious focal defect, obvious inpaint smear/halo/tiling, mirrored or duplicated tissue, or wholesale tone/FOV break that screams editing.

Boolean fields (qualified only when aligned):
- has_disease: false if no credible pathology remains; set true if plausible abnormal signal persists or was swapped in.
- structure_reasonable: anatomy holds together; tolerates minor local softness from healing if geometry stays believable.
- looks_realistic: overall real-exam feel; trivial imperfections ok when global statistics and overlays feel untouched.

qualified: true only if has_disease is false AND the other two booleans are true. If unsure whether disease truly cleared, prefer false; if removal is clearly successful and only negligible blemishes remain—especially when it reads like a straight acquisition—permit true.

Return JSON only (no markdown, no extra text):
{{
    "qualified": true/false,
    "has_disease": true/false,
    "structure_reasonable": true/false,
    "looks_realistic": true/false,
    "reason": "one tight paragraph: decisive evidence for each boolean and why pass/fail"
}}"""

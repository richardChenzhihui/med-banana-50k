# Model Checkpoints

Place downloaded weights under this directory before training or inference.

Expected layout (example):

```text
models/
├── Qwen-Image/
├── Qwen-Image-Edit-2511/
├── FireRed-Image-Edit-1.1/
├── Qwen3.5-9B/
├── Qwen3.5-9B-prompt-refiner/
├── Qwen3.5-9B-prompt-verifier/
└── medgemma-27b-it/          # optional, for MedGemma judge
```

Set `LOCAL_MODELS_ROOT` to this folder when running DiffSynth edit servers or editor LoRA training.

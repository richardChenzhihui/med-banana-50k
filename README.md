# Med-Banana-50K

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Med--Banana--50K-blue)](https://huggingface.co/datasets/RichardChenZH/Med-Banana-50K) [![arXiv](https://img.shields.io/badge/arXiv-2511.00801-b31b1b.svg)](https://arxiv.org/abs/2511.00801)

## 🚀 News
- **[2026-02-08]** 🍌 **FULL DATASET RELEASE!** We are thrilled to announce that the complete **Med-Banana-50K** dataset is now officially open-sourced! This release includes **114,093** records across three modalities:
  - ✅ **50,635** High-quality Successful Edits (Lesion Addition & Removal)
  - ❌ **37,822** Failed Attempts with full conversation logs and LLM-as-Judge feedback (Ideal for RLHF/DPO & Preference Learning!)
  - 🖼️ **25,636** Original clinical source images with precise paired alignment.
  - 🔍 **Enhanced Metadata**: Every record now features `source_image` linking and detailed `judge_verdict` (Compliance, Structure, Realism, Fidelity).
  - 📥 **Explore on Hugging Face:** [RichardChenZH/Med-Banana-50K](https://huggingface.co/datasets/RichardChenZH/Med-Banana-50K)
  - 📄 **Read the Paper:** [arXiv:2511.00801](https://arxiv.org/abs/2511.00801)

This repository contains the official data generation pipeline and data samples for **Med-Banana-50K**, a large-scale, cross-modality dataset for medical image editing.

**[Med-Banana-50K: A Large-Scale Cross-Modality Dataset for Medical Image Editing](https://arxiv.org/abs/2511.00801)**

## About The Project

Recent advances in multimodal large language models have enabled remarkable medical image editing capabilities. However, progress in the research community has been constrained by the absence of large-scale, high-quality, and openly accessible datasets built specifically for medical image editing with strict anatomical and clinical constraints.

To address this, we introduce **Med-Banana-50K**, a comprehensive 50K-image dataset for instruction-based medical image editing spanning three modalities (chest X-ray, brain MRI, and fundus photography) and 23 disease types. Our dataset is constructed by leveraging state-of-the-art multimodal models to generate bidirectional edits (lesion addition and removal) from real medical images.

What distinguishes Med-Banana-50K is our systematic approach to medical quality control: we employ an LLM-as-Judge with a medically grounded rubric (instruction compliance, structural plausibility, realism, and fidelity preservation) and history-aware iterative refinement.

This repository provides the scripts used to generate the dataset and a small sample of the generated images to demonstrate the data structure and quality.

## Dataset Access

**Full Dataset:**

The full Med-Banana-50K dataset, including ~50K successful edits and ~37K failed attempts with full conversation logs, is available on Hugging Face.

**[🤗 Download Med-Banana-50K on Hugging Face](https://huggingface.co/datasets/RichardChenZH/Med-Banana-50K)**

**Data Samples:**

A small subset of the data is available in the `samples/` directory to illustrate the data format and directory structure.

## Getting Started

### Prerequisites

The data generation scripts require Python 3.8+ and several dependencies. You can install them using pip:

```bash
pip install -r requirements.txt
```
*(Note: A `requirements.txt` file will be added soon.)*

### Code Pipeline

This repository contains the core scripts for the data generation pipeline:

*   `add_disease.py`: Script to add pathologies to healthy medical images based on text instructions.
*   `remove_disease.py`: Script to remove pathologies from diseased medical images.
*   `retry_failed.py`: Script to re-process failed attempts from previous runs.

**Setup:**

Before running the scripts, you need to configure your Google AI API key as an environment variable.

```bash
export GOOGLE_API_KEY="YOUR_API_KEY"
```

**Usage:**

Detailed command-line arguments and usage examples are documented within each script. Here is a general example of how to run the lesion addition script:

```bash
python add_disease.py --dataset_name Chest-XRay --disease_name Pneumothorax --output_dir ./output
```

## License

The code in this repository is released under the [MIT License](LICENSE).

The generated images are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and the metadata is released under [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/).

Please refer to the original sources for licenses of the base datasets (MIMIC-CXR, Brain Tumor MRI, ODIR-5K).

## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{chen2025medbanana,
  title={Med-Banana-50K: A Large-Scale Cross-Modality Dataset for Medical Image Editing},
  author={Chen, Zhihui and others},
  journal={arXiv preprint arXiv:2511.00801},
  year={2025}
}
```

## Acknowledgements
*   [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) (Source for Chest-XRay samples)
*   [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Source for Brain-MRI samples)
*   [ODIR-5K](https://www.kaggle.com/datasets/andrewmvd/odir-5k-age-related-macular-degeneration-amd) (Source for Fundus samples)

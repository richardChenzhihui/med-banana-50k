# Med-Banana-50K

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official data generation pipeline and data samples for **Med-Banana-50K**, a large-scale, cross-modality dataset for medical image editing, as presented in our paper:

**[Med-Banana-50K: A Large-Scale Cross-Modality Dataset for Medical Image Editing](Med_banana.pdf)**

## About The Project

Recent advances in multimodal large language models have enabled remarkable medical image editing capabilities. However, progress in the research community has been constrained by the absence of large-scale, high-quality, and openly accessible datasets built specifically for medical image editing with strict anatomical and clinical constraints.

To address this, we introduce **Med-Banana-50K**, a comprehensive 50K-image dataset for instruction-based medical image editing spanning three modalities (chest X-ray, brain MRI, and fundus photography) and 23 disease types. Our dataset is constructed by leveraging state-of-the-art multimodal models to generate bidirectional edits (lesion addition and removal) from real medical images.

What distinguishes Med-Banana-50K is our systematic approach to medical quality control: we employ an LLM-as-Judge with a medically grounded rubric (instruction compliance, structural plausibility, realism, and fidelity preservation) and history-aware iterative refinement.

This repository provides the scripts used to generate the dataset and a small sample of the generated images to demonstrate the data structure and quality.

## Dataset Access

**Data Samples:**

A small subset of the data is available in the `samples/` directory to illustrate the data format and directory structure.

**Full Dataset:**

The full Med-Banana-50K dataset, including ~50K successful edits and ~37K failed attempts with full conversation logs, is currently under review. Upon acceptance, it will be released on Hugging Face.

**Hugging Face Link: [Link will be available here]**

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
@article{medbanana2024,
  title={Med-Banana-50K: A Large-Scale Cross-Modality Dataset for Medical Image Editing},
  author={Anonymous Authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## Acknowledgements
*   [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) (Source for Chest-XRay samples)
*   [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Source for Brain-MRI samples)
*   [ODIR-5K](https://www.kaggle.com/datasets/andrewmvd/odir-5k-age-related-macular-degeneration-amd) (Source for Fundus samples)

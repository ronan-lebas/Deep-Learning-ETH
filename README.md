﻿# Generative Data Augmentation for Detection of Submarine Debris

This repository contains the code and resources for the project *"Generative Data Augmentation for Detection of Submarine Debris"*. Our pipeline introduces novel techniques for enhancing datasets, specifically targeting the task of object detection in the domain of underwater trash and ecosystems.

![Visual Abstract](visual_abstract.svg)

---

## Overview
Deep learning models for object detection demand diverse and large datasets to generalize effectively. Traditional data augmentation methods (e.g., CutMix, GridMask) fall short in introducing new, meaningful visual features.

This project employs a diffusion-based data augmentation pipeline to generate and refine synthetic data:

1. **Inpainting:** A fine-tuned stable diffusion model generates synthetic regions within images.
2. **Bounding Box Refinement:** A neural network, **FocusNet**, refines these regions to improve localization and realism.

We validate our approach using the TrashCan dataset, a labeled underwater dataset, to address the domain-specific challenge of underwater trash detection.

---

## Features

- **Diffusion-Based Inpainting:** Generates diverse and high-fidelity synthetic images.
- **Bounding Box Refinement:** Ensures precise localization for better training results.
- **Performance Evaluation:** Benchmarked using $mAP_{50}$ scores to compare effectiveness.

---

## Results

### Global $mAP_{50}$

| Dataset                        | $mAP_{50}$ |
|-------------------------------|:--------------:|
| Original Dataset              |     0.465      |
| Synthetic Images Only (5k)    |     0.047      |
| Original + Synthetic Images   |  **0.514**     |

### Class-Specific Performance ($mAP_{50}$)

| Category                 | Original Dataset | Synthetic Images | Original + Synthetic | Improvement               |
|--------------------------|------------------|------------------|-----------------------|--------------------------|
| ROV                      | **0.8653**       | 0.0000           | 0.7511               | $\times 0.868 (\downarrow)$ |
| Plant                    | 0.3136           | 0.0000           | **0.4671**           | $\times \textbf{1.489}  (\uparrow)$ |
| Fish                     | 0.3651           | 0.0492           | **0.4393**           | $\times \textbf{1.203}  (\uparrow)$ |
| Starfish                 | 0.4916           | 0.0163           | **0.6627**           | $\times \textbf{1.348}  (\uparrow)$ |
| Shells                   | **0.1949**       | 0.0000           | 0.0734               | $\times 0.377   (\downarrow)$ |
| Crab                     | 0.2434           | 0.0096           | **0.3500**           | $\times \textbf{1.438}  (\uparrow)$ |
| Eel                      | **0.5618**       | 0.0929           | 0.5593               | $\times 0.996   (\downarrow)$ |
| Other Animals            | 0.1477           | 0.0000           | **0.4308**           | $\times \textbf{2.917}  (\uparrow)$ |
| Clothing (Trash)         | 0.3487           | 0.0178           | **0.3673**           | $\times \textbf{1.053}  (\uparrow)$ |
| Pipe (Trash)             | 0.7537           | 0.0087           | **0.8019**           | $\times \textbf{1.064}  (\uparrow)$ |
| Bottle (Trash)           | 0.6285           | 0.0226           | **0.6927**           | $\times \textbf{1.102}  (\uparrow)$ |
| Bag (Trash)              | 0.4596           | 0.1184           | **0.5599**           | $\times \textbf{1.218}  (\uparrow)$ |
| Snack Wrapper (Trash)    | **0.4334**       | 0.0000           | 0.4313               | $\times 0.995   (\downarrow)$ |
| Can (Trash)              | 0.6021           | 0.1908           | **0.6814**           | $\times \textbf{1.132}  (\uparrow)$ |
| Cup (Trash)              | **0.2503**       | 0.0055           | 0.1844               | $\times 0.737   (\downarrow)$ |
| Container (Trash)        | **0.7504**       | 0.1716           | 0.7345               | $\times 0.979   (\downarrow)$ |
| Unknown Instance (Trash) | 0.4412           | 0.0518           | **0.5610**           | $\times \textbf{1.272}  (\uparrow)$ |
| Branch (Trash)           | **0.7672**       | 0.2601           | 0.7165               | $\times 0.934   (\downarrow)$ |
| Wreckage (Trash)         | 0.6385           | 0.0153           | **0.6502**           | $\times \textbf{1.018}  (\uparrow)$ |
| Tarp (Trash)             | 0.0256           | 0.0036           | **0.2156**           | $\times \textbf{8.422}  (\uparrow)$ |
| Rope (Trash)             | 0.5178           | 0.0000           | **0.5824**           | $\times \textbf{1.125}  (\uparrow)$ |
| Net (Trash)              | **0.4334**       | 0.0029           | 0.3934               | $\times 0.908   (\downarrow)$ |

---

## Installation

Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/ronan-lebas/Deep-Learning-ETH
cd Deep-Learning-ETH
```

Install required dependencies using the provided Conda environment file:
```bash
conda env create -f env.yaml
conda activate Augmentation_Pipeline_Env
```

This work was done using CUDA Version: 12.6. The version of the libraries used can be found in the `env.yaml` file.

## Pipeline Usage

### Data Preparation
1. Download the TrashCan dataset from [here](https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7).
2. Place the dataset in the `Full_Pipeline/TrashCan/` directory.

### Running the Pipeline
Run the data augmentation pipeline:
```bash
python main_pipeline.py
```

In the `config.py` file, you can adjust the following parameters:
- Different path to weights and dataset
- Path of the synthetic dataset
- Prompts used to generate synthetic images (to tune with care!)

### Fine Tuning the FocusNet

Fine-tune the FocusNet model on the augmented dataset:
```bash
python fine_tune_lora.py
```

The preprocessed dataset (see `process_dataset.ipynb`) should be placed in the `dataset_finetuning/` directory.

### Training the FocusNet
Train the object detection model on the augmented dataset:
```bash
python main.py
```

Some train hyperparameters such as the learning rate, batch size, and number of epochs can be adjusted in the `main.py` file.

### Evaluation

Further indications to reproduce the detectors training can be found in the `README.md` file in the `Detector_Training` directory.

---

## Acknowledgments

This project uses the TrashCan dataset and builds upon Stable Diffusion and Lora finetuning script provided by HuggingFace.

---

## License
This repository is licensed under the MIT License.

---

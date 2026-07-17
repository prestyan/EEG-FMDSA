# Foundation-Model-Guided Discrete Semantic Reencoding for Aligning EEG Representations with LLMs (EEG-FMDSR)

## Overview
Electroencephalography (EEG) signal decoding plays an important role in brain-computer interfaces and neurological analysis. This repository accompanies our FMDSR framework, which converts EEG signals into LLM-compatible discrete tokens. FMDSR first uses an EEG foundation model to extract generalizable representations, then quantizes them with a pretrained codebook, and finally semantically reencodes these tokens with prototypes derived from the LLM embedding space to better leverage LLM prior knowledge for downstream classification.

## Repository Structure
- `codebook/`: code for pre-training the codebook used to quantize EEG foundation model representations.
- `training/`: code for fine-tuning the LLM using FMDSR for downstream tasks.

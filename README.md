
---

# Transformer-based End-to-End Beat Saber Beatmap Generation

## Introduction

Most existing Beat Saber beatmap generation approaches are **not end-to-end**.
Representative methods such as *DeepSaber* and *DeepSAGE* rely on multiple independently trained components, for example:

* learning word or note embeddings first,
* training separate CNN/RNN (or LSTM) models for
  **when to place notes** and **what notes to place**.

These pipelines are complex, heavily depend on prior heuristics, and often require significant manual tuning, while the final results remain limited.

Unlike previous approaches, this project aims to train an end-to-end model using modern, widely adopted deep learning techniques, resulting in a simpler training pipeline and beatmaps that are visually and structurally comparable to those created by human mappers.

> ⚠️ This project is still under active development and experimentation.
> The data pipeline, baseline model forward pass, and training code are completed, while ongoing work focuses on improving generation quality beyond the baseline.

---

## Project Goal

Given an input music track, the model automatically generates a corresponding **playable Beat Saber beatmap**.

From a data modeling perspective, a Beat Saber beatmap can be represented as a **time series**:

* Each time step corresponds to a **3 × 4 grid**;
* Each grid cell is either empty or contains a note;
* Each note has:

  * **Color** (red / blue), corresponding to left / right hand;
  * **Direction**, specifying the required swing direction.

---

## Dataset Visualization

**Gameplay view (player perspective):**

<video src="dataset_visualization/data_video1.mp4" controls width="600"></video>
**[clip from youtube](https://www.youtube.com/watch?v=LlOlSWnCsQA)**

**Editor view (ground-truth beatmap data):**

<video src="dataset_visualization/data_video2.mp4" controls width="600"></video>

---

## Method Overview

The model follows a **Transformer encoder–decoder architecture**.

### Encoder

* Initialized from **pretrained MusicGen / MAGNeT models**.
* The encoder is **frozen during training** and used purely as a music feature extractor.
* This significantly stabilizes training and accelerates convergence.

### Decoder

* The decoder autoregressively generates beatmaps **along the time dimension**.
* The design is inspired by **Visual Autoregressive Modeling (VAR)**:

  * Tokens are organized with block-wise attention masks;
  * Attention blocks are aligned with generation units (time steps instead of image scales).

Key differences from original VAR:

* VAR autoregresses over **multi-scale image resolutions**;
* This project autoregresses over **beat-aligned time steps**.

### Token Design

* Each token represents a **(color × direction)** note combination.
* An additional **empty token** indicates no note at a given grid position.
* Beatmaps are generated sequentially over time.

---

## Encoder–Decoder Alignment

The encoder and decoder are connected via **local cross-attention**:

* At each beat step, the decoder attends only to the **corresponding temporal window** of encoder features.
* This ensures locality between music and beatmap generation.

To handle the mismatch between:

* beat-based time steps (e.g., 1/4, 1/8 notes),
* and second-based audio features,

the alignment is constructed by analyzing the **receptive field of the MusicGen encoder** and mapping beats to encoder feature ranges.

---

## Key Features

* Transformer-based **end-to-end beatmap generation**
* Pretrained **MusicGen feature extraction**
* Efficient **offline tokenization pipeline**:

  * audio → Encodec tokens (GPU-intensive, done once)
  * beatmap → discrete tokens (CPU-intensive)
* Block-wise Transformer decoder with:

  * Rotary Position Embedding (RoPE)
  * 2D RoPE variants for grid-based attention
* LoRA-based fine-tuning for pretrained models

---

## Project Structure

This project is built on top of the **AudioCraft framework** and reuses **MusicGen** components.

```
audiocraft/
├── models/
│   └── lm_beatmapgen.py
│       # Beatmap generation model (encoder–decoder architecture)
├── solvers/
│   └── beatmapgen.py
│       # Training logic and solver implementation
├── data/
│   ├── audio_dataset_beatmap.py
│   │   # Dataset and tokenizer:
│   │   # beatmap JSON → tokens
│   │   # audio → Encodec tokens
│   ├── beatmap_dataset.ts
│   │   # Beatmap formatter (Beat Saber → JSON)
│   └── beatmap_parser.ts
│       # Beatmap generator (JSON → Beat Saber format)
├── modules/
│   ├── transformer.py
│   │   # Block-wise Transformer (encoder & decoder)
│   ├── lora.py
│   │   # LoRA fine-tuning for pretrained models
│   ├── rope.py
│   │   # Block-wise RoPE embedding
│   ├── rope_ca.py
│   │   # RoPE for cross-attention
│   └── rope_xy.py
│       # 2D RoPE for grid-based self-attention
config/
└── solver/beatmapgen/
    └── beatmapgen_base_32khz.yaml
        # Training configuration
beatmapgen.sh
# Script for data preprocessing, training, and inference
```

---

## Evaluation

* **Automatic metrics**:

  * BLEU score between generated and reference beatmaps
* **Human evaluation**:

  * Visual inspection and playability assessment

Results:

* Generated beatmaps have been visually validated on multiple tracks;
* **Rhythm alignment accuracy exceeds 70%**.

---

## What Works Well

* Leveraging **MusicGen encoder features** leads to **very fast convergence**.
* Heavy preprocessing is performed **before training**:

  * audio is converted to Encodec tokens once,
  * beatmaps are tokenized offline.
* This significantly reduces GPU usage during training and keeps resource consumption low.

---

## Experiments & Extensions

The following improvements have been explored:

* Block-wise RoPE for long sequence modeling
* 2D RoPE for grid-aware attention
* LoRA fine-tuning on pretrained encoders/decoders
* Alternative generation models, including:

  * VAR + MaskGIT inspired methods (e.g., HMAR)

These extensions improve rhythm consistency but provide limited gains beyond the baseline so far.

---

## In-Progress Work

* Exploring **event-level set prediction** instead of pixel-wise autoregression.
* Investigating a **DETR-inspired approach**:

  * Predicting a set of notes per time step,
  * Using cross-attention to music features or intermediate decoder states.
* Improving structural constraints and playability modeling.

---

## License

To be decided.

---
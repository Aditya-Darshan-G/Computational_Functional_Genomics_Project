# Predicting Transcription Factor Binding Potential from Genomic Bins

This repository contains our course project for **Computational Functional Genomics at IISER Pune**. The project focuses on predicting the in vivo binding potential of three transcription factors — **CTCF, REST, and EP300** — in the **K562 cell line**.

The task was motivated by the limitation of sequence-only motif models such as position weight matrices (PWMs). While PWMs capture in vitro DNA-binding preferences, they often fail to fully explain in vivo binding because transcription factor occupancy also depends on chromatin accessibility, genomic context, cell type, cooperative binding, and other regulatory factors.

## Project Goal

The goal was to predict binding scores for **CTCF, REST, and EP300** on held-out genomic regions from chromosomes:

- `chr3`
- `chr10`
- `chr17`

Each genomic region was represented as a 200 bp bin. For the prediction chromosomes, the transcription factor labels were hidden, and the model was required to output a binding score/probability for each TF.

The final output files contain predicted binding scores where larger values indicate higher predicted binding potential.

## Dataset

The dataset consists of 200 bp genomic bins from the human genome build `hg38`.

Each training chromosome file contains:

| Column | Description |
|---|---|
| `chr` | Chromosome |
| `start` | Start coordinate of the 200 bp bin |
| `end` | End coordinate of the 200 bp bin |
| `ATAC` | Chromatin accessibility status |
| `CTCF` | Binding status for CTCF |
| `REST` | Binding status for REST |
| `EP300` | Binding status for EP300 |

The labels use:

- `B` = bound / accessible
- `U` = unbound / inaccessible

The chromosomes `chr3`, `chr10`, and `chr17` contain only the genomic coordinates and ATAC status, and were used for final prediction.

## Methods

The project had two main stages.

### 1. Intermediate Milestone: Markov Model Classifier

For the intermediate milestone, we implemented a Markov model-based classifier from scratch.

The script:

```text
src/Markov_Model_Milestone_Intermediate.py

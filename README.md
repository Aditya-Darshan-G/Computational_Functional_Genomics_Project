# Predicting Transcription Factor Binding Potential in K562 Cells

This repository contains a team project completed for the **Computational Functional Genomics** course at **IISER Pune**.

The objective was to predict the **in vivo binding potential** of three transcription factors—**CTCF**, **REST**, and **EP300**—in the **K562 cell line**. Predictions were required for 200 bp genomic bins from chromosomes **3, 10, and 17**, whose transcription-factor labels were withheld.

## Project overview

Position weight matrices and other sequence-only motif models describe intrinsic DNA-binding preferences, but they do not fully represent in vivo occupancy. Binding in a cell also depends on factors such as chromatin accessibility, local genomic context, TF-specific motif evidence, and cell type.

This project therefore combined genomic-bin information with chromatin-accessibility and motif-derived features to generate a binding score for each TF. Larger output values indicate greater predicted binding potential.

## Dataset

The dataset was derived from **ENCODE ChIP-seq peaks** using the **hg38** human reference genome and was divided into 200 bp genomic bins.

The 19 labelled training-chromosome files contain:

| Column | Description |
|---|---|
| `chr` | Chromosome |
| `start` | Start coordinate of the 200 bp bin |
| `end` | End coordinate of the 200 bp bin |
| `ATAC` | Chromatin-accessibility status |
| `CTCF` | CTCF binding status |
| `REST` | REST binding status |
| `EP300` | EP300 binding status |

The categorical labels are:

- `B`: bound or accessible
- `U`: unbound or inaccessible

The held-out files for `chr3`, `chr10`, and `chr17` contain only `chr`, `start`, `end`, and `ATAC`. The task was to predict continuous scores for the three missing TF columns.

Not every possible 200 bp genomic interval is represented: repeat regions and bins with missing sequence information were removed from the supplied data.

## Methods

### 1. Intermediate milestone: Markov model classifier

The intermediate milestone is implemented in:

```text
src/markov_model_milestone_intermediate.py
```

The script:

- extracts genomic sequences from an external hg38 FASTA file;
- trains separate order-\(m\) Markov models for bound and unbound sequences;
- scores each sequence using the two learned models;
- performs chromosome-level \(k\)-fold cross-validation;
- produces ROC and precision-recall curves and their average areas.

The implementation supports Markov orders from 0 to 10 and 3 to 5 cross-validation folds, as required for the milestone.

A smaller educational implementation of the core Markov-model logic is available in:

```text
src/simpler_version.py
```

### 2. Final prediction pipeline

The final project code is:

```text
src/tf_binding_pipeline.py
```

It trains a separate class-balanced `GradientBoostingClassifier` for each TF. Depending on the TF and command-line options, the features include:

- ATAC accessibility of the current genomic bin;
- ATAC accessibility of one or two adjacent retained bins on each side;
- an ATAC accessibility sum across the local window;
- optional FIMO motif-hit features for CTCF and REST:
  - motif-hit indicator;
  - maximum FIMO score;
  - transformed minimum p-value;
  - transformed minimum q-value.

EP300 is modelled using ATAC and neighbouring-bin context without FIMO features.

## Repository structure

```text
Computational_Functional_Genomics_Project/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .gitattributes
│
├── src/
│   ├── tf_binding_pipeline.py
│   ├── simpler_version.py
│   └── markov_model_milestone_intermediate.py
│
├── data/
│   ├── chr1_200bp_bins.tsv
│   ├── chr2_200bp_bins.tsv
│   ├── ...
│   ├── chr22_200bp_bins.tsv
│   ├── chr3_200bp_bins_unknown.tsv
│   ├── chr10_200bp_bins_unknown.tsv
│   └── chr17_200bp_bins_unknown.tsv
│
└── results/
    ├── final_predictions/
    │   ├── chr3.tsv
    │   ├── chr10.tsv
    │   ├── chr17.tsv
    │   ├── chr3_binding_analysis.txt
    │   ├── chr10_binding_analysis.txt
    │   └── chr17_binding_analysis.txt
    │
    └── per_tf_predictions/
        ├── CTCF_predictions.tsv
        ├── REST_predictions.tsv
        └── EP300_predictions.tsv
```

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are:

- NumPy
- pandas
- scikit-learn
- Matplotlib
- pyfaidx

## Running the code

### Final Gradient Boosting pipeline

The pipeline can be run one TF at a time to reduce memory use. The following example trains the CTCF model using two neighbouring retained bins on each side:

```bash
python src/tf_binding_pipeline.py \
    --data-dir data \
    --train-chromosomes 1 2 4 5 6 7 8 9 11 12 13 14 15 16 18 19 20 21 22 \
    --predict-chromosomes 3 10 17 \
    --ctcf-fimo path/to/CTCF_fimo.tsv \
    --rest-fimo path/to/REST_fimo.tsv \
    --neighbors 2 \
    --output-dir results/per_tf_predictions \
    --tf CTCF
```

Change `--tf` to `REST` or `EP300` to train the other models. The FIMO paths are used for CTCF and REST; EP300 uses ATAC-derived features only.

### Intermediate Markov model

```bash
python src/markov_model_milestone_intermediate.py \
    --m 6 \
    --k 5 \
    --tsv data/chr1_200bp_bins.tsv \
    --tf CTCF \
    --genome_path path/to/hg38.fa
```

Optional arguments include `--pseudocount`, `--unseen_kmer_prob`, and `--n_jobs`.

### Simplified Markov model

```bash
python src/simpler_version.py path/to/sequences.fa 3
```

Here, `3` is the chosen Markov-model order.

## FIMO inputs

The final CTCF and REST models use optional motif-hit features derived from **FIMO** output files. These external FIMO files are not included in the repository and must be generated or supplied separately in the tab-separated format expected by `tf_binding_pipeline.py`.

The expected FIMO columns include:

```text
sequence_name
start
score
p-value
q-value
```

The pipeline aggregates motif hits into the corresponding 200 bp genomic bins before model training and prediction.

## Prediction outputs

For each TF, the final pipeline writes a table such as:

```text
results/per_tf_predictions/CTCF_predictions.tsv
```

with the columns:

```text
chr    start    end    ATAC    CTCF
```

The equivalent files for REST and EP300 contain their respective score columns.

The final chromosome-specific files were created through simple post-processing: the TF-specific prediction tables were filtered by chromosome and their score columns were aligned using the genomic coordinates.

The final repository copies are:

```text
results/final_predictions/chr3.tsv
results/final_predictions/chr10.tsv
results/final_predictions/chr17.tsv
```

Each has the format:

```text
chr    start    end    ATAC    CTCF    REST    EP300
```

The TF columns contain continuous predicted binding scores between 0 and 1.

## Binding-combination summaries

The accompanying `*_binding_analysis.txt` files summarise predicted TF-binding combinations using both chromatin accessibility and the model scores.

A TF is counted as predicted bound only when:

1. the bin is accessible (`ATAC = B`); and
2. its predicted score is at least `0.5`.

Bins with `ATAC = U` are therefore classified as **None bound** in these summaries.

| Chromosome | Total bins | Accessible bins | All three predicted bound | None bound |
|---|---:|---:|---:|---:|
| chr3 | 307,939 | 11,334 | 11,331 | 296,605 |
| chr10 | 212,978 | 9,106 | 9,104 | 203,872 |
| chr17 | 122,530 | 10,106 | 10,095 | 112,424 |

The text files also list the highest-scoring bins for each TF.

## Methodological notes

- The intermediate Markov-model script reports performance from \(k\)-fold cross-validation.
- The AUROC and AUPRC printed by the final Gradient Boosting script are training diagnostics; the held-out chromosomes had no public TF labels.
- Neighbour features use adjacent retained rows within each chromosome. Because some genomic bins were removed from the dataset, adjacent rows are not always immediately contiguous 200 bp intervals.
- Scikit-learn's `GradientBoostingClassifier` is single-threaded; the `--n-jobs` option is retained for resource reporting and other workflow control.

## Reproducibility

To reproduce the intermediate milestone, an indexed **hg38 FASTA** file is required. The reference genome is not included in this repository because of its size.

To reproduce the final CTCF and REST models exactly, the corresponding FIMO output files and the same neighbour-window setting must also be supplied.

## Course context

This work was completed for the **Computational Functional Genomics** course at **IISER Pune**.

The course project consisted of:

1. an intermediate Markov model-based classifier evaluated using ROC and precision-recall curves; and
2. a final pipeline that generated CTCF, REST, and EP300 binding scores for chromosomes 3, 10, and 17.

## Data and licensing

The source code in this repository is released under the terms in the repository's `LICENSE` file.

The code licence does not transfer ownership of the genomic datasets. The supplied genomic data and derived resources retain the provenance, attribution requirements, and usage terms of their original sources and course distribution. Users should verify those terms before redistributing or reusing the data outside the context of this project.

## Acknowledgements

- IISER Pune Computational Functional Genomics course
- ENCODE Project for the ChIP-seq-derived binding information
- hg38 human reference genome
- FIMO for motif scanning

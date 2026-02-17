Predicts transcription factor binding sites using Markov models with k-fold cross-validation.

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn pyfaidx
```

Python: 3.7+

## Data Requirements

### 1. Human Genome (hg38)

Download from UCSC Genome Browser:

```bash
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa.gz
```

### 2. TSV Input File

Format: `chr<N>_200bp_bins.tsv`

```text
chr     start   end     ATAC    CTCF    REST    EP300
chr4    0       200     U       B       U       U
chr4    200     400     U       U       B       U
```

Columns: chr, start, end, ATAC, CTCF, REST, EP300

Labels: B (bound) or U (unbound)

## File Structure

```text
project/
├── cfg_project_markov_model_intermediate_milestone.py
├── data/
│   └── chr4_200bp_bins.tsv
├── hg38.fa
└── outputs/                          # Created automatically
    ├── CTCF_chr4_markov_model_B/     # Bound models
    ├── CTCF_chr4_markov_model_U/     # Unbound models
    ├── CTCF_chr4_predicted_values/   # Predictions per fold
    └── CTCF_chr4_outputs/            # Metrics & plots
```

## Usage

### Basic Run

```bash
python cfg_project_markov_model_intermediate_milestone.py     --m 6     --k 5     --tsv data/chr4_200bp_bins.tsv     --tf CTCF     --genome_path hg38.fa
```

### With All Parameters

```bash
python cfg_project_markov_model_intermediate_milestone.py     --m 6     --k 5     --tsv data/chr4_200bp_bins.tsv     --tf CTCF     --genome_path hg38.fa     --pseudocount 0.01     --unseen_kmer_prob 0.01     --n_jobs 4
```

## Parameters

| Parameter | Required | Default | Description |
|---|---:|---:|---|
| `--m` | ✓ | - | Markov model order (0-10) |
| `--k` | ✓ | - | Number of CV folds (3-5) |
| `--tsv` | ✓ | - | Path to TSV input file |
| `--tf` | ✓ | - | TF name: CTCF, REST, or EP300 |
| `--genome_path` | ✓ | - | Path to hg38.fa |
| `--pseudocount` | ✗ | 0.01 | Laplace smoothing value |
| `--unseen_kmer_prob` | ✗ | 0.01 | Probability for unseen k-mers |
| `--n_jobs` | ✗ | 1 | CPU cores (use >1 for parallel) |

## Output Files

Per fold (in `{TF}_{chr}_outputs/`):

- `{TF}_{chr}_m{m}_k{k}ROC_fold{N}.csv` - ROC curve data
- `{TF}_{chr}_m{m}_k{k}PR_fold{N}.csv` - PR curve data

Aggregate:

- `{TF}_{chr}_m{m}_k{k}_ROC_all_folds.png` - Combined ROC plot
- `{TF}_{chr}_m{m}_k{k}_PR_all_folds.png` - Combined PR plot
- `{TF}_{chr}_m{m}_k{k}_summary.txt` - Performance summary

Models (JSON):

- `{TF}{chr}markov_model{B|U}/markov_model{m}fold{N}.json`
- `{TF}{chr}markov_model{B|U}/markov_model_values{m}fold{N}.json`

End Of File

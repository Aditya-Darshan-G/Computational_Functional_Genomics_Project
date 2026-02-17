# Markov Model for Transcription Factor Binding Site Prediction

## Overview

This pipeline implements a **k-mer Markov model** to predict transcription factor (TF) binding sites in the human genome. It trains separate models on bound (B) and unbound (U) genomic regions, performs k-fold cross-validation, and evaluates performance using ROC and precision-recall curves.

**Key Features:**
- Variable-order Markov models (m = 0-10)
- K-fold cross-validation (k = 3-5)
- Parallel processing support
- Laplace smoothing for unseen k-mers
- Comprehensive performance metrics (AUROC, AUPRC, confusion matrices)

---

## Requirements

### Software Dependencies
```bash
pip install numpy pandas matplotlib scikit-learn pyfaidx
Package	Version	Purpose
numpy	≥1.20	Numerical operations
pandas	≥1.3	Data manipulation
matplotlib	≥3.4	Plotting ROC/PR curves
scikit-learn	≥1.0	Cross-validation, metrics
pyfaidx	≥0.6	Fast FASTA indexing
System Requirements
Python: 3.8+

RAM: 8GB minimum (16GB+ recommended for m ≥ 7)

Disk Space: ~50GB for hg38 genome + outputs

CPU: Multi-core recommended for parallel processing

Data Requirements
1. Human Genome (hg38)
Download from UCSC Genome Browser:

bash
# Option 1: Direct download (RECOMMENDED)
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa.gz

# Option 2: Via rsync
rsync -avzP \
  rsync://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz .
gunzip hg38.fa.gz
Alternative Sources:

NCBI: https://www.ncbi.nlm.nih.gov/genome/guide/human/

Ensembl: http://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/dna/

Note: The genome file (~3.1 GB uncompressed) must be indexed by pyfaidx on first use (automatic).

2. TF Binding Data (TSV Format)
Required columns:

text
chr     start    end      ATAC   CTCF   REST   EP300
chr1    10000    10200    U      B      U      U
chr1    10200    10400    U      B      B      U
...
chr: Chromosome name (e.g., chr1, chr2, ..., chrX)

start/end: 0-based genomic coordinates

TF columns: Binding labels (B = bound, U = unbound)

File Structure
Input Files
text
project/
├── cfg_project_markov_model_intermediate_milestone.py
├── data/
│   ├── chr1_200bp_bins.tsv
│   ├── chr2_200bp_bins.tsv
│   └── ...
└── hg38.fa
Output Structure (Auto-generated)
text
CTCF_chr1_markov_model_B/
├── markov_model_3_fold_1.json          # Probabilities for bound model
├── markov_model_values_3_fold_1.json   # Raw counts for bound model
└── ...

CTCF_chr1_markov_model_U/
├── markov_model_3_fold_1.json          # Probabilities for unbound model
└── ...

CTCF_chr1_predicted_values/
├── chr1_200bp_bins_predictions_fold_1.tsv
└── ...

CTCF_chr1_outputs/
├── CTCF_chr1_m3_k5_ROC_fold_1.csv      # ROC curve data
├── CTCF_chr1_m3_k5_PR_fold_1.csv       # Precision-Recall data
├── CTCF_chr1_m3_k5_ROC_all_folds.png   # ROC plot
├── CTCF_chr1_m3_k5_PR_all_folds.png    # PR plot
└── CTCF_chr1_m3_k5_summary.txt         # Performance summary
Usage
Basic Command
bash
python cfg_project_markov_model_intermediate_milestone.py \
    --m 5 \
    --k 5 \
    --tsv data/chr1_200bp_bins.tsv \
    --tf CTCF \
    --genome_path hg38.fa \
    --pseudocount 0.01 \
    --unseen_kmer_prob 0.01 \
    --n_jobs 4
Parameters
Parameter	Type	Required	Default	Description
--m	int	✓	-	Markov model order (0-10)
--k	int	✓	-	Number of CV folds (3-5)
--tsv	str	✓	-	Path to TSV file
--tf	str	✓	-	TF name (CTCF/REST/EP300)
--genome_path	str	✓	-	Path to hg38.fa
--pseudocount	float	✗	0.01	Laplace smoothing parameter
--unseen_kmer_prob	float	✗	0.01	Probability for unseen k-mers
--n_jobs	int	✗	1	CPU cores (1=serial, >1=parallel)
Running Multiple Models (Overnight Batch)
Test Multiple Markov Orders (m = 1 to 10)
bash
# Create batch script
cat > run_all_m_values.sh << 'EOF'
#!/bin/bash
for m in {1..10}; do
    echo "Starting m=$m at $(date)"
    python cfg_project_markov_model_intermediate_milestone.py \
        --m $m \
        --k 5 \
        --tsv data/chr4_200bp_bins.tsv \
        --tf CTCF \
        --genome_path hg38.fa \
        --pseudocount 0.01 \
        --unseen_kmer_prob 0.01 \
        --n_jobs 4 \
        > log_m${m}.txt 2>&1
    echo "Completed m=$m at $(date)"
done
EOF

# Make executable and run
chmod +x run_all_m_values.sh
nohup ./run_all_m_values.sh &

# Monitor progress
tail -f log_m5.txt
Output Files
1. Markov Models (JSON)
json
{
  "AAGAGCA": [0.346, 0.142, 0.270, 0.242],
  ...
}
Format: [P(A|context), P(C|context), P(G|context), P(T|context)]

2. Predictions (TSV)
text
chr     start    end      prediction  score
chr1    10000    10200    B           2.34
chr1    10200    10400    U           -1.12
3. Performance Metrics (CSV)
ROC Curve:

text
FPR,TPR
0.0,0.0
0.001,0.125
...
Precision-Recall Curve:

text
Recall,Precision
1.0,0.008363
0.995,0.012456
...
4. Summary Report (TXT)
text
Parameters:
  Markov Model Order (m): 5
  Number of Folds (k): 5
  
Results:
  Average ROC AUC: 0.8765
  Average PR AUC: 0.6543
  
Per-Fold Results:
  Fold 1: ROC AUC = 0.8823, PR AUC = 0.6612
  ...
Algorithm Overview
Training Phase (Each Fold)
Extract sequences from hg38 for training regions

Count k-mer transitions: For each sequence, count P(next_nucleotide | prev_m_nucleotides)

Apply Laplace smoothing: P = (count + pseudocount) / (total + 4*pseudocount)

Build separate models for bound (B) and unbound (U) sequences

Prediction Phase
Calculate log-likelihoods: log P(sequence | model_B) and log P(sequence | model_U)

Compute log-odds score: score = log P(seq|B) - log P(seq|U)

Classify: Predict B if score > 0, else U

Evaluation
Confusion matrix: TP, TN, FP, FN

ROC curve: TPR vs FPR at different thresholds

PR curve: Precision vs Recall

Performance Tips
Memory Optimization
Low m values (0-5): ~2-4 GB RAM

High m values (7-10): 8-16 GB RAM

Use --n_jobs 1 if memory-limited

Speed Optimization
Parallel processing: Set --n_jobs to # of CPU cores

Higher m values: Exponentially slower (4^m k-mers)

Expected runtime (chr1, m=5, k=5, 4 cores): ~15-30 minutes

Recommended Settings
Genome Size	m	k	n_jobs	Runtime
Single chr	5	5	4	~30 min
Single chr	7	5	8	~2 hours
Whole genome	5	5	16	~4-6 hours
Troubleshooting
Issue: KeyError during prediction
Cause: Unseen k-mer in test set
Solution: Already handled via --unseen_kmer_prob parameter

Issue: MemoryError for high m values
Solution:

bash
# Reduce parallelism
--n_jobs 1

# Use swap space
sudo fallocate -l 16G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
Issue: pyfaidx index error
Solution:

bash
rm hg38.fa.fai  # Remove corrupted index
# Re-run script (will re-index automatically)
Issue: Genome download fails
Solution:

bash
# Try alternative UCSC mirror
wget https://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz

# Or use aria2c for resume support
aria2c -x 5 https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
Citation
If you use this code, please cite:

text
Markov Model for TF Binding Site Prediction
Computational Functional Genomics Project
2026
License
MIT License - See LICENSE file for details

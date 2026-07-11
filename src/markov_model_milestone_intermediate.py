"""
markov_model_milestone_intermediate.py


Markov Model-based Transcription Factor Binding Site Predictor
Computational Functional Genomics Project - Intermediate Milestone


This script builds separate Markov models for bound (B) and unbound (U) sequences,
performs k-fold cross-validation, and evaluates prediction performance.
"""


# ===========================
# IMPORT REQUIRED LIBRARIES
# ===========================


import argparse  # For parsing command-line arguments
import json  # For reading/writing JSON files
import os  # For file and directory operations
import sys  # For system operations and error handling
from collections import defaultdict  # For creating dictionaries with default values
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation with DataFrames
import matplotlib.pyplot as plt  # For plotting ROC and PR curves
from sklearn.model_selection import KFold  # For k-fold cross-validation splitting
from sklearn.metrics import roc_curve, auc, precision_recall_curve  # For computing ROC and PR curves
from pyfaidx import Fasta  # For efficient FASTA file access
import warnings  # For suppressing warnings
warnings.filterwarnings('ignore')  # Suppress sklearn warnings
import time  # for timing the code
from datetime import timedelta  # For pretty time formatting
from multiprocessing import Pool  # For parallel processing across folds



# ===========================
# PARSE COMMAND-LINE ARGUMENTS
# ===========================
# This section defines what arguments the user must provide when running the script


def parse_arguments():
    """
    Parse command-line arguments for the Markov model script.
    
    Required arguments:
    - m: Order of Markov model (0-10)
    - k: Number of folds for cross-validation (3-5)
    - tsv: Path to TSV file containing chromosome data
    - tf: Name of transcription factor (CTCF, REST, or EP300)
    - genome_path: Path to hg38.fa genome file
    
    Optional arguments:
    - pseudocount: Laplace smoothing pseudocount (default: 0.01)
    - unseen_kmer_prob: Probability for unseen k-mers (default: 0.01)
    - n_jobs: Number of CPU cores for parallel processing (default: -1)
    
    Returns:
    - Parsed arguments object
    """
    parser = argparse.ArgumentParser(
        description='Markov Model for TF Binding Site Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--m', type=int, required=True,
                        help='Order of Markov model (0-10)')
    parser.add_argument('--k', type=int, required=True,
                        help='Number of folds for cross-validation (3-5)')
    parser.add_argument('--tsv', type=str, required=True,
                        help='Path to TSV file (e.g., chr1_200bp_bins.tsv)')
    parser.add_argument('--tf', type=str, required=True, choices=['CTCF', 'REST', 'EP300'],
                        help='Transcription factor name (CTCF, REST, or EP300)')
    parser.add_argument('--genome_path', type=str, required=True,
                        help='Path to hg38.fa genome file')
    
    # Optional arguments
    parser.add_argument('--pseudocount', type=float, default=0.01,
                        help='Pseudocount for Laplace smoothing (default: 0.01)')
    parser.add_argument('--unseen_kmer_prob', type=float, default=0.01,
                        help='Probability for unseen k-mers (default: 0.01)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of CPU cores for parallel processing (default: 1)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.m < 0 or args.m > 10:
        parser.error("m must be between 0 and 10")
    if args.k < 3 or args.k > 5:
        parser.error("k must be between 3 and 5")
    
    return args



# ===========================
# INITIALIZE MARKOV MODEL STRUCTURES
# ===========================
# This section creates the initial data structures for storing Markov model parameters


def initialize_markov_model(m):
    """
    Initialize a Markov model of order m with all transition probabilities set to 0.
    
    For a Markov model of order m, we need to track all possible k-mers of length m,
    and for each k-mer, track the probability of seeing A, C, G, or T next.
    
    Args:
    - m: Order of Markov model
    
    Returns:
    - markov_model: Dictionary with k-mers as keys and [P(A), P(C), P(G), P(T)] as values
    - markov_counts: Dictionary with k-mers as keys and (count_A, count_C, count_G, count_T) as values
    """
    # Generate all possible k-mers of length m
    # For m=0, we have 1 k-mer (empty string "")
    # For m=1, we have 4 k-mers (A, C, G, T)
    # For m=2, we have 16 k-mers (AA, AC, AG, AT, CA, CC, ...)
    # For m=n, we have 4^n k-mers
    
    nucleotides = ['A', 'C', 'G', 'T']
    
    # Generate all k-mers recursively
    if m == 0:
        kmers = ['']  # For order 0, use empty string as key
    else:
        # Start with single nucleotides
        kmers = nucleotides.copy()
        # Extend to length m by adding nucleotides
        for _ in range(m - 1):
            new_kmers = []
            for kmer in kmers:
                for nuc in nucleotides:
                    new_kmers.append(kmer + nuc)
            kmers = new_kmers
    
    # Initialize markov_model with all probabilities set to 0
    # Each k-mer maps to [P(A), P(C), P(G), P(T)]
    markov_model = {kmer: [0.0, 0.0, 0.0, 0.0] for kmer in kmers}
    
    # Initialize markov_counts as a defaultdict so that only observed k-mers
    # are stored; unseen k-mers will be created lazily when first encountered.
    # Each k-mer maps to (count_A, count_C, count_G, count_T)
    markov_counts = defaultdict(lambda: [0, 0, 0, 0])
    
    return markov_model, markov_counts



# ===========================
# LOAD AND PARSE TSV DATA
# ===========================
# This section reads the chromosome data from the TSV file


def load_tsv_data(tsv_path, tf_name):
    """
    Load chromosome data from TSV file and extract relevant columns.
    
    The TSV file has columns: chr, start, end, ATAC, CTCF, REST, EP300
    We need to extract genomic coordinates and the binding status for the specified TF.
    
    Args:
    - tsv_path: Path to TSV file
    - tf_name: Name of transcription factor (CTCF, REST, or EP300)
    
    Returns:
    - DataFrame with columns: chr, start, end, label (B or U)
    """
    print(f"Loading data from {tsv_path}...")
    
    # Read TSV file using pandas
    # The file is tab-separated with a header row
    df = pd.read_csv(tsv_path, sep='\t')
    
    # Extract only the columns we need
    # We need: chr, start, end, and the label for the specified TF
    data = df[['chr', 'start', 'end', tf_name]].copy()
    
    # Rename the TF column to 'label' for easier access
    data.rename(columns={tf_name: 'label'}, inplace=True)
    
    print(f"Loaded {len(data)} genomic bins for {tf_name}")
    print(f"Bound (B) regions: {(data['label'] == 'B').sum()}")
    print(f"Unbound (U) regions: {(data['label'] == 'U').sum()}")
    
    return data



# ===========================
# EXTRACT DNA SEQUENCES FROM GENOME
# ===========================
# This section extracts DNA sequences for each genomic bin using pyfaidx


def extract_sequence(genome, chrom, start, end):
    """
    Extract DNA sequence from genome for a given genomic region.
    
    Uses pyfaidx for efficient sequence extraction from the FASTA file.
    
    Args:
    - genome: pyfaidx.Fasta object
    - chrom: Chromosome name (e.g., 'chr1')
    - start: Start position (0-based, inclusive)
    - end: End position (0-based, exclusive)
    
    Returns:
    - sequence: DNA sequence as uppercase string, or None if region contains 'N'
    """
    try:
        # Extract sequence using pyfaidx (converts to uppercase automatically)
        # Note: pyfaidx uses 1-based coordinates, but our data is 0-based
        # So we add 1 to start for pyfaidx
        sequence = str(genome[chrom][start:end]).upper()
        
        # Check if sequence contains 'N' (unmapped bases)
        # If it does, return None to skip this region
        if 'N' in sequence:
            return None
        
        return sequence
    
    except Exception as e:
        # If there's any error (e.g., invalid coordinates), return None
        print(f"Warning: Could not extract sequence for {chrom}:{start}-{end}: {e}")
        return None



# ===========================
# TRAIN MARKOV MODEL ON SEQUENCES
# ===========================
# This section trains the Markov model by counting k-mer transitions


def train_markov_model(sequences, m, markov_counts):
    """
    Train Markov model by counting k-mer transitions in the given sequences.
    
    For each sequence, we scan through it and count how many times each nucleotide
    follows each k-mer of length m.
    
    Args:
    - sequences: List of DNA sequences (strings)
    - m: Order of Markov model
    - markov_counts: Dictionary to update with counts (modified in-place)
    
    Returns:
    - Updated markov_counts dictionary
    """
    # Mapping from nucleotide to index in counts tuple
    # A -> index 0, C -> index 1, G -> index 2, T -> index 3
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Process each sequence
    for seq in sequences:
        # Skip sequences that are too short
        # We need at least m+1 bases to have prev_m_nucl and next_nucl
        if len(seq) < m + 1:
            continue
        
        # Scan through the sequence
        # We skip the first m nucleotides (as per instructions)
        # We go from position m to len(seq)-1 (so we always have a next nucleotide)
        for i in range(m, len(seq)):
            # Extract previous m nucleotides (the context)
            # For m=0, this is an empty string
            # For m=1, this is the current nucleotide
            # For m=2, this is the current and previous nucleotide, etc.
            if m == 0:
                prev_m_nucl = ''
            else:
                prev_m_nucl = seq[i-m:i]
            
            # Extract next nucleotide (the one we're predicting)
            next_nucl = seq[i]
            
            # Skip if we encounter any unexpected characters
            if next_nucl not in nuc_to_idx:
                continue
            if m > 0 and not all(n in nuc_to_idx for n in prev_m_nucl):
                continue
            
            # Increment the count for this transition
            # We increment the count at the index corresponding to next_nucl
            nuc_idx = nuc_to_idx[next_nucl]
            markov_counts[prev_m_nucl][nuc_idx] += 1
    
    return markov_counts



# ===========================
# COMPUTE TRANSITION PROBABILITIES
# ===========================
# This section converts counts to probabilities using Laplace smoothing


def compute_probabilities(markov_counts, pseudocount):
    """
    Convert k-mer counts to transition probabilities using Laplace smoothing.
    
    For each k-mer, we compute:
    P(nucleotide | k-mer) = (count + pseudocount) / (total_count + 4 * pseudocount)
    
    Args:
    - markov_counts: Dictionary with k-mers as keys and count lists as values
    - pseudocount: Pseudocount for Laplace smoothing (default: 0.01)
    
    Returns:
    - markov_model: Dictionary with k-mers as keys and probability lists as values
    """
    markov_model = {}
    
    # For each k-mer, compute transition probabilities
    for kmer, counts in markov_counts.items():
        # Calculate total count for this k-mer
        total = sum(counts)
        
        # Apply Laplace smoothing
        # Add pseudocount to each count and 4*pseudocount to total
        probabilities = [
            (counts[i] + pseudocount) / (total + 4 * pseudocount)
            for i in range(4)
        ]
        
        markov_model[kmer] = probabilities
    
    return markov_model



# ===========================
# CALCULATE LOG PROBABILITY OF SEQUENCE
# ===========================
# This section calculates the log probability of a sequence under a Markov model


def calculate_log_probability(sequence, markov_model, m, unseen_kmer_prob=0.01):
    """
    Calculate the log probability of a sequence under a given Markov model.
    
    We sum the log probabilities of each nucleotide given its context:
    log P(sequence) = Σ log P(nucleotide_i | prev_m_nucleotides)
    
    Args:
    - sequence: DNA sequence (string)
    - markov_model: Dictionary with k-mers as keys and probability lists as values
    - m: Order of Markov model
    - unseen_kmer_prob: Probability to use for unseen k-mers (default: 0.01)
    
    Returns:
    - log_prob: Log probability of the sequence
    """
    # Mapping from nucleotide to index
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Initialize log probability
    log_prob = 0.0
    
    # Skip sequences that are too short
    if len(sequence) < m + 1:
        return -np.inf  # Return very negative log probability
    
    # Scan through the sequence (skipping first m nucleotides)
    for i in range(m, len(sequence)):
        # Extract previous m nucleotides (context)
        if m == 0:
            prev_m_nucl = ''
        else:
            prev_m_nucl = sequence[i-m:i]
        
        # Extract current nucleotide
        next_nucl = sequence[i]
        
        # Skip if we encounter unexpected characters
        if next_nucl not in nuc_to_idx:
            continue
        if m > 0 and not all(n in nuc_to_idx for n in prev_m_nucl):
            continue
        
        # Get transition probability from model
        nuc_idx = nuc_to_idx[next_nucl]
        # Handle unseen k-mers gracefully with user-specified probability
        prob_list = markov_model.get(prev_m_nucl)
        if prob_list is not None:
            prob = prob_list[nuc_idx]
        else:
            prob = unseen_kmer_prob
        
        # Add log probability (handle case where prob is 0)
        if prob > 0:
            log_prob += np.log(prob)
        else:
            # This should not happen with Laplace smoothing, but just in case
            log_prob += np.log(1e-10)
    
    return log_prob



# ===========================
# PREDICT BINDING FOR TEST SET
# ===========================
# This section predicts binding labels for test sequences using log-odds scores


def predict_binding(test_data, genome, markov_model_B, markov_model_U, m, unseen_kmer_prob=0.01):
    """
    Predict binding labels for test sequences using log-odds scores.
    
    For each sequence:
    1. Calculate log P(sequence | model_B)
    2. Calculate log P(sequence | model_U)
    3. Compute log-odds score = log P(seq | B) - log P(seq | U)
    4. Predict B if score > 0, else predict U
    
    Args:
    - test_data: DataFrame with columns chr, start, end, label
    - genome: pyfaidx.Fasta object
    - markov_model_B: Markov model trained on bound sequences
    - markov_model_U: Markov model trained on unbound sequences
    - m: Order of Markov model
    - unseen_kmer_prob: Probability to use for unseen k-mers (default: 0.01)
    
    Returns:
    - predictions: List of predicted labels ('B' or 'U')
    - scores: List of log-odds scores
    - true_labels: List of true labels ('B' or 'U')
    """
    predictions = []
    scores = []
    true_labels = []
    
    # Process each test sequence
    for idx, row in test_data.iterrows():
        chrom = row['chr']
        start = row['start']
        end = row['end']
        true_label = row['label']
        
        # Extract sequence
        sequence = extract_sequence(genome, chrom, start, end)
        
        # Skip if sequence contains 'N' or is invalid
        if sequence is None:
            continue
        
        # Calculate log probabilities under both models
        log_prob_B = calculate_log_probability(sequence, markov_model_B, m, unseen_kmer_prob)
        log_prob_U = calculate_log_probability(sequence, markov_model_U, m, unseen_kmer_prob)
        
        # Calculate log-odds score
        score = log_prob_B - log_prob_U
        
        # Predict label based on score
        # If score > 0, predict B (more likely under model_B)
        # If score <= 0, predict U (more likely under model_U)
        predicted_label = 'B' if score > 0 else 'U'
        
        # Store results
        predictions.append(predicted_label)
        scores.append(score)
        true_labels.append(true_label)
    
    return predictions, scores, true_labels



# ===========================
# CALCULATE EVALUATION METRICS
# ===========================
# This section calculates TP, TN, FP, FN, ROC, and PR metrics


def calculate_metrics(true_labels, predictions, scores):
    """
    Calculate evaluation metrics: TP, TN, FP, FN, ROC curve, PR curve.
    
    Args:
    - true_labels: List of true labels ('B' or 'U')
    - predictions: List of predicted labels ('B' or 'U')
    - scores: List of log-odds scores
    
    Returns:
    - metrics: Dictionary containing TP, TN, FP, FN, ROC data, PR data
    """
    # Convert labels to binary (1 for B, 0 for U)
    y_true = np.array([1 if label == 'B' else 0 for label in true_labels])
    y_pred = np.array([1 if label == 'B' else 0 for label in predictions])
    y_scores = np.array(scores)
    
    # Calculate confusion matrix values
    TP = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    TN = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
    FP = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    FN = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
    
    # Calculate ROC curve
    # ROC curve plots True Positive Rate (TPR) vs False Positive Rate (FPR)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)  # Area under ROC curve
    
    # Calculate Precision-Recall curve
    # PR curve plots Precision vs Recall
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)  # Area under PR curve
    
    # Store all metrics in a dictionary
    metrics = {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'pr_auc': pr_auc
    }
    
    return metrics



# ===========================
# SAVE RESULTS TO FILES
# ===========================
# This section saves models, predictions, and metrics to files


def save_model(markov_model, markov_counts, filepath_model, filepath_counts):
    """
    Save Markov model and counts to JSON files.
    
    Args:
    - markov_model: Dictionary with k-mers and probabilities
    - markov_counts: Dictionary with k-mers and counts
    - filepath_model: Path to save model JSON file
    - filepath_counts: Path to save counts JSON file
    """
    # Save model (probabilities)
    with open(filepath_model, 'w') as f:
        json.dump(markov_model, f, indent=2)
    
    # Save counts
    with open(filepath_counts, 'w') as f:
        json.dump(markov_counts, f, indent=2)



def save_predictions(test_data, predictions, scores, filepath):
    """
    Save predictions to TSV file.
    
    Args:
    - test_data: DataFrame with chr, start, end columns
    - predictions: List of predicted labels
    - scores: List of log-odds scores
    - filepath: Path to save predictions TSV file
    """
    # Create output DataFrame
    output_df = test_data[['chr', 'start', 'end']].copy()
    output_df['prediction'] = predictions
    output_df['score'] = scores
    
    # Save to TSV
    output_df.to_csv(filepath, sep='\t', index=False)



def save_metrics_csv(metrics, fold_num, output_dir, prefix):
    """
    Save ROC and PR curve data to CSV files.
    
    Args:
    - metrics: Dictionary with ROC and PR data
    - fold_num: Fold number (1 to k)
    - output_dir: Directory to save files
    - prefix: Prefix for filenames (includes TF, chr, m, k)
    """
    # Save ROC curve data
    roc_df = pd.DataFrame({
        'FPR': metrics['fpr'],
        'TPR': metrics['tpr']
    })
    roc_csv_path = os.path.join(output_dir, f"{prefix}_ROC_fold_{fold_num}.csv")
    roc_df.to_csv(roc_csv_path, index=False)
    
    # Save PR curve data
    pr_df = pd.DataFrame({
        'Recall': metrics['recall'],
        'Precision': metrics['precision']
    })
    pr_csv_path = os.path.join(output_dir, f"{prefix}_PR_fold_{fold_num}.csv")
    pr_df.to_csv(pr_csv_path, index=False)



def plot_curves(all_metrics, k, output_dir, prefix):
    """
    Plot ROC and PR curves for all folds and save as PNG files.
    
    Args:
    - all_metrics: List of metrics dictionaries (one per fold)
    - k: Number of folds
    - output_dir: Directory to save plots
    - prefix: Prefix for filenames
    """
    # Create ROC curve plot
    plt.figure(figsize=(10, 8))
    for i, metrics in enumerate(all_metrics):
        plt.plot(metrics['fpr'], metrics['tpr'], 
                label=f"Fold {i+1} (AUC = {metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {k}-Fold Cross-Validation', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_plot_path = os.path.join(output_dir, f"{prefix}_ROC_all_folds.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    
    # Create PR curve plot
    plt.figure(figsize=(10, 8))
    for i, metrics in enumerate(all_metrics):
        plt.plot(metrics['recall'], metrics['precision'],
                label=f"Fold {i+1} (AUC = {metrics['pr_auc']:.3f})")
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curves - {k}-Fold Cross-Validation', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_plot_path = os.path.join(output_dir, f"{prefix}_PR_all_folds.png")
    plt.savefig(pr_plot_path, dpi=300)
    plt.close()



# ===========================
# PROCESS SINGLE FOLD (FOR PARALLEL EXECUTION)
# ===========================
# This function processes a single fold and can be called in parallel


def process_single_fold(fold_num, train_idx, test_idx, data, m, pseudocount, 
                       unseen_kmer_prob, genome_path, tf_name, chr_name, 
                       markov_model_B_dir, markov_model_U_dir, 
                       predicted_values_dir, outputs_dir, results_prefix):
    """
    Process a single fold: train models, predict, and calculate metrics.
    This function is designed to be called in parallel.
    
    Args:
    - fold_num: Fold number (1 to k)
    - train_idx: Training indices
    - test_idx: Testing indices
    - data: Full dataset DataFrame
    - m: Markov model order
    - pseudocount: Laplace smoothing pseudocount
    - unseen_kmer_prob: Probability for unseen k-mers
    - genome_path: Path to genome FASTA file
    - tf_name: Transcription factor name
    - chr_name: Chromosome name
    - markov_model_B_dir: Directory to save B models
    - markov_model_U_dir: Directory to save U models
    - predicted_values_dir: Directory to save predictions
    - outputs_dir: Directory to save metrics
    - results_prefix: Prefix for output filenames
    
    Returns:
    - Dictionary with metrics and timing information
    """
    fold_start = time.time()
    
    print(f"\n{'-'*80}")
    print(f"FOLD {fold_num}")
    print(f"{'-'*80}")
    
    # Load genome (each worker loads its own copy)
    genome = Fasta(genome_path)
    
    # Split data into train and test sets
    train_data = data.iloc[train_idx].reset_index(drop=True)
    test_data = data.iloc[test_idx].reset_index(drop=True)
    
    print(f"  Training set size: {len(train_data)}")
    print(f"  Test set size: {len(test_data)}")
    
    # Initialize Markov models
    print(f"\n  Initializing Markov models (order m={m})...")
    markov_model_B, markov_counts_B = initialize_markov_model(m)
    markov_model_U, markov_counts_U = initialize_markov_model(m)
    print(f"    Models initialized with {len(markov_model_B)} k-mers")
    
    # Extract sequences and train models
    print(f"\n  Extracting sequences and training models...")
    
    train_data_B = train_data[train_data['label'] == 'B']
    train_data_U = train_data[train_data['label'] == 'U']
    
    print(f"    Bound (B) training sequences: {len(train_data_B)}")
    print(f"    Unbound (U) training sequences: {len(train_data_U)}")
    
    # Extract and train B model
    print(f"    Extracting sequences for model B...")
    sequences_B = []
    for idx, row in train_data_B.iterrows():
        seq = extract_sequence(genome, row['chr'], row['start'], row['end'])
        if seq is not None:
            sequences_B.append(seq)
    print(f"      Valid sequences (no N's): {len(sequences_B)}")
    
    print(f"    Training model B...")
    markov_counts_B = train_markov_model(sequences_B, m, markov_counts_B)
    markov_model_B = compute_probabilities(markov_counts_B, pseudocount)
    
    # Extract and train U model
    print(f"    Extracting sequences for model U...")
    sequences_U = []
    for idx, row in train_data_U.iterrows():
        seq = extract_sequence(genome, row['chr'], row['start'], row['end'])
        if seq is not None:
            sequences_U.append(seq)
    print(f"      Valid sequences (no N's): {len(sequences_U)}")
    
    print(f"    Training model U...")
    markov_counts_U = train_markov_model(sequences_U, m, markov_counts_U)
    markov_model_U = compute_probabilities(markov_counts_U, pseudocount)
    
    # Save trained models
    print(f"\n  Saving trained models...")
    model_B_path = os.path.join(markov_model_B_dir, f"markov_model_{m}_fold_{fold_num}.json")
    counts_B_path = os.path.join(markov_model_B_dir, f"markov_model_values_{m}_fold_{fold_num}.json")
    save_model(markov_model_B, markov_counts_B, model_B_path, counts_B_path)
    
    model_U_path = os.path.join(markov_model_U_dir, f"markov_model_{m}_fold_{fold_num}.json")
    counts_U_path = os.path.join(markov_model_U_dir, f"markov_model_values_{m}_fold_{fold_num}.json")
    save_model(markov_model_U, markov_counts_U, model_U_path, counts_U_path)
    print(f"    Models saved for fold {fold_num}")
    
    # Predict on test set
    print(f"\n  Predicting on test set...")
    predictions, scores, true_labels = predict_binding(
        test_data, genome, markov_model_B, markov_model_U, m, unseen_kmer_prob
    )
    print(f"    Predictions made for {len(predictions)} sequences")
    
    # Calculate evaluation metrics
    print(f"\n  Calculating evaluation metrics...")
    metrics = calculate_metrics(true_labels, predictions, scores)
    
    print(f"    True Positives (TP): {metrics['TP']}")
    print(f"    True Negatives (TN): {metrics['TN']}")
    print(f"    False Positives (FP): {metrics['FP']}")
    print(f"    False Negatives (FN): {metrics['FN']}")
    print(f"    ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"    PR AUC: {metrics['pr_auc']:.4f}")
    
    # Save predictions and metrics
    print(f"\n  Saving predictions and metrics...")
    pred_path = os.path.join(predicted_values_dir, 
                            f"{chr_name}_200bp_bins_predictions_fold_{fold_num}.tsv")
    save_predictions(test_data, predictions, scores, pred_path)
    
    save_metrics_csv(metrics, fold_num, outputs_dir, results_prefix)
    print(f"    Results saved for fold {fold_num}")
    
    fold_time = time.time() - fold_start
    print(f"    Fold {fold_num} completed in {fold_time:.2f} seconds")
    
    # Return metrics and timing
    return {
        'fold': fold_num,
        'metrics': metrics,
        'fold_time': fold_time,
        'confusion': {
            'TP': metrics['TP'],
            'TN': metrics['TN'],
            'FP': metrics['FP'],
            'FN': metrics['FN']
        }
    }



# ===========================
# MAIN FUNCTION
# ===========================
# This is the main function that orchestrates the entire workflow


def main():
    """
    Main function to run the Markov model pipeline.
    
    Workflow:
    1. Parse command-line arguments
    2. Load genome and data
    3. Perform k-fold cross-validation
    4. For each fold:
       a. Split data into train and test
       b. Train Markov models on train set
       c. Predict on test set
       d. Calculate metrics
    5. Save results and plot curves
    """


    # START TIMING
    start_time = time.time()
    start_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    
    # ----- STEP 1: PARSE COMMAND-LINE ARGUMENTS -----
    # Parse command-line arguments to get m, k, TSV file, TF name, and genome path
    print("="*80)
    print("MARKOV MODEL FOR TF BINDING SITE PREDICTION")
    print("="*80)
    print(f"Start time: {start_time_readable}\n")
    
    args = parse_arguments()
    
    m = args.m
    k = args.k
    tsv_path = args.tsv
    tf_name = args.tf
    genome_path = args.genome_path
    pseudocount = args.pseudocount
    unseen_kmer_prob = args.unseen_kmer_prob
    n_jobs = args.n_jobs
    
    # Extract chromosome number from TSV filename
    # Example: chr1_200bp_bins.tsv -> chr1
    tsv_filename = os.path.basename(tsv_path)
    chr_name = tsv_filename.split('_')[0]  # Extract 'chr1', 'chr2', etc.
    
    print(f"\nParameters:")
    print(f"  Markov model order (m): {m}")
    print(f"  Number of folds (k): {k}")
    print(f"  Transcription factor: {tf_name}")
    print(f"  Chromosome: {chr_name}")
    print(f"  TSV file: {tsv_path}")
    print(f"  Genome file: {genome_path}")
    print(f"  Pseudocount: {pseudocount}")
    print(f"  Unseen k-mer probability: {unseen_kmer_prob}")
    print(f"  Number of parallel jobs: {n_jobs}")
    
    # ----- STEP 2: CREATE DIRECTORY STRUCTURE -----
    # Create folders for storing models and results
    # Format: {TF}_{chr}_folder_name
    print(f"\n{'='*80}")
    print("CREATING DIRECTORY STRUCTURE")
    print(f"{'='*80}")
    
    prefix = f"{tf_name}_{chr_name}"
    
    # Create main directories
    markov_model_B_dir = f"{prefix}_markov_model_B"
    markov_model_U_dir = f"{prefix}_markov_model_U"
    scripts_dir = f"{prefix}_scripts"
    predicted_values_dir = f"{prefix}_predicted_values"
    outputs_dir = f"{prefix}_outputs"
    
    for directory in [markov_model_B_dir, markov_model_U_dir, scripts_dir, 
                      predicted_values_dir, outputs_dir]:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created directory: {directory}")
    
    # ----- STEP 3: LOAD GENOME AND DATA -----
    # Load the genome FASTA file using pyfaidx
    # Load the TSV data containing genomic bins and binding labels
    print(f"\n{'='*80}")
    print("LOADING GENOME AND DATA")
    print(f"{'='*80}")

    # Track individual step times
    genome_load_start = time.time()
    print(f"Loading genome from {genome_path}...")
    genome = Fasta(genome_path)
    genome_load_time = time.time() - genome_load_start
    print(f"  Genome loaded successfully ({genome_load_time:.2f} seconds)")
    
    # Load TSV data
    data = load_tsv_data(tsv_path, tf_name)
    
    # ----- STEP 4: K-FOLD CROSS-VALIDATION -----
    # Split data into k folds and perform cross-validation
    print(f"\n{'='*80}")
    print(f"PERFORMING {k}-FOLD CROSS-VALIDATION")
    print(f"{'='*80}")
    
    # Initialize KFold splitter
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)  # Shuffle for better randomness
    
    # Create results prefix
    results_prefix = f"{prefix}_m{m}_k{k}"
    
    # Prepare fold data for processing
    fold_data = []
    for fold_num, (train_idx, test_idx) in enumerate(kfold.split(data), start=1):
        fold_data.append((
            fold_num, train_idx, test_idx, data, m, pseudocount, 
            unseen_kmer_prob, genome_path, tf_name, chr_name,
            markov_model_B_dir, markov_model_U_dir, 
            predicted_values_dir, outputs_dir, results_prefix
        ))
    
    # Process folds (parallel or serial)
    if n_jobs > 1:
        print(f"\nProcessing {k} folds in parallel using {n_jobs} CPU cores...")
        with Pool(processes=n_jobs) as pool:
            fold_results = pool.starmap(process_single_fold, fold_data)
    else:
        print(f"\nProcessing {k} folds serially...")
        fold_results = [process_single_fold(*args) for args in fold_data]
    
    # Extract metrics from results
    all_metrics = [result['metrics'] for result in fold_results]
    all_confusion_matrices = [result['confusion'] for result in fold_results]
    fold_times = [result['fold_time'] for result in fold_results]
    
    # Update confusion matrices to include fold number
    for i, cm in enumerate(all_confusion_matrices):
        cm['fold'] = i + 1
    
    # ----- STEP 5: AGGREGATE RESULTS ACROSS FOLDS -----
    # Calculate average metrics and plot curves
    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS ACROSS ALL FOLDS")
    print(f"{'='*80}")
    
    # Calculate average AUROC and AUPRC
    avg_roc_auc = np.mean([m['roc_auc'] for m in all_metrics])
    avg_pr_auc = np.mean([m['pr_auc'] for m in all_metrics])
    
    print(f"\nAverage ROC AUC across {k} folds: {avg_roc_auc:.4f}")
    print(f"Average PR AUC across {k} folds: {avg_pr_auc:.4f}")
    
    # Print confusion matrices for all folds
    print(f"\nConfusion Matrices:")
    for cm in all_confusion_matrices:
        print(f"  Fold {cm['fold']}: TP={cm['TP']}, TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}")

    total_time = time.time() - start_time
    end_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    print(f"\nTotal execution time: {str(timedelta(seconds=int(total_time)))}")
    
    # ----- STEP 6: PLOT AND SAVE CURVES -----
    # Plot ROC and PR curves for all folds
    print(f"\nPlotting ROC and PR curves...")
    plot_curves(all_metrics, k, outputs_dir, results_prefix)
    print(f"  Plots saved in {outputs_dir}")
    
    # ----- STEP 7: SAVE SUMMARY STATISTICS -----
    # Save summary statistics to a text file
    print(f"\nSaving summary statistics...")
    summary_path = os.path.join(outputs_dir, f"{results_prefix}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MARKOV MODEL CROSS-VALIDATION SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write("Execution Time:\n")
        f.write(f"  Start time: {start_time_readable}\n")
        f.write(f"  End time: {end_time_readable}\n")
        f.write(f"  Total time: {str(timedelta(seconds=int(total_time)))} ")
        f.write(f"({total_time:.2f} seconds)\n")
        f.write(f"  Average time per fold: {np.mean(fold_times):.2f} seconds\n")
        f.write(f"  Genome loading time: {genome_load_time:.2f} seconds\n\n")

        f.write(f"Parameters:\n")
        f.write(f"  Transcription Factor: {tf_name}\n")
        f.write(f"  Chromosome: {chr_name}\n")
        f.write(f"  Markov Model Order (m): {m}\n")
        f.write(f"  Number of Folds (k): {k}\n")
        f.write(f"  Pseudocount: {pseudocount}\n")
        f.write(f"  Unseen k-mer probability: {unseen_kmer_prob}\n")
        f.write(f"  Number of parallel jobs: {n_jobs}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Average ROC AUC: {avg_roc_auc:.4f}\n")
        f.write(f"  Average PR AUC: {avg_pr_auc:.4f}\n\n")
        f.write(f"Per-Fold Results:\n")
        for i, metrics in enumerate(all_metrics, start=1):
            f.write(f"  Fold {i}:\n")
            f.write(f"    Execution time: {fold_times[i-1]:.2f} seconds\n")
            f.write(f"    ROC AUC: {metrics['roc_auc']:.4f}\n")
            f.write(f"    PR AUC: {metrics['pr_auc']:.4f}\n")
            f.write(f"    TP: {metrics['TP']}, TN: {metrics['TN']}, ")
            f.write(f"FP: {metrics['FP']}, FN: {metrics['FN']}\n")
    print(f"  Summary saved to {summary_path}")
    
    # ----- COMPLETION -----
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"\nTotal execution time: {str(timedelta(seconds=int(total_time)))}")
    print(f"\nAll results saved in:")
    print(f"  Models: {markov_model_B_dir}, {markov_model_U_dir}")
    print(f"  Predictions: {predicted_values_dir}")
    print(f"  Outputs: {outputs_dir}")
    print()



# ===========================
# ENTRY POINT
# ===========================
# This ensures main() runs only when the script is executed directly


if __name__ == "__main__":
    main()


# ------ End of Script ------

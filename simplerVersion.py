#!/usr/bin/env python3
"""
simplerVersion.py

Takes 2 arguments: FASTA file and Markov model order m
Trains a Markov model on all sequences and prints log likelihood scores

To run the code:
python simplerVersion.py <fasta_file> <m>

"""

import sys
import numpy as np


def read_fasta(fasta_file):
    """Read sequences from FASTA file"""
    sequences = []
    current_seq = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_seq:
                    sequences.append(''.join(current_seq).upper())
                    current_seq = []
            else:
                current_seq.append(line)
        
        if current_seq:
            sequences.append(''.join(current_seq).upper())
    
    return sequences


def initialize_markov_model(m):
    """Initialize Markov model structures"""
    nucleotides = ['A', 'C', 'G', 'T']
    
    if m == 0:
        kmers = ['']
    else:
        kmers = nucleotides.copy()
        for _ in range(m - 1):
            new_kmers = []
            for kmer in kmers:
                for nuc in nucleotides:
                    new_kmers.append(kmer + nuc)
            kmers = new_kmers
    
    markov_model = {kmer: [0.0, 0.0, 0.0, 0.0] for kmer in kmers}
    markov_counts = {kmer: [0, 0, 0, 0] for kmer in kmers}
    
    return markov_model, markov_counts


def train_markov_model(sequences, m, markov_counts):
    """Train Markov model by counting k-mer transitions"""
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    for seq in sequences:
        if len(seq) < m + 1:
            continue
        
        for i in range(m, len(seq)):
            if m == 0:
                prev_m_nucl = ''
            else:
                prev_m_nucl = seq[i-m:i]
            
            next_nucl = seq[i]
            
            if next_nucl not in nuc_to_idx:
                continue
            if m > 0 and not all(n in nuc_to_idx for n in prev_m_nucl):
                continue
            
            nuc_idx = nuc_to_idx[next_nucl]
            markov_counts[prev_m_nucl][nuc_idx] += 1
    
    return markov_counts


def compute_probabilities(markov_counts, pseudocount=0.01):
    """Convert counts to probabilities with Laplace smoothing"""
    markov_model = {}
    
    for kmer, counts in markov_counts.items():
        total = sum(counts)
        probabilities = [
            (counts[i] + pseudocount) / (total + 4 * pseudocount)
            for i in range(4)
        ]
        markov_model[kmer] = probabilities
    
    return markov_model


def calculate_log_probability(sequence, markov_model, m):
    """Calculate log probability of sequence under model"""
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    log_prob = 0.0
    
    if len(sequence) < m + 1:
        return -np.inf
    
    for i in range(m, len(sequence)):
        if m == 0:
            prev_m_nucl = ''
        else:
            prev_m_nucl = sequence[i-m:i]
        
        next_nucl = sequence[i]
        
        if next_nucl not in nuc_to_idx:
            continue
        if m > 0 and not all(n in nuc_to_idx for n in prev_m_nucl):
            continue
        
        nuc_idx = nuc_to_idx[next_nucl]
        prob = markov_model[prev_m_nucl][nuc_idx]
        
        if prob > 0:
            log_prob += np.log(prob)
        else:
            log_prob += np.log(1e-10)
    
    return log_prob


def main():
    """Main function"""
    if len(sys.argv) != 3:
        print("Usage: python simplerVersion.py <fasta_file> <m>", file=sys.stderr)
        sys.exit(1)
    
    fasta_file = sys.argv[1]
    m = int(sys.argv[2])
    
    # Read sequences
    sequences = read_fasta(fasta_file)
    
    # Initialize and train model
    markov_model, markov_counts = initialize_markov_model(m)
    markov_counts = train_markov_model(sequences, m, markov_counts)
    markov_model = compute_probabilities(markov_counts)
    
    # Print log likelihood for each sequence
    for seq in sequences:
        log_likelihood = calculate_log_probability(seq, markov_model, m)
        print(log_likelihood)


if __name__ == "__main__":
    main()

import json
import numpy as np
import matplotlib.pyplot as plt

def load_percentile_ranks(filename, exclude_nan=True):
    percentile_ranks = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                pr = data.get('percentile_rank', None)
                if pr is not None:
                    if exclude_nan:
                        if not np.isnan(pr):
                            percentile_ranks.append(pr)
                    else:
                        percentile_ranks.append(pr)
            except json.JSONDecodeError:
                continue
    return np.array(percentile_ranks)

def plot_percentile_distributions(filenames, labels, exclude_nan=True):
    plt.figure(figsize=(8,6))
    
    for filename, label in zip(filenames, labels):
        pranks = load_percentile_ranks(filename, exclude_nan=exclude_nan)
        if len(pranks) == 0:
            print(f"No valid percentile ranks found in {filename}. Skipping plot for this dataset.")
            continue
        plt.hist(pranks, bins=20, range=(0,1), alpha=0.5, density=True, label=label)

    plt.xlabel('Percentile Rank of Best-fit $E_G$')
    plt.ylabel('Normalized Frequency')
    plt.title('Distribution of Percentile Ranks for Posterior Predictive $E_G$ Fits')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../plots/percentile_ranks_fr-5_cmbprior_300runs.pdf')

if __name__ == "__main__":
    # Example filenames - change to your actual output files for Y1 and Y10
    files = [
        "../txtfiles/post_pred_test_fR-5_CMBPrior_LSSTY1_gc_seed_300runs_percrank.jsonl",
        "../txtfiles/post_pred_test_fR-5_CMBPrior_LSSTY10_gc_seed_300runs_percrank.jsonl"
    ]
    labels = ['Y1', 'Y10']

    plot_percentile_distributions(files, labels, exclude_nan=True)

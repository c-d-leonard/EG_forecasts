import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_results(filename, year_label):
    """Load results from JSONL and tag with year label."""
    rows = []
    with open(filename, "r") as f:
        for line in f:
            data = json.loads(line)
            # Skip entries with missing fit values
            if data.get("OmegaM_fit_mean") is not None and not np.isnan(data["OmegaM_fit_mean"]):
                rows.append({
                    "OmegaM_fit_mean": data["OmegaM_fit_mean"],
                    "OmegaM_true": data.get("OmegaM_true"),
                    "outside_95": data["outside_95"],
                    "const_bad_fit": data["const_bad_fit"],
                    "run": data["run"],
                    "year": year_label
                })
    return pd.DataFrame(rows)

def plot_rejection_vs_fit(df_y1, df_y10, prior_mean=None):
    """Scatter plot of Ωm_fit_mean vs Ωm_true, coloured by rejection."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, (df, label) in zip(axes, [(df_y1, "Y1"), (df_y10, "Y10")]):

        accepted_mask = df['outside_95'] == False
        rejected_mask = df['outside_95'] == True

        ax.scatter(
            df.loc[accepted_mask, 'OmegaM_fit_mean'],
            df.loc[accepted_mask, 'OmegaM_true'],
            color='blue', alpha=0.5, label='Accepted'
        )
        ax.scatter(
            df.loc[rejected_mask, 'OmegaM_fit_mean'],
            df.loc[rejected_mask, 'OmegaM_true'],
            color='red', alpha=0.7, label='Rejected'
        )
        if prior_mean is not None:
            ax.axvline(prior_mean, color='black', linestyle='--', label='Prior mean')
        ax.set_title(label)
        ax.set_xlabel(r"$\Omega_m$ fit (GR)")
        ax.set_ylabel(r"$\Omega_m$ true")
        ax.legend()

    plt.tight_layout()
    plt.savefig('../plots/rejection_vs_OmMfit_debug_fR-5_CMBprior_300runs.pdf')

# Example usage:
df_y1 = load_results("../txtfiles/post_pred_test_fR-5_CMBPrior_LSSTY1_gc_seed_300runs.jsonl", "Y1")
df_y10 = load_results("../txtfiles/post_pred_test_fR-5_CMBPrior_LSSTY10_gc_seed_300runs.jsonl", "Y10")

plot_rejection_vs_fit(df_y1, df_y10, prior_mean=0.292)  # change prior_mean as needed

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ===== User settings =====
file_y1 = "../txtfiles/post_pred_test_fR-5_CMBPrior_LSSTY1_gc_seed_300runs.jsonl"
file_y10 = "../txtfiles/post_pred_test_fR-5_CMBPrior_LSSTY10_gc_seed_300runs.jsonl"
plot_dir = "../plots/"

# Make sure plot directory exists
os.makedirs(plot_dir, exist_ok=True)

# ===== Load data =====
y1 = pd.read_json(file_y1, lines=True)
y10 = pd.read_json(file_y10, lines=True)

# Filter out bad fits explicitly
mask_good_y1 = y1['const_bad_fit'] == 0
mask_good_y10 = y10['const_bad_fit'] == 0

y1_good = y1[mask_good_y1].copy()
y10_good = y10[mask_good_y10].copy()

# Separate fit values
y1_fit_all = y1_good['OmegaM_fit_mean']
y10_fit_all = y10_good['OmegaM_fit_mean']

# Masks for rejections
mask_rej_y1 = y1_good['outside_95'] == 1
mask_rej_y10 = y10_good['outside_95'] == 1

y1_fit_rej = y1_good.loc[mask_rej_y1, 'OmegaM_fit_mean']
y10_fit_rej = y10_good.loc[mask_rej_y10, 'OmegaM_fit_mean']

# Find rejection zone
rej_min = min(y1_fit_rej.min(), y10_fit_rej.min())
rej_max = max(y1_fit_rej.max(), y10_fit_rej.max())

# ===== Plot histogram comparison =====
plt.figure(figsize=(8,5))
bins = np.linspace(min(y1_fit_all.min(), y10_fit_all.min()),
                   max(y1_fit_all.max(), y10_fit_all.max()), 30)

plt.hist(y1_fit_all, bins=bins, alpha=0.5, label='Y1 all', color='tab:blue', density=True)
#plt.hist(y1_fit_rej, bins=bins, alpha=0.7, label='Y1 rejected', color='navy', density=True)

plt.hist(y10_fit_all, bins=bins, alpha=0.5, label='Y10 all', color='tab:orange', density=True)
#plt.hist(y10_fit_rej, bins=bins, alpha=0.7, label='Y10 rejected', color='darkred', density=True)

#plt.axvspan(rej_min, rej_max, color='red', alpha=0.1, label='Rejection zone')

plt.xlabel(r'$\Omega_m$ fit (GR)')
plt.ylabel('Density')
plt.legend()
plt.title('OmegaM_fit distributions and rejection zone')
plt.tight_layout()

plt.savefig(os.path.join(plot_dir, "OmegaM_fit_histogram_with_rejections_300runs.pdf"), dpi=300)
plt.close()

# ===== Plot rejection fraction vs Ωₘ_fit =====
def rejection_fraction(df, nbins=10):
    bins = np.linspace(df['OmegaM_fit_mean'].min(), df['OmegaM_fit_mean'].max(), nbins+1)
    centers = 0.5*(bins[:-1] + bins[1:])
    fracs = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask_bin = (df['OmegaM_fit_mean'] >= lo) & (df['OmegaM_fit_mean'] < hi)
        bin_data = df[mask_bin]
        if len(bin_data) > 0:
            frac_rej = bin_data['outside_95'].mean()
        else:
            frac_rej = np.nan
        fracs.append(frac_rej)
    return centers, fracs

centers_y1, fracs_y1 = rejection_fraction(y1_good, nbins=12)
centers_y10, fracs_y10 = rejection_fraction(y10_good, nbins=12)

plt.figure(figsize=(8,5))
plt.plot(centers_y1, fracs_y1, '-o', label='Y1', color='tab:blue')
plt.plot(centers_y10, fracs_y10, '-o', label='Y10', color='tab:orange')
plt.xlabel(r'$\Omega_m$ fit (GR)')
plt.ylabel('Rejection fraction')
plt.title('Rejection fraction vs OmegaM_fit')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(plot_dir, "Rejection_fraction_vs_OmegaM_fit_300runs.pdf"), dpi=300)
plt.close()

# ===== Summary stats =====
def summary_stats(name, df):
    bias = (df['OmegaM_fit_mean'] - df['OmegaM_true']).mean()
    std = df['OmegaM_fit_mean'].std()
    mean_val = df['OmegaM_fit_mean'].mean()
    print(f"{name}: mean Ωₘ_fit = {mean_val:.4f}, std = {std:.4f}, bias = {bias:.4f}")

summary_stats("Y1", y1_good)
summary_stats("Y10", y10_good)

print(f"Plots saved in {os.path.abspath(plot_dir)}")


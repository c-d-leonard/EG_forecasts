import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# File locations
# -----------------------------
file1 = '../txtfiles/EG_data_realisation_fR0-4_DESY3prior_LSSTY10.dat'
file2 = '../txtfiles/EG_fit_data_realisation_fR0-4_DESY3prior_LSSTY10.dat'
file3 = '../txtfiles/OmMlikelihood_fR0-4_DESY3prior_LSSTY10.dat'
file4 = '../txtfiles/EG_replicated_fR0-4_DESY3prior_LSSTY10.dat'

# -----------------------------
# Load data
# -----------------------------
# file1: rp, EG_draw, EG_err
rp, EG_draw, EG_err = np.loadtxt(file1, unpack=True)

# file2: EG_fit_GR (single number)
EG_fit_GR = np.loadtxt(file2)

# file3: OmM, likeOmM
OmM, likeOmM = np.loadtxt(file3, unpack=True)

# file4: EG_rep samples
EG_rep = np.loadtxt(file4)

# -----------------------------
# Colours (colour-blind friendly)
# -----------------------------
col_data = plt.cm.tab10(0)   # blue
col_fit = plt.cm.tab10(1)    # orange

# -----------------------------
# Font sizes
# -----------------------------
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 18
})

# -----------------------------
# Create figure with 3 panels
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# ---- Panel 1: EG_draw vs rp ----
axes[0].errorbar(
    rp, EG_draw, yerr=EG_err,
    fmt='o', color=col_data, label='Data realisation'
)
axes[0].axhline(
    EG_fit_GR, color=col_fit, linestyle='--', label=r'Best fit $E_G$'
)
axes[0].set_xscale('log')  # log scale for rp
axes[0].set_xlabel(r"$r_p$ (Mpc$/h$)")
axes[0].set_ylabel(r"$E_G$")
axes[0].legend()

# ---- Panel 2: Posterior of OmM (histogram style) ----
# Normalise likeOmM to integrate to 1 (PDF)
likeOmM_norm = likeOmM / np.trapz(likeOmM, OmM)

# Sample proportional to PDF
n_samples = 100000
samples = np.random.choice(OmM, size=n_samples, p=likeOmM_norm/likeOmM_norm.sum())

# Restrict to [0.24, 0.29]
samples = samples[(samples >= 0.2) & (samples <= 0.29)]

# Match number of bins to Panel 3 (30 bins)
n_bins = 30
axes[1].hist(
    samples, bins=n_bins, density=True,
    range=(0.22, 0.28),
    color=col_data, alpha=0.7, label='Posterior under GR model'
)

axes[1].set_xlim(0.22, 0.28)
axes[1].set_xlabel(r"$\Omega_{\rm M}^{0,{\rm fit}}$")
axes[1].set_ylabel("Density")
axes[1].legend(loc='upper left')

# extend y axis slightly for visual padding
ymin, ymax = axes[1].get_ylim()
axes[1].set_ylim(ymin, ymax * 1.15)

# ---- Panel 3: Histogram of EG_rep ----
axes[2].hist(
    EG_rep, bins=30, density=True,
    color=col_data, alpha=0.7, label=r'Replicated $E_G$'
)
axes[2].axvline(
    EG_fit_GR, color=col_fit, linestyle='--', label=r'Best fit $E_G$'
)

# Add 95% CI (two-sided)
low, high = np.percentile(EG_rep, [2.5, 97.5])
axes[2].axvspan(
    low, high, color=col_data, alpha=0.2, label="95%"
)

axes[2].set_xlabel(r"$E_G$")
axes[2].set_ylabel("Density")
axes[2].legend()

# -----------------------------
# Title and save
# -----------------------------
fig.suptitle(r"Example realisation: $f_{R0}= 10^{-4}$, Stage III $3\times2$pt $\Omega_{\rm M}^0$ prior, LSST Y10 sources. GR accepted.")

plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for title
plt.savefig("eg_3panel_fr-4_LSSTY10_DESY3.pdf")  # PDF output
plt.show()

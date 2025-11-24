import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# -----------------------------
# File locations (DESY3prior)
# -----------------------------
file1 = '../txtfiles/EG_data_realisation_fR0-4_DESY3prior_LSSTY10.dat'
file2 = '../txtfiles/EG_fit_data_realisation_fR0-4_DESY3prior_LSSTY10.dat'
file3 = '../txtfiles/OmMlikelihood_fR0-4_DESY3prior_LSSTY10.dat'
file4 = '../txtfiles/EG_replicated_fR0-4_DESY3prior_LSSTY10.dat'

# -----------------------------
# Load data
# -----------------------------
rp, EG_draw, EG_err = np.loadtxt(file1, unpack=True)
EG_fit_GR = np.loadtxt(file2)
OmM, likeOmM = np.loadtxt(file3, unpack=True)
EG_rep = np.loadtxt(file4)

# -----------------------------
# Colours (colour-blind friendly)
# -----------------------------
col_data = plt.cm.tab10(0)   # blue
col_fit = plt.cm.tab10(1)    # orange (used for prior and fit lines)

# -----------------------------
# Font settings (serif, 50% larger)
# -----------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 21,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.titlesize": 27
})

# -----------------------------
# Create figure with 3 panels
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# ---- Panel 1: EG_draw vs rp ----
axes[0].errorbar(
    rp, EG_draw, yerr=EG_err,
    fmt='o', color=col_data, label='Data realisation'
)
axes[0].axhline(
    EG_fit_GR, color=col_fit, linestyle='--', label=r'Best fit $E_G$'
)
axes[0].set_xscale('log')
axes[0].set_xlabel(r"$r_p$ (Mpc$/h$)")
axes[0].set_ylabel(r"$E_G$")
axes[0].legend()
ymin1, _ = axes[0].get_ylim()
axes[0].set_ylim(0.26, 0.36)  # updated y max

# ---- Panel 2: Posterior of OmM (histogram) ----
likeOmM_norm = likeOmM / np.trapz(likeOmM, OmM)
n_samples = 100000
samples = np.random.choice(OmM, size=n_samples, p=likeOmM_norm/likeOmM_norm.sum())
samples = samples[samples <= 0.29]

n_bins = 30
axes[1].hist(
    samples, bins=n_bins, density=True,
    range=(0.18, 0.29),
    color=col_data, alpha=0.7, label='Posterior under GR model'
)

# Gaussian prior
mean_prior = 0.292
std_prior = 0.03
x_prior = np.linspace(0.15, 0.5, 500)  # updated x max to avoid legend overlap
y_prior = norm.pdf(x_prior, loc=mean_prior, scale=std_prior)
axes[1].plot(x_prior, y_prior, color=col_fit, linestyle='-', linewidth=2, label='Prior')

axes[1].set_xlim(0.15, 0.4)
axes[1].set_xlabel(r"$\Omega_{\rm M}^{0,{\rm fit}}$")
axes[1].set_ylabel("Density")
axes[1].legend(loc='upper left')

# Reduce y-axis max to 80
axes[1].set_ylim(0, 40)

# ---- Panel 3: Histogram of EG_rep ----
axes[2].hist(
    EG_rep, bins=30, density=True,
    color=col_data, alpha=0.7, label=r'Replicated $E_G$'
)
axes[2].axvline(
    EG_fit_GR, color=col_fit, linestyle='--', label=r'Best fit $E_G$'
)

low, high = np.percentile(EG_rep, [2.5, 97.5])
axes[2].axvspan(
    low, high, color=col_data, alpha=0.2, label="95%"
)

axes[2].set_xlabel(r"$E_G$")
axes[2].set_ylabel("Density")
ymin3, ymax3 = axes[2].get_ylim()
axes[2].set_ylim(ymin3, 55)
axes[2].set_xlim(0.25,0.375)
axes[2].legend()

# -----------------------------
# Title and save
# -----------------------------
fig.suptitle(
    r"Example realisation: $f(R)$ gravity, $f_{R0}= 10^{-4}$, Stage III 3$\times$2pt $\Omega_{\rm M}^0$ prior, LSST Y10 sources. GR accepted.",
    fontsize=27
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("../plots/eg_3panel_fr-4_LSSTY10_DESY3prior_simscov.pdf")
plt.show()

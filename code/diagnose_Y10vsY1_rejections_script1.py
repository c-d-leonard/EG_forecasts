import json, math
import numpy as np
import matplotlib.pyplot as plt

def quick_diag(filename):
    OmM_true = []
    OmM_fit = []
    outside = []
    const_bad = []
    runs = []

    with open(filename) as f:
        for line in f:
            d = json.loads(line)
            if 'error' in d:
                continue
            runs.append(d.get('run', None))
            OmM_true.append(d['OmegaM_true'])
            OmM_fit.append(d['OmegaM_fit_mean'] if not math.isnan(d['OmegaM_fit_mean']) else np.nan)
            outside.append(d['outside_95'])
            const_bad.append(d['const_bad_fit'])

    OmM_true = np.array(OmM_true)
    OmM_fit = np.array(OmM_fit)
    outside = np.array(outside)
    const_bad = np.array(const_bad)

    # scatter with rejection colour
    plt.figure(figsize=(6,6))
    mask_nan = ~np.isnan(OmM_fit)
    plt.scatter(OmM_true[mask_nan], OmM_fit[mask_nan], c=outside[mask_nan], cmap='coolwarm', s=12, alpha=0.7)
    plt.plot([OmM_true.min(),OmM_true.max()],[OmM_true.min(),OmM_true.max()],'k--')
    plt.xlabel('OmM_true'); plt.ylabel('OmM_fit_mean')
    plt.title('Colour by outside_95 (1=reject)')
    plt.colorbar(label='outside_95')
    plt.savefig('../plots/diagnostic_scatter_LSSTY1_CMB_colour_rejections_fR-5.pdf')

    # histograms of delta for rejected vs non-rejected
    delta = OmM_fit - OmM_true
    plt.hist(delta[mask_nan & (outside==1)], bins=40, alpha=0.6, label='outside_95')
    plt.hist(delta[mask_nan & (outside==0)], bins=40, alpha=0.6, label='non-reject')
    plt.legend(); plt.xlabel('OmM_fit - OmM_true')
    plt.title('Delta OmM')
    plt.savefig('../plots/diagnostic_histogram_LSSTY1_CMB_colour_rejections_fR-5.pdf')

    # counts:
    print("Total runs:", len(runs))
    print("outside_95:", outside.sum(), "const_bad_fit:", const_bad.sum())

# usage
quick_diag("../txtfiles/post_pred_test_fR-5_CMBPrior_LSSTY1_gc_seed_100runs.jsonl")

"""Figure 7.4 (fig:real-mock): detection curves + mock posteriors."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/mnt/agents/output/code/estimation")
from model import g_closed  # noqa: E402

BASE = "/mnt/agents/output/"
RES = BASE + "realdata/results/"
OUT = BASE + "paper_draft/figures/fig74_mock_validation.png"

COLS = ["#4c72b0", "#55a868", "#c44e52"]
C_P, C_A = "#1b9e77", "#d95f02"

fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.3))

# ================= (a) detection curves =================
# ---- IBDMDB representatives
df = pd.read_csv(RES + "identifiability_ibdmdb.csv")
z = np.load(RES.replace("results/", "data/") + "ibdmdb_genus.npz",
            allow_pickle=True)
Y, dep, taxa = z["Y"], z["depths"].astype(float), z["taxa"].astype(str)
phi_ib = pd.read_csv(RES + "fit_ibdmdb_summary.csv").iloc[0]["phi_hat"]
ident = df[df["zone"] == "identifiable"].sort_values("prevalence")
reps = [ident.iloc[-1], ident.iloc[max(len(ident) // 10, 0)]]
ridge = df[df["zone"] == "ridge"].sort_values("e_j")
reps.append(ridge.iloc[len(ridge) // 2])

ax = axes[0, 0]
Ngrid = np.logspace(np.log10(dep.min()), np.log10(dep.max()), 200)
edges = np.unique(np.quantile(np.log10(dep), np.linspace(0, 1, 9)))
binc = pd.cut(np.log10(dep), bins=edges)
for c, r in zip(COLS, reps):
    j = int(np.where(taxa == r["taxon"])[0][0])
    det = (Y[:, j] > 0).astype(float)
    emp = pd.Series(det).groupby(binc, observed=True).mean()
    nmid = np.array([np.median(dep[binc == cc]) for cc in emp.index])
    ax.scatter(nmid, emp.to_numpy(), s=26, color=c, zorder=3)
    ax.plot(Ngrid, r["pi_hat"] * (1 - g_closed(Ngrid, r["theta_hat"], phi_ib)),
            color=c, lw=1.5, label=f"{r['taxon']} ({r['zone']})")
ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("sequencing depth $N$ (log)", fontsize=8.5)
ax.set_ylabel("detection rate", fontsize=8.5)
ax.set_title("IBDMDB biopsy ($n$=178)", fontsize=8.5)
ax.legend(fontsize=6.6, loc="center right")
ax.tick_params(labelsize=7.5)

# ---- MBQC fecal mock members
cu = np.load(RES + "mbqc_mock_depthcurve.npz", allow_pickle=True)
ax = axes[0, 1]
for k, (t, c) in enumerate(zip([str(t) for t in cu["fecal_rep"]], COLS)):
    ax.scatter(cu["fecal_centers"], cu["fecal_emp"][:, k], s=26, color=c,
               zorder=3)
    ax.plot(cu["fecal_centers"], cu["fecal_theo"][:, k], color=c, lw=1.5,
            label=f"{t} (Table 1)")
ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("sequencing depth $N$ (log)", fontsize=8.5)
ax.set_title("MBQC fecal mock ($n$=1,158), known members", fontsize=8.5)
ax.legend(fontsize=6.6, loc="center right")
ax.tick_params(labelsize=7.5)

# ================= (b) genus-level structural-absence posterior =================
zz = np.load(RES.replace("results/", "data/") + "mbqc_mockblank_genus.npz",
             allow_pickle=True)
Ym_, taxa_m, samples = (zz["Y"].astype(np.int64), zz["taxa"].astype(str),
                        zz["samples"].astype(str))
meta = pd.read_csv(BASE + "datasets/mbqc/mbqc_sample_metadata.csv",
                   usecols=["Unnamed: 0", "specimen_type"])
stype = meta.set_index("Unnamed: 0")["specimen_type"].loc[samples].to_numpy()
depths = Ym_.sum(axis=1)
auc_tab = pd.read_csv(RES + "mbqc_mock_auc.csv").set_index("mock")

ST = {"fecal": "Fecal artificial colony", "oral": "Oral artificial colony"}
rng = np.random.default_rng(7)
for col, tag in enumerate(["fecal", "oral"]):
    m = (stype == ST[tag]) & (depths > 0)
    Ym, Nm = Ym_[m], depths[m].astype(float)
    D = (Ym > 0).astype(float)
    fit = pd.read_csv(RES + f"mbqc_mock_fit_{tag}.csv").set_index("taxon")
    fit = fit.loc[taxa_m]
    phi = float(auc_tab.loc[tag, "phi_hat"])
    pi = np.nan_to_num(fit["pi_hat"].to_numpy(), nan=0.0)
    theta = np.nan_to_num(fit["theta_hat"].to_numpy(), nan=1e-9)
    g = g_closed(Nm[:, None], theta[None, :], phi)
    num = 1.0 - pi[None, :]
    post = np.where(D == 0, num / (num + pi[None, :] * g), np.nan)
    with np.errstate(invalid="ignore"):
        mpost = np.nanmean(post, axis=0)
    mpost = np.where(np.isnan(mpost), 0.0, mpost)
    pres = fit["known_present"].to_numpy()
    ax = axes[1, col]
    for x, msk, ccol, lab in ((0, pres, C_P, "Table 1 present"),
                              (1, ~pres, C_A, "absent")):
        ax.scatter(rng.normal(x, 0.055, int(msk.sum())), mpost[msk], s=10,
                   alpha=0.5, color=ccol, edgecolor="none", label=lab,
                   rasterized=True)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"present\n(n={int(pres.sum())})",
                        f"absent\n(n={int((~pres).sum())})"], fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    a = auc_tab.loc[tag]
    ax.set_title(f"{tag} mock — genus mean structural-absence posterior\n"
                 f"AUC = {a['auc_genus_mean_posterior']:.3f}", fontsize=8.5)
    ax.legend(fontsize=7, loc="center left", markerscale=1.6)
    ax.tick_params(labelsize=7.5)
    if col == 0:
        ax.set_ylabel(r"mean $P(Z_{ij}=0\mid Y_{ij}=0,N_i)$", fontsize=8.5)
    # sanity: recompute AUC
    from scipy.stats import rankdata
    sc = 1.0 - mpost
    rk = rankdata(sc[pres | ~pres])
    auc = (rk[pres].sum() - pres.sum() * (pres.sum() + 1) / 2) / (
        pres.sum() * (~pres).sum())
    print(tag, "recomputed genus AUC:", round(auc, 3),
          "table:", round(a["auc_genus_mean_posterior"], 3))

# panel tags
fig.text(0.015, 0.975, "(a)", fontsize=11, fontweight="bold", va="top")
fig.text(0.015, 0.495, "(b)", fontsize=11, fontweight="bold", va="top")
fig.tight_layout(h_pad=2.0, w_pad=2.0, rect=(0.03, 0.01, 1, 0.99))
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved", OUT)

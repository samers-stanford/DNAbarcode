"""
Deep-dive: abundance comparison across runs, spectral plots for top candidates.
Focuses on peaks enriched in enzyme runs (76, 79) vs. no-enzyme control (82).
"""

import base64, struct, xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

RESULTS = "/home/user/DNAbarcode/results"
DATA_DIR = "/home/user/DNAbarcode/mzData_files"
os.makedirs(RESULTS, exist_ok=True)

# ── same parsing as gcms_analysis.py ──────────────────────────────────────────

def decode_array(b64str, precision_bits, endian):
    raw = base64.b64decode(b64str.strip())
    fmt_char = "d" if precision_bits == 64 else "f"
    byte_size = precision_bits // 8
    n = len(raw) // byte_size
    end_prefix = "<" if endian == "little" else ">"
    return np.array(struct.unpack(f"{end_prefix}{n}{fmt_char}", raw), dtype=np.float64)

def parse_mzdata(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    spectra = []
    for spec_el in root.find("spectrumList").findall("spectrum"):
        rt = None
        for cvp in spec_el.iter("cvParam"):
            if cvp.get("name") == "TimeInMinutes":
                rt = float(cvp.get("value")); break
        if rt is None: continue
        mz_el  = spec_el.find(".//mzArrayBinary/data")
        int_el = spec_el.find(".//intenArrayBinary/data")
        if mz_el is None or int_el is None: continue
        mz_arr = decode_array(mz_el.text,  int(mz_el.get("precision",64)),  mz_el.get("endian","little"))
        in_arr = decode_array(int_el.text, int(int_el.get("precision",32)), int_el.get("endian","little"))
        if len(mz_arr) == len(in_arr):
            spectra.append({"rt_min": rt, "mz": mz_arr, "intensity": in_arr})
    return spectra

def build_tic(spectra):
    return (np.array([s["rt_min"] for s in spectra]),
            np.array([s["intensity"].sum() for s in spectra]))

# ── load all runs ──────────────────────────────────────────────────────────────
FILES = {76: "Rxn_splitless_076.mzdata.xml",
         79: "Rxn_splitless_079.mzdata.xml",
         82: "Rxn_splitless_082.mzdata.xml"}

runs = {}
for rnum, fname in FILES.items():
    sp = parse_mzdata(os.path.join(DATA_DIR, fname))
    rts, tics = build_tic(sp)
    runs[rnum] = {"spectra": sp, "rts": rts, "tics": tics}
    print(f"Run {rnum}: {len(sp)} scans, RT {rts[0]:.2f}–{rts[-1]:.2f} min")

# ── build a common RT grid (interpolated TIC) to compare abundances ────────────
def interp_tic(rts_src, tics_src, rts_target):
    return np.interp(rts_target, rts_src, tics_src)

# Use run-76 RT grid as reference
ref_rts = runs[76]["rts"]
tic76 = runs[76]["tics"]
tic79 = interp_tic(runs[79]["rts"], runs[79]["tics"], ref_rts)
tic82 = interp_tic(runs[82]["rts"], runs[82]["tics"], ref_rts)

# enzyme mean TIC vs control
tic_enzyme_mean = (tic76 + tic79) / 2.0
tic_ratio = np.where(tic82 > 1e3, tic_enzyme_mean / tic82, np.nan)

# ── overlay TIC plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1]})

ax = axes[0]
ax.plot(ref_rts,  tic76  / 1e6, color="royalblue",   lw=0.9, label="Run 76 (enzyme)")
ax.plot(ref_rts,  tic79  / 1e6, color="cornflowerblue", lw=0.9, ls="--", label="Run 79 (enzyme)")
ax.plot(ref_rts,  tic82  / 1e6, color="tomato",       lw=0.9, label="Run 82 (no-enzyme ctrl)")
ax.set_ylabel("TIC Intensity (×10⁶)")
ax.set_title("TIC overlay – all three runs")
ax.legend(loc="upper right")

ax2 = axes[1]
valid = ~np.isnan(tic_ratio)
ax2.plot(ref_rts[valid], tic_ratio[valid], color="darkgreen", lw=0.8)
ax2.axhline(1.5, color="gray", ls=":", lw=0.8, label="1.5× threshold")
ax2.set_xlabel("Retention Time (min)")
ax2.set_ylabel("Enzyme/Ctrl ratio")
ax2.set_ylim(0, 10)
ax2.legend(loc="upper right")

plt.tight_layout()
fig.savefig(os.path.join(RESULTS, "TIC_overlay.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved TIC_overlay.png")

# ── find peaks in the enzyme-vs-control ratio ──────────────────────────────────
ratio_clean = np.where(np.isnan(tic_ratio), 0, tic_ratio)
# also look in the enzyme TIC for peaks
smooth_enz = uniform_filter1d(tic_enzyme_mean, size=5)
noise_enz = np.percentile(smooth_enz[smooth_enz > 0], 10) if (smooth_enz > 0).any() else 1.0

enz_peaks, _ = find_peaks(smooth_enz, height=10*noise_enz, distance=15, prominence=5*noise_enz)

print(f"\n{len(enz_peaks)} peaks in enzyme TIC (above background)")

# ── helper: get averaged spectrum around a scan index ─────────────────────────
def avg_spectrum(spectra, rts, target_rt, half_width_min=0.05):
    """Average spectra within ±half_width_min around target_rt."""
    lo_rt = target_rt - half_width_min
    hi_rt = target_rt + half_width_min
    indices = [i for i, rt in enumerate(rts) if lo_rt <= rt <= hi_rt]
    if not indices:
        idx = np.argmin(np.abs(rts - target_rt))
        indices = [idx]
    mz_all = np.concatenate([spectra[i]["mz"] for i in indices])
    in_all  = np.concatenate([spectra[i]["intensity"] for i in indices])
    # bin to integer m/z
    mz_int = np.round(mz_all).astype(int)
    binned = {}
    for m, iv in zip(mz_int, in_all):
        binned[m] = binned.get(m, 0) + iv
    mz_arr = np.array(sorted(binned.keys()), dtype=float)
    in_arr  = np.array([binned[int(m)] for m in mz_arr], dtype=float)
    return mz_arr, in_arr

# ── reference fragments ────────────────────────────────────────────────────────
REF_TARGET = {182:30, 164:100, 153:60, 139:80, 111:70, 99:50, 96:45, 83:65, 71:55, 55:40}

def cosine_sim(obs_mz, obs_int, ref_dict, tol=0.5):
    ref_mzs = np.array(list(ref_dict.keys()), dtype=float)
    ref_ints = np.array(list(ref_dict.values()), dtype=float) / max(ref_dict.values())
    obs_matched = np.zeros(len(ref_mzs))
    for i, rmz in enumerate(ref_mzs):
        mask = np.abs(obs_mz - rmz) <= tol
        if mask.any():
            obs_matched[i] = obs_int[mask].max()
    if obs_matched.max() == 0: return 0.0
    obs_matched /= obs_matched.max()
    dot = np.dot(obs_matched, ref_ints)
    return dot / (np.linalg.norm(obs_matched) * np.linalg.norm(ref_ints) + 1e-12)

# ── per-peak analysis ──────────────────────────────────────────────────────────
RELATED = {
    "3-methylcyclopentanone":  [98, 83, 70, 55, 41],
    "trans-2-pentenal":        [84, 83, 69, 55, 41],
    "dehydrated product MW164":[164, 149, 136, 121, 107, 93, 79],
    "saturated analog MW184":  [184, 166, 153, 141, 113, 99, 85],
    "target MW182":            list(REF_TARGET.keys()),
}

results = []
for idx in enz_peaks:
    rt = ref_rts[idx]
    ratio_val = tic_ratio[idx] if not np.isnan(tic_ratio[idx]) else 0.0
    tic76_val = tic76[idx]
    tic79_val = tic79[idx]
    tic82_val = tic82[idx]

    # spectra from each run
    mz76, in76 = avg_spectrum(runs[76]["spectra"], runs[76]["rts"], rt)
    mz79, in79 = avg_spectrum(runs[79]["spectra"], runs[79]["rts"], rt)
    mz82, in82 = avg_spectrum(runs[82]["spectra"], runs[82]["rts"], rt)

    # use combined enzyme spectrum
    mz_all = np.concatenate([mz76, mz79])
    in_all  = np.concatenate([in76, in79])
    mz_int  = np.round(mz_all).astype(int)
    binned  = {}
    for m, iv in zip(mz_int, in_all):
        binned[m] = binned.get(m, 0) + iv
    mz_e = np.array(sorted(binned.keys()), dtype=float)
    in_e = np.array([binned[int(m)] for m in mz_e], dtype=float)

    cos = cosine_sim(mz_e, in_e, REF_TARGET)
    base_mz = int(mz_e[np.argmax(in_e)]) if in_e.size > 0 else 0

    # fragments observed (≥2% of base peak)
    base_val = in_e.max() if in_e.size > 0 else 1.0
    frags_obs = sorted([int(m) for m, iv in zip(mz_e, in_e)
                        if any(abs(m - ref) <= 0.5 for ref in REF_TARGET)
                        and iv >= 0.02 * base_val])

    # best identity
    best_name, best_score = "unknown", cos
    for name, ions in RELATED.items():
        ref_d = {ion: 100 for ion in ions}
        s = cosine_sim(mz_e, in_e, ref_d)
        if s > best_score:
            best_score, best_name = s, name

    if cos >= 0.3 or 182 in [int(round(m)) for m in mz_e[in_e > 0.01*base_val]]:
        best_name = "target MW182"

    conf = "High" if best_score >= 0.6 else ("Medium" if best_score >= 0.35 else "Low")
    in_ctrl = "Yes" if tic82_val > 0.3 * tic76_val else "No"
    enzyme_flag = ""
    if tic82_val < 0.3 * tic76_val and tic76_val > 1e5:
        enzyme_flag = "ENZYME-ONLY"
    elif ratio_val >= 2.0:
        enzyme_flag = "ENZYME-ENRICHED (%.1fx)" % ratio_val

    results.append({
        "rt_min": round(rt, 3),
        "tic_run76": round(tic76_val/1e6, 3),
        "tic_run79": round(tic79_val/1e6, 3),
        "tic_run82": round(tic82_val/1e6, 3),
        "enzyme_ctrl_ratio": round(ratio_val, 2),
        "base_peak_mz": base_mz,
        "frags_vs_target": frags_obs,
        "putative_identity": best_name,
        "cosine_similarity": round(cos, 4),
        "match_confidence": conf,
        "in_ctrl": in_ctrl,
        "enzyme_flag": enzyme_flag,
        "_mz_e": mz_e, "_in_e": in_e,
        "_mz82": mz82, "_in82": in82,
    })

# ── print table ───────────────────────────────────────────────────────────────
print(f"\n{'RT':>7} {'BP':>5} {'Ratio':>6} {'Cos':>6} {'Conf':>7} {'Flag':>25}  Identity")
print("-"*90)
for r in results:
    print(f"{r['rt_min']:>7.3f} {r['base_peak_mz']:>5} {r['enzyme_ctrl_ratio']:>6.2f} "
          f"{r['cosine_similarity']:>6.3f} {r['match_confidence']:>7} "
          f"{r['enzyme_flag']:>25}  {r['putative_identity']}")

# ── spectral plots for top candidates ─────────────────────────────────────────
# Sort by: enzyme-only flag first, then high cosine, then ratio
def sort_key(r):
    return (0 if "ENZYME" in r["enzyme_flag"] else 1,
            -r["cosine_similarity"],
            -r["enzyme_ctrl_ratio"])

top = sorted(results, key=sort_key)[:12]

fig, axes = plt.subplots(4, 3, figsize=(16, 14))
axes = axes.flatten()

for ax, r in zip(axes, top):
    mz_e = r["_mz_e"]
    in_e = r["_in_e"]
    mz82 = r["_mz82"]
    in82 = r["_in82"]

    # normalize to 100
    def norm(arr): return 100 * arr / arr.max() if arr.max() > 0 else arr

    in_e_n  = norm(in_e)
    in82_n  = norm(in82) if in82.size > 0 else np.zeros_like(in_e_n)

    ax.bar(mz_e, in_e_n, width=0.6, color="steelblue", alpha=0.7, label="enzyme (76+79)")
    if mz82.size > 0:
        ax.bar(mz82, -in82_n, width=0.6, color="tomato", alpha=0.6, label="ctrl (82)")

    # mark reference ions
    for ion in REF_TARGET:
        ax.axvline(ion, color="gold", lw=0.6, alpha=0.5)

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlim(45, 210)
    ax.set_ylim(-120, 120)
    ax.set_xlabel("m/z", fontsize=7)
    ax.set_ylabel("Rel. Int.", fontsize=7)
    ax.set_title(f"RT={r['rt_min']} | BP={r['base_peak_mz']} | cos={r['cosine_similarity']:.3f}\n"
                 f"{r['putative_identity']} | ratio={r['enzyme_ctrl_ratio']:.1f}x",
                 fontsize=7)
    ax.tick_params(labelsize=6)
    if ax is axes[0]:
        ax.legend(fontsize=6, loc="upper right")

    # label key fragment m/z on enzyme bars
    top_idx = np.argsort(in_e_n)[-6:]
    for ti in top_idx:
        ax.text(mz_e[ti], in_e_n[ti]+2, str(int(mz_e[ti])), ha="center", fontsize=5, color="navy")

plt.suptitle("Top candidate spectra (enzyme: blue bars up | control: red bars down)\n"
             "Gold lines = target reference ions", fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS, "candidate_spectra.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved candidate_spectra.png")

# ── updated TIC overlay with candidate annotations ────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(ref_rts, tic76/1e6,  color="royalblue",      lw=0.9, label="Run 76 (enzyme)")
ax.plot(ref_rts, tic79/1e6,  color="cornflowerblue", lw=0.9, ls="--", label="Run 79 (enzyme)")
ax.plot(ref_rts, tic82/1e6,  color="tomato",         lw=0.9, label="Run 82 (no-enzyme ctrl)")

for r in results:
    rt = r["rt_min"]
    y  = max(tic76[np.argmin(np.abs(ref_rts-rt))],
             tic79[np.argmin(np.abs(ref_rts-rt))]) / 1e6
    if "ENZYME" in r["enzyme_flag"]:
        color = "red"
        lbl = f"★ RT={rt:.2f}\n{r['putative_identity']}"
    elif r["cosine_similarity"] >= 0.5:
        color = "darkorange"
        lbl = f"RT={rt:.2f}\n{r['putative_identity']}"
    else:
        continue
    ax.annotate(lbl, xy=(rt, y), xytext=(0, 20),
                textcoords="offset points", ha="center", fontsize=6,
                color=color, arrowprops=dict(arrowstyle="-", color=color, lw=0.5))
    ax.plot(rt, y, "v", color=color, ms=6)

ax.set_xlabel("Retention Time (min)")
ax.set_ylabel("TIC Intensity (×10⁶)")
ax.set_title("TIC Overlay – annotated candidates (★ = enzyme-enriched/only)")
ax.legend(loc="upper right")
plt.tight_layout()
fig.savefig(os.path.join(RESULTS, "TIC_overlay_annotated.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved TIC_overlay_annotated.png")

# ── final CSV ─────────────────────────────────────────────────────────────────
csv_rows = []
for r in results:
    csv_rows.append({
        "Retention Time (min)": r["rt_min"],
        "Base Peak m/z": r["base_peak_mz"],
        "Key Fragment Ions Observed (vs target)": str(r["frags_vs_target"]),
        "Putative Identity": r["putative_identity"],
        "Cosine Similarity Score": r["cosine_similarity"],
        "Match Confidence": r["match_confidence"],
        "TIC Run 76 (1e6)": r["tic_run76"],
        "TIC Run 79 (1e6)": r["tic_run79"],
        "TIC Run 82 ctrl (1e6)": r["tic_run82"],
        "Enzyme/Ctrl TIC Ratio": r["enzyme_ctrl_ratio"],
        "Present in No-Enzyme Control (82)?": r["in_ctrl"],
        "Enzyme Enrichment Flag": r["enzyme_flag"],
    })

df = pd.DataFrame(csv_rows).sort_values("Retention Time (min)")
out_csv = os.path.join(RESULTS, "gcms_deepdive.csv")
df.to_csv(out_csv, index=False)
print(f"\nSummary CSV → {out_csv}")
print(df.to_string(index=False))

# ── highlight enzyme-enriched rows ────────────────────────────────────────────
print("\n\n=== ENZYME-ENRICHED / ENZYME-ONLY PEAKS ===")
enz_df = df[df["Enzyme Enrichment Flag"].str.contains("ENZYME", na=False)]
if enz_df.empty:
    print("  None found — all detected peaks are present in the no-enzyme control at comparable abundance.")
    print("  This suggests the reaction did not produce a clearly separable new peak under these conditions,")
    print("  OR the product co-elutes with a background component present in all runs.")
else:
    print(enz_df.to_string(index=False))

print("\nDone.")

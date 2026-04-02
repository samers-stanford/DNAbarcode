"""
Revised GC-MS analysis incorporating CFM-ID fragmentation predictions.

Key corrections vs. first pass:
  1. Reference spectrum updated: m/z 81 (C6H9+) is the dominant fragment per
     CFM-ID energy0 (29.88%).  m/z 164 (M-18, EI-specific H2O loss from β-OH)
     is retained as a high-weight EI diagnostic — CFM-ID used ESI [M+H]+, so
     it does not predict the dehydration pathway that dominates under EI.
  2. Full-range spectra (m/z 45–320) plotted to expose high-mass ions that
     disqualify peaks from being the target (MW 182).
  3. Extracted Ion Chromatograms (EICs) built for m/z 81, 99, 164, 182 and
     compared across all three runs — these are far more specific than TIC.
  4. Each candidate peak is annotated with what the spectrum actually looks like
     (e.g. FAME, column bleed, sesquiterpene) where the evidence is clear.
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

RESULTS  = "/home/user/DNAbarcode/results"
DATA_DIR = "/home/user/DNAbarcode/mzData_files"
os.makedirs(RESULTS, exist_ok=True)

# ── parsing ────────────────────────────────────────────────────────────────────

def decode_array(b64str, precision_bits, endian):
    raw = base64.b64decode(b64str.strip())
    fmt  = ("d" if precision_bits == 64 else "f")
    n    = len(raw) // (precision_bits // 8)
    pfx  = "<" if endian == "little" else ">"
    return np.array(struct.unpack(f"{pfx}{n}{fmt}", raw), dtype=np.float64)

def parse_mzdata(filepath):
    tree = ET.parse(filepath)
    spectra = []
    for se in tree.getroot().find("spectrumList").findall("spectrum"):
        rt = None
        for cv in se.iter("cvParam"):
            if cv.get("name") == "TimeInMinutes":
                rt = float(cv.get("value")); break
        if rt is None: continue
        me = se.find(".//mzArrayBinary/data")
        ie = se.find(".//intenArrayBinary/data")
        if me is None or ie is None: continue
        mz = decode_array(me.text, int(me.get("precision", 64)), me.get("endian", "little"))
        iv = decode_array(ie.text, int(ie.get("precision", 32)), ie.get("endian", "little"))
        if len(mz) == len(iv):
            spectra.append({"rt_min": rt, "mz": mz, "intensity": iv})
    return spectra

def build_tic(spectra):
    rts  = np.array([s["rt_min"]        for s in spectra])
    tics = np.array([s["intensity"].sum() for s in spectra])
    return rts, tics

# ── EIC builder ───────────────────────────────────────────────────────────────

def build_eic(spectra, target_mz, tol=0.4):
    """Extracted ion chromatogram for a single nominal m/z."""
    rts  = np.array([s["rt_min"] for s in spectra])
    vals = np.zeros(len(spectra))
    for i, s in enumerate(spectra):
        mask = np.abs(s["mz"] - target_mz) <= tol
        if mask.any():
            vals[i] = s["intensity"][mask].max()
    return rts, vals

# ── averaged spectrum around a target RT ──────────────────────────────────────

def avg_spectrum(spectra, rts, target_rt, hw=0.06):
    idx = [i for i, r in enumerate(rts) if abs(r - target_rt) <= hw]
    if not idx:
        idx = [int(np.argmin(np.abs(rts - target_rt)))]
    mz_all = np.concatenate([spectra[i]["mz"]       for i in idx])
    iv_all = np.concatenate([spectra[i]["intensity"] for i in idx])
    mz_int = np.round(mz_all).astype(int)
    binned = {}
    for m, v in zip(mz_int, iv_all):
        binned[m] = binned.get(m, 0) + v
    mz_arr = np.array(sorted(binned), dtype=float)
    iv_arr = np.array([binned[int(m)] for m in mz_arr], dtype=float)
    return mz_arr, iv_arr

# ── reference spectrum ─────────────────────────────────────────────────────────
#
# Combined EI + CFM-ID rationale
# ────────────────────────────────
# CFM-ID (ESI [M+H]+, energy0):
#   81  → 29.9 %  C6H9+         (pentenyl/methylcyclopentyl ring cation)
#   55  → 12.6 %  C4H7+
#   41  → 10.9 %  C3H5+
#   43  →  9.0 %  C2H3O+ / C3H7+
#   99  →  8.5 %  C6H11O+       (3-methylcyclopentenone cation; VERY diagnostic)
#   69  →  4.5 %  C5H9+
#
# EI-specific additions (not produced by ESI/CID but expected under 70 eV EI):
#   182 → ~15 %   M+·            (molecular ion, often weak for β-OH ketones)
#   164 → ~80 %   M−18 (−H2O)   ** BASE PEAK candidate for β-hydroxy ketones under EI **
#   139 → ~40 %   M−43 (loss of propyl / C3H7)
#   153 → ~25 %   M−29 (loss of CHO)
#   111 → ~30 %   ring acylium after α-cleavage
#
# m/z 81 is now the HIGHEST-WEIGHT small-fragment ion.
# m/z 164 is the key EI-specific diagnostic; absence strongly disfavours the target.

REF = {
    164: 80,   # M-18 EI, β-OH ketone water loss ← most diagnostic EI fragment
    182: 15,   # M+· (molecular ion)
    139: 40,   # M-43
    153: 25,   # M-29
    111: 30,   # ring acylium
     81: 30,   # CFM-ID top fragment (C6H9+)
     99: 25,   # 3-methylcyclopentenone+  (both EI and CFM-ID)
     96: 20,
     83: 22,
     71: 18,
     69: 15,   # CFM-ID
     55: 20,   # CFM-ID
     43: 12,   # CFM-ID
     41: 15,   # CFM-ID
}

def cosine_sim(obs_mz, obs_iv, ref, tol=0.5):
    ref_mz = np.array(list(ref), dtype=float)
    ref_iv = np.array(list(ref.values()), dtype=float)
    ref_iv = ref_iv / ref_iv.max()
    obs_m  = np.zeros(len(ref_mz))
    for i, rm in enumerate(ref_mz):
        mask = np.abs(obs_mz - rm) <= tol
        if mask.any():
            obs_m[i] = obs_iv[mask].max()
    if obs_m.max() == 0: return 0.0
    obs_m /= obs_m.max()
    denom  = np.linalg.norm(obs_m) * np.linalg.norm(ref_iv)
    return float(np.dot(obs_m, ref_iv) / denom) if denom else 0.0

def frags_present(obs_mz, obs_iv, ions, tol=0.5, min_rel=0.02):
    base = obs_iv.max() if obs_iv.size else 1.0
    return [int(ion) for ion in ions
            if np.abs(obs_mz - ion).min() <= tol
            and obs_iv[np.abs(obs_mz - ion) <= tol].max() >= min_rel * base]

# ── load data ──────────────────────────────────────────────────────────────────

FILES = {76: "Rxn_splitless_076.mzdata.xml",
         79: "Rxn_splitless_079.mzdata.xml",
         82: "Rxn_splitless_082.mzdata.xml"}

runs = {}
for rn, fn in FILES.items():
    sp       = parse_mzdata(os.path.join(DATA_DIR, fn))
    rts, tic = build_tic(sp)
    runs[rn] = {"sp": sp, "rts": rts, "tic": tic}
    print(f"Run {rn}: {len(sp)} scans  RT {rts[0]:.2f}–{rts[-1]:.2f} min")

# ── EIC plots ─────────────────────────────────────────────────────────────────
#
# Diagnostic ions:
#   m/z 182  M+· of target
#   m/z 164  M-18 (most diagnostic EI fragment)
#   m/z 139  M-43
#   m/z 99   methylcyclopentenone (both CFM-ID and EI)
#   m/z 81   CFM-ID top fragment

EIC_IONS = {
    182: ("M⁺· (182)", "navy"),
    164: ("M−18 H₂O (164)", "crimson"),
    139: ("M−43 (139)", "darkorange"),
     99: ("3-MeCpone⁺ (99)", "forestgreen"),
     81: ("C₆H₉⁺ (81)", "purple"),
}

fig, axes = plt.subplots(len(EIC_IONS), 1, figsize=(15, 14), sharex=True)
colours = {76: "royalblue", 79: "cornflowerblue", 82: "tomato"}
styles  = {76: "-",          79: "--",             82: ":"}
labels  = {76: "Run 76 (enzyme)", 79: "Run 79 (enzyme)", 82: "Run 82 (ctrl)"}

for ax, (ion, (ion_label, ref_colour)) in zip(axes, EIC_IONS.items()):
    for rn in [76, 79, 82]:
        sp   = runs[rn]["sp"]
        rts, eic = build_eic(sp, ion)
        ax.plot(rts, eic / 1e4, color=colours[rn], ls=styles[rn],
                lw=0.9, label=labels[rn])
    ax.set_ylabel(ion_label, fontsize=8, color=ref_colour)
    ax.tick_params(labelsize=7)
    if ax is axes[0]:
        ax.legend(fontsize=7, loc="upper right")

axes[-1].set_xlabel("Retention Time (min)")
plt.suptitle("Extracted Ion Chromatograms for target-diagnostic m/z values\n"
             "(Enzyme runs = blue; No-enzyme control = red dashed)", fontsize=10)
plt.tight_layout()
eic_out = os.path.join(RESULTS, "EIC_diagnostic.png")
fig.savefig(eic_out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {eic_out}")

# ── EIC difference: enzyme mean minus control ──────────────────────────────────

fig, axes = plt.subplots(len(EIC_IONS), 1, figsize=(15, 14), sharex=True)
ref_rts = runs[76]["rts"]

for ax, (ion, (ion_label, ref_colour)) in zip(axes, EIC_IONS.items()):
    _, eic76 = build_eic(runs[76]["sp"], ion)
    _, eic79 = build_eic(runs[79]["sp"], ion)
    eic82_i  = np.interp(ref_rts, runs[82]["rts"],
                         build_eic(runs[82]["sp"], ion)[1])
    eic79_i  = np.interp(ref_rts, runs[79]["rts"], eic79)

    diff = (eic76 + eic79_i) / 2.0 - eic82_i
    ax.fill_between(ref_rts, 0, diff / 1e4,
                    where=(diff > 0), color=ref_colour, alpha=0.5,
                    label="enzyme > ctrl")
    ax.fill_between(ref_rts, 0, diff / 1e4,
                    where=(diff < 0), color="gray", alpha=0.3,
                    label="ctrl > enzyme")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_ylabel(f"Δ {ion_label}\n(×10⁴)", fontsize=7, color=ref_colour)
    ax.tick_params(labelsize=7)
    if ax is axes[0]:
        ax.legend(fontsize=7, loc="upper right")

axes[-1].set_xlabel("Retention Time (min)")
plt.suptitle("EIC difference: enzyme mean − no-enzyme control\n"
             "Positive (coloured) = enriched in enzyme runs", fontsize=10)
plt.tight_layout()
diff_out = os.path.join(RESULTS, "EIC_difference.png")
fig.savefig(diff_out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {diff_out}")

# ── full-range spectra for key peaks with annotation ──────────────────────────
#
# Peaks to inspect (from previous analysis + manual review):
#   8.350  — base peak 138   (aromatic compound, NOT target)
#   9.363  — base peak 191   (sesquiterpene?, NOT target: MW > 182)
#  12.905  — base peak  99   (check carefully)
#  14.261  — base peak  57   (FAME/wax: m/z 239/257/299 >> 182)
#  14.702  — base peak  99   (column bleed: m/z 376/408/420)
#
# Also scan 11–14 min window for any peak with base peak at 164 or 182.

INSPECT_RTS = [8.35, 9.36, 12.91, 14.26, 14.70]

PEAK_NOTES = {
    8.35:  ("background: aromatic ester/terpene\n"
            "MW≈138, no ions >138, loss of CO: 138→108"),
    9.36:  ("background: sesquiterpene or aromatic\n"
            "Base peak 191 > MW 182 → cannot be target"),
    12.91: ("background: terpene mix\n"
            "m/z 99/112/113 alkane/terpene; no M+ at 182"),
    14.26: ("background: FATTY ACID METHYL ESTER\n"
            "m/z 74 (McLafferty), 239 (M−OCH3 for palmitate)\n"
            "m/z > 182 at 62% relative → NOT target (MW 182)"),
    14.70: ("background: COLUMN BLEED / plasticizer\n"
            "m/z 376, 408, 420 >> MW 182 → NOT target"),
}

fig, axes = plt.subplots(len(INSPECT_RTS), 1, figsize=(15, 18))
for ax, rt_t in zip(axes, INSPECT_RTS):
    mz76, iv76 = avg_spectrum(runs[76]["sp"], runs[76]["rts"], rt_t)
    mz82, iv82 = avg_spectrum(runs[82]["sp"], runs[82]["rts"], rt_t)

    norm76 = 100 * iv76 / iv76.max() if iv76.max() > 0 else iv76
    norm82 = 100 * iv82 / iv82.max() if iv82.max() > 0 else iv82

    ax.bar(mz76,  norm76, width=0.7, color="steelblue", alpha=0.7,
           label="Run 76 (enzyme)")
    ax.bar(mz82, -norm82, width=0.7, color="tomato",    alpha=0.5,
           label="Run 82 (ctrl)")

    # vertical lines for target reference ions
    for ion in [182, 164, 139, 99, 81]:
        ax.axvline(ion, color="gold", lw=0.7, ls="--", alpha=0.8)
    ax.axvline(182, color="gold", lw=1.2, ls="-", alpha=1.0,
               label="MW 182 target ions")

    # label top ions on enzyme spectrum (above x-axis)
    top_idx = np.argsort(norm76)[-8:]
    for ti in top_idx:
        if norm76[ti] >= 10:
            ax.text(mz76[ti], norm76[ti] + 2, str(int(mz76[ti])),
                    ha="center", fontsize=6, color="navy")

    # flag any ion > 182 that is prominent
    high_mask = (mz76 > 182) & (norm76 >= 10)
    for mi, vi in zip(mz76[high_mask], norm76[high_mask]):
        ax.text(mi, vi + 2, f"▲{int(mi)}", ha="center", fontsize=6,
                color="red", fontweight="bold")

    note = PEAK_NOTES.get(rt_t, "")
    ax.set_title(f"RT ≈ {rt_t:.2f} min | {note}", fontsize=8, loc="left")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlim(45, 330)
    ax.set_ylim(-115, 115)
    ax.set_ylabel("Rel. Intensity", fontsize=7)
    ax.tick_params(labelsize=7)
    if ax is axes[0]:
        ax.legend(fontsize=7, loc="upper right")

axes[-1].set_xlabel("m/z")
plt.suptitle("Full-range mass spectra at key peaks\n"
             "Blue bars up = enzyme (Run 76) | Red bars down = control (Run 82)\n"
             "Gold dashed = target reference ions | ▲red = ions > MW 182 (disqualifiers)",
             fontsize=9)
plt.tight_layout()
spec_out = os.path.join(RESULTS, "fullrange_spectra.png")
fig.savefig(spec_out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {spec_out}")

# ── systematic search for peaks where m/z 164 or 182 is a MAJOR ion ───────────
#
# Instead of relying on TIC peaks (which are dominated by background), scan
# every spectrum in Run 76 & 79 and flag any scan where:
#   (a) m/z 164 ≥ 5% of base peak, OR m/z 182 ≥ 5% of base peak
#   (b) AND base peak itself is ≤ 200 (discard column bleed at high m/z)
#   (c) AND m/z 81 or 99 also present at ≥ 5% (CFM-ID corroboration)

print("\n\n=== Scanning for spectra containing m/z 164 or 182 as major ions ===")
print(f"{'Run':>5} {'RT':>7} {'164%':>6} {'182%':>6} {'81%':>5} {'99%':>5} "
      f"{'BasePk':>7} {'MaxMz>182':>10}  verdict")
print("-" * 90)

candidates = []
for rn in [76, 79, 82]:
    sp = runs[rn]["sp"]
    for s in sp:
        mz = np.round(s["mz"]).astype(int)
        iv = s["intensity"]
        if iv.max() < 5000:
            continue                        # skip noise scans
        base_val = iv.max()
        base_mz  = mz[np.argmax(iv)]

        # relative abundance of key ions
        def rel(target, tol=0):
            mask = (np.abs(mz - target) <= tol)
            return 100 * iv[mask].max() / base_val if mask.any() else 0.0

        r164 = rel(164)
        r182 = rel(182)
        r81  = rel(81)
        r99  = rel(99)

        # ions above 182 at ≥5%
        high_ions = sorted(set(mz[(mz > 183) & (iv >= 0.05 * base_val)]))
        max_high  = max(high_ions) if high_ions else 0

        if (r164 >= 5 or r182 >= 5) and (r81 >= 5 or r99 >= 5):
            verdict = "CANDIDATE"
            if max_high > 182:
                verdict = "candidate (contam >182)"
            candidates.append({
                "run": rn,
                "rt_min": round(s["rt_min"], 3),
                "rel_164": round(r164, 1),
                "rel_182": round(r182, 1),
                "rel_81":  round(r81, 1),
                "rel_99":  round(r99, 1),
                "base_peak_mz": int(base_mz),
                "max_ion_above_182": int(max_high),
                "verdict": verdict,
            })
            print(f"{rn:>5} {s['rt_min']:>7.3f} {r164:>6.1f} {r182:>6.1f} "
                  f"{r81:>5.1f} {r99:>5.1f} {base_mz:>7} {max_high:>10}  {verdict}")

# ── summarise candidates ───────────────────────────────────────────────────────
df_cand = pd.DataFrame(candidates)

if df_cand.empty:
    print("\n  No scans found with m/z 164/182 ≥5% AND m/z 81/99 ≥5%.")
    print("  Conclusion: the target compound (C11H18O2, MW 182) is NOT detectable")
    print("  above background in any of the three runs at this sensitivity.")
else:
    # cross-run comparison: flag enzyme-only
    df_cand["present_in_ctrl"] = df_cand.apply(
        lambda r: any(abs(df_cand[(df_cand["run"] == 82)]["rt_min"] - r["rt_min"]) < 0.08),
        axis=1
    )
    enz_only = df_cand[
        (df_cand["run"] != 82) &
        (~df_cand["present_in_ctrl"]) &
        (df_cand["max_ion_above_182"] == 0)
    ]
    print(f"\n  {len(df_cand)} candidate scans found across all runs.")
    print(f"  {len(enz_only)} enzyme-only (absent in control, no high-mass contaminant):")
    if not enz_only.empty:
        print(enz_only.to_string(index=False))
    else:
        print("  None — all candidates co-occur in the no-enzyme control.")

# ── EIC peaks specifically on m/z 164 ─────────────────────────────────────────
# Peak-detect on the m/z 164 EIC for each run and report top hits

print("\n\n=== m/z 164 EIC peaks in each run ===")
for rn in [76, 79, 82]:
    rts_r, eic164 = build_eic(runs[rn]["sp"], 164)
    smooth        = uniform_filter1d(eic164, size=5)
    noise         = np.percentile(smooth[smooth > 0], 20) if (smooth > 0).any() else 1.0
    peaks164, _   = find_peaks(smooth, height=10 * noise, distance=10)
    label = "enzyme" if rn != 82 else "ctrl"
    print(f"\n  Run {rn} ({label}) — {len(peaks164)} peaks on m/z 164 EIC:")
    for idx in peaks164:
        # check if m/z 182 also present at this RT
        _, eic182_r = build_eic(runs[rn]["sp"], 182)
        r182 = eic182_r[idx] / eic164[idx] if eic164[idx] > 0 else 0
        print(f"    RT={rts_r[idx]:.3f}  int164={eic164[idx]:.0f}  "
              f"182/164={r182:.2f}  ({'likely product' if r182 >= 0.1 else 'weak 182'})")

# ── final CSV ──────────────────────────────────────────────────────────────────
if not df_cand.empty:
    out_csv = os.path.join(RESULTS, "gcms_revised_candidates.csv")
    df_cand.to_csv(out_csv, index=False)
    print(f"\nCSV → {out_csv}")

# ── updated TIC overlay (individual per-run) showing EIC 164 overlaid ─────────
for rn in [76, 79, 82]:
    rts_r = runs[rn]["rts"]
    tic_r = runs[rn]["tic"]
    _, eic164_r = build_eic(runs[rn]["sp"], 164)
    _, eic182_r = build_eic(runs[rn]["sp"], 182)
    _, eic81_r  = build_eic(runs[rn]["sp"],  81)
    _, eic99_r  = build_eic(runs[rn]["sp"],  99)

    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax2 = ax1.twinx()

    ax1.plot(rts_r, tic_r / 1e6, color="lightsteelblue", lw=0.8,
             label="TIC", zorder=1)
    ax2.plot(rts_r, eic164_r / 1e4, color="crimson",     lw=1.2,
             label="EIC 164 (M−18)", zorder=3)
    ax2.plot(rts_r, eic182_r / 1e4, color="navy",        lw=1.0, ls="--",
             label="EIC 182 (M⁺·)", zorder=3)
    ax2.plot(rts_r, eic81_r  / 1e4, color="purple",      lw=0.8, ls="-.",
             label="EIC  81 (C₆H₉⁺)", zorder=2)
    ax2.plot(rts_r, eic99_r  / 1e4, color="forestgreen", lw=0.8, ls=":",
             label="EIC  99 (MeCp+)", zorder=2)

    ax1.set_xlabel("Retention Time (min)")
    ax1.set_ylabel("TIC (×10⁶)", color="steelblue")
    ax2.set_ylabel("EIC Intensity (×10⁴)", color="crimson")
    lbl  = "enzyme" if rn != 82 else "no-enzyme control"
    ax1.set_title(f"Run {rn} ({lbl}) — TIC and target-diagnostic EICs")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="upper right")

    plt.tight_layout()
    out = os.path.join(RESULTS, f"TIC_EIC_run{rn}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

print("\nDone.")

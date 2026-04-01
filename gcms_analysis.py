"""
GC-MS analysis of Agilent mzData.xml files.
Target: (E)-2-(1-hydroxypent-2-en-1-yl)-3-methylcyclopentan-1-one
        C11H18O2, MW=182.26
Runs 76, 79 = enzyme; Run 82 = no-enzyme control
"""

import base64
import struct
import xml.etree.ElementTree as ET
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

# ---------------------------------------------------------------------------
# Reference spectrum for the target compound (MW 182) and related compounds.
# Fragment ions based on expected EI fragmentation of a β-hydroxy
# cyclopentanone aldol product.
#   182  M+  (molecular ion)
#   164  M-18 (loss of H2O)
#   153  M-29 (loss of CHO / C2H5)
#   139  M-43 (loss of propyl or CH2=CHCH2· / acetyl)
#   111  loss of C4H7O from M  (ring-opened acyl cation)
#    99  3-methylcyclopentenone cation
#    96  C6H8O  cyclopentenone base fragment
#    83  C5H7O+ or C6H11+
#    71  C5H11+ or acylium
#    55  C4H7+ / C3H3O+
# ---------------------------------------------------------------------------
REFERENCE_IONS = {
    182: 30,   # M+ (often weak in β-hydroxy ketones due to easy dehydration)
    164: 100,  # M-18, base peak candidate
    153: 60,   # M-29
    139: 80,   # M-43
    111: 70,   # ring fragment
    99:  50,   # methylcyclopentenone
    96:  45,
    83:  65,
    71:  55,
    55:  40,
}

# Related compounds to flag
RELATED = {
    "3-methylcyclopentanone (SM)": {"mw": 98, "key_ions": [98, 83, 70, 55, 41]},
    "trans-2-pentenal (SM)":       {"mw": 84, "key_ions": [84, 83, 69, 55, 41]},
    "dehydrated product (MW 164)": {"mw": 164, "key_ions": [164, 149, 136, 121, 107, 93, 79]},
    "saturated analog (MW 184)":   {"mw": 184, "key_ions": [184, 166, 153, 141, 113, 99, 85]},
    "target (MW 182)":             {"mw": 182, "key_ions": list(REFERENCE_IONS.keys())},
}

# ---------------------------------------------------------------------------
# mzData parsing
# ---------------------------------------------------------------------------

def decode_array(b64str, precision_bits, endian):
    """Decode base64 binary array from mzData."""
    raw = base64.b64decode(b64str.strip())
    fmt_char = "d" if precision_bits == 64 else "f"
    byte_size = precision_bits // 8
    n = len(raw) // byte_size
    end_prefix = "<" if endian == "little" else ">"
    arr = struct.unpack(f"{end_prefix}{n}{fmt_char}", raw)
    return np.array(arr, dtype=np.float64)


def parse_mzdata(filepath):
    """
    Parse an mzData 1.05 XML file.
    Returns list of dicts: {rt_min, mz, intensity}
    """
    print(f"\n[parse] {os.path.basename(filepath)}")
    tree = ET.parse(filepath)
    root = tree.getroot()

    # mzData 1.05 has no namespace
    spectra = []
    spectrum_list = root.find("spectrumList")
    if spectrum_list is None:
        raise RuntimeError("No <spectrumList> found in file.")

    total = int(spectrum_list.get("count", 0))
    print(f"  Spectrum count declared: {total}")

    for spec_el in spectrum_list.findall("spectrum"):
        # --- retention time ---
        rt = None
        for cvp in spec_el.iter("cvParam"):
            if cvp.get("name") == "TimeInMinutes":
                rt = float(cvp.get("value"))
                break
        if rt is None:
            continue

        # --- m/z array ---
        mz_el = spec_el.find(".//mzArrayBinary/data")
        int_el = spec_el.find(".//intenArrayBinary/data")
        if mz_el is None or int_el is None:
            continue

        mz_prec = int(mz_el.get("precision", 64))
        mz_end  = mz_el.get("endian", "little")
        in_prec = int(int_el.get("precision", 32))
        in_end  = int_el.get("endian", "little")

        mz_arr  = decode_array(mz_el.text, mz_prec, mz_end)
        int_arr = decode_array(int_el.text, in_prec, in_end)

        if len(mz_arr) != len(int_arr):
            print(f"  WARNING scan at RT={rt:.3f}: mz/int length mismatch "
                  f"({len(mz_arr)} vs {len(int_arr)}) — skipping")
            continue

        spectra.append({"rt_min": rt, "mz": mz_arr, "intensity": int_arr})

    print(f"  Successfully parsed {len(spectra)} scans "
          f"(RT {spectra[0]['rt_min']:.2f}–{spectra[-1]['rt_min']:.2f} min)")
    return spectra


# ---------------------------------------------------------------------------
# TIC construction
# ---------------------------------------------------------------------------

def build_tic(spectra):
    rts = np.array([s["rt_min"] for s in spectra])
    tics = np.array([s["intensity"].sum() for s in spectra])
    return rts, tics


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(obs_mz, obs_int, ref_dict, mz_tol=0.5):
    """
    Dot-product cosine similarity between observed spectrum and reference.
    obs_mz, obs_int: numpy arrays
    ref_dict: {mz_nominal: relative_intensity}
    Returns score in [0, 1].
    """
    ref_mzs = np.array(list(ref_dict.keys()), dtype=float)
    ref_ints = np.array(list(ref_dict.values()), dtype=float)
    ref_ints = ref_ints / ref_ints.max()   # normalize

    obs_matched = np.zeros(len(ref_mzs))
    for i, rmz in enumerate(ref_mzs):
        mask = np.abs(obs_mz - rmz) <= mz_tol
        if mask.any():
            obs_matched[i] = obs_int[mask].max()

    if obs_matched.sum() == 0:
        return 0.0

    obs_matched = obs_matched / obs_matched.max()

    dot = np.dot(obs_matched, ref_ints)
    norm = np.sqrt((obs_matched**2).sum()) * np.sqrt((ref_ints**2).sum())
    return float(dot / norm) if norm > 0 else 0.0


# ---------------------------------------------------------------------------
# Fragment ion presence check
# ---------------------------------------------------------------------------

def observed_fragments(obs_mz, obs_int, target_ions, mz_tol=0.5, min_rel=0.02):
    """Return list of target ions found above min_rel * base peak."""
    base = obs_int.max() if obs_int.size > 0 else 1.0
    found = []
    for ion in target_ions:
        mask = np.abs(obs_mz - ion) <= mz_tol
        if mask.any() and obs_int[mask].max() >= min_rel * base:
            found.append(int(ion))
    return found


# ---------------------------------------------------------------------------
# Peak detection and annotation
# ---------------------------------------------------------------------------

def detect_peaks_tic(rts, tics, run_label):
    """
    Detect significant TIC peaks above a rolling background.
    Returns indices of detected peaks.
    """
    # smooth TIC slightly for detection
    smooth = uniform_filter1d(tics, size=5)
    # baseline: rolling minimum (approximated by percentile over window)
    window = 100
    baseline = np.array([
        np.percentile(smooth[max(0, i-window):i+window+1], 10)
        for i in range(len(smooth))
    ])
    above_bg = smooth - baseline
    noise = np.percentile(above_bg[above_bg > 0], 30) if (above_bg > 0).any() else 1.0

    peaks, props = find_peaks(
        above_bg,
        height=5 * noise,
        distance=15,       # min ~1 s separation at ~6 scans/s
        prominence=3 * noise,
    )
    print(f"  [{run_label}] {len(peaks)} TIC peaks detected above background")
    return peaks, above_bg, baseline


def annotate_peaks(spectra, peak_indices, rts, run_label):
    """For each TIC peak, extract spectrum and classify."""
    rows = []
    for idx in peak_indices:
        s = spectra[idx]
        mz = s["mz"]
        inten = s["intensity"]
        rt = s["rt_min"]

        # sum a small window around the peak for better S/N
        lo = max(0, idx - 2)
        hi = min(len(spectra) - 1, idx + 2)
        mz_all = np.concatenate([spectra[j]["mz"] for j in range(lo, hi+1)])
        in_all  = np.concatenate([spectra[j]["intensity"] for j in range(lo, hi+1)])
        # bin into integer m/z
        mz_int = np.round(mz_all).astype(int)
        binned = {}
        for m, iv in zip(mz_int, in_all):
            binned[m] = binned.get(m, 0) + iv
        mz_arr = np.array(list(binned.keys()), dtype=float)
        in_arr  = np.array(list(binned.values()), dtype=float)

        base_peak_mz = int(mz_arr[np.argmax(in_arr)])

        # cosine similarity vs target
        cos_target = cosine_similarity(mz_arr, in_arr, REFERENCE_IONS)

        # check for each related compound
        best_identity = "unknown"
        best_score = cos_target
        best_ions = []

        identity_scores = {}
        for name, info in RELATED.items():
            ref_dict = {ion: 100 for ion in info["key_ions"]}
            score = cosine_similarity(mz_arr, in_arr, ref_dict)
            found = observed_fragments(mz_arr, in_arr, info["key_ions"])
            identity_scores[name] = (score, found)
            if score > best_score:
                best_score = score
                best_identity = name

        if cos_target >= 0.3 or 182 in [int(round(m)) for m in mz_arr[in_arr > 0.01 * in_arr.max()]]:
            best_identity = "target (MW 182)"
            best_score = cos_target

        # key fragments observed (vs target reference)
        key_frags = observed_fragments(mz_arr, in_arr, list(REFERENCE_IONS.keys()))

        # confidence
        if best_score >= 0.6:
            confidence = "High"
        elif best_score >= 0.35:
            confidence = "Medium"
        else:
            confidence = "Low"

        rows.append({
            "run": run_label,
            "rt_min": round(rt, 3),
            "base_peak_mz": base_peak_mz,
            "key_fragments_observed": str(key_frags),
            "putative_identity": best_identity,
            "cosine_similarity": round(best_score, 4),
            "match_confidence": confidence,
            "_peak_idx": idx,
            "_identity_scores": identity_scores,
            "_mz_arr": mz_arr,
            "_in_arr": in_arr,
        })
    return rows


# ---------------------------------------------------------------------------
# TIC plotting
# ---------------------------------------------------------------------------

def plot_tic(rts, tics, peak_indices, rows, run_label, run_num):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rts, tics / 1e6, color="steelblue", lw=0.8, label="TIC")

    # annotate peaks
    peak_rts = {r["_peak_idx"]: r for r in rows}
    for idx in peak_indices:
        rt_pk = rts[idx]
        tic_pk = tics[idx]
        label = ""
        color = "gray"
        if idx in peak_rts:
            r = peak_rts[idx]
            label = f"{r['putative_identity'].split('(')[0].strip()}\nRT={rt_pk:.2f}"
            if "target" in r["putative_identity"]:
                color = "red"
            elif "dehydrated" in r["putative_identity"]:
                color = "orange"
            elif "SM" in r["putative_identity"]:
                color = "green"
            elif "saturated" in r["putative_identity"]:
                color = "purple"
            else:
                color = "gray"
        ax.annotate(
            label,
            xy=(rt_pk, tic_pk / 1e6),
            xytext=(0, 18),
            textcoords="offset points",
            ha="center",
            fontsize=6,
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.6),
        )
        ax.plot(rt_pk, tic_pk / 1e6, "v", color=color, markersize=5)

    ax.set_xlabel("Retention Time (min)")
    ax.set_ylabel("TIC Intensity (×10⁶)")
    ax.set_title(f"Total Ion Chromatogram — Run {run_num} "
                 f"({'enzyme' if run_num != 82 else 'no-enzyme control'})")
    ax.legend(loc="upper right")
    out = os.path.join(RESULTS, f"TIC_run{run_num}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  TIC saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FILES = {
    76: "Rxn_splitless_076.mzdata.xml",
    79: "Rxn_splitless_079.mzdata.xml",
    82: "Rxn_splitless_082.mzdata.xml",
}

all_rows = []
run_data = {}  # store for cross-run comparison

for run_num, fname in FILES.items():
    fpath = os.path.join(DATA_DIR, fname)
    spectra = parse_mzdata(fpath)
    rts, tics = build_tic(spectra)
    run_data[run_num] = {"rts": rts, "tics": tics, "spectra": spectra}

    peak_indices, above_bg, baseline = detect_peaks_tic(rts, tics, f"Run {run_num}")
    rows = annotate_peaks(spectra, peak_indices, rts, f"Run {run_num}")

    # print per-peak detail
    print(f"\n  {'RT':>7}  {'BP':>5}  {'Cos':>6}  {'Conf':>7}  Identity")
    print(f"  {'-'*7}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*40}")
    for r in rows:
        print(f"  {r['rt_min']:>7.3f}  {r['base_peak_mz']:>5}  "
              f"{r['cosine_similarity']:>6.3f}  {r['match_confidence']:>7}  "
              f"{r['putative_identity']}")

    plot_tic(rts, tics, peak_indices, rows, f"Run {run_num}", run_num)
    all_rows.extend(rows)

# ---------------------------------------------------------------------------
# Cross-run comparison
# ---------------------------------------------------------------------------
print("\n\n=== Cross-run comparison ===")

# For each peak in enzyme runs, check if a matching peak (within 0.05 min RT)
# exists in the control run (82).
ctrl_rows = [r for r in all_rows if r["run"] == "Run 82"]
enzyme_rows = [r for r in all_rows if r["run"] != "Run 82"]

RT_TOL_CROSS = 0.08  # min

def find_ctrl_tic(rt, ctrl_spectra, ctrl_rts):
    """Get TIC value nearest to rt in control run."""
    idx = np.argmin(np.abs(ctrl_rts - rt))
    return run_data[82]["tics"][idx]

def find_ctrl_peak(rt, ctrl_rows, tol=RT_TOL_CROSS):
    matches = [r for r in ctrl_rows if abs(r["rt_min"] - rt) <= tol]
    return matches[0] if matches else None

# Build final CSV rows
csv_rows = []
for r in all_rows:
    ctrl_match = find_ctrl_peak(r["rt_min"], ctrl_rows)
    in_ctrl = "Yes" if ctrl_match is not None else "No"
    # flag enzyme-enriched peaks
    flag = ""
    if r["run"] != "Run 82":
        if ctrl_match is None:
            flag = "ENZYME-ONLY"
        elif r["cosine_similarity"] > (ctrl_match["cosine_similarity"] + 0.1):
            flag = "ENZYME-ENRICHED"

    csv_rows.append({
        "Run": r["run"],
        "Retention Time (min)": r["rt_min"],
        "Base Peak m/z": r["base_peak_mz"],
        "Key Fragment Ions Observed": r["key_fragments_observed"],
        "Putative Identity": r["putative_identity"],
        "Cosine Similarity Score": r["cosine_similarity"],
        "Match Confidence": r["match_confidence"],
        "Present in No-Enzyme Control (82)?": in_ctrl,
        "Enzyme Enrichment Flag": flag,
    })

df = pd.DataFrame(csv_rows).sort_values(["Retention Time (min)", "Run"])
csv_out = os.path.join(RESULTS, "gcms_summary.csv")
df.to_csv(csv_out, index=False)
print(f"\nSummary CSV saved → {csv_out}")
print(f"\n{df.to_string(index=False)}")

# ---------------------------------------------------------------------------
# Print detailed hits for target/related
# ---------------------------------------------------------------------------
print("\n\n=== Candidate peaks for target / analogs ===")
for r in sorted(all_rows, key=lambda x: x["cosine_similarity"], reverse=True):
    if r["cosine_similarity"] >= 0.25 or r["base_peak_mz"] in (164, 182, 184, 139, 111):
        print(f"\n  Run {r['run']}  RT={r['rt_min']:.3f} min  "
              f"BP={r['base_peak_mz']}  cos={r['cosine_similarity']:.3f}  "
              f"conf={r['match_confidence']}")
        print(f"    Key fragments vs target observed: {r['key_fragments_observed']}")
        for name, (sc, found) in r["_identity_scores"].items():
            if sc >= 0.2:
                print(f"    vs '{name}': score={sc:.3f}  found={found}")

print("\nDone.")

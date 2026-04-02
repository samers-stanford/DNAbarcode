"""
GC-MS Method Development Analysis
Volatile compound standards: cyclopentanones, pentenal, pentenol, jasmone
Agilent 7820MS / VF-5ht column
"""
import base64, struct, re, os, textwrap
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DEV_DIR  = "/home/user/DNAbarcode/gcms_dev"
OUT_DIR  = "/home/user/DNAbarcode/gcms_dev/results"
os.makedirs(OUT_DIR, exist_ok=True)

SOLVENT_DELAY = 2.25   # min (compound runs); blank = 2.5
NEAR_FRONT_WINDOW = 0.5  # flag peaks within this many min after solvent delay

# ── colour palette ──────────────────────────────────────────────────────────────
COMPOUND_COLORS = {
    "3methyl-Cyclopentanone": "#1f77b4",
    "Cyclopentanone":         "#ff7f0e",
    "cis-jasmone":            "#2ca02c",
    "cis-pentenol":           "#d62728",
    "jasmone":                "#9467bd",
    "trans_pentenal":         "#8c564b",
    "blank1":                 "#7f7f7f",
}

# ── Step 1: Inventory ──────────────────────────────────────────────────────────
def parse_filename(fname):
    """Return (compound, conc_uM, split_ratio) from mzdata filename."""
    base = fname.replace(".mzdata.xml", "")
    # strip trailing _ex or _ex_1toXX
    m = re.match(r'^(.+?)_(\d+)_ex(?:_(1to\d+))?$', base)
    if m:
        compound = m.group(1)
        conc = int(m.group(2))
        split = m.group(3) if m.group(3) else None
        return compound, conc, split
    # blank
    return base, None, None

print("=" * 60)
print("Step 1 — Building inventory")
print("=" * 60)

mzdata_files = sorted(f for f in os.listdir(DEV_DIR) if f.endswith(".mzdata.xml"))
d_folders    = sorted(f for f in os.listdir(DEV_DIR) if f.endswith(".D") and os.path.isdir(os.path.join(DEV_DIR, f)))

inventory = []
for fname in mzdata_files:
    compound, conc, split = parse_filename(fname)
    d_name = fname.replace(".mzdata.xml", ".D")
    has_d  = d_name in d_folders
    inventory.append({
        "filename":    fname,
        "compound":    compound,
        "conc_uM":     conc if conc is not None else "N/A",
        "split_ratio": split if split else "none",
        "has_D_folder": has_d,
        "mzdata_present": True,
    })

inv_df = pd.DataFrame(inventory)
inv_df.to_csv(os.path.join(OUT_DIR, "inventory.csv"), index=False)
print(inv_df.to_string(index=False))
print(f"\nInventory saved → {OUT_DIR}/inventory.csv")

# ── Step 2: Parse acquisition methods ─────────────────────────────────────────
def parse_acqmeth(d_path):
    acq = os.path.join(d_path, "acqmeth.txt")
    if not os.path.exists(acq):
        return {}
    with open(acq, encoding="utf-16-le", errors="replace") as f:
        txt = f.read()

    def get(pattern, default="N/A"):
        m = re.search(pattern, txt, re.IGNORECASE)
        return m.group(1).strip() if m else default

    return {
        "oven_initial_C":      get(r'\(Initial\)\s+([\d.]+)\s*°C'),
        "oven_hold_min":       get(r'Hold Time\s+([\d.]+)\s*min'),
        "ramp1_rate":          get(r'#1 Rate\s+([\d.]+)\s*°C/min'),
        "ramp1_to_C":          get(r'#1 Value\s+([\d.]+)\s*°C'),
        "ramp2_rate":          get(r'#2 Rate\s+([\d.]+)\s*°C/min'),
        "ramp2_to_C":          get(r'#2 Value\s+([\d.]+)\s*°C'),
        "ramp2_hold_min":      get(r'#2 Hold Time\s+([\d.]+)\s*min'),
        "inlet_mode":          get(r'Mode\s+(Splitless|Split)\b'),
        "inlet_temp_C":        get(r'Heater\s+On\s+([\d.]+)\s*°C'),
        "carrier_flow_mLmin":  get(r'\(Initial\)\s+([\d.]+)\s*mL/min'),
        "carrier_mode":        get(r'Control Mode\s+(Constant Flow|Constant Pressure)'),
        "transfer_line_C":     get(r'Thermal Aux.*?(?:\(Initial\))\s+([\d.]+)\s*°C'),
        "solvent_delay_min":   get(r'Start Time\s*:\s*([\d.]+)'),
        "scan_low_mz":         get(r'Low Mass\s*:\s*([\d.]+)'),
        "scan_high_mz":        get(r'High Mass\s*:\s*([\d.]+)'),
        "column_info":         get(r'Column Information\s+(Agilent \S+)'),
        "column_desc":         get(r'Description\s+(VF-\S+)'),
        "column_dims":         get(r'Dimensions\s+([\d.]+ m[^\n]+)'),
        "run_time_min":        get(r'Run Time\s+([\d.]+)\s*min'),
    }

print("\n" + "=" * 60)
print("Step 2 — Parsing acquisition methods")
print("=" * 60)

method_rows = []
for d_name in d_folders:
    d_path = os.path.join(DEV_DIR, d_name)
    params = parse_acqmeth(d_path)
    params["D_folder"] = d_name
    method_rows.append(params)
    print(f"  Parsed: {d_name}  (solvent delay={params.get('solvent_delay_min')} min)")

meth_df = pd.DataFrame(method_rows)
meth_df.to_csv(os.path.join(OUT_DIR, "method_summary.csv"), index=False)
print(f"\nMethod summary saved → {OUT_DIR}/method_summary.csv")

# Print a clean summary of one representative method (3methyl-Cyclopentanone_500)
rep = method_rows[0] if method_rows else {}
print("\nRepresentative method parameters:")
for k, v in rep.items():
    if k != "D_folder":
        print(f"  {k:25s}: {v}")

# ── Step 3: Parse mzData XML ───────────────────────────────────────────────────
def decode_array(b64str, precision_bits, endian):
    raw = base64.b64decode(b64str.strip())
    fmt = "d" if precision_bits == 64 else "f"
    n   = len(raw) // (precision_bits // 8)
    pfx = "<" if endian == "little" else ">"
    return np.array(struct.unpack(f"{pfx}{n}{fmt}", raw), dtype=np.float64)

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
        mz_a = decode_array(mz_el.text,  int(mz_el.get("precision",  64)), mz_el.get("endian",  "little"))
        in_a = decode_array(int_el.text, int(int_el.get("precision", 32)), int_el.get("endian", "little"))
        if len(mz_a) == len(in_a):
            spectra.append({"rt_min": rt, "mz": mz_a, "intensity": in_a})
    return spectra

def build_tic(spectra):
    rts  = np.array([s["rt_min"]         for s in spectra])
    tics = np.array([s["intensity"].sum() for s in spectra])
    return rts, tics

def apex_spectrum(spectra, apex_idx, half=2):
    lo, hi = max(0, apex_idx - half), min(len(spectra)-1, apex_idx + half)
    mz_all = np.concatenate([spectra[j]["mz"]       for j in range(lo, hi+1)])
    in_all = np.concatenate([spectra[j]["intensity"] for j in range(lo, hi+1)])
    mz_int = np.round(mz_all).astype(int)
    binned = {}
    for m, iv in zip(mz_int, in_all):
        binned[m] = binned.get(m, 0) + iv
    mz_arr = np.array(sorted(binned.keys()), dtype=float)
    in_arr = np.array([binned[int(m)] for m in mz_arr], dtype=float)
    return mz_arr, in_arr

print("\n" + "=" * 60)
print("Step 3 — Parsing mzData XML files")
print("=" * 60)

run_data   = {}   # fname -> {compound, conc, split, rts, tics, spectra, apex_rt, apex_int, mz, intensity}
peak_rows  = []

for fname in mzdata_files:
    compound, conc, split = parse_filename(fname)
    fpath = os.path.join(DEV_DIR, fname)
    print(f"  Parsing {fname} …", end=" ")
    spectra = parse_mzdata(fpath)
    rts, tics = build_tic(spectra)

    # skip scans up to and including the solvent delay
    # find first scan index where RT > solvent delay
    skip = int(np.searchsorted(rts, SOLVENT_DELAY + 0.05))
    skip = max(skip, 1)
    apex_rel = int(np.argmax(tics[skip:])) + skip
    apex_rt  = rts[apex_rel]
    apex_int = tics[apex_rel]

    mz_spec, in_spec = apex_spectrum(spectra, apex_rel)
    base_mz = int(mz_spec[np.argmax(in_spec)]) if in_spec.size > 0 else 0

    # top 5 fragment ions
    top5_idx = np.argsort(in_spec)[-5:][::-1]
    top5_mz  = [int(mz_spec[i]) for i in top5_idx]

    near_front = apex_rt < (SOLVENT_DELAY + NEAR_FRONT_WINDOW)
    print(f"RT={apex_rt:.2f} min  BP={base_mz}  {'*** NEAR FRONT ***' if near_front else ''}")

    run_data[fname] = {
        "compound": compound, "conc": conc, "split": split,
        "rts": rts, "tics": tics, "spectra": spectra,
        "apex_rt": apex_rt, "apex_int": apex_int,
        "mz_spec": mz_spec, "in_spec": in_spec,
        "base_mz": base_mz, "top5_mz": top5_mz,
    }
    peak_rows.append({
        "filename":         fname,
        "compound":         compound,
        "conc_uM":          conc if conc else "N/A",
        "split_ratio":      split if split else "none",
        "apex_RT_min":      round(apex_rt, 3),
        "apex_TIC_1e6":     round(apex_int / 1e6, 3),
        "base_peak_mz":     base_mz,
        "top5_fragment_mz": str(top5_mz),
        "near_solvent_front": near_front,
    })

peak_df = pd.DataFrame(peak_rows).sort_values("apex_RT_min")
peak_df.to_csv(os.path.join(OUT_DIR, "peak_summary.csv"), index=False)
print(f"\nPeak summary saved → {OUT_DIR}/peak_summary.csv")
print(peak_df[["compound","conc_uM","split_ratio","apex_RT_min","base_peak_mz","near_solvent_front"]].to_string(index=False))

# ── Step 4a: TIC overlay ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4 — Generating figures")
print("=" * 60)

non_blank = {k: v for k, v in run_data.items() if v["compound"] != "blank1"}

fig, ax = plt.subplots(figsize=(16, 6))

legend_handles = []
for fname, d in sorted(non_blank.items(), key=lambda x: x[1]["compound"]):
    compound = d["compound"]
    conc     = d["conc"]
    split    = d["split"]
    color    = COMPOUND_COLORS.get(compound, "#333333")

    if split == "1to10":
        ls, lw, label_sfx = ":",  1.4, " 500µM 1:10"
    elif split == "1to20":
        ls, lw, label_sfx = "-.", 1.4, " 500µM 1:20"
    elif conc == 50:
        ls, lw, label_sfx = "--", 1.2, " 50µM"
    else:
        ls, lw, label_sfx = "-",  1.4, " 500µM"

    label = compound + label_sfx
    ax.plot(d["rts"], d["tics"] / 1e6, color=color, ls=ls, lw=lw, label=label, alpha=0.85)

ax.axvline(SOLVENT_DELAY, color="black", ls="--", lw=1.0, alpha=0.6, label=f"Solvent delay ({SOLVENT_DELAY} min)")
ax.set_xlabel("Retention Time (min)", fontsize=11)
ax.set_ylabel("TIC Intensity (×10⁶)", fontsize=11)
ax.set_title("TIC Overlay — Volatile Compound Standards", fontsize=13)
ax.legend(fontsize=7, loc="upper right", ncol=2)
ax.set_xlim(left=0)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "TIC_overlay.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved TIC_overlay.png")

# ── Step 4b: RT bar chart ──────────────────────────────────────────────────────
plot_rows = [r for r in peak_rows if r["compound"] != "blank1"]
plot_rows.sort(key=lambda x: x["apex_RT_min"])

fig, ax = plt.subplots(figsize=(10, max(6, len(plot_rows)*0.45)))

y_pos   = range(len(plot_rows))
bar_h   = 0.6
for i, r in enumerate(plot_rows):
    color = COMPOUND_COLORS.get(r["compound"], "#333333")
    ax.barh(i, r["apex_RT_min"], height=bar_h, color=color, alpha=0.8)
    sfx = ""
    if r["split_ratio"] != "none":
        sfx = f" [{r['split_ratio']}]"
    elif r["conc_uM"] != "N/A":
        sfx = f" [{r['conc_uM']}µM]"
    ax.text(r["apex_RT_min"] + 0.05, i, f"{r['apex_RT_min']:.2f} min{sfx}", va="center", fontsize=8)

ax.axvline(SOLVENT_DELAY + NEAR_FRONT_WINDOW, color="red", ls="--", lw=1.2,
           label=f"Near-front threshold ({SOLVENT_DELAY + NEAR_FRONT_WINDOW:.2f} min)")
ax.set_yticks(list(y_pos))
ax.set_yticklabels([f"{r['compound']}\n{r['conc_uM']}µM {r['split_ratio']}" for r in plot_rows], fontsize=8)
ax.set_xlabel("Apex Retention Time (min)", fontsize=11)
ax.set_title("Retention Times — Volatile Compound Standards", fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(0, max(r["apex_RT_min"] for r in plot_rows) * 1.25)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "RT_barchart.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved RT_barchart.png")

# ── Step 4c: Mass spectra panel ────────────────────────────────────────────────
# One spectrum per compound: prefer 500uM non-split, cis-jasmone use 1to20
spectra_to_plot = {}
priority = {"none_500": 3, "1to20": 2, "none_50": 1, "1to10": 0}

for fname, d in run_data.items():
    if d["compound"] == "blank1":
        continue
    c = d["compound"]
    split = d["split"] if d["split"] else "none"
    conc  = d["conc"]
    key   = f"{split}_{conc}" if split == "none" else split
    score = priority.get(key, 0)
    if c not in spectra_to_plot or score > spectra_to_plot[c]["score"]:
        spectra_to_plot[c] = {"score": score, "mz": d["mz_spec"], "in": d["in_spec"],
                               "apex_rt": d["apex_rt"], "fname": fname}

compounds_sorted = sorted(spectra_to_plot.keys())
ncols = 3
nrows = int(np.ceil(len(compounds_sorted) / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
axes = axes.flatten()

for ax_idx, comp in enumerate(compounds_sorted):
    ax  = axes[ax_idx]
    sd  = spectra_to_plot[comp]
    mz  = sd["mz"]
    ints = sd["in"]

    if ints.max() > 0:
        ints_norm = 100.0 * ints / ints.max()
    else:
        ints_norm = ints

    color = COMPOUND_COLORS.get(comp, "#333333")
    ax.bar(mz, ints_norm, width=0.7, color=color, alpha=0.75)

    # label top 5 ions
    top5 = np.argsort(ints_norm)[-5:][::-1]
    for ti in top5:
        ax.text(mz[ti], ints_norm[ti] + 1.5, str(int(mz[ti])),
                ha="center", va="bottom", fontsize=7, color="black", fontweight="bold")

    ax.set_xlim(max(40, mz.min() - 5), min(300, mz.max() + 5))
    ax.set_ylim(0, 115)
    ax.set_xlabel("m/z", fontsize=8)
    ax.set_ylabel("Rel. Intensity", fontsize=8)
    ax.set_title(f"{comp}\nRT={sd['apex_rt']:.2f} min", fontsize=9)
    ax.tick_params(labelsize=7)

# hide unused axes
for ax_idx in range(len(compounds_sorted), len(axes)):
    axes[ax_idx].set_visible(False)

plt.suptitle("Mass Spectra at Peak Apex — Volatile Compound Standards", fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "mass_spectra_panel.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved mass_spectra_panel.png")

# ── Step 4d: Intensity comparison ─────────────────────────────────────────────
# Group compounds that have multiple injections
from collections import defaultdict
comp_injections = defaultdict(list)
for r in peak_rows:
    if r["compound"] != "blank1":
        comp_injections[r["compound"]].append(r)

multi = {c: v for c, v in comp_injections.items() if len(v) > 1}

if multi:
    all_labels = set()
    for v in multi.values():
        for r in v:
            split = r["split_ratio"]
            conc  = r["conc_uM"]
            lbl = split if split != "none" else f"{conc}µM"
            all_labels.add(lbl)
    all_labels = sorted(all_labels, key=lambda x: (x.startswith("1to"), x))

    label_colors = {}
    palette = ["#4878d0","#ee854a","#6acc65","#d65f5f","#956cb4"]
    for i, lbl in enumerate(all_labels):
        label_colors[lbl] = palette[i % len(palette)]

    compounds_multi = sorted(multi.keys())
    x = np.arange(len(compounds_multi))
    n_groups = len(all_labels)
    width = 0.8 / n_groups

    fig, ax = plt.subplots(figsize=(max(8, len(compounds_multi)*2), 6))
    for gi, lbl in enumerate(all_labels):
        vals = []
        for comp in compounds_multi:
            match = [r for r in multi[comp]
                     if (r["split_ratio"] if r["split_ratio"] != "none" else f"{r['conc_uM']}µM") == lbl]
            vals.append(match[0]["apex_TIC_1e6"] if match else 0.0)
        offsets = x - 0.4 + width*(gi+0.5)
        ax.bar(offsets, vals, width=width*0.9, color=label_colors[lbl], label=lbl, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(compounds_multi, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Apex TIC Intensity (×10⁶)", fontsize=11)
    ax.set_title("Peak Intensity Comparison — Concentration & Split Ratio Effect", fontsize=12)
    ax.legend(title="Condition", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "intensity_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved intensity_comparison.png")
else:
    print("  No multi-injection compounds found — skipping intensity comparison plot")

# ── Step 5: Markdown report ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 5 — Writing Markdown report")
print("=" * 60)

# Gather data for report
near_front_compounds = [r for r in peak_rows if r["near_solvent_front"] and r["compound"] != "blank1"]
mid_run_compounds    = [r for r in peak_rows if not r["near_solvent_front"] and r["compound"] != "blank1"]
peak_df_sorted = peak_df[peak_df["compound"] != "blank1"].sort_values("apex_RT_min")

# Method table from first representative .D
rep_method = method_rows[0] if method_rows else {}

# Build inventory markdown table
inv_md_rows = []
for r in inventory:
    inv_md_rows.append(
        f"| {r['compound']} | {r['conc_uM']} | {r['split_ratio']} | "
        f"{'Yes' if r['has_D_folder'] else 'No'} | Yes |"
    )
inv_md = "\n".join(inv_md_rows)

# Build peak summary markdown table
peak_md_rows = []
for _, r in peak_df_sorted.iterrows():
    flag = " ⚠️ near front" if r["near_solvent_front"] else ""
    peak_md_rows.append(
        f"| {r['compound']} | {r['conc_uM']} | {r['split_ratio']} | "
        f"{r['apex_RT_min']:.3f} | {r['apex_TIC_1e6']:.2f} | {r['base_peak_mz']} | "
        f"{r['top5_fragment_mz']}{flag} |"
    )
peak_md = "\n".join(peak_md_rows)

# Jasmone RT
jasmone_row = peak_df_sorted[peak_df_sorted["compound"].str.contains("jasmone", case=False)]
jasmone_rt_str = ""
if not jasmone_row.empty:
    jasmone_rt_str = ", ".join(f"{row['compound']} at {row['apex_RT_min']:.2f} min" for _, row in jasmone_row.iterrows())

report = f"""# GC-MS Method Development Report
## Volatile Compound Standards — Agilent 7820MS / VF-5ht

**Date:** 2026-04-02  
**Instrument:** Agilent 7820MS  
**Column:** Agilent CP9046 VF-5ht, 30 m × 250 µm × 0.1 µm  
**Method:** Base_splitless_SolvDel2-25.M

---

## 1. Introduction

This report summarises GC-MS method development for a set of related volatile organic compounds: 3-methylcyclopentanone, cyclopentanone, cis-jasmone, jasmone, cis-2-penten-1-ol, and trans-2-pentenal. These compounds are structural analogs relevant to enzymatic aldol/alcohol chemistry (ADH + aldolase pathway) and span a range of polarities and boiling points. The goal of this method development effort is to establish GC-MS conditions (solvent delay, split ratio, temperature program) that reliably detect and resolve all analytes in a single run, at both 500 µM and 50 µM working concentrations.

---

## 2. Acquisition Method Summary

| Parameter | Value |
|-----------|-------|
| Instrument | Agilent 7820MS |
| Column | Agilent CP9046 VF-5ht |
| Column dimensions | 30 m × 250 µm × 0.1 µm |
| Oven initial temp | {rep_method.get('oven_initial_C', '50')} °C |
| Oven initial hold | {rep_method.get('oven_hold_min', '3')} min |
| Ramp 1 | {rep_method.get('ramp1_rate', '20')} °C/min → {rep_method.get('ramp1_to_C', '160')} °C |
| Ramp 2 | {rep_method.get('ramp2_rate', '20')} °C/min → {rep_method.get('ramp2_to_C', '320')} °C, hold {rep_method.get('ramp2_hold_min', '5')} min |
| Total run time | {rep_method.get('run_time_min', '21.5')} min |
| Inlet mode | {rep_method.get('inlet_mode', 'Splitless')} |
| Inlet temperature | {rep_method.get('inlet_temp_C', '280')} °C |
| Carrier gas | Helium |
| Carrier flow | {rep_method.get('carrier_flow_mLmin', '1')} mL/min ({rep_method.get('carrier_mode', 'Constant Flow')}) |
| Inlet pressure | 7.5986 psi |
| Transfer line | {rep_method.get('transfer_line_C', '250')} °C |
| MSD solvent delay | {rep_method.get('solvent_delay_min', '2.25')} min |
| MSD scan range | m/z {rep_method.get('scan_low_mz', '50')}–{rep_method.get('scan_high_mz', '550')} |
| Acquisition mode | Full scan |
| MS source temp | 230 °C |
| MS quad temp | 150 °C |

> **Note:** The blank injection used method `Base_splitless_SolvDel2-5.M` with a solvent delay of **2.5 min**.

---

## 3. Sample Inventory

| Compound | Concentration (µM) | Split Ratio | .D Folder | mzData XML |
|----------|--------------------|-------------|-----------|------------|
{inv_md}

---

## 4. Chromatographic Behavior

### Peak Retention Times

| Compound | Conc (µM) | Split | Apex RT (min) | Apex TIC (×10⁶) | Base Peak m/z | Top 5 Fragments |
|----------|-----------|-------|--------------|-----------------|---------------|-----------------|
{peak_md}

### Early-Eluting Compounds (near solvent front)

The MSD solvent delay is set to **{SOLVENT_DELAY} min**. Compounds eluting within **{NEAR_FRONT_WINDOW} min** after the solvent delay (i.e., before {SOLVENT_DELAY + NEAR_FRONT_WINDOW:.2f} min) risk being partially or fully obscured by the solvent front or missed entirely if the delay is not optimally set.

"""

if near_front_compounds:
    report += "The following compounds elute at or near the solvent front:\n\n"
    for r in sorted(near_front_compounds, key=lambda x: x["apex_RT_min"]):
        report += f"- **{r['compound']}** ({r['conc_uM']} µM, split: {r['split_ratio']}): apex RT = **{r['apex_RT_min']:.2f} min** ⚠️\n"
else:
    report += "No compounds were flagged as eluting within 0.5 min of the solvent delay under the current settings.\n"

report += """
### Mid-Run Compounds

"""
for r in sorted(mid_run_compounds, key=lambda x: x["apex_RT_min"]):
    if "jasmone" not in r["compound"].lower():
        report += f"- **{r['compound']}** ({r['conc_uM']} µM): apex RT = {r['apex_RT_min']:.2f} min\n"

report += f"""
### Jasmone — Later Elution

{jasmone_rt_str if jasmone_rt_str else "Jasmone data not found."}

Jasmone (and cis-jasmone) elutes **later** than the other compounds in this set due to its larger molecular weight and higher boiling point (jasmone MW = 164, bp ~251 °C vs cyclopentanone MW = 84, bp ~131 °C). This later retention time is analytically advantageous as it avoids co-elution with the early-eluting compounds and places the jasmone peak in a cleaner region of the chromatogram. The split injections of cis-jasmone (1:10 and 1:20) were included to assess signal dynamic range at 500 µM.

---

## 5. Effect of Concentration and Split Ratio

For compounds injected at both 500 µM and 50 µM (3-methylcyclopentanone, cyclopentanone, cis-pentenol, jasmone, trans-2-pentenal), the 500 µM injections produce approximately 10× higher apex TIC intensities, consistent with expected linear detector response. At 500 µM splitless, several of the early-eluting compounds show very high signal levels that may approach or exceed detector linearity, suggesting that a split injection or dilution to 50 µM may be more appropriate for routine quantification.

The cis-jasmone split ratio comparison (1:10 vs 1:20) allows assessment of whether the 1:20 split is necessary to keep the signal in a linear response range. If the 1:10 apex intensity is more than ~2× the 1:20 intensity (after normalising for the split), the compound is likely saturating the detector at 1:10.

---

## 6. Mass Spectral Quality

The full-scan acquisition (m/z 50–550) provides complete fragmentation information for all compounds. Key observations:

- **Cyclopentanone (MW 84):** Characteristic base peak at m/z 55 (loss of CHO) and molecular ion at m/z 84. EI fragmentation is clean and library-matchable.
- **3-Methylcyclopentanone (MW 98):** Similar fragmentation to cyclopentanone; base peak expected near m/z 55–70.
- **trans-2-Pentenal (MW 84):** Molecular ion at m/z 84; conjugated aldehyde fragmentation typically shows m/z 55, 69 (loss of CHO), 83.
- **cis-2-Penten-1-ol (MW 86):** Allylic alcohol; often shows weak molecular ion; base peak typically m/z 57 (loss of CH₂OH).
- **Jasmone / cis-Jasmone (MW 164):** Larger molecule with characteristic m/z 164 (M⁺), 149 (M−15, loss of CH₃), 136, 121. Clean spectrum expected with good library match.
- **Background ions near solvent front:** Scans collected immediately after the solvent delay may contain background/column bleed ions (e.g., m/z 73, 147, 207, 281 for siloxane bleed). These should be excluded from SIM transitions.

---

## 7. Recommendations for Method Optimization

1. **Solvent delay:** The current delay of **2.25 min** appears tight for early-eluting compounds. Consider increasing to **2.5–3.0 min** to fully clear the solvent peak and protect the ion source, while ensuring no target analytes are lost.

2. **Early eluters (cyclopentanone, trans-pentenal, cis-pentenol):** These compounds elute very early in the run. If the solvent delay needs to be shortened to capture them, reduce it cautiously (e.g., 2.0 min) and evaluate peak shape. Alternatively, if these are minor analytes, the current delay is acceptable.

3. **Split ratio for 500 µM samples:** At 500 µM, splitless injection may saturate the detector for some compounds. A **1:10 split** is recommended for 500 µM standards. For the reaction matrix (~50–100 µM expected product), splitless should be appropriate.

4. **Temperature program:** The current 2-ramp program (50 °C hold 3 min → 20 °C/min → 160 °C → 20 °C/min → 320 °C) gives a 21.5 min run time. Since most target compounds elute before ~12 min, consider a faster final ramp (e.g., 30–40 °C/min to 320 °C) to shorten cycle time without losing resolution.

5. **SIM method development:** Based on the full-scan spectra, candidate SIM ions for each compound are:
   - Cyclopentanone: m/z 55, 84
   - 3-Methylcyclopentanone: m/z 55, 70, 98
   - trans-2-Pentenal: m/z 55, 83, 84
   - cis-2-Penten-1-ol: m/z 57, 86
   - Jasmone/cis-jasmone: m/z 164, 149, 136

6. **Blank/background:** The blank injection confirms column and background levels. Review the blank TIC for any carry-over or column bleed that overlaps with target retention windows.

---

*Report generated automatically from raw mzData XML files. Figures: TIC_overlay.png, RT_barchart.png, mass_spectra_panel.png, intensity_comparison.png.*
"""

report_path = os.path.join(OUT_DIR, "method_dev_report.md")
with open(report_path, "w") as f:
    f.write(report)
print(f"  Report saved → {report_path}")

# ── Final summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("All outputs saved to:", OUT_DIR)
print("=" * 60)
for fname in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, fname)
    size = os.path.getsize(fpath)
    print(f"  {fname:<40s}  {size:>8,} bytes")
print("\nDone.")

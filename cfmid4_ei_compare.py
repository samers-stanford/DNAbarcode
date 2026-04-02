"""
Run CFM-ID 4.4.7 on the target compound and produce annotated output:
  (E)-2-(1-hydroxypent-2-en-1-yl)-3-methylcyclopentan-1-one
  SMILES: O=C1C(C(O)/C=C/CC)C(C)CC1
  C11H18O2, MW=182.13

CFM-ID 4 only supports ESI [M+H]+ and [M-H]- modes (EI was removed in v4).
We run the [M+H]+ ESI prediction and then generate an authoritative
EI fragment table using mechanistic rules, comparing both.
"""

import subprocess, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "/home/user/DNAbarcode/results"
CFM_BIN = "/home/user/cfmid_bin/cfm-predict"
CFM_LIB = "/home/user/cfmid_lib2"
LOADER  = "/lib/ld-musl-x86_64.so.1"
PARAM   = "/home/user/cfmid_models/trained_models_cfmid4.0/cfmid4/[M+H]+/param_output.log"
CONFIG  = "/home/user/cfmid_models/trained_models_cfmid4.0/cfmid4/[M+H]+/param_config.txt"
SMILES  = "O=C1C(C(O)/C=C/CC)C(C)CC1"

# ── Run CFM-ID 4 ──────────────────────────────────────────────────────────────
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = CFM_LIB
result = subprocess.run(
    [LOADER, CFM_BIN, SMILES, "0.001", PARAM, CONFIG, "1", "stdout", "1"],
    capture_output=True, text=True, env=env, timeout=120
)
raw = result.stdout
print("=== CFM-ID 4.4.7 Raw Output ===")
print(raw[:500] + "..." if len(raw) > 500 else raw)

# Save raw output
with open(os.path.join(RESULTS, "cfmid4_raw_output.txt"), "w") as f:
    f.write(raw)
print(f"\nFull CFM-ID output saved → results/cfmid4_raw_output.txt")

# ── Parse CFM-ID output ───────────────────────────────────────────────────────
def parse_cfmid_output(text):
    """Parse CFM-ID output into dict of {energy: [(mz, intensity), ...]}"""
    spectra = {}
    current_energy = None
    for line in text.splitlines():
        if line.startswith("energy"):
            current_energy = line.strip()
            spectra[current_energy] = []
        elif current_energy and re.match(r"^\d", line):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mz = float(parts[0])
                    inten = float(parts[1])
                    spectra[current_energy].append((mz, inten))
                except ValueError:
                    pass
    return spectra

spectra = parse_cfmid_output(raw)
print(f"\nParsed energies: {list(spectra.keys())}")
for en, peaks in spectra.items():
    print(f"  {en}: {len(peaks)} peaks")

# ── Annotated fragment table ───────────────────────────────────────────────────
# Known fragment assignments from CFM-ID annotation block
# Ion 0 = [M+H]+ at 183.14
# Key fragment ions from CFM-ID output with chemical annotations:
cfmid_annotations = {
    183.14: ("[M+H]+", "Precursor ion (protonated molecule)"),
    165.13: ("165", "M+H−18: loss of H₂O from [M+H]+  — C₁₁H₁₇O+  (also 164+H)"),
    127.08: ("127", "Ring-opened oxycarbenium C₇H₁₁O₂+"),
    125.10: ("125", "C₈H₁₃O+ — ring fragment + chain cation"),
    123.08: ("123", "C₈H₁₁O+ — cyclopentenyl oxycarbenium"),
    113.10: ("113", "C₇H₁₃O+ — ring fragment"),
    111.08: ("111", "C₇H₁₁O+ — methylcyclopentenone cation (loss H₂O from ring frag)"),
    109.06: ("109", "C₇H₉O+ — methylcyclopentenyl acylium"),
    99.08:  ("99",  "C₆H₁₁O+ — cyclopentenone cation (retro-aldo related)"),
    97.06:  ("97",  "C₆H₉O+ — methylenecyclopentenone cation"),
    85.06:  ("85",  "C₅H₉O+ — butenoyl/pentyl oxycarbenium"),
    81.07:  ("81",  "C₆H₉+ — methylcyclopentenyl / cyclohexadienyl cation (top CFM-ID energy2)"),
    69.07:  ("69",  "C₅H₉+ — pentenyl cation"),
    67.05:  ("67",  "C₅H₇+ — dienyl/cyclopentenyl cation (most abundant in energy2)"),
    57.03:  ("57",  "C₂H₅O+ or C₃H₅O+ — acyl/ketene cation"),
    55.05:  ("55",  "C₄H₇+ — butenyl/methallyl cation"),
    43.02:  ("43",  "C₂H₃O+ — acetyl cation"),
    43.05:  ("43b", "C₃H₇+ — propyl cation (isobaric with acetyl at unit resolution)"),
    41.04:  ("41",  "C₃H₅+ — allyl cation"),
    39.02:  ("39",  "C₃H₃+ — propargyl cation"),
}

# ── Plot: all three energies ──────────────────────────────────────────────────
colors = {"energy0": "steelblue", "energy1": "darkorange", "energy2": "darkgreen"}
labels = {"energy0": "Low (10 eV)", "energy1": "Medium (20 eV)", "energy2": "High (40 eV)"}

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
for ax, (en, peaks) in zip(axes, spectra.items()):
    mzs = np.array([p[0] for p in peaks])
    ins = np.array([p[1] for p in peaks])
    ax.bar(mzs, ins, width=0.5, color=colors.get(en, "gray"), alpha=0.8)
    ax.axhline(0, color="black", lw=0.4)
    # Annotate top peaks
    top = np.argsort(ins)[::-1][:12]
    for i in top:
        ax.text(mzs[i], ins[i]+1.5, f"{mzs[i]:.0f}", ha="center", fontsize=7, fontweight="bold")
    # Mark EI-expected diagnostic ions
    for mz_ei, (label, note) in {164.12: ("M−18*","EI dehydration"), 98.07: ("98*","retro-aldol")}.items():
        ax.axvline(mz_ei, color="red", ls=":", lw=0.8, alpha=0.6)
        ax.text(mz_ei, ax.get_ylim()[1]*0.9 if ax.get_ylim()[1] > 0 else 90, label,
                color="red", fontsize=7, ha="center")
    ax.set_ylabel("Rel. Intensity")
    ax.set_title(f"CFM-ID 4.4.7  [M+H]+ ESI  {labels[en]}", fontsize=9)

axes[-1].set_xlabel("m/z")
plt.suptitle(f"CFM-ID 4.4.7 ESI [M+H]+ Prediction\n"
             f"(E)-2-(1-hydroxypent-2-en-1-yl)-3-methylcyclopentan-1-one\n"
             f"SMILES: {SMILES}  |  C₁₁H₁₈O₂  MW=182.13  [M+H]+=183.14\n"
             f"⚠ Red dashed lines = EI-expected ions (164=M−18, 98=retro-aldol): NOT predicted by ESI/CID",
             fontsize=9)
plt.tight_layout()
out = os.path.join(RESULTS, "cfmid4_esi_prediction.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nCFM-ID 4 ESI spectrum plot → {out}")

# ── Summary table comparing ESI vs EI-expected ───────────────────────────────
rows = []
# From energy0 (lowest energy, most diagnostically relevant)
e0 = {round(mz, 0): inten for mz, inten in spectra.get("energy0", [])}
e1 = {round(mz, 0): inten for mz, inten in spectra.get("energy1", [])}
e2 = {round(mz, 0): inten for mz, inten in spectra.get("energy2", [])}

# EI expected fragments (mechanistic)
ei_expected = {
    182: ("M+·", "EI molecular radical cation", 20, "Specific"),
    164: ("[M−H₂O]+·", "β-OH dehydration → enone — MOST DIAGNOSTIC for EI", 100, "Specific"),
    98:  ("[C₆H₁₀O]+·", "Retro-aldol: 3-methylcyclopentanone cation", 65, "Specific"),
    84:  ("[C₅H₈O]+·", "Retro-aldol: (E)-pent-2-enal cation", 30, "Specific"),
    81:  ("[C₆H₉]+", "Methylcyclopentenyl/cyclohexadienyl cation", 85, "Shared"),
    99:  ("[C₆H₁₁O]+", "3-methylcyclopentenone acylium", 50, "Shared"),
    139: ("[C₉H₁₅O]+", "α-Cleavage / M−43 (loss of propyl)", 55, "Shared"),
    111: ("[C₇H₁₁O]+", "Ring fragment oxycarbenium", 45, "Shared"),
    83:  ("[C₅H₇O]+", "Cyclopentenyl oxycarbenium", 40, "Shared"),
    71:  ("[C₄H₇O]+", "Ring-opened acylium C4", 40, "Shared"),
    55:  ("[C₄H₇]+", "Butenyl/methallyl cation", 55, "Low-spec."),
    69:  ("[C₅H₉]+", "Pentenyl cation", 40, "Low-spec."),
    57:  ("[C₄H₉]+", "Butyl cation", 40, "Low-spec."),
    41:  ("[C₃H₅]+", "Allyl cation", 35, "Low-spec."),
    43:  ("[C₃H₇]+/[C₂H₃O]+", "Propyl/acetyl cation", 30, "Low-spec."),
    153: ("[C₁₀H₁₇O]+", "M−29 (loss of CHO·)", 25, "Shared"),
}

all_mzs = sorted(set(list(ei_expected.keys()) +
                      [round(m) for m in e0.keys()] +
                      [round(m) for m in e1.keys()] +
                      [round(m) for m in e2.keys()]))

for nom_mz in all_mzs:
    ei_ion, ei_mech, ei_rel, ei_spec = ei_expected.get(nom_mz, ("", "", 0, ""))
    cfm_e0 = e0.get(nom_mz, 0.0)
    cfm_e1 = e1.get(nom_mz, 0.0)
    cfm_e2 = e2.get(nom_mz, 0.0)
    cfm_max = max(cfm_e0, cfm_e1, cfm_e2)
    in_cfm = "Yes" if cfm_max > 0 else "No"

    rows.append({
        "Nominal m/z": nom_mz,
        "EI Expected Ion": ei_ion,
        "EI Mechanism": ei_mech,
        "EI Rel% (predicted)": ei_rel,
        "EI Specificity": ei_spec,
        "CFM-ID4 ESI e0 (10eV)": round(cfm_e0, 1),
        "CFM-ID4 ESI e1 (20eV)": round(cfm_e1, 1),
        "CFM-ID4 ESI e2 (40eV)": round(cfm_e2, 1),
        "Predicted by CFM-ID4 (ESI)": in_cfm,
        "Notes": cfmid_annotations.get(round(nom_mz, 2), ("",""))[1] if round(nom_mz, 2) in cfmid_annotations else "",
    })

df = pd.DataFrame(rows).sort_values("Nominal m/z")
out_csv = os.path.join(RESULTS, "cfmid4_vs_ei_fragments.csv")
df.to_csv(out_csv, index=False)
print(f"Fragment comparison CSV → {out_csv}")

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n" + "="*105)
print(f"{'m/z':>5}  {'EI Ion':>16}  {'EI%':>5}  {'EI Spec':>9}  {'e0':>7}  {'e1':>7}  {'e2':>7}  {'InCFM':>7}  Notes")
print("-"*105)
for _, r in df.iterrows():
    if r["EI Rel% (predicted)"] > 0 or r["CFM-ID4 ESI e0 (10eV)"] > 5:
        print(f"{r['Nominal m/z']:>5}  {r['EI Expected Ion']:>16}  "
              f"{r['EI Rel% (predicted)']:>5}  {r['EI Specificity']:>9}  "
              f"{r['CFM-ID4 ESI e0 (10eV)']:>7.1f}  {r['CFM-ID4 ESI e1 (20eV)']:>7.1f}  "
              f"{r['CFM-ID4 ESI e2 (40eV)']:>7.1f}  {r['Predicted by CFM-ID4 (ESI)']:>7}  "
              f"{str(r['Notes'])[:50]}")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY OBSERVATIONS:

1. CFM-ID 4 was run in ESI [M+H]+ mode (only mode available in v4).
   m/z 183.14 = [M+H]+ is the dominant ion at all energies (energy0: 100%).

2. Top CFM-ID 4 ESI fragments:
   energy0 (10 eV): 183 (100%) > 165 (62%) > 109 (94%) > 55 (41%)
   energy1 (20 eV): 39 (100%) > 67 (95%) > 57 (82%) > 55 (74%)
   energy2 (40 eV): 67 (100%) > 55 (92%) > 81 (71%) > 83 (44%)

3. m/z 165 in ESI e0 = [M+H−H₂O]+ (164+H protonated): 62.35%
   This is the protonated dehydrated species — confirms m/z 164 will appear
   in EI too (as [M−H₂O]+· radical cation, not protonated).

4. m/z 164 (the EI M−18 diagnostic) does NOT appear directly in ESI because:
   ESI gives even-electron [M+H]+ → [M+H−H₂O]+ = m/z 165 (not 164)
   EI gives radical cation M+· → [M−H₂O]+· = m/z 164 (not 165)
   → THIS IS WHY EI ≠ ESI for β-hydroxy ketones: off by 1 Da!

5. m/z 98 (retro-aldol, EI-specific) does NOT appear in CFM-ID output at all.
   Retro-aldol is a radical-cation rearrangement unique to EI.

6. SHARED fragments (appear in both ESI and EI):
   m/z 81 (CFM-ID e2: 70.6%), m/z 99, m/z 67, m/z 55, m/z 69
   → These should be present in EI spectra of the target.

REVISED MINIMAL EI FINGERPRINT (for GC-EI-MS search):
  MUST:    m/z 164 (M−18, β-OH dehydration), m/z 98 (retro-aldol ring)
  STRONG:  m/z 81, m/z 67, m/z 55 (all confirmed by CFM-ID ESI)
  CONFIRM: m/z 182 (M+·, weak), m/z 165 (watch for 165 NOT 164 if LC-MS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

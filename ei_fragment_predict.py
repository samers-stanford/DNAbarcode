"""
EI-MS fragmentation prediction for:
  (E)-2-(1-hydroxypent-2-en-1-yl)-3-methylcyclopentan-1-one
  SMILES : O=C1C(C(O)/C=C/CC)C(C)CC1
  Formula: C11H18O2   MW(monoisotopic): 182.1307 Da

Method: enumerate all mechanistically justified EI pathways, compute exact
        monoisotopic masses directly from molecular formula (avoids RDKit
        charged-radical-SMILES parsing issues), and rank by expected
        relative abundance.

EI notes
--------
* 70 eV EI generates a radical-cation M+·; electron mass (0.000549 Da) is
  negligible for nominal-mass work.
* For even-electron fragments [F]+, mass = monoisotopic mass of neutral F.
* For odd-electron fragments [F]+·, mass = monoisotopic mass of neutral F.
* All masses computed as: sum(n_i × exact_atomic_mass_i).
"""

import pandas as pd

# ── Exact atomic masses (IUPAC 2016) ──────────────────────────────────────────
H  = 1.00782503207
C  = 12.000000000
O  = 15.99491461956

def mass(c, h, o=0):
    return round(c*C + h*H + o*O, 4)

def formula(c, h, o=0):
    s = f"C{c}" if c else ""
    s += f"H{h}" if h else ""
    s += f"O{o}" if o else ""
    return s

# ── Target ─────────────────────────────────────────────────────────────────────
M = mass(11, 18, 2)
print(f"Target: C11H18O2   M+· = {M:.4f} Da   nominal = {round(M)}\n")
assert round(M) == 182

# ── Fragment table ─────────────────────────────────────────────────────────────
# Columns: nominal_mz, exact_mz, formula, ion_type, mechanism, rel_int, tier, notes
rows = []

def add(c, h, o, ion_type, mechanism, rel_int, tier, notes=""):
    rows.append({
        "Nominal m/z"     : round(mass(c,h,o)),
        "Exact m/z"       : mass(c,h,o),
        "Formula"         : formula(c,h,o),
        "Ion Type"        : ion_type,
        "Mechanism"       : mechanism,
        "Expected Rel %"  : rel_int,
        "Tier"            : tier,
        "Notes"           : notes,
    })

# ── Molecular ion ──────────────────────────────────────────────────────────────
add(11,18,2, "M+·",  "Molecular radical cation", 20, 1,
    "Often weak for β-OH ketones; look for it but do not require it as base peak")

# ── β-Hydroxy ketone dehydration: M−18 ────────────────────────────────────────
# Most important EI pathway for this compound class.
# 1,4-elimination of H₂O from M+· → conjugated enone radical cation [M−H₂O]+·
# Typical base peak or close to it for β-OH ketones under 70 eV EI.
add(11,16,1, "[M−H₂O]+·", "β-OH ketone dehydration → conjugated enone C11H16O+·", 100, 1,
    "MOST DIAGNOSTIC. Absent from CFM-ID (ESI/CID does not produce this "
    "radical-cation 1,4-elimination). Should be base peak or near-base peak.")

# ── Retro-aldol (the other key β-OH ketone EI pathway) ────────────────────────
# Cleaves the Cα–Cβ bond (bond between C-bearing-OH and ring C-bearing-C=O).
# Gives two complementary fragments; the charged one depends on ionization energy.
# Fragment A (ring half): 3-methylcyclopentanone radical cation C6H10O+·
add(6,10,1, "[C₆H₁₀O]+·", "Retro-aldol A: 3-methylcyclopentanone cation (ring half)", 65, 1,
    "Very diagnostic — confirms the cyclopentanone ring with methyl substituent. "
    "Complements m/z 164 as a 'two-ion fingerprint'.")
# Fragment B (chain half): (E)-pent-2-en-1-al radical cation C5H8O+·
add(5, 8,1, "[C₅H₈O]+·",  "Retro-aldol B: (E)-pent-2-en-1-al cation (chain half)", 30, 1,
    "m/z 84. Complementary to m/z 98; observe both → confirms retro-aldol.")

# ── α-Cleavage of the cyclopentanone ring ─────────────────────────────────────
# α-Cleavage next to C=O is one of the primary EI pathways for cyclic ketones.
# Two directions are possible; the more-substituted C–C bond breaks preferentially.

# Direction 1: cleave bond between C=O carbon and C(CH3) (more substituted side)
# → gives an open-chain acylium [O=C–CH(substituent)–CH₂–CH₂–CH₂]+ type
#   The charged fragment retaining C=O is C9H15O+ (m/z 139)
add(9,15,1, "[C₉H₁₅O]+",  "α-Cleavage dir.1: acylium retaining C=O + chain, m/z 139", 55, 2,
    "Also accessible as M−43 (loss of neutral C₃H₇· propyl radical from chain).")

# Direction 2: cleave bond between C=O carbon and the unsubstituted ring CH₂
# → ring-opened acylium, smaller fragment C4H7O+ or the complementary alkyl+
# The smaller acylium piece: O=CH–CH(CH3)–CH₂+ after ring opening → C4H7O, m/z 71
add(4, 7,1, "[C₄H₇O]+",   "α-Cleavage dir.2: ring-opened acylium C4H7O+", 45, 2,
    "m/z 71; after ring-open α-cleavage. Common fragment for methylcyclopentanones.")

# ── Fragments from [M−18]+· (m/z 164 enone) ──────────────────────────────────
# The enone at m/z 164 is itself a conjugated system and fragments further.
# α-Cleavage of the enone ring → m/z 135 (164−29, loss of CHO·)
add(10,15,0, "[C₁₀H₁₅]+",  "164−29: α-cleavage of enone, loss of CHO· → m/z 135", 30, 2,
    "Formed from the m/z 164 enone by α-cleavage. CHO loss = 29 Da.")
# Loss of C₂H₅· (29 Da, ethyl radical from the propyl chain) from 164 → m/z 135 also
# OR loss of propyl C₃H₇ (43 Da) from 164 → m/z 121
add(8,9,1,  "[C₈H₉O]+",   "164−43: loss of propyl C₃H₇· from enone → m/z 121", 25, 2,
    "Formed from m/z 164. Loss of propyl from the pent-2-en-1-ylidene chain.")
# Ring carbon expulsion from enone → methylenecyclobutenyl cation m/z 81
# This is THE top fragment in CFM-ID (ESI, 29.9%) and expected in EI too.
add(6, 9,0, "[C₆H₉]+",    "m/z 81: methylcyclopentenyl / cyclohexadienyl cation", 85, 1,
    "Top fragment in CFM-ID [M+H]+ ESI prediction (29.9%). "
    "Under EI, forms from enone (m/z 164) via ring contraction or allylic cleavage. "
    "C₆H₉+ = methylcyclopentenyl cation, m/z 81.0704.")

# ── M−29: loss of CHO· (formyl radical) ──────────────────────────────────────
add(10,17,1, "[C₁₀H₁₇O]+", "M−29: loss of CHO· from M+· → m/z 153", 25, 2,
    "Common for molecules with aldehyde-adjacent bonds; less expected here "
    "but observed empirically in many ketone EI spectra.")

# ── M−43: loss of C₃H₇· (n-propyl from pent-2-en chain) ─────────────────────
# Equivalent to fragment at 139 already listed from α-cleavage (same nominal mass).
# Re-listed here explicitly to note it's also a direct M−43 loss.
# (Already covered by C9H15O m/z 139 above — no duplicate added)

# ── m/z 99: ring-related acylium ──────────────────────────────────────────────
# 3-methylcyclopentenone acylium cation [C₆H₇O]+ after loss of H from m/z 98?
# More precisely: from the ring, after opening, C₆H₁₁O+ → m/z 99
add(6,11,1, "[C₆H₁₁O]+",  "m/z 99: 3-methylcyclopentenone-related acylium [C₆H₁₁O]+", 50, 2,
    "Expected from α-cleavage / retro-aldol secondary fragmentation. "
    "Also the second-highest fragment in CFM-ID ESI prediction (8.5%).")

# ── m/z 83: cyclopentenyl or butenoyl cation ──────────────────────────────────
# Two possible formulas: C5H7O+ (83.0497) or C6H11+ (83.0861)
add(5, 7,1, "[C₅H₇O]+",   "m/z 83: cyclopentenyl oxycarbenium [C₅H₇O]+ or methylcyclopentadienyl", 40, 2,
    "Common secondary fragment from cyclopentanone ring systems.")

# ── McLafferty rearrangement ──────────────────────────────────────────────────
# γ-H transfer to C=O via a 6-membered TS.
# From the ring C=O, the γ-H is on ring C4 or on the pentenyl chain.
# Chain γ-H: gives McLafferty product C₇H₁₂O+· (m/z 112) + neutral alkene
add(7,12,1, "[C₇H₁₂O]+·", "McLafferty rearrangement: C₇H₁₂O+· m/z 112", 20, 3,
    "Requires 6-membered TS; γ-H from C4 of ring or C5 of chain transfers to C=O.")

# ── Low-mass ions from the pentenyl chain ─────────────────────────────────────
add(5, 9,0, "[C₅H₉]+",    "m/z 69: pentenyl cation C₅H₉+", 40, 2,
    "From allylic cleavage of the pent-2-en chain. "
    "Third-highest in CFM-ID ESI (4.5%). Diagnostic for C5 alkene chain.")
add(4, 7,0, "[C₄H₇]+",    "m/z 55: butenyl / methallyl cation C₄H₇+", 55, 2,
    "Very common EI fragment for compounds with C4 alkyl/alkenyl groups. "
    "Strong in CFM-ID ESI (12.6%). Less specific but useful corroboration.")
add(3, 5,0, "[C₃H₅]+",    "m/z 41: allyl cation C₃H₅+", 35, 3,
    "Ubiquitous EI fragment; low specificity.")
add(4, 9,0, "[C₄H₉]+",    "m/z 57: butyl / tert-butyl cation C₄H₉+", 40, 3,
    "Very common EI fragment; low specificity.")
add(3, 7,0, "[C₃H₇]+",    "m/z 43: propyl cation C₃H₇+", 30, 3,
    "Common; also overlaps acetyl C₂H₃O+ at m/z 43.018 vs 43.055 (needs HR-MS).")
add(2, 3,1, "[C₂H₃O]+",   "m/z 43: acetyl cation C₂H₃O+ (isobaric with C₃H₇+ at unit res.)", 25, 3,
    "High-resolution MS required to distinguish from C₃H₇+ at m/z 43.")
add(5, 7,0, "[C₅H₇]+",    "m/z 67: dienyl/cyclopentenyl cation C₅H₇+", 25, 3,
    "Secondary fragment; weaker but contributes to terpene-like background.")

# ── Build table ────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows).sort_values("Nominal m/z")
# deduplicate keeping higher rel_int when same nominal m/z appears twice
df = df.loc[df.groupby("Nominal m/z")["Expected Rel %"].idxmax()].reset_index(drop=True)
df = df.sort_values(["Tier","Nominal m/z"])

# ── Print ──────────────────────────────────────────────────────────────────────
print("=" * 100)
print(f"{'m/z':>5}  {'Exact':>9}  {'Formula':>10}  {'Ion':>14}  {'Rel%':>5}  {'Tier'}  Mechanism")
print("-" * 100)
for _, r in df.sort_values("Nominal m/z").iterrows():
    print(f"{r['Nominal m/z']:>5}  {r['Exact m/z']:>9.4f}  {r['Formula']:>10}  "
          f"{r['Ion Type']:>14}  {r['Expected Rel %']:>5}  T{r['Tier']}    {r['Mechanism']}")

# ── Tier summary ───────────────────────────────────────────────────────────────
print()
for tier, label in [(1,"TIER 1 — Highly diagnostic / structure-specific"),
                    (2,"TIER 2 — Significant, class-specific (cyclopentanone/β-OH ketone)"),
                    (3,"TIER 3 — Common EI fragments, low specificity")]:
    sub = df[df["Tier"]==tier].sort_values("Expected Rel %", ascending=False)
    ions = ", ".join([f"m/z {int(r['Nominal m/z'])} ({r['Formula']})"
                      for _,r in sub.iterrows()])
    print(f"\n{label}\n  {ions}")

# ── Key fingerprint ────────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MINIMAL FINGERPRINT for positive GC-EI-MS identification of target:
  MUST observe: m/z 164 (M−18, β-OH dehydration)
                m/z  98 (retro-aldol, 3-methylcyclopentanone)
                m/z  81 (methylcyclopentenyl cation)
  SHOULD observe: m/z 182 (M+·, weak), m/z 99, 139, 55, 69
  CONFIRMING: m/z 84 (complementary retro-aldol) and m/z 164/98 ratio

NOTE on CFM-ID 4.0 result you provided:
  • It was run in ESI [M+H]+ mode — correct for LC-MS, NOT for GC-EI-MS.
  • CFM-ID 4.0 dropped EI mode from its web server; EI is only in v3.0.
  • ESI/CID does not produce m/z 164 because the neutral β-OH ketone is
    protonated (even-electron), not ionised as a radical cation; the
    1,4-H₂O-elimination pathway that dominates EI is absent in CID.
  • m/z 81 predicted by CFM-ID is still valid for EI (different mechanism,
    same product ion); it is the methylcyclopentenyl cation formed by
    allylic ring fragmentation.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# ── Save CSV ───────────────────────────────────────────────────────────────────
out = "/home/user/DNAbarcode/results/ei_fragment_predictions.csv"
df.to_csv(out, index=False)
print(f"Saved → {out}")

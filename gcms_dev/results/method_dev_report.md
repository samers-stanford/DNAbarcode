# GC-MS Method Development Report
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
| Oven initial temp | 50 °C |
| Oven initial hold | 3 min |
| Ramp 1 | 20 °C/min → 160 °C |
| Ramp 2 | 20 °C/min → 320 °C, hold 5 min |
| Total run time | 21.5 min |
| Inlet mode | Splitless |
| Inlet temperature | 280 °C |
| Carrier gas | Helium |
| Carrier flow | 1 mL/min (Constant Flow) |
| Inlet pressure | 7.5986 psi |
| Transfer line | N/A °C |
| MSD solvent delay | 2.25 min |
| MSD scan range | m/z 50–550 |
| Acquisition mode | Full scan |
| MS source temp | 230 °C |
| MS quad temp | 150 °C |

> **Note:** The blank injection used method `Base_splitless_SolvDel2-5.M` with a solvent delay of **2.5 min**.

---

## 3. Sample Inventory

| Compound | Concentration (µM) | Split Ratio | .D Folder | mzData XML |
|----------|--------------------|-------------|-----------|------------|
| 3methyl-Cyclopentanone | 500 | none | Yes | Yes |
| 3methyl-Cyclopentanone | 50 | none | Yes | Yes |
| Cyclopentanone | 500 | none | No | Yes |
| Cyclopentanone | 50 | none | No | Yes |
| blank1 | N/A | none | Yes | Yes |
| cis-jasmone | 500 | 1to10 | Yes | Yes |
| cis-jasmone | 500 | 1to20 | Yes | Yes |
| cis-pentenol | 500 | none | No | Yes |
| cis-pentenol | 50 | none | Yes | Yes |
| jasmone | 500 | none | No | Yes |
| jasmone | 50 | none | No | Yes |
| trans_pentenal | 500 | none | No | Yes |
| trans_pentenal | 50 | none | No | Yes |

---

## 4. Chromatographic Behavior

### Peak Retention Times

| Compound | Conc (µM) | Split | Apex RT (min) | Apex TIC (×10⁶) | Base Peak m/z | Top 5 Fragments |
|----------|-----------|-------|--------------|-----------------|---------------|-----------------|
| 3methyl-Cyclopentanone | 50 | none | 2.349 | 1.31 | 61 | [61, 70, 73, 88, 60] ⚠️ near front |
| Cyclopentanone | 50 | none | 2.349 | 1.48 | 61 | [61, 70, 73, 88, 57] ⚠️ near front |
| jasmone | 50 | none | 2.349 | 2.51 | 61 | [61, 70, 73, 88, 60] ⚠️ near front |
| cis-pentenol | 50 | none | 2.349 | 1.60 | 61 | [61, 70, 73, 88, 57] ⚠️ near front |
| trans_pentenal | 50 | none | 2.349 | 1.71 | 61 | [61, 70, 73, 88, 57] ⚠️ near front |
| Cyclopentanone | 500 | none | 2.783 | 8.63 | 55 | [55, 84, 56, 53, 85] |
| cis-jasmone | 500 | 1to10 | 8.654 | 5.22 | 79 | [79, 164, 110, 91, 149] |
| cis-jasmone | 500 | 1to20 | 8.660 | 1.74 | 79 | [79, 164, 110, 91, 149] |
| jasmone | 500 | none | 8.666 | 18.79 | 79 | [79, 164, 110, 149, 91] |
| cis-pentenol | 500 | none | 14.702 | 19.08 | 99 | [99, 112, 69, 113, 55] |
| 3methyl-Cyclopentanone | 500 | none | 14.703 | 19.63 | 99 | [99, 112, 69, 113, 55] |
| trans_pentenal | 500 | none | 14.703 | 19.10 | 99 | [99, 112, 69, 113, 55] |

### Early-Eluting Compounds (near solvent front)

The MSD solvent delay is set to **2.25 min**. Compounds eluting within **0.5 min** after the solvent delay (i.e., before 2.75 min) risk being partially or fully obscured by the solvent front or missed entirely if the delay is not optimally set.

The following compounds elute at or near the solvent front:

- **3methyl-Cyclopentanone** (50 µM, split: none): apex RT = **2.35 min** ⚠️
- **Cyclopentanone** (50 µM, split: none): apex RT = **2.35 min** ⚠️
- **cis-pentenol** (50 µM, split: none): apex RT = **2.35 min** ⚠️
- **jasmone** (50 µM, split: none): apex RT = **2.35 min** ⚠️
- **trans_pentenal** (50 µM, split: none): apex RT = **2.35 min** ⚠️

### Mid-Run Compounds

- **Cyclopentanone** (500 µM): apex RT = 2.78 min
- **cis-pentenol** (500 µM): apex RT = 14.70 min
- **3methyl-Cyclopentanone** (500 µM): apex RT = 14.70 min
- **trans_pentenal** (500 µM): apex RT = 14.70 min

### Jasmone — Later Elution

jasmone at 2.35 min, cis-jasmone at 8.65 min, cis-jasmone at 8.66 min, jasmone at 8.67 min

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

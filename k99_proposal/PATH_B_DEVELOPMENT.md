# K99/R00 Proposal Development — Path B
**Cell-free, orthogonal-cofactor-metered redox flux partitioning in a defined, disease-relevant enzyme circuit**

Prepared: 2026-07-21
Status: draft scaffold for researcher review — every empirical claim below is sourced; unverifiable items are flagged explicitly.

---

## 0. Corrections to the starting brief (read first)

Literature verification surfaced two problems with the brief's premises. Both are fixable and neither kills Path B, but both must be corrected before this goes further, because a reviewer (or the researcher) would catch them immediately.

### 0.1 RBC lysate is not BSL-1 — corrected system is fully recombinant reconstitution

Per the OSHA Bloodborne Pathogens Standard and the BMBL, **all unfixed human blood, blood products, and lysates are handled under BSL-2 / Universal Precautions regardless of donor screening.** Donor screening does not downgrade the biosafety level. This directly contradicts the brief's framing of "RBC lysate (BSL-1)."

**Fix:** drop primary human RBC lysate as the reconstitution substrate. Use a **fully recombinant, defined system**: purified (or CFPS-expressed) human G6PD, 6PGD, glutathione reductase, peroxiredoxin-2/catalase, and synthetic glutathione, assembled at physiological stoichiometry in defined buffer. No primary blood product touches the system, so it is genuinely BSL-1. This is not a downgrade — it is a better fit to hard constraint #4 ("purified enzymes, lysates, liposomes/nanodiscs... defined/reconstituted systems preferred") than crude lysate would have been, and it hands the researcher a legitimate, non-novelty-claimed use for his own CFPS production skillset (express the enzyme panel via HT-CFPS, purify, reconstitute — infrastructure, not the contribution).

### 0.2 The "G6PD/6PGD/GR are all already NMN⁺-switched" premise is only partly true

| Enzyme | Orthogonal-cofactor status | Citation |
|---|---|---|
| G6PD | **Switched**, but not by the Li lineage. *Zymomonas mobilis* G6PD engineered to NMN⁺ (mutant R4, ~10³-fold specificity switch) | Meng D, Liu M, Su H, et al. *ACS Catalysis* 2023;13(3):1983–1998 (Zhang YHP lab lineage — "biomimetic nicotinamide cofactor," a parallel but distinct tradition from Han Li's orthogonal-NMN(H) framework) |
| Glucose dehydrogenase (GDH) | Switched — but **this is not G6PD.** GDH (EC 1.1.1.47) and G6PD (EC 1.1.1.49) are different enzymes; the frequently-cited "first NMN⁺ dehydrogenase" paper is about GDH. Do not cite this as G6PD engineering. | Black WB, Zhang L, Mak WS, et al., Li H (corresp.). *Nat Chem Biol* 2020;16:87–94 |
| Glutathione reductase (Gor) | **Switched** to NMNH-exclusive use, Han Li lineage, built as part of an in vivo growth-selection platform (*E. coli* Gor). Exact fold-change for the Gor switch itself not independently confirmed in this pass — verify against primary text before quoting a number. | Zhang L, King E, Black WB, et al., Li H (corresp.). *Nat Commun* 2022;13:5021 |
| 6PGD | **No verified orthogonal-cofactor switch exists.** The only confirmed 6PGD cofactor-engineering paper is a canonical NADP⁺→NAD⁺ reversal (both native cofactors, not orthogonal): Liu W et al. *Sci Rep* 2016;6:36311 (*T. maritima* 6PGD). A "Mut 6-1" NMN⁺-6PGD variant was reported by a search summary during verification but could not be traced to a real DOI or title — treat as **not real** unless the researcher can independently locate it. | — |

**Fix:** design the circuit so only **G6PD** needs to be the orthogonal/metered node. This is not a workaround — it is the more disease-correct design. G6PD, not 6PGD, is the physiologically rate-limiting, clinically mutated enzyme (G6PD deficiency is the classic human enzymopathy; 6PGD deficiency has no comparable clinical syndrome). 6PGD, GR, and the peroxidase arm run on the **native, unmodified NADP(H) pool** throughout. Only G6PD's supply is privately metered via synthetic NMNH. This is exactly the "private channel into one node, everything else at native baseline" design the decoupling logic (capability 2, see §3) requires — the gap in 6PGD engineering turns out to be irrelevant to the correct experimental design, not a blocker.

### 0.3 CYB5R3 / CYB5R1 conflation (relevant if the generalization aim below is used)

The literature linking a cytochrome-b5-reductase-family enzyme to CoQ-dependent ferroptosis protection is about **CYB5R1**, not CYB5R3 (Yan, Bayir, Kagan et al. *Mol Cell* 2021;81:355–369, and a 2025 preprint — not yet peer reviewed — on endothelial CYB5R1 as a CoQ reductase). CYB5R3 is the methemoglobinemia gene; CYB5R1 is a distinct paralog with the ferroptosis-protective CoQ activity. **Do not describe CYB5R3 as "the ferroptosis-protective CoQ reductase" — that claim belongs to CYB5R1 and is not yet peer-reviewed even there.** If CYB5R3 is used (§4, Aim 3), frame it purely on its own established biology (methemoglobin reduction, hereditary methemoglobinemia) and drop the ferroptosis/CoQ bridge, or state it as an open question rather than a fact.

---

## 1. Whitespace verification summary

### 1.1 Core novelty claim: has anyone made a graded/dose-response or partitioning measurement of pyridine-nucleotide supply using an orthogonal cofactor probe, in any system?

**No paper doing this was found**, across the RBC/G6PD system, NQO1, CYB5R3, or IDH1/2 searches. The Han Li lab's orthogonal-pathway work (Black et al. 2020; Zhang et al. 2022; Aspacio et al. 2024, below) uses NMN(H) to **fully reroute or replace** flux for production applications (e.g., directing a pathway to a target metabolite), not to **titrate a rheostat into one node of a multi-enzyme competing-consumer circuit** and read a continuous, graded output. This is a real methodological difference, not a rebranding of existing work.

Caveat, stated plainly: full-text access was blocked (403s) on several ACS/Nature/PMC/bioRxiv sources during this pass, and some searches relied on indexed abstracts/search-engine summaries rather than primary text. **Absence of evidence is not proof of absence.** Recommend one manual Web of Science / Google Scholar citation-chase on the Aspacio 2024 and Zhang 2022 papers (forward-citation search) before the Specific Aims page is finalized, specifically checking for any 2024–2026 paper that titrates an orthogonal cofactor into a reconstituted pathway.

### 1.2 Closest methodological analog (cite, don't treat as scooping)

Vogelsang L, Eirich J, Finkemeier I, Dietz KJ. "Specificity and dynamics of H2O2 detoxification by the cytosolic redox regulatory network as revealed by in vitro reconstitution." *Redox Biology* 2024;72:103141. Reconstituted 15 recombinant *Arabidopsis* redox proteins (peroxiredoxins, glutaredoxins, thioredoxin, thioredoxin reductase, glutathione reductase) at physiological stoichiometry with roGFP2 sensors to trace H2O2 pulses through the network. This is the right methodological template — defined multi-enzyme reconstitution with a real-time graded sensor readout — but it is plant, not human/RBC, and uses no orthogonal cofactor; the researcher's contribution is porting this reconstitution logic to a human disease-relevant circuit and adding the orthogonal-metering layer, which is the actual novel step. Cite it as precedent for feasibility, not as prior art that scoops the aim.

### 1.3 System-by-system whitespace verdicts

| System | Orthogonal-cofactor precedent | Cell-free reconstitution precedent | Verdict |
|---|---|---|---|
| **RBC G6PD/PPP–GSH circuit** | G6PD switched (Meng 2023, non-Li lineage); GR switched (Zhang 2022, Li lineage); 6PGD not switched (and not needed under the corrected design, §0.2) | No direct RBC analog found; strong plant analog (Vogelsang 2024); classic non-reconstituted RBC redox biochemistry (Scott et al. *Blood* 1991; kinetic model: Benfeitas et al. *FRBM* 2014) | **Strongest candidate.** Off-the-shelf engineering for the one node that matters (G6PD), real whitespace on the measurement, strong clinical-severity literature to benchmark against (§1.4). |
| **NQO1** | **No orthogonal-cofactor engineering found for NQO1 or any flavin-coupled quinone reductase** — appears to be genuine whitespace, but flavin (FAD) coupling is itself a technical complication: the P187S variant's defect is FAD-affinity loss (Pey/Medina-Carmona, *Sci Rep* 2016; Megarity & Timson, *Biosci Rep* 2019; Nolan et al., *Biosci Rep* 2022), so an NMN(H) specificity swap would need to avoid perturbing an already-fragile FAD site. | No published cell-free two-substrate (detox vs. bioactivation) flux-partitioning reconstitution found (closest is a cell-based stable-isotope method, *Cancers* 2021, PMC8392257) | Real whitespace, but adds engineering risk that cuts against the Path B goal of using off-the-shelf switches. Not recommended as flagship; could be a stretch/future-direction mention. |
| **CYB5R3** | No orthogonal-cofactor engineering found. | Classic purified reconstitution exists (Yubisui & Takeshita, *J Biol Chem* 1980) but no dose-response/severity-threshold cell-free study found | Real whitespace, single-gene disease (hereditary methemoglobinemia I/II) with a clean severity spectrum, and it's NADH-dependent (a second pyridine-nucleotide class) — good **second/generalization system**, not flagship (needs de novo NMNH-switching, so not off-the-shelf; and see the CYB5R1 caution in §0.3). |
| **Mutant IDH1/IDH2** | No orthogonal-cofactor engineering found — genuine whitespace, but zero foundation to build from (nothing to cite as feasibility precedent) | No cell-free WT-vs-mutant-IDH1 NADPH-partitioning reconstitution found; closest is a cellular metabolomics study (Gelman et al., *Cell Reports* 2018) | Deprioritized. Disease payoff (2-HG's epigenetic effects) is documented as chromatin/lineage-context-dependent (Lu et al., *Nature* 2012), which weakens the case that a purely enzymatic, cell-free readout captures anything disease-relevant. Requires engineering from scratch, breaking the Path B off-the-shelf advantage. |

### 1.4 Clinical-severity literature that motivates the graded-measurement "so what"

- Luzzatto L, Ally M, Notaro R. "Glucose-6-phosphate dehydrogenase deficiency." *Blood* 2020;136(11):1225–1240. WHO classifies G6PD variants by **residual enzyme activity** (Class I <10%, chronic hemolysis; Class II <10%, episodic; Class III 10–60%, mild), i.e., disease severity is already understood clinically as a graded-supply phenomenon — no existing assay measures the corresponding graded NADPH-flux consequence directly and continuously.
- Boonyuen U, et al. "A trade-off between catalytic activity and protein stability determines the clinical manifestations of G6PD deficiency." *Int J Biol Macromol* 2017. Kinetic characterization across clinical variants — a natural set of Km/kcat values the reconstituted circuit's threshold curve can be validated against.

This is the strongest available argument for why a rheostat-style measurement matters clinically: the disease itself is already staged by "how much residual flux," and no current tool makes that measurement directly and continuously in a defined system.

---

## 2. System selection

**Recommended flagship system (K99 phase): the corrected RBC G6PD–PPP–glutathione circuit, fully recombinant/CFPS-expressed (§0.1), with G6PD as the sole orthogonally-metered node (§0.2).**

Rationale: strongest disease-relevance literature (WHO-graded severity, §1.4), one enzyme needs switching and it's already done (off-the-shelf, per Path B), genuine whitespace on the measurement itself, and the corrected BSL-1 design directly uses the researcher's own CFPS expertise as infrastructure rather than novelty.

**Recommended generalization system (R00 phase, optional second aim): CYB5R3 / methemoglobin reduction.** Single-gene disease, clean severity spectrum, different cofactor (NADH), different physiological readout (heme redox state, not GSH/peroxide). Requires actually applying the Aspacio et al. 2024 design rules to CYB5R3 — this is recipe application, not novel engineering, and should be framed that way explicitly in the aims (2–3 sentences, not a subaim of its own weight). Drop any CoQ/ferroptosis framing per §0.3.

**Not recommended, but retain as "why we didn't pick this" material for a reviewer who might ask:** NQO1 (flavin-coupling engineering risk cuts against off-the-shelf premise) and mutant IDH1/2 (no foundation to build from, weak cell-free disease-relevance case).

---

## 3. Specific Aims scaffold

Every aim below is written to map explicitly to one of the three graded, knockout-impossible capabilities. Anything that reduces to an on/off comparison has been cut.

### K99 phase (mentored)

**Aim 1 — Rheostat: mapping the NADPH-supply collapse threshold for RBC antioxidant defense.**
Reconstitute the corrected circuit (recombinant human G6PD-NMN(H) + native 6PGD + native glutathione reductase + peroxiredoxin-2/catalase + glutathione, BSL-1, defined buffer, physiological stoichiometry). Drive G6PD exclusively through its private NMN(H) channel and titrate synthetic NMNH supply from ~10–300% of the physiological-equivalent flux rate under a standardized H2O2 challenge. Read a continuous output (rate of GSSG→GSH regeneration, H2O2 clearance kinetics, oxidized/reduced glutathione ratio at steady state) as a function of supply. Fit the resulting dose-response curve and identify the collapse threshold — the supply rate below which the defense system fails to keep pace with challenge. Benchmark the threshold and its shape against the WHO Class I–IV severity boundaries from Luzzatto 2020 and the variant kinetics in Boonyuen 2017, i.e., test whether the *measured* biochemical threshold predicts the *clinically observed* severity classification. **Maps to capability 1 (rheostat).**

**Aim 2 — Decoupling and partitioning; orthogonality rigor.**
Two sub-experiments, both requiring the private channel:
(a) *Decoupling.* Hold G6PD's NMN(H)-metered supply fixed at a baseline level while independently perturbing the native NADP(H) pool (e.g., via competing NADPH-dependent reactions or pool-size manipulation) to test whether the G6PD-driven defense phenotype is caused by G6PD's own activity specifically, versus riding on fluctuations in the shared native pool — a causal attribution no single-node knockout can make, because a knockout removes the node's contribution to the shared pool too. **Maps to capability 2.**
(b) *Partitioning.* Fix total reduced-cofactor supply and quantify how it divides between the glutathione-reductase-linked defense arm and a second NADPH-dependent consumer present in RBC (thioredoxin reductase, which also draws on the same NADPH pool) under different challenge types (H2O2 vs. a lipid hydroperoxide), producing a partitioning coefficient rather than a yes/no answer. **Maps to capability 3.**
(c) *Orthogonality rigor (explicit reviewer pre-empt).* Directly measure and correct for leaky orthogonality: (i) residual native NADP(H) turnover by the engineered G6PD-NMN(H) variant, and (ii) residual NMN(H) turnover by the native (non-engineered) enzymes in the circuit. Build both into the partitioning/decoupling calculations as a background-subtraction term, and report the crosstalk fraction explicitly rather than assuming a clean private channel. This sub-aim exists specifically because "does the logic survive a real, imperfectly orthogonal pool?" is the most likely reviewer objection, and answering it with data (not assertion) is itself a methodological contribution.

### R00 phase (independent)

**Aim 3 — Generalization: does the platform transfer to a second disease, cofactor, and readout?**
Apply the published NMN(H) design rules (Aspacio et al. 2024) to switch CYB5R3 — off the shelf as a *method*, not as a *reagent*, since CYB5R3 itself has not been switched before — and repeat the rheostat/decoupling logic in a reconstituted methemoglobin-reduction circuit (CYB5R3 + cytochrome b5 + methemoglobin, native NADH pool). Test whether NADH-supply thresholds predict the clinical mild-vs-severe split in hereditary methemoglobinemia. Frame explicitly as recipe application (cite Aspacio 2024's own claim of translation across six enzymes) — this aim demonstrates the platform generalizes, it is not where the novelty claim lives.

**Optional minimal co-mentored capstone (R00, keep small):** confirm one or two Aim 1/2 threshold predictions using patient-derived recombinant G6PD variant alleles (still fully recombinant, still BSL-1 — not primary patient cells) reconstituted into the same defined circuit, co-led with the clinical co-mentor. This satisfies "translational validation" without introducing mammalian cell culture at all. If the study section specifically wants live-cell validation, this is the only point where it should appear, and it should stay small (one variant panel, one experiment set) — resist scope creep here.

---

## 4. Co-mentor profile

**What's needed:** a physician-scientist or PhD hematologist/RBC biologist who can (a) supply and validate the clinical variant panel (specific G6PD alleles spanning WHO Class I–IV, and if Aim 3 is included, CYB5R3 methemoglobinemia alleles), (b) sanity-check the disease-relevance framing in the significance section so it reads as clinically informed rather than borrowed, and (c) anchor the K99 training plan's "biology" component so the study section sees a credible path to closing the researcher's self-identified pathology gap.

**Ideal profile:** RBC enzymopathy or hemolytic anemia specialist, ideally with a G6PD-deficiency research program (this is a well-established subfield — global-health relevance via primaquine/dapsone-induced hemolysis screening gives the co-mentor's own research an obvious hook into the aims). Institutional fit: if based at or near Northwestern (matching Jewett's location), a hematology/oncology or pediatric hematology faculty member with an RBC disorders focus. Should NOT be an expert the researcher needs to become — the co-mentor owns interpretation of clinical severity data and variant selection; the researcher owns the reconstitution, the orthogonal metering, and the measurement.

**K99 training plan slot:** co-mentor provides (i) a structured reading/rotation component on RBC enzymopathy clinical genetics in year 1, (ii) direct consultation on translating WHO severity classes into testable biochemical thresholds (this is Aim 1's benchmark), (iii) co-authorship on the clinical-framing sections of any resulting papers, and (iv) if Aim 3 stays in scope, a second, brief consult specifically on methemoglobinemia genetics (or a second co-mentor if that's a stretch too far for one person — flag this as a decision point, don't assume one person covers both diseases well).

---

## 5. Framing text

### 5.1 NIGMS-fit significance paragraph (mechanism + tool, not therapy)

> Reduced pyridine-nucleotide cofactors (NADPH, NADH) are shared currencies that multiple, often competing, cellular processes draw on simultaneously — yet no existing tool can measure how a fixed supply of reducing equivalents is partitioned among those competing demands, or where the graded threshold lies between adequate and insufficient supply, without resorting to genetic knockouts that eliminate a node's contribution to the shared pool entirely and confound supply effects with structural loss. This proposal develops a generalizable, fully reconstituted (cell-free) platform that uses an orthogonal, bioorthogonal pyridine-nucleotide cofactor (NMN⁺/NMNH) to privately meter reducing-equivalent supply into a single node of a defined, multi-enzyme redox circuit, enabling three measurements no knockout can make: dose-response (rheostat) behavior, supply-decoupled causal attribution, and quantitative partitioning of a shared cofactor pool among competing consumers. The platform is demonstrated in circuits built from enzymes implicated in well-characterized human enzymopathies (G6PD deficiency; hereditary methemoglobinemia), which provide graded, clinically staged severity spectra against which the biochemical measurements can be validated — but the deliverable is the measurement platform and the underlying mechanistic insight into cofactor-supply partitioning, not a therapeutic or diagnostic claim. This is fundamental redox enzymology and quantitative measurement science with broad applicability across the many NAD(P)(H)-dependent pathways NIGMS supports, using disease systems as rigorous, well-characterized test cases rather than as the endpoint.

### 5.2 Innovation / independence paragraph (destination-based independence from both mentors)

> The engineering method underlying this platform — switching NAD(P)-dependent dehydrogenases to accept the orthogonal cofactor NMN⁺/NMNH via established design rules — originates in Dr. Li's laboratory and is applied here as a mature, published tool, not extended or re-engineered as a methodological contribution. Likewise, high-throughput cell-free protein synthesis, the production infrastructure used to express the enzyme panel, is Dr. Jewett's laboratory's signature technology, used here for reagent generation rather than as the object of study. The scientific destination of this proposal — quantitative measurement of how a shared reduced-cofactor pool is partitioned among competing physiological demands in a defined, disease-relevant circuit, using a bioorthogonal probe as a metering instrument rather than a pathway-rerouting tool — is a question neither laboratory works on. The applicant's independent contribution is the conceptual reframing of the orthogonal cofactor from a production/engineering tool into a measurement instrument, the experimental design that makes graded and partitioning measurements possible where only on/off comparisons existed before, and the specific circuits and disease anchors (RBC antioxidant defense, methemoglobin reduction) that neither mentor's research program addresses.

---

## 6. Annotated citation list

**Orthogonal-cofactor engineering (verified):**
- Black WB, Zhang L, Mak WS, et al., Li H (corresponding). "Engineering a nicotinamide mononucleotide redox cofactor system for biocatalysis." *Nat Chem Biol* 2020;16:87–94. Founding NMN⁺-orthogonal paper; engineered enzyme is glucose dehydrogenase (GDH), **not G6PD** — do not conflate (§0.2).
- Zhang L, King E, Black WB, et al., Li H (corresponding). "Directed evolution of phosphite dehydrogenase to cycle noncanonical redox cofactors via universal growth selection platform." *Nat Commun* 2022;13:5021. Contains the *E. coli* glutathione reductase → NMNH-exclusive switch used as an off-the-shelf reagent basis in Aim 1/2.
- Aspacio D, Zhang Y, Cui Y, et al., Li H (corresponding). "Shifting redox reaction equilibria on demand using an orthogonal redox cofactor." *Nat Chem Biol* 2024. Establishes generalized NMN(H)-orthogonality design rules, translated across six enzymes with ~10³–10⁶-fold specificity switches — the method Aim 3 applies to CYB5R3.
- Meng D, Liu M, Su H, et al. "Coenzyme Engineering of Glucose-6-phosphate Dehydrogenase on a Nicotinamide-Based Biomimic and Its Application as a Glucose Biosensor." *ACS Catalysis* 2023;13(3):1983–1998. The actual G6PD → NMN⁺ switch (*Zymomonas mobilis* G6PD, ~10³-fold), a distinct lineage from Li lab work — cite accurately as such.
- Liu W, et al. "Coenzyme Engineering of a Hyperthermophilic 6-Phosphogluconate Dehydrogenase from NADP+ to NAD+..." *Sci Rep* 2016;6:36311. Only confirmed 6PGD cofactor-engineering paper; canonical swap, not orthogonal — cite only to support "6PGD orthogonal switching does not yet exist" if needed, and note the corrected design (§0.2) doesn't require it.

**Cell-free protein synthesis (Jewett lineage, verified):**
- Wong DA, Shaver ZM, Cabezas MD, et al., Jewett MC (corresponding/senior). "Characterizing and engineering post-translational modifications with high-throughput cell-free expression." *Nat Commun* 2025;16. The "2025 cell-free PTM paper."
- Ekas HM, Wang B, Silverman AD, et al., Jewett MC. "An Automated Cell-Free Workflow for Transcription Factor Engineering." *ACS Synth Biol* 2024. Direct precedent for the HT-CFPS screening throughput claim (127+134 variants, 3,682 reactions, <48h).
- Hunt AC, Rasor BJ, Seki K, et al., Jewett MC. "Cell-Free Gene Expression: Methods and Applications." *Chem Rev* 2025;125(1):91–149. General HT-CFPS methods citation.

**Reconstitution methodology / precedent:**
- Vogelsang L, Eirich J, Finkemeier I, Dietz KJ. "Specificity and dynamics of H2O2 detoxification by the cytosolic redox regulatory network as revealed by in vitro reconstitution." *Redox Biology* 2024;72:103141. Closest methodological analog (15-protein reconstitution, roGFP2 sensor readout); plant system, no orthogonal cofactor — cite as feasibility precedent, not prior art on the novel claim.

**Disease relevance / clinical severity:**
- Luzzatto L, Ally M, Notaro R. "Glucose-6-phosphate dehydrogenase deficiency." *Blood* 2020;136(11):1225–1240. WHO Class I–IV severity framework — primary benchmark for Aim 1.
- Boonyuen U, et al. "A trade-off between catalytic activity and protein stability determines the clinical manifestations of G6PD deficiency." *Int J Biol Macromol* 2017. Variant kinetics.
- Scott MD, Zuo L, Lubin BH, Chiu DT. "NADPH, not glutathione, status modulates oxidant sensitivity in normal and G6PD-deficient erythrocytes." *Blood* 1991;77(9):2059–2064. Classic (intact-cell) precedent that NADPH supply, not GSH pool size, is the limiting variable — directly motivates Aim 1's design.
- Benfeitas R, Selvaggio G, Antunes F, Coelho PMBM, Salvador A. "Hydrogen peroxide metabolism and sensing in human erythrocytes: a validated kinetic model..." *Free Radic Biol Med* 2014. Kinetic-parameter source for circuit design/simulation.
- Yubisui T, Takeshita M. "Characterization of the purified NADH-cytochrome b5 reductase of human erythrocytes." *J Biol Chem* 1980. Classic CYB5R3 purified reconstitution, relevant if Aim 3 is developed.

**Cautions — verify before citing, or avoid:**
- Any claim that CYB5R3 (rather than CYB5R1) reduces CoQ to protect against ferroptosis (§0.3) — the primary and 2025-preprint evidence is for CYB5R1.
- P187S NQO1 exact Km/kcat numbers — qualitative destabilization/FAD-loss finding is well supported across ≥4 papers (Traver 1997; Siegel 2001; Pey/Medina-Carmona 2016; Megarity & Timson 2019; Nolan 2022), but exact numeric values were not independently pulled from primary text in this pass.
- The Gor (glutathione reductase) NMN(H) switch's exact fold-change (Zhang et al. 2022) — the paper's headline fold-change number applies to phosphite dehydrogenase evolved in that background, not to Gor itself; confirm the Gor-specific number against primary text before quoting it.

**Not recommended for the flagship, kept for reference:**
- NQO1 P187S / prodrug bioactivation literature (§1.3) — real whitespace but added engineering risk (FAD-site fragility).
- Mutant IDH1/2 (Gelman et al. *Cell Reports* 2018; Lu et al. *Nature* 2012) — real orthogonal-cofactor whitespace but no foundation to build from and a documented cell-context-dependence problem for the disease-relevance claim.

---

## 7. Open items for the researcher

1. Confirm the exact Gor NMN(H) fold-change and the P187S kinetic table against primary text (both flagged above) before either goes into a written Specific Aims page.
2. Decide whether Aim 3 (CYB5R3 generalization) belongs in the K99 or is pushed entirely to R00 — as scoped above it's R00-only, but eligibility/timeline considerations may argue for at least preliminary CYB5R3 data in the K99 if reviewers want to see platform generality earlier.
3. Identify a specific co-mentor candidate (§4) — this needs a name and institutional confirmation before the training plan section can be written.
4. Run the recommended forward-citation chase on Aspacio 2024 and Zhang 2022 (§1.1) to close the residual whitespace-verification gap before the Specific Aims page is finalized.

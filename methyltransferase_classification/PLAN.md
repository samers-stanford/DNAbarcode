# Plan: classify all methyltransferases (EC 2.1.1.-) by acceptor-atom bond type

Status: DRAFT — pausing here for review before any pipeline code is written,
per the working-style instructions in the source prompt.

## 0. Scope note on repo placement

This project lives entirely under `methyltransferase_classification/` in the
`DNAbarcode` repo (on branch `claude/new-session-6tsbug`). It is unrelated to
the existing barcode-QC / GC-MS code at the repo root and will not touch
those files. Everything below (`data/`, `analysis/`, `src/`, its own
`requirements.txt`, `README.md`) is scoped to this subdirectory.

## 1. Objective (restated)

For every EC 2.1.1.- entry, determine which atom accepts the methyl group
(C, N, O, S, Se, As, P, Co, or halide) using **structure-first** evidence
(reaction SMILES via RDKit), with name-based regex as a fallback/cross-check
only. Output a curated TSV, a flagged-for-review set, and a short analysis
of how bond class correlates with fold, cofactor, and taxonomy.

Key hazard to design against: name-based classification systematically
undercounts C-methyltransferases (many are named for their product, not
their chemistry — e.g. DNA cytosine-5-MTase is C–C, not N–C). Structural
evidence is authoritative; disagreements are flagged, never silently
resolved in favor of one tier.

## 2. Directory layout

```
methyltransferase_classification/
  PLAN.md
  README.md                    # re-run instructions, source versions, download dates
  requirements.txt             # pinned: rdkit, pandas, requests, etc.
  pyproject.toml or venv via uv
  data/
    raw/                       # untouched downloads, cached, never re-fetched if present
      rhea-reaction-smiles.tsv
      rhea-chebi-smiles.tsv
      rhea2ec.tsv
      rhea2uniprot_sprot.tsv
      rhea-directions.tsv
      enzyme.dat
      enzclass.txt
      uniprot_ec_2.1.1.tsv
      ...
    interim/                   # parsed/normalized intermediate tables
    PROVENANCE.json            # source URL, release/version, download date, file hash per source
    methyltransferases.tsv     # final curated output
    needs_review.tsv           # flagged/ambiguous entries
    gold_set_confusion_matrix.tsv
  analysis/
    summary.md
    figures/
      bond_class_distribution.png
      fold_x_bond_heatmap.png
  src/
    fetch.py                   # download + cache all raw sources
    parse_ec.py                # parse enzyme.dat/enzclass.txt -> EC metadata, detect deleted/transferred entries
    parse_rhea.py               # join rhea-reaction-smiles, rhea2ec, rhea-chebi-smiles, rhea-directions
    donors.py                  # canonical list of methyl-donor/byproduct SMARTS pairs (SAM/SAH, THF variants, methylcobalamin, methyl-CoM, betaine)
    classify_structural.py     # Tier 1: MCS-based acceptor-atom detection + formula-delta sanity check
    classify_name.py           # Tier 2: regex over accepted/systematic names + synonyms
    reconcile.py                # Tier 3: merge tiers, agreement check, confidence scoring, flagging
    enrich_uniprot.py          # protein counts, taxonomy, cofactor/fold annotations (InterPro/Pfam)
    build_output.py             # assemble methyltransferases.tsv / needs_review.tsv
    validate_gold_set.py        # run pipeline against the 21-entry gold set, print confusion matrix
    analyze.py                  # summary stats + figures
  tests/
    test_classify_structural.py # unit tests on synthetic donor/substrate/product SMILES
```

## 3. Data acquisition (`fetch.py`)

- Idempotent: every fetch checks `data/raw/<file>` exists before requesting;
  skip if present. Record `{url, resolved_release_version_if_available,
  download_date, sha256}` for each file into `data/PROVENANCE.json`.
- Sources pulled verbatim from the prompt: Rhea TSV bundle (or the
  `rhea-tsv.tar.gz` archive, unpacked once), ExPASy `enzyme.dat` +
  `enzclass.txt`, UniProt REST TSV stream for `ec:2.1.1.*` (paginated via
  `Link` header, not per-accession), ExplorEnz/IUBMB dump only if the
  ExPASy `enzyme.dat` names/systematic-names turn out incomplete for any
  EC number (check first, don't fetch redundantly).
- BRENDA bulk file: deferred until Tier 1+2+3 are working end-to-end; only
  pulled if it's shown to add mechanism/organism info the other sources lack.
- UniProt requests batched/streamed; BRENDA (if used) rate-limited to
  ≥1 req/s with an explicit sleep between calls.

## 4. EC-level normalization (`parse_ec.py`)

- Parse `enzyme.dat` for accepted name, systematic name (where present),
  synonyms, and status flags. Detect `DE   Deleted entry.` and
  `DE   Transferred entry: ...` lines — exclude these from the working set,
  and report the excluded count/list in `analysis/summary.md`.
- Cross-reference against `enzclass.txt` to confirm 2.1.1.- coverage and
  catch any EC numbers Rhea doesn't map to (record those as
  "no structural evidence available" rather than dropping them silently).

## 5. Reaction-level normalization (`parse_rhea.py`)

- Join `rhea2ec.tsv` → `rhea-reaction-smiles.tsv` → `rhea-directions.tsv`
  (resolve to the defined/master direction) → `rhea-chebi-smiles.tsv` for
  participant-level SMILES where the whole-reaction SMILES has wildcard
  (`*`) attachment atoms (protein/nucleic-acid residues).
- Keep wildcard-containing reactions rather than dropping them — they cover
  a large share of protein/DNA/RNA N- and C-methyltransferases. The MCS
  step (Tier 1) needs to tolerate `*` dummy atoms.
- One EC number can map to multiple Rhea IDs (different directions,
  different specific substrates, or generic vs. specific reaction entries).
  Handle 1:many explicitly: classify each Rhea reaction independently, then
  roll up to one EC-level call if all resolved reactions agree; flag if
  they disagree.

## 6. Donor/acceptor stripping (`donors.py`)

Canonical donor→byproduct pairs to match against each reaction's
participants (as ChEBI IDs and/or SMARTS), not just SAM→SAH:

| Donor | Byproduct |
|---|---|
| S-adenosyl-L-methionine (SAM) | S-adenosyl-L-homocysteine (SAH) |
| 5-methyltetrahydrofolate | tetrahydrofolate |
| 5,10-methylene-THF | dihydrofolate (thymidylate synthase: methylene C transferred, not a simple methyl) |
| methylcobalamin | cob(I)alamin |
| methyl-coenzyme M | coenzyme M (demethylated) |
| betaine | dimethylglycine / N,N-dimethylglycine |

Thymidylate synthase (2.1.1.45) is a known special case: the one-carbon
unit comes from methylene-THF and is reduced during transfer, and the
gold set says the product bond is C–C — this needs its own handling path,
not the generic "product minus substrate = CH2" rule, since the THF also
gets oxidized in the same step. Document this explicitly rather than
special-casing it silently.

## 7. Tier 1 — structural classification (`classify_structural.py`)

Per Rhea reaction:
1. Identify donor/byproduct pair from the table above among the reaction's
   ChEBI participants.
2. Remove donor+byproduct, leaving substrate(s)→product(s).
3. Run `rdFMCS.FindMCS([substrate, product])` (with matching parameters
   tolerant of the wildcard atoms from step 5) to align substrate and
   product; identify the product atom(s) not present in the MCS.
4. Confirm the added atom is a methyl carbon (degree consistent with CH3
   after considering aromaticity/valence — can't always rely on explicit
   H-count from SMILES) and identify its heavy-atom neighbor's element.
   That element is the acceptor atom / bond type.
5. Sanity check: molecular formula delta product − substrate should be
   CH2 net (methyl added, one proton lost to the leaving group/base) —
   flag (not silently accept) any reaction where this doesn't hold, since
   it usually means the donor-stripping step or MCS alignment picked the
   wrong atoms.
6. Record: rhea_id(s) used, donor identified, acceptor atom, acceptor
   context (aromatic_C/sp3_C/amine_N/etc. inferred from local environment),
   and a confidence flag (e.g., MCS ambiguity, multiple candidate
   added-atoms, wildcard atoms involved).

If RDKit `rdFMCS` proves too coarse for edge cases (multi-atom donors, ring
formation as in cyclopropane synthase), fall back to RXNMapper atom-mapping
for those specific reactions — decide this after seeing how Tier 1 performs
on the 20-reaction pilot, not upfront.

## 8. Tier 2 — name-based classification (`classify_name.py`)

- Regex over accepted name, systematic name, and synonyms for
  `\b([CNOS]|Se|As|Te|Sb|Hg|P)-methyltransferase\b` and positional/atom
  hints in parentheses (`cytosine-5`, `adenine-N6`, `2'-O`, `N-terminal`,
  etc.), applied to *every* entry (including ones Tier 1 already resolved),
  purely to measure agreement — never to overrule Tier 1.
- Output a per-entry Tier 2 call + the matched text span, so disagreements
  are auditable.

## 9. Tier 3 — reconciliation (`reconcile.py`)

- Agreement (Tier 1 and Tier 2 give the same bond class): confidence =
  high, evidence_tier = "1+2 agree".
- Tier 1 resolves, Tier 2 silent/ambiguous: confidence = medium-high,
  evidence_tier = "1 only".
- Tier 1 fails (no Rhea mapping, MCS failure, formula-delta mismatch) but
  Tier 2 resolves cleanly: confidence = medium, evidence_tier = "2 only",
  goes in main table but flagged for spot-checking.
- Disagreement between tiers, or neither resolves: → `needs_review.tsv`
  with both tiers' evidence, not a forced pick.
- Explicit handling for known double-counting risks called out in the
  prompt: methyl-CoM/methanogenesis reactions and corrinoid Co–C
  intermediates that may span multiple EC entries — detect by shared
  Rhea/ChEBI participants across EC numbers and note the linkage rather
  than counting each EC independently in the summary stats.

## 10. UniProt / fold / taxonomy enrichment (`enrich_uniprot.py`)

- Stream `ec:2.1.1.*` from UniProt REST (paginated via `Link` header) for
  accession, protein name, organism/lineage, InterPro, Pfam, cofactor and
  catalytic-activity comments, and Rhea cross-refs.
- Roll up counts at both EC level and UniProt-protein level (Swiss-Prot vs
  TrEMBL) — these will diverge substantially (a few EC numbers account for
  huge numbers of TrEMBL sequences, e.g. DNA MTases); both are reported in
  `analysis/summary.md` with an explanation of why they differ.
- Map InterPro/Pfam families to the coarse MTase fold classes in the
  schema (Rossmann_I / MetH_II / precorrin_III / SPOUT_IV / SET_V /
  radical_SAM / TM / unknown) via a small curated family→fold lookup table
  (built from known Pfam clan assignments), falling back to "unknown"
  rather than guessing.

## 11. Output schema (`data/methyltransferases.tsv`)

Exactly the columns specified in the prompt:

```
ec_number, accepted_name, systematic_name, bond_formed, acceptor_atom,
acceptor_context, methyl_donor, rhea_ids, reaction_smiles, substrate_chebi,
product_chebi, n_swissprot, n_trembl, representative_uniprot,
interpro_families, pfam, mtase_fold_class, is_radical_sam,
taxonomic_span, evidence_tier, confidence, notes
```

Plus `data/needs_review.tsv`, `analysis/summary.md`, `analysis/figures/`,
and this project's own `README.md`.

## 12. Validation gate (must pass before full-scale run)

- Hand-run the pipeline (fetch → parse → Tier 1 → Tier 2 → Tier 3) against
  the 21-entry gold set embedded in the prompt.
- Produce a confusion matrix (predicted bond class × gold bond class) and
  show it before proceeding further.
- Target ≥90% accuracy on the gold set. Below that: fix the pipeline logic
  (donor list, MCS parameters, formula-delta check), not the gold set.
- **Stop here and show the confusion matrix — do not run the full
  EC 2.1.1.- sweep until it's reviewed**, per the source prompt's working
  style.

## 13. Full-scale run + analysis (only after gate above is cleared)

- Run Tier 1–3 over all EC 2.1.1.- entries (minus deleted/transferred).
- Generate `analysis/summary.md`: counts/percentages per bond class at EC
  level and at UniProt-protein level, cofactor/fold breakdown per class,
  and 3–5 genuinely interesting observations (not just descriptive
  restatements of the tables) — e.g., whether C-methylation clusters in
  specific fold families or radical-SAM mechanisms disproportionately.
- Generate `analysis/figures/bond_class_distribution.png` and a
  `fold_class × bond_class` heatmap.
- Write `README.md`: exact re-run commands, source release versions and
  download dates (pulled from `PROVENANCE.json`), and known limitations.

## 14. Tooling / environment

- Python + `uv` (or venv) with a pinned `requirements.txt`
  (rdkit, pandas, requests, matplotlib/seaborn for figures, pytest).
- All network fetches idempotent against `data/raw/`.
- Unit tests for the structural classifier on synthetic SAM→SAH +
  substrate/product SMILES covering each bond class in the gold set,
  independent of live network access.

## 15. Open questions before I start building

1. RXNMapper as a Tier 1 fallback pulls in a transformer model dependency
   (torch) — OK to add if `rdFMCS` alone proves insufficient on the pilot
   set, or would you rather I stay RDKit-only and just flag anything MCS
   can't resolve?
2. Fold-class assignment (Rossmann_I/MetH_II/etc.) needs a curated
   Pfam/InterPro→fold-class mapping table since no single source encodes
   this directly — fine for me to hand-build a small lookup table from
   published MTase fold reviews, with "unknown" as the honest fallback for
   anything not in it?
3. Any preference between `uv` and a plain `venv` for this subdirectory?

## Next step

On approval of this plan, I'll start with `fetch.py` + `parse_rhea.py` +
`classify_structural.py` on a ~20-reaction pilot slice, then run the
gold-set validation and show the confusion matrix before scaling up —
no full pipeline run until that's reviewed.

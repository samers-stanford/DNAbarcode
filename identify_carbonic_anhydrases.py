#!/usr/bin/env python3
"""
identify_carbonic_anhydrases.py – Find carbonic anhydrase proteins in organism genomes

For each supplied organism name the script:
  1. Resolves the organism to an NCBI Taxonomy ID (TaxID).
  2. Searches the NCBI RefSeq protein database for annotated carbonic anhydrase
     sequences (keyword: "carbonic anhydrase" OR "carbonate dehydratase").
  3. Classifies each hit by CA family (α, β, γ, δ, ζ, η, θ, ι).
  4. Writes a CSV summary; optionally writes a multi-FASTA file.

Carbonic anhydrase families detected
-------------------------------------
  α (alpha)  — Eukaryotic/vertebrate CAs (CA1–CA15 in humans, etc.)
  β (beta)   — Plant and prokaryotic CAs
  γ (gamma)  — Archaeal and bacterial CAs
  δ (delta)  — Marine diatom CAs
  ζ (zeta)   — Marine diatom CAs
  η (eta)    — Pathogenic protozoa CAs
  θ (theta)  — Diatom CAs
  ι (iota)   — Bacterial / marine eukaryote CAs
  ? (unknown)— Cannot be classified from description alone

Usage
-----
  # Organisms listed on the command line
  python identify_carbonic_anhydrases.py --email you@example.com \\
      "Homo sapiens" "Arabidopsis thaliana" "Methanobacterium thermoautotrophicum"

  # Organisms from a file (one per line; lines starting with # are comments)
  python identify_carbonic_anhydrases.py --email you@example.com \\
      --organism-file organisms.txt

  # Also save protein sequences to a FASTA file
  python identify_carbonic_anhydrases.py --email you@example.com \\
      --fasta-out ca_sequences.fasta --output results.csv \\
      "Chlamydomonas reinhardtii"

  # Broaden search beyond RefSeq (slower, more hits)
  python identify_carbonic_anhydrases.py --email you@example.com \\
      --no-refseq-only "Drosophila melanogaster"

Requirements
------------
  pip install biopython
  (auto-installed if missing)
"""

import argparse
import csv
import re
import sys
import time
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Auto-install Biopython
# ---------------------------------------------------------------------------

def _install_biopython() -> None:
    import subprocess
    print("[setup] Biopython not found – installing (pip install biopython)…",
          flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "biopython"]
    )
    print("[setup] Biopython installed.\n", flush=True)


try:
    from Bio import Entrez, SeqIO
except ImportError:
    _install_biopython()
    from Bio import Entrez, SeqIO


# ---------------------------------------------------------------------------
# CA family classification
# ---------------------------------------------------------------------------

# Each rule: (compiled regex, symbol, full name).
# Rules are tested in order; the first match wins.
_CA_RULES: list[tuple[re.Pattern, str, str]] = []

_RAW_RULES = [
    # Greek-letter family names in the description
    (r'\b(alpha|α[-\s]?class|α)\b',    'α', 'alpha'),
    (r'\b(beta|β[-\s]?class|β)\b',     'β', 'beta'),
    (r'\b(gamma|γ[-\s]?class|γ)\b',    'γ', 'gamma'),
    (r'\b(delta|δ[-\s]?class|δ)\b',    'δ', 'delta'),
    (r'\b(zeta|ζ[-\s]?class|ζ)\b',     'ζ', 'zeta'),
    (r'\b(eta|η[-\s]?class|η)\b',      'η', 'eta'),
    (r'\b(theta|θ[-\s]?class|θ)\b',    'θ', 'theta'),
    (r'\b(iota|ι[-\s]?class|ι)\b',     'ι', 'iota'),
    # Vertebrate isozyme numbering → alpha family
    # Matches "carbonic anhydrase II", "carbonic anhydrase 9", "CA-14", "CA IX"
    (r'carboni[cq]\s+anhydrase\s+(?:type\s+)?'
     r'(I{1,3}V?|VI{0,3}|XI{0,2}|XIV|XV|[1-9][0-9]?)\b',
     'α', 'alpha'),
    (r'\bca[-\s]?(?:type[-\s]?)?([1-9][0-9]?)\b',   'α', 'alpha'),
    # Prokaryotic / plant-type → beta
    (r'prokaryoti[c]|plant[\s-]type',   'β', 'beta'),
    # Eukaryotic-type → alpha
    (r'eukaryoti[c][\s-]type',          'α', 'alpha'),
    # Related / domain-only proteins (not catalytically active)
    (r'carbonic\s+anhydrase[\s-]related|carbonic\s+anhydrase\s+domain|'
     r'\bCARP\b',
     '?', 'CA-related'),
]

for _pat, _sym, _name in _RAW_RULES:
    _CA_RULES.append((re.compile(_pat, re.IGNORECASE), _sym, _name))


def classify_ca_family(description: str) -> tuple[str, str]:
    """Return ``(symbol, full_name)`` for the CA family found in *description*.

    Returns ``('?', 'unknown')`` if no family can be determined.
    """
    for pattern, symbol, name in _CA_RULES:
        if pattern.search(description):
            return symbol, name
    return '?', 'unknown'


# ---------------------------------------------------------------------------
# NCBI Entrez helpers
# ---------------------------------------------------------------------------

_BATCH_SIZE = 200  # records per efetch call


def _retry(fn, *args, max_attempts: int = 3, **kwargs):
    """Call *fn* with exponential back-off on network exceptions."""
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == max_attempts - 1:
                raise
            wait = 2 ** attempt
            print(f"  [warn] Request failed ({exc}), retrying in {wait}s…",
                  flush=True)
            time.sleep(wait)


def resolve_taxid(organism_name: str,
                  delay: float) -> tuple[str | None, str | None]:
    """Return ``(taxid, scientific_name)`` for *organism_name*.

    Returns ``(None, None)`` if not found in NCBI Taxonomy.
    """
    handle = _retry(Entrez.esearch, db="taxonomy",
                    term=organism_name, retmax=1)
    record = Entrez.read(handle)
    handle.close()

    ids = record.get("IdList", [])
    if not ids:
        return None, None

    time.sleep(delay)
    handle = _retry(Entrez.efetch, db="taxonomy",
                    id=ids[0], rettype="xml", retmode="xml")
    tax_records = Entrez.read(handle)
    handle.close()

    sci_name: str = (
        tax_records[0].get("ScientificName", organism_name)
        if tax_records else organism_name
    )
    return ids[0], sci_name


def search_carbonic_anhydrases(
    taxid: str,
    sci_name: str,
    *,
    max_results: int,
    refseq_only: bool,
    delay: float,
) -> list[dict]:
    """Search NCBI protein for carbonic anhydrases in *taxid*.

    Returns a list of result dicts, one per protein record.
    """
    # Build the Entrez query
    name_clause = (
        '"carbonic anhydrase"[Protein Name] '
        'OR "carbonate dehydratase"[Protein Name]'
    )
    parts = [f'({name_clause})', f'txid{taxid}[Organism:exp]']
    if refseq_only:
        parts.append('refseq[filter]')
    query = ' AND '.join(parts)
    print(f"  Query : {query}")

    handle = _retry(Entrez.esearch, db="protein", term=query,
                    retmax=max_results, usehistory="y")
    record = Entrez.read(handle)
    handle.close()

    total_found = int(record.get("Count", 0))
    webenv     = record["WebEnv"]
    query_key  = record["QueryKey"]
    n_to_fetch = min(total_found, max_results)
    print(f"  Hits  : {total_found} (fetching up to {n_to_fetch})")

    if n_to_fetch == 0:
        return []

    results: list[dict] = []

    for retstart in range(0, n_to_fetch, _BATCH_SIZE):
        batch_size = min(_BATCH_SIZE, n_to_fetch - retstart)
        time.sleep(delay)

        handle = _retry(
            Entrez.efetch,
            db="protein",
            webenv=webenv,
            query_key=query_key,
            retstart=retstart,
            retmax=batch_size,
            rettype="gb",
            retmode="text",
        )
        try:
            for seq_rec in SeqIO.parse(handle, "genbank"):
                ca_sym, ca_name = classify_ca_family(seq_rec.description)

                # Pull gene name and locus_tag from CDS / gene features
                gene_name = ""
                locus_tag = ""
                for feat in seq_rec.features:
                    if feat.type in ("CDS", "gene", "Protein"):
                        gene_name  = (feat.qualifiers.get("gene",      [""])[0]
                                      or gene_name)
                        locus_tag  = (feat.qualifiers.get("locus_tag", [""])[0]
                                      or locus_tag)
                    if gene_name and locus_tag:
                        break

                results.append({
                    "organism_query":    sci_name,
                    "taxid":             taxid,
                    "accession":         seq_rec.id,
                    "description":       seq_rec.description,
                    "gene_name":         gene_name,
                    "locus_tag":         locus_tag,
                    "protein_length_aa": len(seq_rec.seq),
                    "ca_family_symbol":  ca_sym,
                    "ca_family_name":    ca_name,
                    "sequence":          str(seq_rec.seq),
                })
        finally:
            handle.close()

        retrieved = min(retstart + batch_size, n_to_fetch)
        print(f"  Fetched {retrieved}/{n_to_fetch} …", end="\r", flush=True)

    print()  # newline after progress indicator
    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "organism_query",
    "taxid",
    "accession",
    "description",
    "gene_name",
    "locus_tag",
    "protein_length_aa",
    "ca_family_symbol",
    "ca_family_name",
]


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_fasta(rows: list[dict], path: Path) -> None:
    with path.open("w") as fh:
        for row in rows:
            seq = row.get("sequence", "")
            if not seq:
                continue
            header = (
                f">{row['accession']} "
                f"[{row['organism_query']}] "
                f"family={row['ca_family_symbol']} "
                f"{row['description']}"
            )
            fh.write(header + "\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i : i + 60] + "\n")


def print_summary(all_results: list[dict]) -> None:
    print(f"\nTotal CA sequences found : {len(all_results)}")

    if not all_results:
        return

    print("\nBy organism:")
    for org, n in Counter(r["organism_query"] for r in all_results).most_common():
        print(f"  {n:5d}  {org}")

    print("\nBy CA family:")
    for label, n in Counter(
        f"{r['ca_family_symbol']}  ({r['ca_family_name']})"
        for r in all_results
    ).most_common():
        print(f"  {n:5d}  {label}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify carbonic anhydrases in organism genomes via NCBI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "organisms",
        nargs="*",
        metavar="ORGANISM",
        help="Organism name(s) — quote multi-word names, e.g. 'Homo sapiens'",
    )
    parser.add_argument(
        "--email", required=True,
        help="Your e-mail address (required by NCBI Entrez policy)",
    )
    parser.add_argument(
        "--organism-file", metavar="FILE",
        help="Text file with one organism per line (lines starting with # "
             "are treated as comments and ignored)",
    )
    parser.add_argument(
        "--output", metavar="FILE", default="carbonic_anhydrases.csv",
        help="Output CSV path (default: carbonic_anhydrases.csv)",
    )
    parser.add_argument(
        "--fasta-out", metavar="FILE",
        help="Optional FASTA file for all retrieved protein sequences",
    )
    parser.add_argument(
        "--max-per-organism", type=int, default=500, metavar="N",
        help="Maximum protein records to retrieve per organism (default: 500)",
    )
    parser.add_argument(
        "--no-refseq-only", action="store_true",
        help="Also include non-RefSeq sequences (broader search, more hits)",
    )
    parser.add_argument(
        "--api-key", metavar="KEY",
        help="NCBI API key – raises rate limit from 3 to 10 requests/second",
    )
    return parser.parse_args()


def load_organisms_from_file(path: str) -> list[str]:
    organisms = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            organisms.append(line)
    return organisms


def main() -> None:
    args = parse_args()

    # Collect organisms from CLI + optional file
    organisms: list[str] = list(args.organisms)
    if args.organism_file:
        organisms.extend(load_organisms_from_file(args.organism_file))

    if not organisms:
        print(
            "Error: supply at least one organism name or use --organism-file.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Configure Entrez
    Entrez.email = args.email
    if args.api_key:
        Entrez.api_key = args.api_key

    # NCBI allows 3 req/s without a key, 10 req/s with one
    delay = 0.11 if args.api_key else 0.34
    refseq_only = not args.no_refseq_only
    out_path   = Path(args.output)
    fasta_path = Path(args.fasta_out) if args.fasta_out else None

    all_results: list[dict] = []

    for organism in organisms:
        print(f"\n{'='*60}")
        print(f"Organism : {organism}")
        print("=" * 60)

        # Step 1 – resolve organism name → TaxID
        print("  Resolving taxonomy ID…")
        taxid, sci_name = resolve_taxid(organism, delay=delay)
        if taxid is None:
            print(f"  [!] '{organism}' not found in NCBI Taxonomy — skipping.")
            continue
        print(f"  Resolved : {sci_name}  (TaxID {taxid})")
        time.sleep(delay)

        # Step 2 – search carbonic anhydrases
        try:
            results = search_carbonic_anhydrases(
                taxid,
                sci_name,
                max_results=args.max_per_organism,
                refseq_only=refseq_only,
                delay=delay,
            )
        except Exception as exc:
            print(f"  [!] Search failed for '{organism}': {exc}")
            continue

        print(f"  Retrieved {len(results)} CA sequences for {sci_name}")
        all_results.extend(results)

    # Print summary to stdout
    print_summary(all_results)

    # Write outputs
    if all_results:
        write_csv(all_results, out_path)
        print(f"\nCSV   → {out_path}")
        if fasta_path:
            write_fasta(all_results, fasta_path)
            print(f"FASTA → {fasta_path}")
    else:
        print("\nNo results to write.")


if __name__ == "__main__":
    main()

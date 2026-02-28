#!/usr/bin/env python3
"""
barcode_qc.py
=============
Standalone quality-control script for DNA barcode libraries designed for
Sidewinder junction addressing.  Works independently of SeqWalk.

Usage
-----
    # 25-nt library (defaults match 25-nt design)
    python barcode_qc.py sidewinder_25nt_barcodes.csv

    # 20-nt library – adjust length and distance threshold
    python barcode_qc.py sidewinder_20nt_barcodes.csv --barcode-length 20 --min-hamming 6

    # See all configurable options
    python barcode_qc.py --help

Required CSV columns
--------------------
    junction_id                – integer label for each barcode pair
    barcode_forward            – N-nt sequence (A/C/G/T)
    barcode_reverse_complement – pre-stored reverse complement

Output files (written to the current directory; names are configurable)
-----------------------------------------------------------------------
    barcodes_qc_pass.csv  – rows that pass ALL QC checks
    barcodes_qc_fail.csv  – rows failing ≥1 check, with per-check boolean columns

QC checks performed
-------------------
    1. Length uniformity
    2. Reverse-complement integrity (stored RC == computed RC)
    3. GC content (fraction within configurable window)
    4. Homopolymer runs (> configurable threshold)
    5. Simple tandem repeats (period 2–4, total length ≥ configurable threshold)
    6. Hairpin heuristic (self-complementary stem ≥ configurable length)
    7. Pairwise Hamming distance within the forward-barcode set
    8. Cross-talk: Hamming distance between each forward barcode and every
       OTHER barcode's reverse complement

Dependencies: pandas, numpy  (no Biopython or SeqWalk required)
"""

# ============================================================
# § 0  Configurable parameters – edit here or override via CLI
# ============================================================

BARCODE_LENGTH        = 25    # expected sequence length (nt)
GC_MIN                = 0.40  # minimum GC fraction (inclusive)
GC_MAX                = 0.60  # maximum GC fraction (inclusive)
HOMOPOLYMER_MAX       = 4     # flag runs LONGER than this (>4 flags AAAAA etc.)
TANDEM_REPEAT_MIN_LEN = 8     # flag tandem repeats with total length ≥ this (nt)
MIN_HAMMING_DISTANCE  = 8     # minimum acceptable pairwise Hamming distance
HAIRPIN_MIN_STEM      = 6     # stem length (nt) that triggers a hairpin flag
HAIRPIN_MIN_LOOP      = 3     # minimum loop length (nt) between stem arms
CHUNK_SIZE            = 200   # rows per chunk for vectorised Hamming (tune for RAM)

PASS_OUTPUT = "barcodes_qc_pass.csv"
FAIL_OUTPUT = "barcodes_qc_fail.csv"

# ============================================================
# § 1  Imports
# ============================================================

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# § 2  Sequence utility functions
# ============================================================

_COMP_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement(seq: str) -> str:
    """Return the Watson-Crick reverse complement of *seq*."""
    return seq.translate(_COMP_TABLE)[::-1]


def gc_fraction(seq: str) -> float:
    """Return GC fraction (0–1) of *seq*."""
    return sum(1 for b in seq if b in "GC") / len(seq)


# ============================================================
# § 3  Per-sequence QC check functions
# ============================================================

def check_homopolymer(seq: str, max_run: int) -> bool:
    """Return True if *seq* contains a homopolymer run LONGER than *max_run* bases.

    E.g. max_run=4 flags 'AAAAA' (run of 5) but passes 'AAAA' (run of 4).
    Regex: one character captured, then ≥ max_run additional identical characters.
    """
    pattern = r"(.)\1{" + str(max_run) + r",}"
    return bool(re.search(pattern, seq))


def check_tandem_repeat(seq: str, min_total: int) -> bool:
    """Return True if *seq* contains a tandem repeat of total length ≥ *min_total* nt.

    Checks repeat periods 2–4 nt.  Period-1 (homopolymers) is handled separately.

    With min_total=8:
        period 2 → flags 'ATATATAT'  (AT  × 4 =  8 nt)
        period 3 → flags 'ATCATCATC' (ATC × 3 =  9 nt, ceil(8/3)=3)
        period 4 → flags 'ACGTACGT'  (ACGT× 2 =  8 nt)
    """
    for period in range(2, 5):
        total_reps = -(-min_total // period)  # ceiling division
        additional = max(total_reps - 1, 1)   # repetitions after the first capture
        pattern = r"(.{" + str(period) + r"})\1{" + str(additional) + r",}"
        if re.search(pattern, seq):
            return True
    return False


def check_hairpin(seq: str, min_stem: int, min_loop: int) -> bool:
    """Heuristic hairpin screen – no thermodynamic model needed.

    Flags *seq* if any sub-window of length ≥ *min_stem* has its exact reverse
    complement appearing later in the sequence, separated by at least *min_loop*
    bases (the loop).  This detects obvious stretches of strong self-complementarity
    that could form stable hairpin structures.
    """
    L = len(seq)
    for stem_len in range(min_stem, L // 2 + 1):
        for i in range(L - stem_len):
            rc_stem = reverse_complement(seq[i : i + stem_len])
            # Search for rc_stem starting at least (stem + loop) bases downstream.
            # str.find(sub, start, end) only returns a hit if the full sub fits in
            # seq[start:end], so bounds are handled automatically.
            j_start = i + stem_len + min_loop
            if seq.find(rc_stem, j_start, L) != -1:
                return True
    return False


# ============================================================
# § 4  Vectorised Hamming distance routines (NumPy)
# ============================================================

def _encode(seqs: list) -> np.ndarray:
    """Encode equal-length DNA strings as a uint8 NumPy array of shape (N, L)."""
    return np.frombuffer(
        "".join(seqs).encode("ascii"), dtype=np.uint8
    ).reshape(len(seqs), len(seqs[0]))


def compute_pairwise_hamming(mat: np.ndarray, chunk_size: int):
    """Pairwise Hamming distances for an (N, L) encoded sequence matrix.

    Processes rows in chunks to keep peak RAM usage O(chunk_size × N × L).
    Self-distances (diagonal) are excluded from all outputs.

    Returns
    -------
    per_seq_min : ndarray, shape (N,)
        Minimum Hamming distance from each sequence to any other sequence.
    hist : ndarray, shape (L+1,)
        Count of unique upper-triangle pairs at each integer distance value.
    global_min : int
        Minimum pairwise distance over the entire library.
    """
    N, L = mat.shape
    per_seq_min = np.full(N, L, dtype=np.int32)
    hist = np.zeros(L + 1, dtype=np.int64)

    for i in range(0, N, chunk_size):
        end   = min(i + chunk_size, N)
        chunk = mat[i:end]                         # (chunk_sz, L)

        # Compute element-wise mismatches → (chunk_sz, N) distance matrix.
        # Broadcasting: chunk[:, None, :] vs mat[None, :, :] → (chunk_sz, N, L)
        diffs = np.sum(chunk[:, None, :] != mat[None, :, :], axis=2)

        # Replace self-distances with L so they don't affect the per-row minimum.
        for k in range(end - i):
            diffs[k, i + k] = L

        # Row minimum gives the nearest-neighbour distance for each chunk row.
        per_seq_min[i:end] = diffs.min(axis=1)

        # Accumulate histogram using only upper-triangle pairs (j > global row i+k)
        # to avoid counting each pair twice.
        rows = np.arange(i, end)[:, None]   # (chunk_sz, 1)
        cols = np.arange(N)[None, :]        # (1, N)
        hist += np.bincount(diffs[cols > rows], minlength=L + 1)

        print(f"  Pairwise Hamming … {end}/{N} rows", end="\r", file=sys.stderr)

    print(file=sys.stderr)
    return per_seq_min, hist, int(per_seq_min.min())


def compute_crosstalk_hamming(fwd_mat: np.ndarray, rc_mat: np.ndarray, chunk_size: int):
    """Minimum Hamming distance from each forward barcode to all OTHER barcodes' RCs.

    The own-pair (fwd[i] vs rc[i]) is excluded, as it represents the intended
    complement pair rather than genuine cross-talk.

    Returns
    -------
    per_fwd_min : ndarray, shape (N,)
    global_min  : int
    """
    N, L = fwd_mat.shape
    per_fwd_min = np.full(N, L, dtype=np.int32)

    for i in range(0, N, chunk_size):
        end   = min(i + chunk_size, N)
        chunk = fwd_mat[i:end]

        # (chunk_sz, N) matrix: fwd rows vs all RC columns
        diffs = np.sum(chunk[:, None, :] != rc_mat[None, :, :], axis=2)

        # Exclude own pair: fwd[i+k] vs rc[i+k]
        for k in range(end - i):
            diffs[k, i + k] = L

        per_fwd_min[i:end] = diffs.min(axis=1)
        print(f"  Cross-talk Hamming … {end}/{N} rows", end="\r", file=sys.stderr)

    print(file=sys.stderr)
    return per_fwd_min, int(per_fwd_min.min())


# ============================================================
# § 5  Summary report (stdout)
# ============================================================

def print_summary(df: pd.DataFrame, args, global_min_ham, global_min_ct, hist):
    """Write the QC summary table to stdout."""
    N  = len(df)
    gc = df["gc_fraction"]

    LINE = "─" * 62

    print("=" * 62)
    print("  DNA Barcode Library QC Report")
    print("=" * 62)
    print(f"  Input file      : {args.input}")
    print(f"  Barcodes loaded : {N:,}")
    print()

    # ── 1. Basic integrity ────────────────────────────────────
    print(LINE)
    print("  1. Basic Integrity")
    print(LINE)
    n_fl = int(df["fail_length"].sum())
    n_rc = int(df["fail_rc_mismatch"].sum())
    print(f"  Length == {args.barcode_length} nt  :  {N - n_fl:,} pass,  {n_fl:,} fail")
    print(f"  RC mismatch      :  {n_rc:,} fail")
    print()

    # ── 2. GC content ─────────────────────────────────────────
    print(LINE)
    print("  2. GC Content  (barcode_forward)")
    print(LINE)
    print(f"  Min  : {gc.min():.4f}  ({gc.min()*100:.1f} %)")
    print(f"  Max  : {gc.max():.4f}  ({gc.max()*100:.1f} %)")
    print(f"  Mean : {gc.mean():.4f}  ({gc.mean()*100:.1f} %)")
    print(f"  Std  : {gc.std():.4f}")
    n_gc = int(df["fail_gc"].sum())
    print(f"  Flagged (outside {args.gc_min:.0%}–{args.gc_max:.0%}) : {n_gc:,}")
    print()

    # ── 3. Sequence complexity ────────────────────────────────
    print(LINE)
    print("  3. Sequence Complexity")
    print(LINE)
    n_hp  = int(df["fail_homopolymer"].sum())
    n_rep = int(df["fail_tandem_repeat"].sum())
    n_pin = int(df["fail_hairpin"].sum())
    print(f"  Homopolymer run > {args.homopolymer_max}              : {n_hp:,}")
    print(f"  Tandem repeat ≥ {args.tandem_repeat_min_len} nt (period 2–4)  : {n_rep:,}")
    print(f"  Potential hairpin (stem ≥ {args.hairpin_min_stem} nt)      : {n_pin:,}")
    print()

    # ── 4. Pairwise Hamming ───────────────────────────────────
    print(LINE)
    print("  4. Pairwise Hamming Distance  (forward ↔ forward)")
    print(LINE)
    if global_min_ham is not None:
        L = args.barcode_length
        flag = (
            f"  *** WARNING: global min {global_min_ham} is below threshold "
            f"{args.min_hamming_distance} ***"
            if global_min_ham < args.min_hamming_distance
            else ""
        )
        print(f"  Global minimum : {global_min_ham}{' ← BELOW THRESHOLD' if flag else ''}")
        if flag:
            print(flag)
        print()
        # Distance distribution table
        nz    = np.nonzero(hist)[0]
        total = int(hist.sum())
        cumul = 0
        print(f"  {'Dist':>4}  {'Pairs':>14}  {'Cumul %':>8}")
        print(f"  {'----':>4}  {'----------':>14}  {'--------':>8}")
        for d in range(int(nz[0]), int(nz[-1]) + 1):
            c = int(hist[d])
            if c:
                cumul += c
                print(f"  {d:4d}  {c:>14,}  {100*cumul/total:>7.2f}%")
        print()
        n_fh = int(df["fail_low_hamming"].sum())
        print(f"  Sequences with min dist < {args.min_hamming_distance} : {n_fh:,}")
    else:
        print("  [skipped – fewer than 2 valid-length sequences]")
    print()

    # ── 5. Cross-talk ─────────────────────────────────────────
    print(LINE)
    print("  5. Cross-talk  (forward[i] ↔ RC[j ≠ i])")
    print(LINE)
    if global_min_ct is not None:
        flag_ct = global_min_ct < args.min_hamming_distance
        print(f"  Global min cross-talk distance : {global_min_ct}"
              f"{'  ← BELOW THRESHOLD' if flag_ct else ''}")
        n_ct = int(df["fail_low_crosstalk"].sum())
        print(f"  Sequences with cross-talk dist < {args.min_hamming_distance} : {n_ct:,}")
    else:
        print("  [skipped]")
    print()

    # ── Summary ───────────────────────────────────────────────
    print(LINE)
    print("  QC Summary")
    print(LINE)
    fail_cols = [
        "fail_length", "fail_rc_mismatch", "fail_gc",
        "fail_homopolymer", "fail_tandem_repeat", "fail_hairpin",
        "fail_low_hamming", "fail_low_crosstalk",
    ]
    for col in fail_cols:
        label = col.replace("fail_", "").replace("_", " ")
        n = int(df[col].sum())
        if n:
            print(f"  {label:<28} : {n:,} flagged")
    n_pass = int((~df["any_fail"]).sum())
    n_fail = int(df["any_fail"].sum())
    print()
    print(f"  PASS (all checks)    : {n_pass:,}")
    print(f"  FAIL (≥1 check)      : {n_fail:,}")
    print()
    print(f"  {args.pass_output}  →  {n_pass:,} sequences")
    print(f"  {args.fail_output}  →  {n_fail:,} sequences")
    print("=" * 62)


# ============================================================
# § 6  Argument parser
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="barcode_qc.py",
        description="QC checks for a DNA barcode library (Sidewinder junctions).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "input", metavar="CSV",
        help="Input CSV file with junction_id, barcode_forward, "
             "barcode_reverse_complement columns.",
    )

    g = p.add_argument_group("QC thresholds")
    g.add_argument("--barcode-length",        type=int,   default=BARCODE_LENGTH,
                   metavar="INT",
                   help="Expected barcode length (nt).")
    g.add_argument("--gc-min",                type=float, default=GC_MIN,
                   metavar="FLOAT",
                   help="Minimum GC fraction (inclusive).")
    g.add_argument("--gc-max",                type=float, default=GC_MAX,
                   metavar="FLOAT",
                   help="Maximum GC fraction (inclusive).")
    g.add_argument("--homopolymer-max",       type=int,   default=HOMOPOLYMER_MAX,
                   metavar="INT",
                   help="Flag runs LONGER than this many identical bases.")
    g.add_argument("--tandem-repeat-min-len", type=int,   default=TANDEM_REPEAT_MIN_LEN,
                   metavar="INT",
                   help="Flag tandem repeats (period 2–4) with total length ≥ this.")
    g.add_argument("--min-hamming",           type=int,   default=MIN_HAMMING_DISTANCE,
                   dest="min_hamming_distance", metavar="INT",
                   help="Minimum acceptable pairwise Hamming distance.")
    g.add_argument("--hairpin-min-stem",      type=int,   default=HAIRPIN_MIN_STEM,
                   metavar="INT",
                   help="Stem length (nt) that triggers a hairpin flag.")
    g.add_argument("--hairpin-min-loop",      type=int,   default=HAIRPIN_MIN_LOOP,
                   metavar="INT",
                   help="Minimum loop length (nt) for hairpin detection.")

    g2 = p.add_argument_group("performance / output")
    g2.add_argument("--chunk-size",   type=int, default=CHUNK_SIZE,
                    metavar="INT",
                    help="Rows per chunk for vectorised Hamming (tune for RAM).")
    g2.add_argument("--pass-output",  default=PASS_OUTPUT,
                    help="Output CSV for barcodes that pass all checks.")
    g2.add_argument("--fail-output",  default=FAIL_OUTPUT,
                    help="Output CSV for barcodes that fail ≥1 check.")
    return p


# ============================================================
# § 7  Main
# ============================================================

def main() -> None:
    args = build_parser().parse_args()

    # ── Load CSV ──────────────────────────────────────────────
    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"ERROR: file not found: {in_path}")

    df = pd.read_csv(in_path, dtype={"junction_id": str})

    required = {"junction_id", "barcode_forward", "barcode_reverse_complement"}
    missing  = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: CSV is missing required columns: {missing}")

    # Normalise to uppercase so comparisons are case-insensitive.
    df["barcode_forward"]            = df["barcode_forward"].str.upper()
    df["barcode_reverse_complement"] = df["barcode_reverse_complement"].str.upper()

    N = len(df)
    print(f"Loaded {N:,} barcodes from '{in_path.name}'", file=sys.stderr)

    # Warn if non-ACGT characters are present (assumed not to occur per spec).
    bad = df["barcode_forward"].str.contains(r"[^ACGT]", regex=True)
    if bad.any():
        print(f"WARNING: {bad.sum()} barcode_forward sequences contain non-ACGT "
              "characters.  Hamming distances for those rows may be unreliable.",
              file=sys.stderr)

    # ── Per-sequence checks ───────────────────────────────────

    # Compute GC fraction and true reverse complement once.
    df["gc_fraction"] = df["barcode_forward"].map(gc_fraction)
    df["computed_rc"] = df["barcode_forward"].map(reverse_complement)

    # 1. Length uniformity
    df["fail_length"] = df["barcode_forward"].str.len() != args.barcode_length

    # 2. Reverse-complement integrity
    df["fail_rc_mismatch"] = df["barcode_reverse_complement"] != df["computed_rc"]

    # 3. GC content outside the allowed window
    df["fail_gc"] = (
        (df["gc_fraction"] < args.gc_min) | (df["gc_fraction"] > args.gc_max)
    )

    # 4. Homopolymer runs exceeding the threshold
    df["fail_homopolymer"] = df["barcode_forward"].map(
        lambda s: check_homopolymer(s, args.homopolymer_max)
    )

    # 5. Simple tandem repeats (period 2–4)
    df["fail_tandem_repeat"] = df["barcode_forward"].map(
        lambda s: check_tandem_repeat(s, args.tandem_repeat_min_len)
    )

    # 6. Potential hairpin (self-complementarity heuristic)
    print("Running hairpin screen …", file=sys.stderr)
    df["fail_hairpin"] = df["barcode_forward"].map(
        lambda s: check_hairpin(s, args.hairpin_min_stem, args.hairpin_min_loop)
    )

    # ── Pairwise and cross-talk Hamming ──────────────────────
    # Restrict to valid-length rows only; different-length sequences cannot be
    # compared by Hamming distance.
    valid_mask = ~df["fail_length"]
    fwd_valid  = df.loc[valid_mask, "barcode_forward"].tolist()
    rc_valid   = df.loc[valid_mask, "computed_rc"].tolist()   # use computed RC

    # Initialise columns for the whole dataframe.
    df["fail_low_hamming"]   = False
    df["fail_low_crosstalk"] = False
    df["min_hamming_dist"]   = pd.NA
    df["min_crosstalk_dist"] = pd.NA

    global_min_ham = None
    global_min_ct  = None
    hist           = None

    if len(fwd_valid) >= 2:
        fwd_mat = _encode(fwd_valid)
        rc_mat  = _encode(rc_valid)

        print("Computing pairwise Hamming distances …", file=sys.stderr)
        per_min, hist, global_min_ham = compute_pairwise_hamming(
            fwd_mat, args.chunk_size
        )

        print("Computing cross-talk Hamming distances …", file=sys.stderr)
        per_ct, global_min_ct = compute_crosstalk_hamming(
            fwd_mat, rc_mat, args.chunk_size
        )

        # Write per-sequence results back to the dataframe.
        valid_idx = df.index[valid_mask]
        df.loc[valid_idx, "min_hamming_dist"]   = per_min.astype(float)
        df.loc[valid_idx, "min_crosstalk_dist"] = per_ct.astype(float)
        df.loc[valid_idx, "fail_low_hamming"]   = per_min < args.min_hamming_distance
        df.loc[valid_idx, "fail_low_crosstalk"] = per_ct  < args.min_hamming_distance
    else:
        print(
            "WARNING: fewer than 2 valid-length sequences — "
            "Hamming checks skipped.",
            file=sys.stderr,
        )

    # ── Aggregate overall pass / fail ─────────────────────────
    fail_cols = [
        "fail_length", "fail_rc_mismatch", "fail_gc",
        "fail_homopolymer", "fail_tandem_repeat", "fail_hairpin",
        "fail_low_hamming", "fail_low_crosstalk",
    ]
    df["any_fail"] = df[fail_cols].any(axis=1)

    # ── Print QC report ───────────────────────────────────────
    print_summary(df, args, global_min_ham, global_min_ct, hist)

    # ── Write output CSVs ─────────────────────────────────────
    info_cols = [
        "junction_id", "barcode_forward", "barcode_reverse_complement",
        "gc_fraction", "min_hamming_dist", "min_crosstalk_dist",
    ]

    # Pass CSV: informational columns only (no fail_ columns needed).
    df_pass = df.loc[~df["any_fail"], info_cols].copy()
    df_pass.to_csv(args.pass_output, index=False)

    # Fail CSV: informational columns + per-check reason flags.
    df_fail = df.loc[df["any_fail"], info_cols + fail_cols].copy()
    df_fail.to_csv(args.fail_output, index=False)


if __name__ == "__main__":
    main()

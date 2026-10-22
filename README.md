# seqconsistency

`seqconsistency` assesses whether multiple sequencing datasets are structurally valid,
technically consistent, and suitable for pooling under an explicit, configurable
policy. It targets multiple technical sequencing runs declared to originate from the
**same biological sample**. It never modifies or merges source files.

- **Inputs**: single-end and paired-end FASTQ (plain or gzip), coordinate-sorted BAM
  (via HTSlib through `pysam`).
- **Comparison profiles**: `paired_short`, `merged_short`, `ont`, `hifi`.
- **Outputs**: five Parquet tables + provenance JSON (each with its own required CLI
  path), optional self-contained HTML report.
- **Status**: Phase 1 (mandatory MVP) of the project roadmap. See
  [Implemented / not implemented](#implemented--not-implemented).

> **Pooling** here means combining sequencing datasets, not merging overlapping mate
> sequences.

---

## Installation

Python ≥ 3.11, Linux (the development platform; no Windows support).

```bash
# from the repository (pixi workspace)
pixi install
pixi run test
# or plain pip
pip install .
```

Dependencies: `numpy`, `scipy`, `pyarrow`, `pysam`, `PyYAML`.

## Quick start

```bash
seqconsistency check \
    --manifest inputs.tsv \
    --policy policies/example-illustrative.yaml \
    --metrics-out metrics.parquet \
    --distributions-out distributions.parquet \
    --comparisons-out comparisons.parquet \
    --findings-out findings.parquet \
    --decisions-out decisions.parquet \
    --provenance-out provenance.json \
    --report-out report.html \
    --workers 4 --threads 8 --sample-size 100000 --seed 42
```

Every persistent output has its own explicit CLI path. There is no
`--output-prefix`, no `--output-dir`, no implicit caches, sketches, or sidecar files.
Outputs whose parent directory does not exist are rejected (directories are never
created implicitly); existing outputs are rejected unless `--overwrite` is passed;
duplicate output paths (including resolved aliases such as symlinks) and outputs
colliding with input files are rejected. Collision protection covers **every consumed input** — the manifest, the policy file, all `path1`/`path2` data files, and fastp reports — so `--overwrite` can never destroy an input. Writes are staged to temporary sibling files
and published with `os.replace` only after all computation succeeded. Publication
across multiple independent filesystem paths is **not transactional** — on failure the
run removes what it published, but a crash between renames can leave a partial set.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Run completed; every group is `COMPATIBLE`. |
| 1 | Run completed; one or more groups are not `COMPATIBLE` (scientific result). |
| 2 | Input, configuration, dependency, or execution failure. |

Scientific outcomes (`REVIEW`, `INCOMPATIBLE`, `INSUFFICIENT_EVIDENCE`) are results in
the tables and exit code 1 — never exit code 2.

## Input manifest

Parquet (`.parquet`/`.pq`, primary) or TSV (`.tsv`/`.tab`/`.txt`). One row per logical
dataset. Relative paths resolve against the manifest's parent directory.

| Column | Required | Meaning |
|---|---|---|
| `dataset_id` | yes | Globally unique dataset identifier |
| `pool_group` | yes | Proposed pooling group (≥ 2 datasets per group) |
| `sample_id` | yes | Declared biological sample |
| `library_id` | yes | Library identifier |
| `assay` | yes | WGS, WES, RNA-seq, amplicon, … |
| `platform` | yes | Sequencing platform (free text; compared within groups) |
| `read_type` | yes | `paired_short` \| `merged_short` \| `ont` \| `hifi` |
| `input_format` | yes | `fastq` \| `bam` |
| `path1` | yes | SE/R1 FASTQ or BAM |
| `path2` | PE FASTQ only | R2 FASTQ; required for `fastq`+`paired_short`, forbidden otherwise |
| `processing_id` | no | Identifier for preprocessing settings |
| `reference_id` | no | Reference identifier (BAM) |
| `fastp_json` | no | fastp report for merged reads |

Unknown columns are rejected. Validation (all failures are errors, exit 2): duplicate
`dataset_id`; group with < 2 datasets; mixed `input_format` or `read_type` within a
group (a v1 restriction — the error says so); wrong path arity; missing/unreadable
files; the same underlying file used twice within one group (alias-safe via device and
inode). The same file used in **different** groups is allowed and recorded as an info
finding.

Biological identity is never inferred from filenames. Identical `sample_id` values
establish **declared** identity only.

## Policy (YAML)

See [`policies/example-illustrative.yaml`](policies/example-illustrative.yaml). Strictly
validated: unknown keys are rejected, never silently ignored; booleans must be real
YAML booleans (`calibrated: "false"` is rejected — string coercion would silently
flip it to true); threshold `max` must be finite and non-negative; every
`(metric, method)` pair must be one this version implements (typos fail at load,
not silently never-match); scope selectors must be string lists; and two rules for
the same metric with overlapping profile/stratum/assay scopes are rejected
(repeating a metric across disjoint profile scopes is fine). Top-level sections:

```yaml
policy_id: my-policy
policy_version: "1"
identity:
  accept_declared_sample_id: true    # declared sample_id accepted as identity evidence
  require_independent_verification: false  # true => INSUFFICIENT_EVIDENCE in v1 (no method)
metadata:
  same_sample_id_required: true
  same_assay_required: true
  distinct_library_ids: review       # allowed | review | incompatible
  distinct_processing_ids: review
  reference_mismatch: incompatible   # review | incompatible
  require_fastp_evidence: false
technical:
  min_observations: 1000
  require_thresholds: true           # metrics without thresholds block COMPATIBLE
  thresholds:
    - metric: read_length
      method: wasserstein1
      max: 10.0          # distance must be <= max
      action: review     # review | incompatible
      calibrated: false  # MUST be false until you calibrate with real data
      profiles: [paired_short]   # optional scoping
      strata: [R1, R2]           # optional scoping
      assays: [WGS]              # optional scoping
```

**The shipped thresholds are ILLUSTRATIVE AND UNCALIBRATED.** They demonstrate
mechanics only and are never described as validated. Per the specification ("if no
threshold is configured **or calibrated**, report the metric but do not use it to
automatically approve pooling"):

1. A metric with **no threshold** is reported but never approves: with
   `require_thresholds: true` (default) it blocks `COMPATIBLE`
   (`TECH_NO_THRESHOLD_CONFIGURED`, group `INSUFFICIENT_EVIDENCE`). With
   `require_thresholds: false` each unthresholded distance is reported with an INFO
   finding instead — but if **no** comparison in the group was assessed against a
   threshold at all, the technical component is `MISSING_EVIDENCE` and the group is
   `INSUFFICIENT_EVIDENCE` regardless: the opt-out makes the run a report, never an
   approval.
2. A **PASS against a `calibrated: false` threshold** is reported with
   `threshold_calibrated=false` in the comparisons table and withholds automatic
   approval: the group falls to `REVIEW` with a `TECH_THRESHOLD_UNCALIBRATED`
   finding. The distances were assessed; only the verdict is untrusted — so this is
   a REVIEW, not INSUFFICIENT_EVIDENCE. An **EXCEED** stands regardless of
   calibration: a measured distance above its maximum is real evidence of a shift.
3. Consequently the shipped example policy can never yield `COMPATIBLE` — by
   design. Calibrate thresholds against genuine between-run data and mark them
   `calibrated: true` to enable automatic approval.

### Threshold calibration

1. Collect genuine compatible material: multiple independent sequencing runs of the
   same library, and different libraries of the same sample, on **your** platform and
   assay.
2. Collect known incompatibilities: different samples, degraded runs, wrong-platform
   data.
3. Compute the distances on those sets (`seqconsistency check` on manifests of each
   category; the comparisons table carries every value).
4. Choose thresholds that separate the two empirical distributions for your context.
   Mark them `calibrated: true` only after this.
5. **Random splits of one FASTQ estimate sampling variation, not real between-run
   variation.** They must not be used alone to claim thresholds are calibrated for
   independent sequencing runs.

No threshold in this package is scientifically validated, and the package never claims
biological validation from synthetic software tests.

## FASTQ validation behavior

- Gzip compression is detected by the `.gz` filename suffix (there is no content
  sniffing).
- **Strict four-line records** (`@header`, sequence, `+`, qualities). Wrapped /
  multi-line FASTQ records are **rejected** with a clear error — a documented v1
  restriction (chosen over partial multiline support).
- The **entire file is always scanned** for structural validation and full-file
  counters; sampling reduces memory and per-read statistics, never validation
  coverage. There is no first-N-reads mode.
- Structural checks (first defect aborts with record number and read name): header
  and separator lines, sequence/quality length equality, complete records at EOF,
  nonempty file, gzip integrity (corrupt gzip aborts), supported alphabet
  (`A/C/G/T/N`, upper or lower case; other IUPAC codes are unsupported in v1 and
  abort), quality bytes within ASCII `[33, 126]`.
- **Quality encoding: Phred+33 only.** ASCII quality ranges alone cannot always
  unambiguously distinguish Phred+33 from Phred+64. Bytes outside `[33, 126]` abort
  (provably not Phred+33). A file is additionally rejected as carrying the classic
  Phred+64 signature when its full-file minimum quality byte is ≥ 64 (`@`) **and**
  maximum ≥ 90 (`Z`): under Phred+33 that means every base ≥ Q31 with some > Q57,
  which no platform emits, while under Phred+64 it means Q26, which is routine.
  Uniform high-quality Phred+33 files (for example every byte `K`, uniform Q42)
  are accepted. Files with any low-quality base (byte < 64) are unambiguous
  Phred+33 and accepted. The exact decision rule: a file is rejected only when
  **every** quality byte in the whole file is ≥ 64 **and** at least one is ≥ 90 —
  the rejection window is *entirely* uniform-high files (every base ≥ Q31, some
  ≥ Q58), which no standard platform or common simulator emits; a mixed file
  containing Q60 records alongside ordinary ones is accepted. A rejected file
  must be resolved by **verifying its true encoding**, never by modifying the
  data under assessment: a genuinely Phred+64 file must be re-exported as
  Phred+33 from its source pipeline (the tool would otherwise read every
  quality silently shifted by +31), and a genuinely uniform-high Phred+33 file
  is a case to report upstream. This tool never alters or advises altering
  input data. v1 has no encoding override flag by design, because the same
  signature is produced by both encodings and a flag would only move the guess
  from the tool to the operator.
- **Paired-end synchronization** compares corresponding records — not just counts.
  Read names are normalized by taking the first whitespace-delimited token and
  removing a trailing `/1` or `/2` (covering legacy suffixes and Illumina/CASAVA
  `name 1:N:0:INDEX` comments). Arbitrary substrings are never stripped to force
  synchronization. Mate annotations are recognized from **both** the legacy
  `/1`/`/2` suffix and the CASAVA comment mate number; the run aborts on the first
  mismatched pair, on contradictory annotations (the R1 file carrying a mate-2
  annotation or the R2 file a mate-1 annotation — from suffix or CASAVA comment),
  on a header whose suffix and CASAVA comment disagree with each other, or on
  unequal mate counts.
- PE sampling unit is the **mate pair**: R1 and R2 features of one pair always come
  from the same pair.
- R1 and R2 metrics are computed separately and compared R1-vs-R1, R2-vs-R2. R1 and R2
  distributions are never required to match each other.

## BAM validation behavior

- `pysam`/HTSlib; sequential scans only; no index required and none created.
- The header must declare `SO:coordinate`, **and** the observed (tid, pos) sequence
  must be non-decreasing with unplaced (tid −1) records allowed only at the end. A
  header declaration alone never proves order — the wrong-order error names both
  coordinates.
- Alignment categories are counted separately: primary, secondary, supplementary,
  unmapped, duplicate-marked, QC-failed, paired, properly-paired. Supplementary and
  secondary records are **never** counted as independent sequencing reads.
- Read-level metrics (GC, per-read quality) use eligible primary records. Missing
  base qualities or sequence count as **unavailable (NaN), never zero**, with
  dedicated counters.
- Insert size: **one observation per pair** — primary, paired, properly-paired,
  first-in-pair, `TLEN != 0` (absolute value used). The SAM spec defines TLEN as
  negative on the rightmost segment, so a proper pair whose first segment is the
  rightmost one carries `TLEN < 0`; requiring `TLEN > 0` would silently drop those
  pairs and bias the distribution. This never builds a mate dictionary. Documented
  limitation: pairs without the proper-pair flag or with `TLEN == 0` (unknown
  template length) are excluded, so insert-size observations can undercount pairs
  in libraries with many discordant mates; the observation count is reported.
  Insert-size comparison applies only to the `paired_short` profile — unpaired and
  long-read BAMs get no insert-size requirement (inapplicable evidence must never
  block).
- Reference dictionaries are compared within a group: contig names, lengths, and
  order must match (mismatch ⇒ `INCOMPATIBLE` finding). `M5` checksums are
  aggregated per contig across **every** BAM in the group, independent of manifest
  order: two or more distinct values for one contig ⇒ `INCOMPATIBLE`
  (`BAM_REFERENCE_CHECKSUM_CONFLICT`). When any checksum is absent the group gets a
  `BAM_REFERENCE_IDENTITY_UNVERIFIED` warning — matching names and lengths do not
  prove sequence identity.
- Read-group IDs may legitimately differ across runs; an ID **redefined
  differently** in two datasets is a `BAM_READ_GROUP_COLLISION` incompatibility.
- Header sample metadata is reconciled with the manifest: several distinct
  non-empty `SM` values across a group is a declared-identity conflict
  (`BAM_SAMPLE_METADATA_CONFLICT` ⇒ `INCOMPATIBLE`); one `SM` contradicting the
  manifest `sample_id` is a `BAM_SM_MISMATCHES_MANIFEST` review finding. Records
  referencing read-group IDs missing from the header are counted full-scan and
  reported (`BAM_UNDEFINED_RG_RECORDS`) — merge tools drop or misattribute them.
- Clipping summaries are collected full-scan on primary records: soft+hard clipped
  bases per read and the clipped-record fraction.
- Duplicate-marked fractions depend entirely on upstream processing; they are never
  interpreted as independently measured PCR duplication rates.
- Per-read GC/N/mean-quality histograms and quality comparisons for BAM come from the
  sampled eligible primary records (`sampling_mode=sample`); category counters,
  length, MAPQ, and insert-size histograms are full-scan.

## Metrics

Every metrics row states: unit of observation, denominator, observation count,
sampling mode (`full_scan` or `sample`), and availability
(`available`/`not_applicable`/`insufficient_observations`/`missing_data`). Missing or
undefined values are null — never zero.

FASTQ (per stratum R1/R2/SE, full-scan unless noted):

- `reads/total`, `bases/total`
- `read_length` mean/std (population, full-scan sums), and median/p10/p90 from the
  sample
- `gc_content/per_read_mean` — mean over reads of per-read GC fraction
  (G+C)/(A+C+G+T); reads with no ACGT bases are excluded from the denominator
- `n_content/base_fraction` — N bases / total bases
- `base_quality` `q20_fraction`, `q30_fraction` — bases ≥ Q20 / Q30 over total bases
- `read_quality`:
  - `mean_of_arithmetic_mean` — mean over reads of the arithmetic mean Phred of the
    read's bases
  - `mean_of_error_based_mean` — mean over reads of
    −10·log10(mean<sub>bases</sub> 10^(−Q/10))
  Both definitions are reported because they differ systematically: the arithmetic
  mean over-weights high qualities; the error-based mean is dominated by low-quality
  bases.
- Nucleotide composition (A/C/G/T/N base fractions)
- Per-cycle profiles for short reads (`paired_short` ≤ 512 cycles, `merged_short` ≤
  1024): per-cycle coverage, mean quality, and base composition; positions beyond the
  cap are counted in an overflow counter. Long-read profiles (`ont`, `hifi`) have no
  per-cycle metrics (no fixed cycles exist) — reported as not applicable, never
  faked.

BAM (strata `READS` pooled, `R1`/`R2` per mate for `paired_short`, plus `PAIR` for
insert size): category counts and fractions with exact denominators (primary
records; paired fraction over primary; properly-paired over paired), mapped/unmapped
fractions, duplicate/QC-fail fractions, mean/std MAPQ, insert-size mean/std
(full-scan sums) plus sample quantiles, eligible-primary read count,
missing-quality/sequence counts, and the FASTQ-equivalent sampled read metrics. For
`paired_short`, the sampled read metrics (length, mean quality, GC, N) are kept
**per mate** (`R1`, `R2`) so comparisons are R1-vs-R1 and R2-vs-R2 — mirroring PE
FASTQ; the pooled `READS` sample then holds only unpaired records, and `mapq`
(full-scan over every mapped primary) stays on `READS`.

`merged_short` datasets may carry `fastp_json` evidence: before/after total reads,
Q30 rates, and a merge rate **only when the report contains an explicit merged-read
count under a recognized key**. A merged FASTQ alone never implies a merge rate
(merged-read distributions carry strong selection bias and are never interpreted as
raw R1/R2 distributions). `fastp --include_unmerged` changes output composition and
must be represented through `processing_id`. fastp schemas vary by version; missing
fields stay null. Pairwise processing comparison prefers **`merge_rate`**
(merged/before — isolates merging) when both reports carry merged-read counts, and
falls back to **`merge_retained_fraction`** (after/before — conflates filtering and
merging; the name says so). When fastp evidence is absent it is `NOT_ASSESSED`
unless the policy sets `require_fastp_evidence: true`, in which case it is
`MISSING_EVIDENCE` — optional evidence never blocks compatibility.

A joint read-length × mean-quality summary (spec §6.1) is computed from each
dataset's sample: 16 length bins (short profiles: 64 bp bins to 1024; long reads:
a log-style ladder) × 13 quality bins of 8 Phred, emitted as
`joint_length_quality` distribution rows (`bin_index` = length bin, `category` =
`q00`..`q12`).

## Sampling

- **Algorithm**: key-based reservoir — every eligible observation receives an iid
  uniform key; the retained sample is the k smallest keys (a uniform sample without
  replacement). Equivalent to algorithm R in distribution, but the key draw is
  vectorized per block.
- **Determinism**: per-dataset seed = `blake2b-128("{master_seed}:{dataset_id}")`;
  PCG64 stream in stream order. Python's randomized `hash()` is never used. Results
  are identical across `--workers` values (tested) and independent of scheduling.
- **Sampling units**: read (SE FASTQ), mate pair (PE FASTQ — one key per pair, shared
  by R1 and R2), eligible primary record (BAM), insert-eligible pair (BAM insert
  sample).
- **Long-read memory**: only compact numeric features (length, GC, N fraction, mean
  qualities) are retained — never sequences. Retained memory scales with sample size
  and fixed histogram bins, not with read count (verified by test and benchmark).
- Reported per dataset in provenance: full eligible count, sampled count, fraction,
  unit, algorithm, and the per-dataset seed.
- The beginning of a coordinate-sorted BAM is never used as a sample: the reservoir
  spans the whole stream.

## Distributions and binning

All histograms use deterministic shared bins (module
`seqconsistency.distributions`): bin *i* covers `[edges[i], edges[i+1])`; the final
bin is closed on the right; values beyond the last edge are counted as overflow
(never dropped); non-finite values count as underflow so
`bins + overflow + underflow = observations`.

| Histogram | Bins |
|---|---|
| per-read GC | 50 × 0.02 over [0, 1] |
| per-read N fraction | 20 × 0.05 over [0, 1] |
| read length (short) | unit bins 0–1025 bp, overflow above |
| read length (long) | log2-style ladder 0–1,048,576 bp, overflow above |
| base quality | one bin per Phred 0–93 |
| MAPQ | one bin per value 0–255 |
| insert size | 10 bp bins to 20,000 bp, overflow above |
| per-read mean quality | unit Phred bins 0–94 |

Nucleotide composition is categorical. Per-cycle rows use absolute cycles
(`bin_start` = cycle, `bin_end` = cycle+1) with the per-cycle statistic in the
`value` column.

## Statistical distances

- **Wasserstein-1** (`wasserstein1`): empirical, on the per-read sample of each
  dataset. Each dataset's sample is finite-filtered and sorted **once**; pairs are
  compared with an exact sorted-input implementation of the same quantity SciPy
  computes (`statistics.wasserstein_distance_sorted`, method version
  `sorted-merge-w1-jsd-base2-v2`) — SciPy re-sorts both inputs on every call, which
  dominated comparison cost for large groups. Equivalence against
  `scipy.stats.wasserstein_distance` is locked by test (random sizes, heavy ties,
  identical arrays); the reference implementation remains available in
  `statistics.wasserstein_distance`. Units: bp (lengths), Phred (qualities).
  Undefined per-read values (NaN) are excluded pairwise; observation counts after
  exclusion are reported.
- **Jensen–Shannon distance** (`jensenshannon_base2`): the actual JS **distance**
  (square root of the divergence), logarithm base 2, bounded [0, 1]
  (`scipy.spatial.distance.jensenshannon`). Inputs are count histograms over the
  identical deterministic bins above. No pseudocount: zero-count bins on both sides
  contribute zero; an all-zero histogram on either side is missing evidence and
  raises — it is never treated as distance 0 (which would silently pass).
- p-values are deliberately not used: at sequencing-data sample sizes, microscopic
  differences become "significant". Assessment uses distance magnitudes against
  policy thresholds only.
- Comparisons are emitted for every applicable (metric, stratum) of every pair;
  `dataset_a < dataset_b` lexicographically, so no mirrored duplicates exist.
- **Pairwise consistency is not transitive.** A~B and B~C do not imply A~C; the
  engine never assumes transitivity (tested).

## Decision engine

Components per pooling group, each with its own status:

| Component | Covers |
|---|---|
| `structural_status` | Input structural validity. A structural failure aborts the run (exit 2) before any decision exists, so in a completed report this is always `PASS`. |
| `technical_status` | Comparison outcomes, min-observation requirements, threshold presence |
| `metadata_status` | sample/assay/library/platform/processing/reference consistency, BAM dictionaries and read groups |
| `identity_status` | Declared vs conflicting vs independently verified identity |

Check statuses (never collapsed): `PASS`, `REVIEW`, `FAIL`, `MISSING_EVIDENCE`,
`INSUFFICIENT_OBSERVATIONS`, `NOT_REQUESTED`, `UNSUPPORTED`.

Group decision = aggregation with fixed precedence:

1. any hard incompatibility (any component `FAIL`) ⇒ `INCOMPATIBLE`;
2. else any missing required evidence ⇒ `INSUFFICIENT_EVIDENCE`;
3. else any actionable warning ⇒ `REVIEW`;
4. else `COMPATIBLE`.

Optional unrequested checks never block compatibility.

### Identity semantics

- Identical declared `sample_id` + `identity.accept_declared_sample_id: true` ⇒
  identity PASS with evidence level `declared_not_verified`, and the report states:
  **"Sample identity accepted from supplied metadata; not independently verified."**
  Biological identity is never claimed to be proven.
- Different `sample_id` within a group ⇒ identity and metadata `FAIL`
  (`GROUP_DIFFERENT_SAMPLE_IDS`) ⇒ `INCOMPATIBLE`.
- `identity.require_independent_verification: true` ⇒ v1 implements no
  genotype-based verification, so the group is `INSUFFICIENT_EVIDENCE` with an
  `IDENTITY_VERIFICATION_NOT_IMPLEMENTED` finding (never a silent pass).
- Similar GC/quality/length/k-mer distributions would not establish sample identity
  (stated in every report).

### Pooling compatibility rules

- PE FASTQ pools mate-wise (R1 with R1, R2 with R2) when structure and intended
  processing are compatible.
- SE and PE FASTQ cannot be concatenated into an ordinary PE pair — mixed layouts in
  one group are rejected at manifest validation.
- Different SE technologies may be syntactically concatenable but are not
  homogeneous data: mixed platforms in a group produce a `GROUP_DIFFERENT_PLATFORMS`
  warning.
- Coordinate-sorted BAM must be combined with BAM-aware merging (headers,
  references, read-group collision handling) — never byte concatenation. The
  reference/read-group findings above exist precisely to flag merge hazards.
- Technical pooling is not statistical combination of biological replicates; the
  tool does not approve pooling biological replicates automatically, and every
  decision carries the `technical_pooling_not_statistical_merge_of_biological_replicates`
  assumption.

## Outputs

All tables are Parquet (zstd), explicit versioned Arrow schemas (`schema_version=1`
in schema metadata), normalized long format (no opaque JSON columns), readable by
pyarrow/pandas/polars/duckdb, with valid schemas even when empty.

- `--metrics-out`: one row per dataset/metric/statistic/stratum (columns per spec
  §10.1 including `sampling_mode` and `availability`).
- `--distributions-out`: one row per bin; adds a nullable `value` column used for
  per-cycle statistics.
- `--comparisons-out`: one row per pair/stratum/metric with method, value, unit,
  threshold, direction, status, observation counts, reason code, method version.
- `--findings-out`: one row per finding: stable `reason_code`, severity, message,
  evidence metric/value, recommended action, affected datasets (typed list).
- `--decisions-out`: one row per group: five statuses, decision, identity evidence
  level, assumptions and reason codes (typed lists), policy id/version.
- `--provenance-out`: JSON — package/dependency versions, CLI arguments, policy id
  and SHA-256, manifest path/SHA-256/schema version, per-input file metadata
  (size/mtime; SHA-256 only with `--hash-inputs`), per-dataset sampling config and
  seeds, histogram parameters, worker/thread allocation, timings, output schema
  versions.
- `--report-out` (optional): self-contained HTML (inline CSS/SVG, no network).
  Display limits (findings ≤ 200, comparison rows ≤ 500, histogram bars ≤ 60) are
  stated on the page; Parquet outputs always carry complete results.

Staged writes: each file is written to a `.name.tmp-<pid>` sibling and published with
`os.replace` after all computations succeeded. On failure, staged temporaries are
removed and anything already published by this run is removed best-effort.

## Performance, memory, and CPU control

- FASTQ processing is one streaming pass per file (validation + counters +
  histograms + sampling combined); per-base work is vectorized numpy over
  concatenated record buffers (cumsum differencing) — there are no per-base Python
  loops. Batches flush at 16,384 records or 8 MiB of sequence, whichever comes first;
  peak memory is bounded by the batch budget, not by file size or read count.
- Long reads: only compact numeric features are retained per sampled read; no
  sequences. Memory scales with `--sample-size`, histogram bins, and output batches.
- All-pairs comparisons are quadratic in dataset count; `--max-pairs` (default
  10,000) aborts the run **before** expensive processing when the configured pair
  count is exceeded. Pairs are never silently omitted.
- The comparison stage streams: each pair's rows go straight into the staged
  Parquet writers (comparisons and findings), the technical assessment aggregates
  incrementally, and the optional HTML report retains only a bounded preview
  (findings ≤ 200, comparison rows ≤ 500 — the same limits the page displays).
  In-memory comparison state is one pair's rows plus one group's finding objects;
  the Parquet outputs always carry every row. Metric rows are written as a
  generator, never a whole-run list.
- `--workers` = maximum simultaneous dataset scans (processes);
  `--threads` = total logical CPU budget. Deterministic resolution rule:
  `workers = min(--workers, --threads)`,
  `threads_per_worker = max(1, threads // workers)`, remainder to the main process
  (comparisons, Parquet writing). Native thread pools (BLAS/OpenMP) are capped **at
  runtime** with `threadpoolctl` — per worker in the scan processes and to
  `main_threads` around the comparison stage — and pyarrow through its own API.
  Runtime limits are used because setting `OMP/OPENBLAS/MKL_NUM_THREADS` after
  NumPy loads caps nothing (measured: 12 OpenBLAS threads before and after late
  env changes); the actual pool settings are verified by test
  (`tests/test_resources.py`). Contradictory settings are resolved by this
  documented rule, not rejected.
- Worker exceptions propagate with dataset context, cancel pending futures, and no
  outputs are published (tested).
- Measured throughput/memory for this code version: see
  [benchmarks/results-2026-10-21.md](benchmarks/results-2026-10-21.md) (real
  measurements recorded by `benchmarks/bench.py`; regenerate with
  `pixi run -- python benchmarks/bench.py run --data-dir <dir> --results-out <json>`).
  Headline (x86_64 Linux, Python 3.12, single scan process): SE FASTQ 128k reads/s
  (12.8 M bases/s; gzip ≈ same), PE FASTQ 57.4k pairs/s including per-pair name
  synchronization, long-read FASTQ 23.3 M bases/s at ≈0.5 GiB peak RSS, BAM 64k
  records/s (pysam per-record access dominates; the per-mate reservoirs cost
  nothing measurable), peak RSS flat when reads grow 4× at fixed
  `--sample-size`. After the sorted-input Wasserstein and streaming comparison
  stage, pair-scaling wall time dropped 32–65% across 5–40-dataset groups versus
  the previous measurement of the same workloads (same host).

## Python API

Thin, typed, import-stable modules; the CLI is a wrapper around
`seqconsistency.pipeline.run_check(CheckConfig)`.

```python
from pathlib import Path
from seqconsistency.fastq import scan_fastq_single
from seqconsistency.models import ReadType, SamplingConfig

stats, sample, eligible, sampled = scan_fastq_single(
    "run1", Path("reads.fastq"), ReadType.ONT,
    SamplingConfig(sample_size=50_000, master_seed=42), 50_000,
)
print(stats.total_reads, stats.total_bases, eligible, sampled)
```

## Examples

Runnable end-to-end examples live in [`examples/`](examples/) with a generator
script (`examples/make_examples.py`) that writes synthetic FASTQ/BAM inputs,
manifests, and policies for the scenarios below:

1. Multiple Illumina PE runs of one library → `examples/01_paired_short/`
   (REVIEW — the example policy's thresholds are uncalibrated, so approval is
   withheld with `TECH_THRESHOLD_UNCALIBRATED` findings)
2. Merged-read FASTQ datasets (`merged_short`) → `examples/02_merged_short/`
3. ONT / HiFi long-read datasets → `examples/03_ont/`
4. Coordinate-sorted BAM datasets → `examples/04_bam/`
5. Shifted GC content producing an exceed finding → `examples/05_shifted_gc/`
6. Genuine technical incompatibility (mixed samples) → `examples/06_incompatible/`
   (INCOMPATIBLE — hard failure outranks the uncalibrated REVIEW)
7. Missing identity evidence → INSUFFICIENT_EVIDENCE (policy requiring independent
   verification) → `examples/07_insufficient_identity/`
8. Multiple pooling groups in one invocation → `examples/08_multi_group/`

With the shipped uncalibrated policy, scenarios 1–5 and 8 end in `REVIEW`; a
`COMPATIBLE` outcome requires thresholds marked `calibrated: true` after genuine
calibration.

```bash
python examples/make_examples.py --out-dir /tmp/seqc-examples
seqconsistency check --manifest /tmp/seqc-examples/01_paired_short/manifest.tsv \
    --policy policies/example-illustrative.yaml \
    --metrics-out /tmp/m.parquet --distributions-out /tmp/d.parquet \
    --comparisons-out /tmp/c.parquet --findings-out /tmp/f.parquet \
    --decisions-out /tmp/dec.parquet --provenance-out /tmp/p.json
```

## Limitations

- **No independent biological identity verification.** No SNP/genotype concordance
  (NGSCheckMate-style), no contamination screening. Identity status always says
  declared-not-verified unless an external tool supplies evidence. (Phase 3.)
- **No sequence-content (k-mer/sourmash) comparison.** High k-mer similarity would
  not prove identity anyway. (Phase 2.)
- **No reference-aware coverage comparison.** A naive genome-wide correlation would
  mislead for targeted/WGBS-style libraries; deferred until assay-aware binning
  exists. (Phase 2.)
- Strict four-line FASTQ; no wrapped records. Phred+33 only. `A/C/G/T/N` alphabet
  only.
- v1 pooling groups must be format- and profile-homogeneous (clearly reported as a
  v1 restriction, not an impossibility claim).
- Per-position metrics cap at 512 (`paired_short`) / 1024 (`merged_short`) cycles.
- Pairwise decisions are not transitive; coverage of "poolability" beyond pairwise
  structure is out of scope.
- Thresholds are never scientifically validated by this package; calibration is the
  user's responsibility (see above).
- Cross-technology comparisons are not attempted: profiles define separate metric
  sets and thresholds; there is no universal distribution threshold.
- Linux only.

## Implemented / not implemented

**Implemented (Phase 1, tested):** installable package + CLI; strict manifest
(Parquet/TSV) and policy parsing; explicit output-path validation; single-pass FASTQ
streaming/validation/metrics (SE/PE/gzip); BAM streaming/validation/metrics;
deterministic bounded-memory reservoir sampling; profile-aware histograms;
Wasserstein-1 and Jensen–Shannon (base-2 distance) comparisons; all-pairs guard;
policy decision engine with stable reason codes; five Parquet tables + provenance
JSON; optional HTML report; fastp JSON evidence (basic); benchmarks with measured
results.

**Not implemented (documented placeholders, no silent simulation):** sourmash
sequence-content comparison; reference-aware coverage comparison; processing-aware
merged-read checks beyond basic fastp fields; group outlier visualization beyond
per-dataset histograms; genotype-based identity verification.

## Development

```bash
pixi run test         # full suite (includes slow memory tests)
pixi run test-fast    # skip slow tests
pixi run lint         # ruff check
pixi run format       # ruff format
pixi run typecheck    # mypy
```

## License

MIT.

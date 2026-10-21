#!/usr/bin/env bash
# Run every generated example end-to-end and print each group's decision.
# Usage: examples/run_examples.sh <examples-root> [work-dir]
# Requires: seqconsistency on PATH (e.g. `pixi shell`), python3 with pyarrow.
set -euo pipefail

ROOT="${1:?usage: run_examples.sh <examples-root> [work-dir]}"
WORK="${2:-$(mktemp -d)}"
POLICY="$(dirname "$0")/../policies/example-illustrative.yaml"

for manifest in "$ROOT"/*/manifest.tsv; do
    ex_dir="$(dirname "$manifest")"
    name="$(basename "$ex_dir")"
    policy="$POLICY"
    if [ -f "$ex_dir/policy-strict.yaml" ]; then
        policy="$ex_dir/policy-strict.yaml"
    fi
    out="$WORK/$name"
    mkdir -p "$out"
    echo "== $name (policy: $(basename "$policy"))"
    seqconsistency check \
        --manifest "$manifest" \
        --policy "$policy" \
        --metrics-out "$out/metrics.parquet" \
        --distributions-out "$out/distributions.parquet" \
        --comparisons-out "$out/comparisons.parquet" \
        --findings-out "$out/findings.parquet" \
        --decisions-out "$out/decisions.parquet" \
        --provenance-out "$out/provenance.json" \
        --workers 2 --threads 2 --sample-size 20000 --seed 42 \
        || code=$?
    code="${code:-0}"
    echo "   exit=$code"
    python3 - "$out/decisions.parquet" <<'PY'
import sys
import pyarrow.parquet as pq
for row in pq.read_table(sys.argv[1]).to_pylist():
    print(f"   {row['pool_group']}: {row['overall_decision']} "
          f"(identity={row['identity_evidence_level']}, "
          f"technical={row['technical_status']}, metadata={row['metadata_status']})")
PY
    unset code || true
done
echo "Done. Outputs under $WORK."

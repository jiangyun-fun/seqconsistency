"""Optional self-contained HTML report (--report-out).

No network resources, no JavaScript dependencies: inline CSS and inline SVG only.
Display limits are explicit — the Parquet outputs always carry the complete results;
this page summarizes and truncates for readability.
"""

from __future__ import annotations

import html
from typing import Any

from seqconsistency import distributions as dist
from seqconsistency.models import DatasetOutcome
from seqconsistency.policy import Finding, Policy

MAX_FINDINGS = 200
MAX_COMPARISONS = 500
MAX_HIST_BARS = 60

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { font-family: ui-sans-serif, system-ui, "Segoe UI", Helvetica, Arial,
  sans-serif; margin: 0; background: #f6f7f9; color: #1c2430; line-height: 1.45; }
main { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 { font-size: 1.15rem; margin-top: 2.2rem; border-bottom: 2px solid #d8dee6;
  padding-bottom: .3rem; }
h3 { font-size: 1rem; margin: 1.4rem 0 .4rem; }
.meta { color: #5a6a7a; font-size: .9rem; }
table { border-collapse: collapse; width: 100%; font-size: .85rem; background: #fff; }
th, td { border: 1px solid #d8dee6; padding: .35rem .55rem; text-align: left;
  vertical-align: top; }
th { background: #eef1f5; }
.tag { display: inline-block; padding: .05rem .45rem; border-radius: 999px;
  font-size: .75rem; font-weight: 600; white-space: nowrap; }
.tag.COMPATIBLE, .tag.PASS { background: #dcfce7; color: #14532d; }
.tag.REVIEW { background: #fef9c3; color: #713f12; }
.tag.INCOMPATIBLE, .tag.FAIL, .tag.EXCEED { background: #fee2e2; color: #7f1d1d; }
.tag.INSUFFICIENT_EVIDENCE, .tag.MISSING_EVIDENCE,
.tag.INSUFFICIENT_OBSERVATIONS { background: #e0e7ff; color: #312e81; }
.tag.NOT_ASSESSED, .tag.NO_THRESHOLD { background: #e5e7eb; color: #374151; }
.tag.info { background: #dbeafe; color: #1e3a8a; }
.tag.warning { background: #fef9c3; color: #713f12; }
.tag.incompatible { background: #fee2e2; color: #7f1d1d; }
.finding { background: #fff; border: 1px solid #d8dee6; border-left: 4px solid
  #94a3b8; padding: .6rem .8rem; margin: .6rem 0; }
svg text { font-size: 9px; fill: #334155; }
.note { background: #fffbeb; border: 1px solid #fde68a; padding: .6rem .9rem;
  font-size: .88rem; }
code { background: #eef1f5; padding: .05rem .3rem; border-radius: 4px; }
"""


def _esc(x: Any) -> str:
    return html.escape(str(x if x is not None else ""))


def _mini_hist_svg(hist: Any) -> str:
    counts = hist.counts
    if not counts or max(counts) == 0:
        return "<p class='meta'>empty</p>"
    step = max(1, len(counts) // MAX_HIST_BARS)
    shown = counts[::step]
    width = 520
    height = 90
    bw = width / max(len(shown), 1)
    maxc = max(shown)
    bars = []
    for i, c in enumerate(shown):
        h = (c / maxc) * (height - 14) if maxc else 0
        bars.append(
            f"<rect x='{i * bw:.1f}' y='{height - h:.1f}' width='{bw - 0.5:.1f}' "
            f"height='{h:.1f}' fill='#4f7cac'/>"
        )
    axis = [
        f"<text x='0' y='{height - 1}'>{hist.edges[0]:g}</text>",
        f"<text x='{width - 60}' y='{height - 1}'>{hist.edges[-1]:g}</text>",
    ]
    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' "
        f"role='img' aria-label='histogram {hist.name}'>" + "".join(bars) + "".join(axis) + "</svg>"
    )


def _hist_for(outcome: DatasetOutcome, stratum: str) -> Any:
    if outcome.fastq is not None:
        st = outcome.fastq.r2 if stratum == "R2" else outcome.fastq.r1
        if st is None:
            return None
        return st.length_hist if st.length_hist is not None else None
    if outcome.bam is not None:
        return outcome.bam.length_hist
    return None


def render_html_report(
    decisions_rows: list[dict[str, Any]],
    findings: list[Finding],
    comparison_rows: list[dict[str, Any]],
    outcomes: list[DatasetOutcome],
    policy: Policy,
    provenance: dict[str, Any],
) -> str:
    parts: list[str] = []
    parts.append("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append(f"<title>seqconsistency report — {_esc(policy.policy_id)}</title>")
    parts.append(f"<style>{_CSS}</style></head><body><main>")
    parts.append(
        f"<h1>seqconsistency pooling-consistency report</h1>"
        f"<p class='meta'>policy <code>{_esc(policy.policy_id)}</code> v"
        f"{_esc(policy.policy_version)} · generated "
        f"{_esc(provenance.get('created_utc', ''))} · seqconsistency "
        f"{_esc(provenance.get('seqconsistency_version', ''))}</p>"
    )

    # Decisions.
    parts.append("<h2>Group decisions</h2>")
    parts.append(
        "<table><tr><th>pool_group</th><th>structural</th><th>technical</th>"
        "<th>metadata</th><th>identity</th><th>overall</th>"
        "<th>identity evidence</th><th>reason codes</th></tr>"
    )
    for d in decisions_rows:
        parts.append(
            f"<tr><td><b>{_esc(d['pool_group'])}</b></td>"
            f"<td><span class='tag {_esc(d['structural_status'])}'>"
            f"{_esc(d['structural_status'])}</span></td>"
            f"<td><span class='tag {_esc(d['technical_status'])}'>"
            f"{_esc(d['technical_status'])}</span></td>"
            f"<td><span class='tag {_esc(d['metadata_status'])}'>"
            f"{_esc(d['metadata_status'])}</span></td>"
            f"<td><span class='tag {_esc(d['identity_status'])}'>"
            f"{_esc(d['identity_status'])}</span></td>"
            f"<td><span class='tag {_esc(d['overall_decision'])}'>"
            f"<b>{_esc(d['overall_decision'])}</b></span></td>"
            f"<td>{_esc(d['identity_evidence_level'])}</td>"
            f"<td>{_esc(', '.join(d['reason_codes']) or '—')}</td></tr>"
        )
    parts.append("</table>")
    parts.append(
        "<p class='meta'>A structural failure aborts the run before any decision "
        "row is written; structural_status is PASS for every group in a "
        "completed report.</p>"
    )

    # Assumptions.
    parts.append("<h2>Assumptions and limitations</h2>")
    parts.append("<ul>")
    for d in decisions_rows:
        for a in d["assumptions"]:
            parts.append(f"<li>{_esc(a)} ({_esc(d['pool_group'])})</li>")
    parts.append(
        "<li>Sample identity is accepted from supplied metadata; it is not "
        "independently verified (no genotype-based verification is implemented in "
        "this version).</li>"
        "<li>Sequence-content (k-mer) comparison and reference-aware coverage "
        "comparison are NOT implemented in this version.</li>"
        "<li>Similar GC, quality, length, or k-mer distributions would not "
        "establish biological sample identity.</li>"
        "<li>Pairwise compatibility is not transitive: A~B and B~C do not imply "
        "A~C.</li>"
        "<li>Shipped example thresholds are illustrative and uncalibrated; they "
        "must be calibrated with genuine compatible runs and known "
        "incompatibilities before reuse (README § Threshold calibration).</li>"
        "</ul>"
    )

    # Findings.
    parts.append("<h2>Findings</h2>")
    shown = findings[:MAX_FINDINGS]
    if not findings:
        parts.append("<p class='meta'>None.</p>")
    for f in shown:
        parts.append(
            f"<div class='finding'><b>{_esc(f.finding_id)}</b> "
            f"<span class='tag {_esc(f.severity.value)}'>{_esc(f.severity.value)}"
            f"</span> <code>{_esc(f.reason_code)}</code> "
            f"<span class='meta'>({_esc(f.component)} · group "
            f"{_esc(f.pool_group)})</span><br>{_esc(f.message)}<br>"
            f"<span class='meta'>Action: {_esc(f.recommended_action)}</span></div>"
        )
    if len(findings) > MAX_FINDINGS:
        parts.append(
            f"<p class='meta'>Showing {MAX_FINDINGS} of {len(findings)} findings — "
            "the findings Parquet output carries all of them.</p>"
        )

    # Comparisons.
    parts.append("<h2>Pairwise comparisons</h2>")
    parts.append(
        f"<p class='meta'>Showing up to {MAX_COMPARISONS} rows of "
        f"{len(comparison_rows)}; the comparisons Parquet output is complete.</p>"
    )
    parts.append(
        "<table><tr><th>group</th><th>A</th><th>B</th><th>metric</th>"
        "<th>stratum</th><th>method</th><th>value</th><th>threshold</th>"
        "<th>status</th><th>obs A/B</th></tr>"
    )
    for r in comparison_rows[:MAX_COMPARISONS]:
        value = f"{r['value']:.4g}" if r["value"] is not None else "—"
        thr = f"{r['threshold']:.4g}" if r["threshold"] is not None else "—"
        parts.append(
            f"<tr><td>{_esc(r['pool_group'])}</td><td>{_esc(r['dataset_a'])}</td>"
            f"<td>{_esc(r['dataset_b'])}</td><td>{_esc(r['metric'])}</td>"
            f"<td>{_esc(r['stratum'])}</td><td>{_esc(r['method'])}</td>"
            f"<td>{value}</td><td>{thr}</td>"
            f"<td><span class='tag {_esc(r['status'])}'>{_esc(r['status'])}"
            f"</span></td><td>{_esc(r['observations_a'])}/{_esc(r['observations_b'])}"
            f"</td></tr>"
        )
    parts.append("</table>")

    # Distributions (one length histogram per dataset stratum, bounded).
    parts.append("<h2>Read-length distributions (sample of datasets)</h2>")
    parts.append(
        "<p class='meta'>One inline SVG per dataset strand; truncated histograms are "
        "for overview only — the distributions Parquet output is complete.</p>"
    )
    for o in outcomes[:24]:
        strata = ["R1", "R2"] if o.fastq is not None and o.fastq.r2 is not None else ["SE"]
        for s in strata:
            hist = _hist_for(o, s)
            if hist is not None:
                parts.append(f"<h3>{_esc(o.spec.dataset_id)} / {s}</h3>" + _mini_hist_svg(hist))

    # Sampling.
    parts.append("<h2>Sampling</h2>")
    parts.append(
        "<table><tr><th>dataset</th><th>eligible</th><th>sampled</th>"
        "<th>fraction</th><th>unit</th><th>algorithm</th></tr>"
    )
    for entry in provenance.get("inputs", []):
        eligible = entry.get("eligible_observations", 0)
        sampled = entry.get("sampled_count", 0)
        frac = (sampled / eligible) if eligible else 0.0
        unit = (
            "mate_pair"
            if entry.get("pool_group")
            and any(
                oo.spec.dataset_id == entry["dataset_id"] and oo.spec.is_paired_fastq
                for oo in outcomes
            )
            else ("primary_record" if "references" in entry else "read")
        )
        parts.append(
            f"<tr><td>{_esc(entry['dataset_id'])}</td><td>{eligible}</td>"
            f"<td>{sampled}</td><td>{frac:.4g}</td><td>{unit}</td>"
            "<td>uniform_key_reservoir</td></tr>"
        )
    parts.append("</table>")

    # Policy.
    parts.append("<h2>Applied policy</h2>")
    parts.append(
        f"<p class='meta'>policy_id {_esc(policy.policy_id)}, version "
        f"{_esc(policy.policy_version)}, sha256 "
        f"<code>{_esc(policy.sha256[:16])}…</code></p>"
    )
    if policy.thresholds:
        parts.append(
            "<table><tr><th>metric</th><th>method</th><th>max</th><th>action</th>"
            "<th>calibrated</th><th>profiles</th><th>strata</th></tr>"
        )
        for t in policy.thresholds:
            parts.append(
                f"<tr><td>{_esc(t.metric)}</td><td>{_esc(t.method)}</td>"
                f"<td>{t.max:g}</td><td>{_esc(t.action)}</td>"
                f"<td>{'yes' if t.calibrated else '<b>NO</b>'}</td>"
                f"<td>{_esc(', '.join(t.profiles or []) or 'all')}</td>"
                f"<td>{_esc(', '.join(t.strata or []) or 'all')}</td></tr>"
            )
        parts.append("</table>")
    else:
        parts.append(
            "<p class='meta'>No thresholds configured; distances are reported but not assessed.</p>"
        )

    parts.append(
        "<p class='meta'>Display limits: findings ≤200, comparisons ≤500, "
        f"histogram bars ≤{MAX_HIST_BARS}. Parquet outputs are complete.</p>"
    )
    parts.append("</main></body></html>")
    return "".join(parts)


_ = dist  # distributions referenced indirectly via histograms; keep import honest

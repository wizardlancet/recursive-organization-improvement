# Verification records

## Current manuscript package v3

- `manuscript_checks.json`: current PDF compilation, page counts, reference
  resolution, and machine-readable layout checks.
- `saved_results_audit.json`: reconstruction of statistics from saved trajectories,
  CSV-to-table checks, acquisition budgets, and the five-of-six cost comparison.
- `numeric_claim_ledger.json`: result claims with full-precision values and sources.
- `numeric_source_inventory.json`: every numeral-bearing line in the actual
  manuscript input graph, including settings, formulas, identifiers and provenance.
- `protocol_provenance.json`: protocol hashes, archival commit, and evidence limits.
- `literature_verification.json`: primary-source metadata/access-depth checks for
  the newly cited classical literature and the software-workflow acknowledgment.
- `package_verification.json`: independent build from the arXiv source ZIP.

No experiment was rerun to produce these revision checks. The scientific
validation records under `study2/qa/` were generated during the existing Study 2
work and are preserved as such.

## Historical records retained from research-2026-09-30 (v1)

`audit.json`, `submission_checks.json`, and `release_verification.json` refer to
the v1 manuscript/release, including its page count and PDF hash. They do not
describe the current PDF. `learning_checks.json`, `learning_validation.json`,
`public_trace_checks.json`, and `audit_extensions_checks.json` document the
original implementation/data validation. The Study 1 scientific files and
result tables are unchanged; these records are retained without relabeling
them as fresh experimental runs.

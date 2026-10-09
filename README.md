# Recursive Organization Improvement

**A Modeling Specification for Human–Agent Organizations**

Zilong Wang · CataX AI · wangzilong@cata-x.ai

This repository accompanies [the manuscript](main.pdf), its executable
change-contract checker, public-record feasibility mapping, and two synthetic
simulation studies:

- **Study 1: evidence acquisition and retention** (Section 4, Appendix C).
  Fixed label quality; alternative acquisition, memory, and evaluator-selection rules.
- **Study 2: review arrangements and evidence generation** (Section 5, Appendix E).
  Individual private judgments, shared errors, copying, cost, and authority determine
  outcomes of human–agent review arrangements. Fixed paid observation policies
  select arrangements under a resource ceiling.

Repository: https://github.com/wizardlancet/recursive-organization-improvement

## Manuscript and release versions

| Manuscript package | Repository tag | Status |
|---|---|---|
| v1 | `research-2026-09-30` | First public manuscript package |
| v3 | `research-2026-10-09` | Two-study manuscript and complete supplementary tables |

The intermediate v2 package was a local review draft, without a public repository
tag. These are manuscript-package identifiers; uploading an arXiv replacement is
a separate action and arXiv assigns its own version number. Suggested replacement
comments are in [ARXIV_COMMENTS.txt](ARXIV_COMMENTS.txt).

## Results and interpretation

In Study 1, retained evidence largely removes the repeated-assessment penalty
seen with reset memory. After workflow reversal, indefinite retention delays
adaptation. Matching trial acquisition and label reuse narrows the gain
attributable to evaluator replacement. Successive rejection has the highest
unrounded core mean in five of nine environment–memory cells; two further
Uniform gain cells differ from the highest mean by less than 0.00004.

In Study 2, copying can make visible agreement conceal errors. Shared private
errors and exposure-induced copying can be observationally equivalent in the
specified two-agent public record, yet respond differently to blinding. A
gold-labeled measurement differs from granting an imperfect human final authority.
Under the stated costs and cumulative memory, paired observation lowers error in
all six environments compared with the fixed blind A+3A baseline, but has lower
net value in five. This is a specific policy/comparator/cost comparison, not a
general result that learning is unprofitable.

All actor abilities and costs are synthetic. Study 2 and the acquisition-matched
controls are exploratory. These simulations do not measure deployed LLMs, human
participants, real PR defect escape rates, or organizational productivity.

## Supplementary tables and source files

Tables S1–S5 are delivered as the arXiv ancillary file
[`anc/supplementary_tables.pdf`](anc/supplementary_tables.pdf).
The [HTML copy](supplement/supplementary_tables.html) permits text search and
copying. Its source is `supplement/supplementary_tables.tex`.

| Table | Content | Repository source |
|---|---|---|
| S1 A–B | Study 1: all 54 core conditions, uncertainty and secondary outcomes | `results/learning_summary.csv` |
| S1 C | Core and exploratory paired contrasts | `results/learning_contrasts.csv` |
| S2 | All three retained programs and all three memories, discover-once and repeated discovery, across all three environments; 189 descriptive groups | `results/learning_conditional.csv` |
| S3 | All 54 audit-policy/window/environment conditions | `results/robustness_summary.csv` |
| S4 | All 72 EVSI prior/threshold/generation-cost conditions | `results/audit_extensions_summary.csv` |
| S5 A–D | Study 2: all 84 policy/memory/environment conditions, full and late outcomes, costs, regret | `study2/results/learning_summary.csv` |
| S5 E | 30 paired policy/memory contrasts | `study2/results/paired_contrasts.csv` |

Main Tables 6–9 use these same files or the paired trajectories in
`results/learning_sensitivity_runs.csv`; Table 12 uses the cumulative/fixed subset
of S5. Figure 4 uses `study2/results/illustrative_curve.csv` and `phase.csv`.
All source rows are retained. S2's conditional groups are descriptive and do not
have Monte Carlo intervals. Study 1 late outcomes cover eight rounds; Study 2
late outcomes cover sixteen rounds.

## Reproducibility and protocol provenance

Study 1's protocol is `examples/learning_protocol.json`; its first public
commit `75d312e` includes both protocol and results and does **not** prove a
pre-run Git timestamp. `examples/learning_controls.json` explicitly identifies
controls designed after core-result inspection, using fresh streams.
Study 2's `study2/protocol.json` matches the retained pre-run hash record at
`study2/qa/pre_run_record.json`. Its archive commit `fdf5ec1` postdates the run.
Neither study is externally preregistered. See `qa/protocol_provenance.json` for
hashes and the limits of this evidence. Historical machine-readable design IDs
are preserved and are not manuscript-version labels.

**No new experiments were run for the v3 manuscript revision.** Its numerical
audit reconstructs means, intervals, contrasts, and cost identities from saved
trajectories. Existing independent validation records are retained; their
simulation checks were not rerun for v3.

Using a Python environment with NumPy, inspect archived results without running
experiments:

```sh
python study2/summarize.py
python scripts/audit_saved_results.py
```

The first command checks the two Study 2 summary CSVs. The second writes
`qa/saved_results_audit.json`, `qa/numeric_claim_ledger.json`, and
`qa/numeric_source_inventory.json`. It checks every stored summary and paired
contrast used by the tables, all main numeric table rows, key prose values, and
the five-of-six cost decomposition. The inventory also identifies protocol
settings, mathematical illustrations, cross-references and provenance numerals;
these are not empirical result estimates.

To redraw from saved results, with Matplotlib installed:

```sh
python scripts/build_evidence.py
```

This rendering script does not sample outcomes or rerun an exact grid.

### Running the experiments separately

The following commands reproduce Study 1 and **overwrite generated results**.
Use its pinned root `requirements.txt` in a separate environment; the recorded
Study 1 runtime was Python 3.12.14.

```sh
python -m pip install -r requirements.txt
python scripts/contracts.py
python scripts/learning_experiment.py
python scripts/learning_controls.py
python scripts/learning_analysis.py
python scripts/validate_learning.py
python scripts/public_trace.py
```

The core, allocation/memory sensitivity, and exploratory-control matrices have
6,912, 4,608, and 6,144 trajectories respectively. For Study 2, use the separate
CPython 3.13.1 environment and instructions in [study2/README.md](study2/README.md).

The supplementary audit and diagnostic code remains available:

```sh
python scripts/experiments.py
python scripts/robustness.py
python scripts/review_analysis.py
python scripts/coverage_experiment.py
python scripts/audit_extensions.py
python scripts/v5_analysis.py
```

`v5_analysis.py` is a historical implementation identifier, not a public
manuscript version. `coverage_*` data describe a diagnostic reset-only model,
not the crossed-memory Study 1. The public snapshot adapter runs offline by
default. `--fetch` refreshes live public sources and intentionally does not
reproduce the stored snapshot. Public PR links allow re-identification: the
snapshot is pseudonymized, not irreversibly anonymized.

## Build the manuscript and supplement

```sh
tectonic --keep-logs --keep-intermediates --reruns 2 main.tex
tectonic --keep-logs --keep-intermediates --reruns 2 supplement/supplementary_tables.tex --outdir supplement
# Copy the resulting supplementary PDF to anc/supplementary_tables.pdf for arXiv.
```

`latexmk -pdf main.tex` is an alternative with a full TeX installation. The
community single-column template is `arxiv.sty`. The arXiv source release asset
includes the manuscript inputs and `anc/supplementary_tables.pdf`, and excludes
the experiment datasets. The complete research repository retains the latter.

`qa/manuscript_checks.json` records current compilation and reference checks.
`qa/release_verification.json` and the checks explicitly labeled v1 are retained
historical validation of the original release, not evidence of a new v3 run.
`MANIFEST.sha256` covers the current tracked delivery files except itself.

## License and citation

All original implementation, including `study2/` and the saved-result audit and
rendering utilities, is licensed under MIT (`LICENSE-CODE`). The manuscript,
original figures, protocols, and synthetic data in both studies are licensed
under CC BY 4.0 (`LICENSE-CONTENT.md`). Third-party materials retain their terms,
including `template/License.txt`. Use `CITATION.cff` for author/release metadata.

OpenAI Codex assisted with drafting, implementation, analysis, and figure
preparation, as disclosed in the manuscript. The academic-paper-writing and PDF
skills supported revision and document checking. Figure preparation also used
Scientific Agent Skills: Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel,
and Aubrey M. Brueckner (2026), *Scientific Agent Skills: A Library of Procedural
Knowledge for Research Agents*, https://doi.org/10.48550/arXiv.2609.00065.

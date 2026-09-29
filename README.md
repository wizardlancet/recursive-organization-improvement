# Recursive Organization Improvement

**A Modeling Specification for Human–Agent Organizations**

Zilong Wang · CataX AI · wangzilong@cata-x.ai

This repository accompanies the research manuscript in `main.pdf`.
It contains an executable change-contract checker, a three-PR feasibility
mapping, and controlled simulations of evidence acquisition, retention,
and evaluation-program reassessment.

Repository: https://github.com/wizardlancet/recursive-organization-improvement

Release: `research-2026-09-30`. No Zenodo DOI has been issued for this release.

## Main result

Evidence accumulation largely removes the repeated-assessment penalty observed
when evidence is reset each round. After a workflow-ranking reversal, indefinite
retention delays adaptation. A finite window recovers late performance but has
a transition cost. Matching trial acquisition and label reuse narrows the
apparent reassessment gain; replacing the evaluator adds no stable benefit
across the tested reversal times.

The contribution is an explicit modeling specification and a synthetic mechanism
study. The simulations do not measure real organizational productivity or
deployed LLM performance. See the manuscript for exact assumptions and uncertainty.

## Reproduce

Python 3.12 or later (verified with 3.12.14) and the dependencies pinned in `requirements.txt` are needed.

```sh
python -m pip install -r requirements.txt
python scripts/contracts.py
python scripts/learning_experiment.py
python scripts/learning_controls.py
python scripts/learning_analysis.py
python scripts/validate_learning.py
python scripts/public_trace.py
```

The primary matrix has 6,912 trajectories. The allocation grid and memory
ablations add 4,608. Exploratory acquisition-matched/timing controls add 6,144,
using fresh random streams. `examples/learning_protocol.json` is the frozen
core protocol; `examples/learning_controls.json` explicitly identifies the
post-core exploratory controls. Their original machine-readable design IDs are
preserved for provenance. They are not externally preregistered protocols.

`results/learning_*.csv` retains all trajectories, summaries, per-round means,
conditional program outcomes and paired contrasts. The uniform fixed mixture
averages the conditional values of Biased, Balanced and Neyman. Its uncertainty
integrates over this prior rather than sampling a random program for each run.

To reproduce supplementary audit and diagnostic results:

```sh
python scripts/experiments.py
python scripts/robustness.py
python scripts/review_analysis.py
python scripts/coverage_experiment.py
python scripts/audit_extensions.py
python scripts/v5_analysis.py
```

The filename `v5_analysis.py` is retained as a historical implementation
identifier; it is not a public manuscript version. `coverage_*` CSVs describe
the diagnostic reset-only implementation, not the main crossed-memory study.
`audit_*` and `robustness_*` contain the supplementary evidence-acquisition model.
Figures not included by `main.tex` are diagnostic outputs retained for audit.
On Windows, use `python -X utf8` when the console encoding requires it.

The public snapshot adapter runs offline by default. It retains source links,
timestamps, and snapshot-scoped actor pseudonyms, with no usernames, account IDs,
comment bodies or emails. Public PR links allow source re-identification, so
the snapshot is pseudonymized, not irreversibly anonymized. `--fetch` refreshes
live source data and intentionally does not reproduce the frozen snapshot.

## Build the paper

```sh
latexmk -pdf main.tex
# Alternative:
tectonic --keep-logs --keep-intermediates --reruns 2 main.tex
```

The included `arxiv.sty` is a community single-column preprint style.
The clean source archive has been rebuilt and compared with the supplied PDF.
`qa/audit.json` records citations, page count and layout diagnostics;
`qa/release_verification.json` records offline reproduction of all 33 CSVs.
`MANIFEST.sha256` fixes the public release's file contents.

The standalone validation reconstructs 36 Balanced trajectories in scalar form
and checks the memory-window boundary and successive-rejection allocation.
Every simulation block checks its resource ceiling, and every round checks its
cost identity. Intervals measure Monte Carlo uncertainty under model assumptions.

## License and citation

Original implementation: MIT (`LICENSE-CODE`).
Manuscript, original figures and synthetic data: CC BY 4.0
(`LICENSE-CONTENT.md`). Third-party material retains its original terms,
including the template license in `template/License.txt`.
Use `CITATION.cff` for author and repository metadata.

OpenAI Codex assisted with manuscript drafting, implementation, analysis and
figure preparation, as disclosed in the paper.

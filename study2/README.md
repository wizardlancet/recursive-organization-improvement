# Study 2: review arrangements and evidence generation

This directory contains the synthetic study described in Section 5 and Appendix E.
Private error dependence, upstream copying, actor cost, and final decision rights
jointly determine the outcomes of finite-catalog review arrangements. The paid
observation policies are fixed treatments; they do not search over observation
procedures or learn when to stop purchasing evidence.

## Recorded design and results

`protocol.json` is preserved byte for byte. Its SHA-256 matches the retained
`qa/pre_run_record.json` dated 2026-10-02 13:06:04 UTC. This local record is not
an independently timestamped preregistration. Study 2 was designed after Study 1
and is exploratory. Commit `fdf5ec1` archives the existing protocol, scientific
implementation, results, and validation; it postdates execution.
The 21-point illustrative curve was selected after the core run.

- `model.py`: private-signal mixture, copying rule, exact probabilities, sampling,
  and observation estimators. Human errors are conditionally independent; agent
  errors have the specified shared-error mixture. Blinding removes copying by
  construction but does not remove intrinsic error dependence.
- `oracle_experiment.py`: 86 candidates; 16,200 budget cells, 25,200 fixed-roster
  comparisons, and 6,450 source-specific copying comparisons. The finite-catalog
  minimum-error objective differs from the deployment-loss benchmark.
- `learning_experiment.py`: 58 feasible arrangements; 6 environments,
  4 policies by 3 memories plus 2 fixed baselines; 128 trajectories of 48 rounds
  per condition, totaling 10,752 trajectories.
- `results/learning_raw.npz`: axes are environment, policy/memory, replicate,
  round. `learning_metadata.json` defines the axis order and arrangement catalog.
  `estimates` appends the parameter axis qH, qA, rho, etaH, etaA; `choice` indexes
  the catalog. Production error and cost are exact expectations after each choice.
- `results/learning_runs.csv`, `learning_periods.csv`: full/late trajectory
  summaries and per-round summaries. Late means the final 16 rounds.
- `results/learning_summary.csv`, `paired_contrasts.csv`: all 84 conditions and
  30 paired comparisons. Intervals are 1.96 standard errors across trajectories.
- `qa/validation.json`, `monte_carlo_checks.json`: retained independent enumeration
  and Monte Carlo validation records. These retain the original execution dates and validation scope.

An agent reviewer's copying probability eta_H or eta_A refers to the *upstream*
human or agent, respectively. All reviewers in this study are agents. A human
arbiter supplies an imperfect final decision; a gold label supplies correct
measurement at a cost. These are different operations.

## Reproduce

The experiment used CPython 3.13.1 and the pinned `requirements.txt`.
From the repository root, the following commands **run simulations** and replace
generated results in this directory:

```sh
python -m pip install -r study2/requirements.txt
python study2/oracle_experiment.py
python study2/learning_experiment.py
python study2/summarize.py --write
python study2/validate.py
```

For a check using only the archived trajectories, with no simulation:

```sh
python study2/summarize.py
python scripts/audit_saved_results.py
```

`python scripts/build_evidence.py` renders Figure 4, Table 12, and supplementary
tables from saved CSVs only. The exploratory 21-point curve is retained in
`results/illustrative_curve.csv`; it is not a new prespecified condition grid.

## Cost and information boundaries

Paired observation buys 2 human private responses, 6 agent private responses,
10 post-exposure agent responses, a gold label, and 6 private-commitment overheads:
7.3 units per task, or 233.6 for 32 tasks per round. Randomized exposure buys
50 tasks, with exactly 25 blind tasks: 232.5 units per round. The ceiling is
240 per round and 1920 per eight-round block, without borrowing.
Window 8 includes eight previous rounds and the current round. Acquisition
precedes deployment. Deployed judgments do not provide free learning feedback.
The production count 4096 amortizes acquisition and switching costs; it is not
4096 additional sampled production outcomes. Parameters are synthetic.

## Licensing

All code here, including the summary utility, is covered by the repository's
MIT `LICENSE-CODE`. Protocol prose, synthetic data, and original figures are
covered by CC BY 4.0 in `LICENSE-CONTENT.md`. Third-party terms remain unchanged.

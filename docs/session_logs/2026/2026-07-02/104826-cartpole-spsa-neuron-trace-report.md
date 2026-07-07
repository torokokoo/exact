---
schema_version: 1
title: "CartPole SPSA neuron trace report"
trigger: "milestone"
timestamp: "2026-07-02T10:48:26-04:00"
timezone: "-04"
branch: "main"
head: "bec5503cff27e18ebe3503a340413491f1870519"
dirty: true
tags: []
git_status:
  - " M docs/rl_tools_snn.md"
  - " M docs/use_snn_implementations.md"
  - " M exact_rl_tools/rl_evaluator.cxx"
  - " M exact_rl_tools/rl_evaluator.hxx"
  - " M exact_rl_tools/rl_local_search.cxx"
  - " M exact_rl_tools/rl_local_search.hxx"
  - " M multithreaded/snn_rl_mt.cxx"
  - " M rnn/lif_node.cxx"
  - " M rnn/lif_node.hxx"
  - " M rnn/rnn.cxx"
  - " M rnn/rnn.hxx"
  - "?? docs/session_logs/2026/2026-06-24/"
diff_stat:
  - " docs/rl_tools_snn.md               |  2 ++"
  - " docs/use_snn_implementations.md    |  2 ++"
  - " exact_rl_tools/rl_evaluator.cxx    | 22 ++++++++++--"
  - " exact_rl_tools/rl_evaluator.hxx    | 14 +++++---"
  - " exact_rl_tools/rl_local_search.cxx | 44 +++++++++++++++--------"
  - " exact_rl_tools/rl_local_search.hxx |  4 +++"
  - " multithreaded/snn_rl_mt.cxx        | 73 +++++++++++++++++++++++++++++++-------"
  - " rnn/lif_node.cxx                   |  8 +++++"
  - " rnn/lif_node.hxx                   |  3 ++"
  - " rnn/rnn.cxx                        | 19 ++++++++++"
  - " rnn/rnn.hxx                        | 13 +++++++"
  - " 11 files changed, 170 insertions(+), 34 deletions(-)"
recent_commits:
  - "bec5503 (HEAD -> main, origin/main, origin/HEAD) Add thesis session logging workflow"
  - "41af552 Document EXACT and SNN training paths"
  - "c672935 Refactor SNN RL runner onto rl-tools"
  - "0cf8f51 Add rl-tools environment integration"
  - "6711498 Ignore generated build and run artifacts"
---

# CartPole SPSA neuron trace report

## Goal
Train a fresh CartPole SNN/RL run with SPSA and generate validation figures/HTML

## What Changed
- Executed a new CartPole SPSA run in test_output/cartpole_spsa_neuron_trace_20260702_104611_run2 and built a thesis visualizer report with reward, architecture, episode trace, and LIF current plots.

## Decisions
- Used a new output directory and wrote the report under test_output to preserve prior runs and avoid modifying earlier artifacts.

## Code State
- Branch: `main`
- HEAD: `bec5503cff27e18ebe3503a340413491f1870519`
- Dirty worktree: `yes`

### Changed Files
```text
M docs/rl_tools_snn.md
 M docs/use_snn_implementations.md
 M exact_rl_tools/rl_evaluator.cxx
 M exact_rl_tools/rl_evaluator.hxx
 M exact_rl_tools/rl_local_search.cxx
 M exact_rl_tools/rl_local_search.hxx
 M multithreaded/snn_rl_mt.cxx
 M rnn/lif_node.cxx
 M rnn/lif_node.hxx
 M rnn/rnn.cxx
 M rnn/rnn.hxx
?? docs/session_logs/2026/2026-06-24/
```

### Diff Summary
```text
docs/rl_tools_snn.md               |  2 ++
 docs/use_snn_implementations.md    |  2 ++
 exact_rl_tools/rl_evaluator.cxx    | 22 ++++++++++--
 exact_rl_tools/rl_evaluator.hxx    | 14 +++++---
 exact_rl_tools/rl_local_search.cxx | 44 +++++++++++++++--------
 exact_rl_tools/rl_local_search.hxx |  4 +++
 multithreaded/snn_rl_mt.cxx        | 73 +++++++++++++++++++++++++++++++-------
 rnn/lif_node.cxx                   |  8 +++++
 rnn/lif_node.hxx                   |  3 ++
 rnn/rnn.cxx                        | 19 ++++++++++
 rnn/rnn.hxx                        | 13 +++++++
 11 files changed, 170 insertions(+), 34 deletions(-)
```

### Recent Commits
```text
bec5503 (HEAD -> main, origin/main, origin/HEAD) Add thesis session logging workflow
41af552 Document EXACT and SNN training paths
c672935 Refactor SNN RL runner onto rl-tools
0cf8f51 Add rl-tools environment integration
6711498 Ignore generated build and run artifacts
```

## Verification
- snn_rl_mt completed successfully; visualizer build generated index.html, thesis_snippets.md, summary tables, and non-empty PNG figures including lif_current_heatmap and lif_current_detail.

## Open Threads
_Not recorded._

## Next Steps
- Open the generated index.html and inspect whether the new neuron current plots communicate the intended SNN behavior for the thesis.

---
schema_version: 1
title: "RL evaluation statistics rerun"
trigger: "milestone"
timestamp: "2026-06-24T10:30:15-04:00"
timezone: "-04"
branch: "main"
head: "bec5503cff27e18ebe3503a340413491f1870519"
dirty: true
tags: []
git_status:
  - " M exact_rl_tools/rl_evaluator.cxx"
  - " M exact_rl_tools/rl_evaluator.hxx"
  - " M exact_rl_tools/rl_local_search.cxx"
  - " M exact_rl_tools/rl_local_search.hxx"
  - " M multithreaded/snn_rl_mt.cxx"
  - "?? docs/session_logs/2026/2026-06-24/"
diff_stat:
  - " exact_rl_tools/rl_evaluator.cxx    |  7 +++++-"
  - " exact_rl_tools/rl_evaluator.hxx    |  8 ++++---"
  - " exact_rl_tools/rl_local_search.cxx | 44 ++++++++++++++++++++++++++------------"
  - " exact_rl_tools/rl_local_search.hxx |  4 ++++"
  - " multithreaded/snn_rl_mt.cxx        | 30 +++++++++++++++-----------"
  - " 5 files changed, 63 insertions(+), 30 deletions(-)"
recent_commits:
  - "bec5503 (HEAD -> main, origin/main, origin/HEAD) Add thesis session logging workflow"
  - "41af552 Document EXACT and SNN training paths"
  - "c672935 Refactor SNN RL runner onto rl-tools"
  - "0cf8f51 Add rl-tools environment integration"
  - "6711498 Ignore generated build and run artifacts"
---

# RL evaluation statistics rerun

## Goal
Registrar estadisticas reales de evaluaciones internas por genoma y regenerar graficos desde nuevas corridas

## What Changed
- Se agregaron evaluation_best_reward, evaluation_mean_reward y evaluation_worst_reward a rl_fitness_log.csv; exact-thesis-viz ahora omite el fallback movil y grafica esas columnas reales. Se corrieron cartpole_rl_evalstats_none y cartpole_rl_evalstats_spsa.

## Decisions
- Se mantuvo la configuracion de estabilidad anterior para aislar el cambio de logging; SPSA registra 7 evaluaciones internas por genoma porque iterations=2 implica 1+3*2.

## Code State
- Branch: `main`
- HEAD: `bec5503cff27e18ebe3503a340413491f1870519`
- Dirty worktree: `yes`

### Changed Files
```text
M exact_rl_tools/rl_evaluator.cxx
 M exact_rl_tools/rl_evaluator.hxx
 M exact_rl_tools/rl_local_search.cxx
 M exact_rl_tools/rl_local_search.hxx
 M multithreaded/snn_rl_mt.cxx
?? docs/session_logs/2026/2026-06-24/
```

### Diff Summary
```text
exact_rl_tools/rl_evaluator.cxx    |  7 +++++-
 exact_rl_tools/rl_evaluator.hxx    |  8 ++++---
 exact_rl_tools/rl_local_search.cxx | 44 ++++++++++++++++++++++++++------------
 exact_rl_tools/rl_local_search.hxx |  4 ++++
 multithreaded/snn_rl_mt.cxx        | 30 +++++++++++++++-----------
 5 files changed, 63 insertions(+), 30 deletions(-)
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
- cmake --build build_rl --target snn_rl_mt -j 4; PYTHONPATH=src python3 -m unittest tests.test_loaders tests.test_report.ReportTests.test_sample_report_builds -q; reporte cartpole_rl_evalstats_compare generado y verificado con figuras reward_evaluation_trend e island_reward.

## Open Threads
_Not recorded._

## Next Steps
- Revisar el nuevo reporte en exact-thesis-viz/reports/cartpole_rl_evalstats_compare/index.html y decidir si repetir con mas presupuesto o SPSA iterations=3 para alrededor de 10 evaluaciones internas.

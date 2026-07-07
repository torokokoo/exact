---
schema_version: 1
title: "RL island and reward trend visualizations"
trigger: "milestone"
timestamp: "2026-06-24T08:49:58-04:00"
timezone: "-04"
branch: "main"
head: "bec5503cff27e18ebe3503a340413491f1870519"
dirty: true
tags: []
git_status:
  - " M exact_rl_tools/rl_evaluator.cxx"
  - " M exact_rl_tools/rl_evaluator.hxx"
  - " M multithreaded/snn_rl_mt.cxx"
diff_stat:
  - " exact_rl_tools/rl_evaluator.cxx |  7 ++++++-"
  - " exact_rl_tools/rl_evaluator.hxx |  8 +++++---"
  - " multithreaded/snn_rl_mt.cxx     | 10 ++++++----"
  - " 3 files changed, 17 insertions(+), 8 deletions(-)"
recent_commits:
  - "bec5503 (HEAD -> main, origin/main, origin/HEAD) Add thesis session logging workflow"
  - "41af552 Document EXACT and SNN training paths"
  - "c672935 Refactor SNN RL runner onto rl-tools"
  - "0cf8f51 Add rl-tools environment integration"
  - "6711498 Ignore generated build and run artifacts"
---

# RL island and reward trend visualizations

## Goal
Agregar graficos de islas y tendencia best/promedio/peor a exact-thesis-viz para corridas SNN/RL

## What Changed
- Se extendio el visualizador para usar fitness_log.csv en corridas RL, generar reward por isla, comparacion por isla, y tendencia de evaluacion reward; ademas snn_rl_mt ahora registra best_reward y worst_reward por genoma.

## Decisions
- Los logs historicos no contienen extremos por episodio, por lo que el reporte usa max/media/min moviles sobre avg_reward como fallback; las corridas nuevas usaran best_reward/worst_reward reales.

## Code State
- Branch: `main`
- HEAD: `bec5503cff27e18ebe3503a340413491f1870519`
- Dirty worktree: `yes`

### Changed Files
```text
M exact_rl_tools/rl_evaluator.cxx
 M exact_rl_tools/rl_evaluator.hxx
 M multithreaded/snn_rl_mt.cxx
```

### Diff Summary
```text
exact_rl_tools/rl_evaluator.cxx |  7 ++++++-
 exact_rl_tools/rl_evaluator.hxx |  8 +++++---
 multithreaded/snn_rl_mt.cxx     | 10 ++++++----
 3 files changed, 17 insertions(+), 8 deletions(-)
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
- PYTHONPATH=src python3 -m unittest tests.test_loaders tests.test_report.ReportTests.test_sample_report_builds -q; cmake --build build_rl --target snn_rl_mt -j 4; smoke run en /tmp con cabecera avg_reward,best_reward,worst_reward; rebuild de cartpole_rl_stability_compare.

## Open Threads
_Not recorded._

## Next Steps
- Para obtener el grafico best/promedio/peor real en corridas largas, volver a entrenar con el binario snn_rl_mt actualizado.

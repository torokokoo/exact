---
schema_version: 1
title: "Stabilize Week 7 CartPole SNN-RL"
trigger: "milestone"
timestamp: "2026-06-18T20:26:36-04:00"
timezone: "-04"
branch: "main"
head: "0b6427b94b645073f9300bfde40dee693d1bbfc6"
dirty: true
tags: []
git_status:
  - " M .gitignore"
  - "A  .gitmodules"
  - " M AGENTS.md"
  - " M CMakeLists.txt"
  - " M docs/executables.md"
  - "A  external/rl_tools"
  - " M multithreaded/CMakeLists.txt"
  - " D multithreaded/cartpole_env.cxx"
  - " D multithreaded/cartpole_env.hxx"
  - " M multithreaded/snn_rl_mt.cxx"
  - " M rnn/rnn.cxx"
  - "?? docs/rl_tools_snn.md"
  - "?? docs/session_logs/"
  - "?? docs/train_vanilla_exact.md"
  - "?? docs/use_snn_implementations.md"
  - "?? exact_rl_tools/"
  - "?? scripts/base_run/snn_rl_cartpole_mt.sh"
  - "?? scripts/base_run/snn_rl_mt.sh"
  - "?? scripts/base_run/snn_rl_mt_spsa.sh"
  - "?? scripts/session_log.py"
diff_stat:
  - " .gitignore                     |  17 ++-"
  - " .gitmodules                    |   3 +"
  - " AGENTS.md                      |  16 +++"
  - " CMakeLists.txt                 |  24 +++-"
  - " docs/executables.md            |   8 ++"
  - " external/rl_tools              |   1 +"
  - " multithreaded/CMakeLists.txt   |   7 +-"
  - " multithreaded/cartpole_env.cxx |  53 -------"
  - " multithreaded/cartpole_env.hxx |  41 ------"
  - " multithreaded/snn_rl_mt.cxx    | 304 ++++++++++++++++-------------------------"
  - " rnn/rnn.cxx                    |   5 +-"
  - " 11 files changed, 188 insertions(+), 291 deletions(-)"
recent_commits:
  - "0b6427b (HEAD -> main, origin/main, origin/HEAD) Add SNN LIF support"
  - "16c1999 added sweet code (#70)"
  - "6cfa5be Diana/iterations ramp up (#64)"
  - "a40824a Added a new parent selection strategy (#65)"
  - "a82ff7a Diana/iterations ramp up (#63)"
---

# Stabilize Week 7 CartPole SNN-RL

## Goal
Stabilize the Week 7 CartPole-first SNN-RL prototype around snn_rl_mt and rl_fitness_log.csv.

## What Changed
- Added submodule guard for external/rl_tools, official CartPole run script, generated-output ignore rules, docs clarifying CartPole as Week 7 scope, and removed the stale multithreaded cartpole_env sources.

## Decisions
- Keep rl_fitness_log.csv as the official RL analysis log while fitness_log.csv remains an EXAMM compatibility artifact; record rl_tools as a Git submodule at e1143309be7c5f0c655248b4b03763102f8b486c.

## Code State
- Branch: `main`
- HEAD: `0b6427b94b645073f9300bfde40dee693d1bbfc6`
- Dirty worktree: `yes`

### Changed Files
```text
M .gitignore
A  .gitmodules
 M AGENTS.md
 M CMakeLists.txt
 M docs/executables.md
A  external/rl_tools
 M multithreaded/CMakeLists.txt
 D multithreaded/cartpole_env.cxx
 D multithreaded/cartpole_env.hxx
 M multithreaded/snn_rl_mt.cxx
 M rnn/rnn.cxx
?? docs/rl_tools_snn.md
?? docs/session_logs/
?? docs/train_vanilla_exact.md
?? docs/use_snn_implementations.md
?? exact_rl_tools/
?? scripts/base_run/snn_rl_cartpole_mt.sh
?? scripts/base_run/snn_rl_mt.sh
?? scripts/base_run/snn_rl_mt_spsa.sh
?? scripts/session_log.py
```

### Diff Summary
```text
.gitignore                     |  17 ++-
 .gitmodules                    |   3 +
 AGENTS.md                      |  16 +++
 CMakeLists.txt                 |  24 +++-
 docs/executables.md            |   8 ++
 external/rl_tools              |   1 +
 multithreaded/CMakeLists.txt   |   7 +-
 multithreaded/cartpole_env.cxx |  53 -------
 multithreaded/cartpole_env.hxx |  41 ------
 multithreaded/snn_rl_mt.cxx    | 304 ++++++++++++++++-------------------------
 rnn/rnn.cxx                    |   5 +-
 11 files changed, 188 insertions(+), 291 deletions(-)
```

### Recent Commits
```text
0b6427b (HEAD -> main, origin/main, origin/HEAD) Add SNN LIF support
16c1999 added sweet code (#70)
6cfa5be Diana/iterations ramp up (#64)
a40824a Added a new parent selection strategy (#65)
a82ff7a Diana/iterations ramp up (#63)
```

## Verification
- clang-format dry-run passed; default ASan CMake target snn_rl_mt built; non-ASan snn_rl_mt and test_rl_tools_environments built; test_rl_tools_environments passed; tiny CartPole max_genomes=2 smoke run wrote completed and rl_fitness_log.csv rows.

## Open Threads
_Not recorded._

## Next Steps
- Review/stage the remaining source and docs changes, then run the official scripts/base_run/snn_rl_cartpole_mt.sh for the full Week 7 artifact.


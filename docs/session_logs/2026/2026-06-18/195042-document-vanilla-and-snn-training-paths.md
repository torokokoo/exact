---
schema_version: 1
title: "Document vanilla and SNN training paths"
trigger: "milestone"
timestamp: "2026-06-18T19:50:42-04:00"
timezone: "-04"
branch: "main"
head: "0b6427b94b645073f9300bfde40dee693d1bbfc6"
dirty: true
tags: []
git_status:
  - " M AGENTS.md"
  - " M CMakeLists.txt"
  - " M docs/executables.md"
  - " M multithreaded/CMakeLists.txt"
  - " M multithreaded/snn_rl_mt.cxx"
  - " M rnn/rnn.cxx"
  - "?? .gitmodules"
  - "?? build_mpi_lyu2021/"
  - "?? build_rl/"
  - "?? docs/rl_tools_snn.md"
  - "?? docs/session_logs/"
  - "?? docs/train_vanilla_exact.md"
  - "?? docs/use_snn_implementations.md"
  - "?? exact_rl_tools/"
  - "?? external/"
  - "?? outputs/"
  - "?? scripts/base_run/snn_rl_mt.sh"
  - "?? scripts/base_run/snn_rl_mt_spsa.sh"
  - "?? scripts/session_log.py"
diff_stat:
  - " AGENTS.md                    |  16 +++"
  - " CMakeLists.txt               |  17 ++-"
  - " docs/executables.md          |   8 ++"
  - " multithreaded/CMakeLists.txt |   7 +-"
  - " multithreaded/snn_rl_mt.cxx  | 304 ++++++++++++++++---------------------------"
  - " rnn/rnn.cxx                  |   5 +-"
  - " 6 files changed, 163 insertions(+), 194 deletions(-)"
recent_commits:
  - "0b6427b (HEAD -> main, origin/main, origin/HEAD) Add SNN LIF support"
  - "16c1999 added sweet code (#70)"
  - "6cfa5be Diana/iterations ramp up (#64)"
  - "a40824a Added a new parent selection strategy (#65)"
  - "a82ff7a Diana/iterations ramp up (#63)"
---

# Document vanilla and SNN training paths

## Goal
Create practical documentation for vanilla EXACT/EXAMM training and the available SNN usage modes.

## What Changed
- Added docs/train_vanilla_exact.md and docs/use_snn_implementations.md with build, run, flag, output, surrogate-gradient LIF, RL reward, SPSA, and Python reference instructions.

## Decisions
- Kept the documentation as separate runbooks so existing architecture docs and dirty worktree files were not rewritten; distinguished supervised surrogate-gradient LIF from derivative-free RL local search.

## Code State
- Branch: `main`
- HEAD: `0b6427b94b645073f9300bfde40dee693d1bbfc6`
- Dirty worktree: `yes`

### Changed Files
```text
M AGENTS.md
 M CMakeLists.txt
 M docs/executables.md
 M multithreaded/CMakeLists.txt
 M multithreaded/snn_rl_mt.cxx
 M rnn/rnn.cxx
?? .gitmodules
?? build_mpi_lyu2021/
?? build_rl/
?? docs/rl_tools_snn.md
?? docs/session_logs/
?? docs/train_vanilla_exact.md
?? docs/use_snn_implementations.md
?? exact_rl_tools/
?? external/
?? outputs/
?? scripts/base_run/snn_rl_mt.sh
?? scripts/base_run/snn_rl_mt_spsa.sh
?? scripts/session_log.py
```

### Diff Summary
```text
AGENTS.md                    |  16 +++
 CMakeLists.txt               |  17 ++-
 docs/executables.md          |   8 ++
 multithreaded/CMakeLists.txt |   7 +-
 multithreaded/snn_rl_mt.cxx  | 304 ++++++++++++++++---------------------------
 rnn/rnn.cxx                  |   5 +-
 6 files changed, 163 insertions(+), 194 deletions(-)
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
- Verified the new Markdown files for ASCII-only content and reviewed commands against scripts/base_run and the relevant C++ sources.

## Open Threads
_Not recorded._

## Next Steps
- Run the smoke commands after the project is built, then tune max_genomes/bp_iterations or RL local-search iterations for actual experiments.


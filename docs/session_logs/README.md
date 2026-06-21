# Session Logs

This directory is the thesis development ledger for EXACT. It records important
Codex coding sessions, decisions, verification steps, and code-state pointers in
plain Markdown.

Each entry is stored as:

```text
docs/session_logs/YYYY/YYYY-MM-DD/HHMMSS-slug.md
```

Entries use a YAML-style front matter block for machine parsing, followed by
human-readable sections:

- Goal
- What Changed
- Decisions
- Code State
- Verification
- Open Threads
- Next Steps

The log intentionally stores code-state metadata rather than full patches. Exact
code reconstruction should use Git commits and diffs; these entries explain why
the work happened and what was verified.

## Commands

Add a manual snapshot:

```bash
python3 scripts/session_log.py add \
  --trigger manual \
  --title "LIF node integration checkpoint" \
  --goal "Add LIF support to the C++ RNN node system" \
  --decision "Use surrogate gradients instead of changing the optimizer pipeline" \
  --test "./build/rnn_tests/test_lif_gradients: passing"
```

Add a milestone when a goal or turning point is reached:

```bash
python3 scripts/session_log.py add \
  --trigger milestone \
  --title "Session log system implemented" \
  --goal "Create a thesis-friendly development ledger" \
  --change "Added a no-dependency CLI for session logs" \
  --next "Use this command at future implementation checkpoints"
```

Preview without writing:

```bash
python3 scripts/session_log.py add --dry-run \
  --title "Preview entry" \
  --goal "Check the log format"
```

Create a daily recap from that day's entries:

```bash
python3 scripts/session_log.py daily --date today
```

List recent entries:

```bash
python3 scripts/session_log.py list --limit 10
```

## What Gets Captured Automatically

- local timestamp and timezone
- Git branch and HEAD commit
- dirty or clean worktree state
- `git status --short --untracked-files=normal`
- `git diff --stat HEAD --`
- recent commits from `git log -5 --oneline --decorate`

Untracked build or experiment directories may appear as path-level Git status
entries, but their contents are not copied into the log.

## When To Log

Create a milestone entry when a coding session reaches a clear goal, a design
choice becomes durable, a test result changes the plan, or a long session is
about to end. Create a manual entry anytime with "log this session so far".

Use daily recaps as a fallback when several smaller steps happened without
individual milestone entries.

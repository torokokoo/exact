#!/usr/bin/env python3
"""Create thesis-oriented development session logs for EXACT.

The log format is intentionally plain Markdown with a small YAML-style front
matter block. Humans should be able to read it directly, while future tools can
parse the metadata without depending on third-party Python packages.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json
import re
import subprocess
import sys


LOG_ROOT = Path("docs/session_logs")
SCHEMA_VERSION = 1
TRIGGERS = ("manual", "milestone", "daily")


@dataclass
class GitState:
    timestamp: str
    timezone: str
    branch: str
    head: str
    dirty: bool
    status_lines: list[str]
    diff_stat_lines: list[str]
    recent_commit_lines: list[str]


def run_command(args: list[str], cwd: Path) -> tuple[int, str, str]:
    result = subprocess.run(
        args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout.rstrip("\n"), result.stderr.rstrip("\n")


def find_repo_root() -> Path:
    code, stdout, _ = run_command(["git", "rev-parse", "--show-toplevel"], Path.cwd())
    if code == 0 and stdout:
        return Path(stdout).resolve()
    return Path(__file__).resolve().parents[1]


def git_output(root: Path, args: list[str]) -> str:
    code, stdout, stderr = run_command(["git", *args], root)
    if code == 0:
        return stdout
    if stderr:
        return f"(git command failed: {stderr})"
    return "(git command failed)"


def collect_git_state(root: Path, now: datetime) -> GitState:
    status = git_output(root, ["status", "--short", "--untracked-files=normal"])
    diff_stat = git_output(root, ["diff", "--stat", "HEAD", "--"])
    recent_commits = git_output(root, ["log", "-5", "--oneline", "--decorate"])

    branch = git_output(root, ["branch", "--show-current"])
    if not branch:
        branch = git_output(root, ["rev-parse", "--short", "HEAD"])
        if branch and not branch.startswith("(git command failed"):
            branch = f"(detached at {branch})"
    if not branch:
        branch = "(unknown)"

    head = git_output(root, ["rev-parse", "HEAD"])
    if not head:
        head = "(unknown)"

    return GitState(
        timestamp=now.isoformat(timespec="seconds"),
        timezone=now.tzname() or now.strftime("%z"),
        branch=branch,
        head=head,
        dirty=bool(status),
        status_lines=split_lines(status),
        diff_stat_lines=split_lines(diff_stat),
        recent_commit_lines=split_lines(recent_commits),
    )


def split_lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.splitlines() if line.strip()]


def slugify(title: str) -> str:
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    if not slug:
        return "session-log"
    return slug[:72].strip("-")


def unique_entry_path(root: Path, now: datetime, title: str) -> Path:
    date = now.date().isoformat()
    directory = root / LOG_ROOT / f"{now.year:04d}" / date
    stem = f"{now.strftime('%H%M%S')}-{slugify(title)}"
    path = directory / f"{stem}.md"
    suffix = 2
    while path.exists():
        path = directory / f"{stem}-{suffix}.md"
        suffix += 1
    return path


def yaml_quote(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def render_front_matter(items: dict[str, object]) -> str:
    lines = ["---"]
    for key, value in items.items():
        if isinstance(value, bool):
            lines.append(f"{key}: {'true' if value else 'false'}")
        elif isinstance(value, int):
            lines.append(f"{key}: {value}")
        elif isinstance(value, list):
            if value:
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {yaml_quote(item)}")
            else:
                lines.append(f"{key}: []")
        else:
            lines.append(f"{key}: {yaml_quote(value)}")
    lines.append("---")
    return "\n".join(lines)


def parse_scalar(value: str) -> object:
    value = value.strip()
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "[]":
        return []
    if value.isdigit():
        return int(value)
    if value.startswith('"') and value.endswith('"'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value.strip('"')
    return value


def parse_front_matter(text: str) -> dict[str, object]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    data: dict[str, object] = {}
    current_key: str | None = None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line.startswith("  - ") and current_key:
            data.setdefault(current_key, [])
            value = parse_scalar(line[4:])
            if isinstance(data[current_key], list):
                data[current_key].append(value)
            continue
        current_key = None
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            data[key] = parse_scalar(value)
        else:
            data[key] = []
            current_key = key
    return data


def markdown_bullets(items: list[str], empty_text: str = "_Not recorded._") -> str:
    cleaned = [item.strip() for item in items if item and item.strip()]
    if not cleaned:
        return empty_text
    return "\n".join(f"- {item.replace(chr(10), chr(10) + '  ')}" for item in cleaned)


def text_block(lines: list[str], empty_text: str = "(none)") -> str:
    text = "\n".join(lines).strip()
    if not text:
        text = empty_text
    return f"```text\n{text}\n```"


def render_entry(
    *,
    title: str,
    trigger: str,
    goal: str,
    changes: list[str],
    decisions: list[str],
    tests: list[str],
    open_threads: list[str],
    next_steps: list[str],
    tags: list[str],
    git_state: GitState,
) -> str:
    front_matter = render_front_matter(
        {
            "schema_version": SCHEMA_VERSION,
            "title": title,
            "trigger": trigger,
            "timestamp": git_state.timestamp,
            "timezone": git_state.timezone,
            "branch": git_state.branch,
            "head": git_state.head,
            "dirty": git_state.dirty,
            "tags": tags,
            "git_status": git_state.status_lines,
            "diff_stat": git_state.diff_stat_lines,
            "recent_commits": git_state.recent_commit_lines,
        }
    )

    dirty_text = "yes" if git_state.dirty else "no"
    goal_text = goal.strip() if goal and goal.strip() else "_Not recorded._"

    return "\n\n".join(
        [
            front_matter,
            f"# {title}",
            "## Goal\n" + goal_text,
            "## What Changed\n" + markdown_bullets(changes),
            "## Decisions\n" + markdown_bullets(decisions),
            (
                "## Code State\n"
                f"- Branch: `{git_state.branch}`\n"
                f"- HEAD: `{git_state.head}`\n"
                f"- Dirty worktree: `{dirty_text}`\n\n"
                "### Changed Files\n"
                + text_block(git_state.status_lines)
                + "\n\n### Diff Summary\n"
                + text_block(git_state.diff_stat_lines)
                + "\n\n### Recent Commits\n"
                + text_block(git_state.recent_commit_lines)
            ),
            "## Verification\n" + markdown_bullets(tests),
            "## Open Threads\n" + markdown_bullets(open_threads),
            "## Next Steps\n" + markdown_bullets(next_steps),
            "",
        ]
    )


def extract_section(text: str, name: str) -> str:
    heading = f"## {name}"
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip() == heading:
            start = index + 1
            break
    if start is None:
        return ""

    collected: list[str] = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        collected.append(line)
    return "\n".join(collected).strip()


def meaningful_section_lines(text: str) -> list[str]:
    ignored = {"_Not recorded._", "(none)"}
    result: list[str] = []
    in_code_block = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or not line or line in ignored:
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        if line and line not in ignored:
            result.append(line)
    return result


def relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_or_preview(path: Path, content: str, root: Path, dry_run: bool) -> None:
    rel = relative_path(root, path)
    if dry_run:
        print(f"[dry-run] would write {rel}")
        print()
        print(content, end="")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(rel)


def parse_date(value: str, now: datetime) -> str:
    if value == "today":
        return now.date().isoformat()
    if value == "yesterday":
        return (now.date() - timedelta(days=1)).isoformat()
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except ValueError:
        raise SystemExit(f"invalid date '{value}', expected today, yesterday, or YYYY-MM-DD")


def read_day_entries(root: Path, date: str) -> list[tuple[Path, dict[str, object], str]]:
    year = date[:4]
    directory = root / LOG_ROOT / year / date
    if not directory.exists():
        return []

    entries: list[tuple[Path, dict[str, object], str]] = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        metadata = parse_front_matter(text)
        if metadata.get("trigger") == "daily":
            continue
        entries.append((path, metadata, text))

    entries.sort(key=lambda item: str(item[1].get("timestamp", "")))
    return entries


def render_daily_entry(
    *,
    date: str,
    title: str,
    entries: list[tuple[Path, dict[str, object], str]],
    git_state: GitState,
    root: Path,
) -> str:
    source_entries = [relative_path(root, path) for path, _, _ in entries]
    front_matter = render_front_matter(
        {
            "schema_version": SCHEMA_VERSION,
            "title": title,
            "trigger": "daily",
            "timestamp": git_state.timestamp,
            "timezone": git_state.timezone,
            "branch": git_state.branch,
            "head": git_state.head,
            "dirty": git_state.dirty,
            "date": date,
            "source_entry_count": len(entries),
            "source_entries": source_entries,
            "git_status": git_state.status_lines,
            "diff_stat": git_state.diff_stat_lines,
            "recent_commits": git_state.recent_commit_lines,
        }
    )

    source_lines: list[str] = []
    goal_lines: list[str] = []
    decision_lines: list[str] = []
    verification_lines: list[str] = []
    open_thread_lines: list[str] = []
    next_step_lines: list[str] = []

    for path, metadata, text in entries:
        timestamp = str(metadata.get("timestamp", "unknown time"))
        short_time = timestamp[11:19] if len(timestamp) >= 19 else timestamp
        entry_title = str(metadata.get("title", path.stem))
        trigger = str(metadata.get("trigger", "unknown"))
        label = f"{short_time} {entry_title}"
        source_lines.append(f"- `{relative_path(root, path)}` - {entry_title} ({trigger})")

        goals = meaningful_section_lines(extract_section(text, "Goal"))
        if goals:
            goal_lines.append(f"- {label}: {goals[0]}")

        for decision in meaningful_section_lines(extract_section(text, "Decisions")):
            decision_lines.append(f"- {label}: {decision}")
        for verification in meaningful_section_lines(extract_section(text, "Verification")):
            verification_lines.append(f"- {label}: {verification}")
        for thread in meaningful_section_lines(extract_section(text, "Open Threads")):
            open_thread_lines.append(f"- {label}: {thread}")
        for step in meaningful_section_lines(extract_section(text, "Next Steps")):
            next_step_lines.append(f"- {label}: {step}")

    if not entries:
        source_lines.append("- No session entries found for this date.")
        goal_lines.append("- No session entries found for this date.")

    dirty_text = "yes" if git_state.dirty else "no"

    return "\n\n".join(
        [
            front_matter,
            f"# {title}",
            "## Source Entries\n" + "\n".join(source_lines),
            "## Goal\n" + "\n".join(goal_lines),
            "## What Changed\n"
            + "_Daily recap generated from existing session entries; no semantic summary was inferred._",
            "## Decisions\n" + ("\n".join(decision_lines) if decision_lines else "_Not recorded._"),
            (
                "## Code State\n"
                f"- Branch: `{git_state.branch}`\n"
                f"- HEAD: `{git_state.head}`\n"
                f"- Dirty worktree: `{dirty_text}`\n\n"
                "### Changed Files\n"
                + text_block(git_state.status_lines)
                + "\n\n### Diff Summary\n"
                + text_block(git_state.diff_stat_lines)
                + "\n\n### Recent Commits\n"
                + text_block(git_state.recent_commit_lines)
            ),
            "## Verification\n" + ("\n".join(verification_lines) if verification_lines else "_Not recorded._"),
            "## Open Threads\n" + ("\n".join(open_thread_lines) if open_thread_lines else "_Not recorded._"),
            "## Next Steps\n" + ("\n".join(next_step_lines) if next_step_lines else "_Not recorded._"),
            "",
        ]
    )


def command_add(args: argparse.Namespace) -> int:
    root = find_repo_root()
    now = datetime.now().astimezone()
    title = args.title or f"{args.trigger.capitalize()} session log"
    git_state = collect_git_state(root, now)
    path = unique_entry_path(root, now, title)
    content = render_entry(
        title=title,
        trigger=args.trigger,
        goal=args.goal or "",
        changes=args.change,
        decisions=args.decision,
        tests=args.test,
        open_threads=args.open_thread,
        next_steps=args.next,
        tags=args.tag,
        git_state=git_state,
    )
    write_or_preview(path, content, root, args.dry_run)
    return 0


def command_daily(args: argparse.Namespace) -> int:
    root = find_repo_root()
    now = datetime.now().astimezone()
    date = parse_date(args.date, now)
    title = args.title or f"Daily Recap: {date}"
    entries = read_day_entries(root, date)
    git_state = collect_git_state(root, now)
    path = unique_entry_path(root, now, title)
    content = render_daily_entry(
        date=date,
        title=title,
        entries=entries,
        git_state=git_state,
        root=root,
    )
    write_or_preview(path, content, root, args.dry_run)
    return 0


def iter_log_files(root: Path) -> list[Path]:
    log_root = root / LOG_ROOT
    if not log_root.exists():
        return []
    return sorted(path for path in log_root.rglob("*.md") if path.name != "README.md")


def command_list(args: argparse.Namespace) -> int:
    root = find_repo_root()
    entries: list[tuple[str, Path, dict[str, object]]] = []
    for path in iter_log_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        metadata = parse_front_matter(text)
        timestamp = str(metadata.get("timestamp", ""))
        entries.append((timestamp, path, metadata))

    entries.sort(key=lambda item: item[0], reverse=True)
    for timestamp, path, metadata in entries[: args.limit]:
        title = str(metadata.get("title", path.stem))
        trigger = str(metadata.get("trigger", "unknown"))
        rel = relative_path(root, path)
        print(f"{timestamp or 'unknown time'} [{trigger}] {title} - {rel}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true", help="print the entry without writing it")

    parser = argparse.ArgumentParser(
        description="Create thesis-friendly Codex session logs for EXACT."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser(
        "add",
        parents=[common],
        help="add a manual or milestone session entry",
    )
    add_parser.add_argument("--trigger", choices=TRIGGERS, default="manual")
    add_parser.add_argument("--title", help="short title for this session entry")
    add_parser.add_argument("--goal", help="goal or checkpoint reached")
    add_parser.add_argument(
        "--change",
        "--changed",
        action="append",
        default=[],
        help="notable code, documentation, or workflow change; repeat as needed",
    )
    add_parser.add_argument(
        "--decision",
        action="append",
        default=[],
        help="decision made and rationale; repeat as needed",
    )
    add_parser.add_argument(
        "--test",
        action="append",
        default=[],
        help="command, check, or result used for verification; repeat as needed",
    )
    add_parser.add_argument(
        "--open-thread",
        action="append",
        default=[],
        help="unresolved risk, question, or follow-up thread; repeat as needed",
    )
    add_parser.add_argument(
        "--next",
        action="append",
        default=[],
        help="recommended next action; repeat as needed",
    )
    add_parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="machine-searchable tag; repeat as needed",
    )
    add_parser.set_defaults(func=command_add)

    daily_parser = subparsers.add_parser(
        "daily",
        parents=[common],
        help="create a daily recap entry from that day's session logs",
    )
    daily_parser.add_argument("--date", default="today", help="today, yesterday, or YYYY-MM-DD")
    daily_parser.add_argument("--title", help="override the recap title")
    daily_parser.set_defaults(func=command_daily)

    list_parser = subparsers.add_parser("list", help="list recent session log entries")
    list_parser.add_argument("--limit", type=int, default=20, help="maximum number of entries to show")
    list_parser.set_defaults(func=command_list)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

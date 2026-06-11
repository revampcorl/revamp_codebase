from __future__ import annotations


def print_box(title: str, lines: list[str]) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    for line in lines:
        print(line)

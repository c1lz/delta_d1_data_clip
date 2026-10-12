#!/usr/bin/env python3
"""Export data_clip inference boundaries to a flat CSV table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re


SEQUENCE_PATTERN = re.compile(r"_(\d+)(?:_|$)")
FIELDS = ("task_sequence", "task_name", "source_video", "fps", "frame_count",
          "cut_index", "frame_index", "timestamp_seconds", "score", "origin")
SUMMARY_FIELDS = ("task_sequence", "task_name", "source_video", "fps", "frame_count",
                  "cut_count", "cut_points")


def sequence_from_directory(directory: Path) -> str:
    match = SEQUENCE_PATTERN.search(directory.name)
    return match.group(1) if match else ""


def rows(root: Path):
    for inference_path in sorted(root.glob("*/inference.json")):
        payload = json.loads(inference_path.read_text(encoding="utf-8"))
        common = {
            "task_sequence": sequence_from_directory(inference_path.parent),
            "task_name": inference_path.parent.name,
            "source_video": payload.get("source_video", ""),
            "fps": payload.get("fps", ""),
            "frame_count": payload.get("frame_count", ""),
        }
        boundaries = payload.get("boundaries", [])
        if not boundaries:
            yield {**common, "cut_index": 0, "frame_index": "", "timestamp_seconds": "",
                   "score": "", "origin": "no_boundary"}
            continue
        for index, boundary in enumerate(boundaries, start=1):
            yield {**common, "cut_index": index, "frame_index": boundary.get("frame_index", ""),
                   "timestamp_seconds": boundary.get("timestamp_seconds", ""),
                   "score": boundary.get("score", ""), "origin": boundary.get("origin", "")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export inference.json cut points to CSV")
    parser.add_argument("input_root", type=Path, help="Directory containing task subdirectories")
    parser.add_argument("output_csv", type=Path, help="CSV output path")
    parser.add_argument("--summary-csv", type=Path,
                        help="Optional one-row-per-task CSV output path")
    args = parser.parse_args()
    if not args.input_root.is_dir():
        parser.error(f"input root does not exist: {args.input_root}")
    values = list(rows(args.input_root))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(values)
    print(f"wrote {len(values)} cut-point rows to {args.output_csv}")
    if args.summary_csv:
        grouped: dict[str, list[dict]] = {}
        for row in values:
            grouped.setdefault(row["task_name"], []).append(row)
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_csv.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            for task_rows in grouped.values():
                first = task_rows[0]
                points = [row for row in task_rows if row["origin"] != "no_boundary"]
                formatted = " | ".join(
                    f"{row['timestamp_seconds']:.3f}s (frame {row['frame_index']}; "
                    f"score {row['score']:.6f}; {row['origin']})" for row in points)
                writer.writerow({key: first[key] for key in SUMMARY_FIELDS[:5]} |
                                {"cut_count": len(points), "cut_points": formatted})
        print(f"wrote {len(grouped)} task-summary rows to {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

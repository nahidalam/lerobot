import json
import jsonlines
import os
from pathlib import Path


def load_tasks_json(tasks_json_path: Path) -> list[dict]:
    with jsonlines.open(tasks_json_path, "r") as f:
        return list(f)

def main():

    dataset_path = Path("/home/leonardo/.cache/huggingface/lerobot/2025-challenge-demos")
    annotations_path = dataset_path / "annotations"
    data_path = dataset_path / "data"

    tasks_json = load_tasks_json(annotations_path / "tasks.jsonl")


if __name__ == "__main__":
    main()
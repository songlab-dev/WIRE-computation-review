"""Verify a retained data copy without regenerating any observations."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
args = parser.parse_args()
root = args.root.resolve()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


ready = json.loads((root / "READY.json").read_text())
for name in ("datasets", "tasks"):
    assert digest(root / (name + ".csv")) == ready[name + "_sha256"]
assert digest(root / "data_config.json") == ready["config_sha256"]
datasets = list(csv.DictReader((root / "datasets.csv").open()))
assert len(datasets) == 600
assert [int(x["data_seed"]) for x in datasets] == list(range(20226091, 20226691))
counts = collections.Counter((x["setting"], int(x["n"])) for x in datasets)
assert len(counts) == 6 and set(counts.values()) == {100}
file_count = 0
for row in datasets:
    directory = root / "data" / row["dataset_id"]
    meta = json.loads((directory / "metadata.json").read_text())
    assert meta["data_seed"] == int(row["data_seed"])
    assert meta["training_seed"] == 2026091 + (int(row["replication"]) - 1) % 5
    for name, expected in meta["files_sha256"].items():
        assert digest(directory / name) == expected, str(directory / name)
        file_count += 1
    splits = list(csv.DictReader((directory / "splits.csv").open()))
    n = int(row["n"])
    assert len(splits) == n
    assert {int(x["row_id"]) for x in splits} == set(range(n))
    assert collections.Counter(x["split"] for x in splits) == {
        "train": int(n * 0.6),
        "val": int(n * 0.2),
        "test": int(n * 0.2),
    }
report = {
    "datasets_verified": len(datasets),
    "csv_hashes_verified": file_count,
    "unique_seeds": 600,
    "disjoint_complete_splits": True,
    "data_config_sha256": ready["config_sha256"],
}
(root / "audit").mkdir(exist_ok=True)
(root / "audit/data_integrity.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))

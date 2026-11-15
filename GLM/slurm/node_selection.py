#!/usr/bin/env python3
"""Restrict workers to verified nodes when Slurm has no CPU features."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def explicit_nodes(value):
    nodes = value.split(",") if value else []
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", n) for n in nodes):
        raise ValueError("Use comma-separated explicit node names, without ranges or whitespace")
    if len(set(nodes)) != len(nodes):
        raise ValueError("Duplicate allowed node names")
    return sorted(nodes)


def select_nodes(allowed, partition_nodes, partition):
    allowed = set(allowed)
    available = set(partition_nodes)
    if not allowed:
        raise ValueError("The allowed node pool is empty")
    if not available:
        raise ValueError("Slurm returned no nodes for partition " + partition)
    missing = allowed - available
    if missing:
        raise ValueError("Allowed nodes are absent from the selected partition: " + ",".join(sorted(missing)))
    return {"partition": partition, "allowed_nodes": sorted(allowed),
            "excluded_nodes": sorted(available - allowed)}


def partition_nodes(partition):
    result = subprocess.run(["sinfo", "--noheader", "--Node", "--partition", partition,
                             "--format", "%N"], check=True, capture_output=True, text=True)
    hostlists = result.stdout.split()
    if not hostlists:
        return []
    expanded = subprocess.run(["scontrol", "show", "hostnames", ",".join(hostlists)],
                              check=True, capture_output=True, text=True)
    return sorted(set(expanded.stdout.split()))


def cpu_model():
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    raise ValueError("Could not read the actual CPU model")


def check_node(node, allowed, cpu, expected_cpu):
    # Each array element requests exactly one node. New nodes added to the
    # partition after submission must never silently enter the benchmark.
    explicit_nodes(node)
    if not node or "," in node:
        raise ValueError("Expected exactly one explicit allocated node name")
    if allowed and node not in allowed:
        raise ValueError("Allocated node is outside the verified pool: " + node)
    if expected_cpu and cpu != expected_cpu:
        raise ValueError("Allocated CPU differs from the required CPU: " + cpu + " vs " + expected_cpu)
    return {"node": node, "cpu": cpu, "node_policy_passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--allow", required=True)
    select.add_argument("--partition", required=True)
    check = sub.add_parser("check")
    check.add_argument("--allow", default="")
    check.add_argument("--node", required=True)
    check.add_argument("--expected-cpu", default="")
    args = parser.parse_args()
    allowed = explicit_nodes(args.allow)
    if args.command == "select":
        result = select_nodes(allowed, partition_nodes(args.partition), args.partition)
    else:
        result = check_node(args.node, allowed, cpu_model(), args.expected_cpu)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

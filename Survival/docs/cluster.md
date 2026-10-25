# Great Lakes setup

Set SURV_PYTHON, DEEP_PYTHON, SBATCH_ACCOUNT, and EXPERIMENT_ROOT as shown in the
main README. R_MODULE defaults to Rtidyverse/4.5.1. You can save these exports in
settings.local.sh, which is ignored by Git.

Each fit requests one node, 4 CPUs, 32 GiB, and a one-hour limit. Numerical
libraries, randomForestSRC, and PyTorch are limited to one computational thread.
Start with CONCURRENCY=16 and increase it only within your allocation and
current scheduler limits.

## Use one processor model

The published run used Intel Xeon Gold 6154 processors. The worker checks the
actual CPU before fitting. Great Lakes nodes in this run did not advertise
CPU-specific Slurm features, so the original job used a verified node pool.

Ask the scheduler for the current standard-node list and verify processor
models before submission. Put all nodes outside your verified 6154 pool in:

```text
$EXPERIMENT_ROOT/audit/exclude_nodes.txt
```

The file contains one comma-separated node list. submit.py passes it to Slurm
for fitting arrays. Do not reuse an old exclusion list without checking it.
For a small check on one known node, use Slurm's --nodelist option when submitting
the batch file. The CPU check still runs in every worker.

## Stages and retries

prepare writes each dataset once. pilot submits the native checks and 64 fits:
one replication for each of 48 reporting cells, plus DeepSurv seeds 2–5 at
n=1500 in both settings and languages. validate requires all of these checks
before formal submission.

run submits the 4,800 tasks. Each task writes to its own directory.
A lock prevents duplicate execution of the same task, and successful records
with a matching code fingerprint are skipped. retry recomputes the missing or
failed list and submits only those tasks.

summary submits result aggregation, final checks, and Slurm accounting.
For the retained original experiment, use its original source releases and
release registry when resuming; the edited publication copy has a different
fingerprint. See [provenance](../provenance/README.md).

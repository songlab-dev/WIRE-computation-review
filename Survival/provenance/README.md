# Source and result provenance

The paper uses 4,800 selected successful fits. The exact runtime source files are
preserved here:

| Source | Selected results | SHA-256 |
| --- | --- | --- |
| source_v1 | 3,600 non-neural fits | 8e9cf0f44fcad799209d502177ff0b4c342a680fda5e5367b770c33f147478b5 |
| source_v2 | 1,200 DeepSurv fits | fd1a34ce62c073a8e650709cac10dfc534b6d643b98e2d7b1707ad5b5da16b16 |

The fingerprint hashes each filename followed by its bytes, in this order:
config.json, common.py, fit_python.py, fit_r.R, run_task.py, prepare.py,
make_splits.R. These copies are unchanged, including the original machine paths.

The publication code uses the final protocol for every model. Its changes are
formatting, documentation, machine-path configuration, and submission helpers.
The source comparison and small rerun check are recorded in
publication_validation.json. It is a separate code version; the original time
and memory measurements remain attached to the original releases.

## DeepSurv correction

Batch size 128 left a final batch of four training observations at n=1500.
An event-free batch produced a non-finite Cox loss, although early stopping
could restore finite predictions. Batch size 150 avoids that remainder at all
three sample sizes. Both languages were rerun with this setting and finite-loss
checks.

The 228 completed older DeepSurv fits are excluded from the paper. Their
records, canceled pending tasks, and the replacement release are recorded in
protocol_amendment.json. The final experiment had one failed startup-monitor
attempt, which succeeded on retry, and no unresolved failures.

## Stored material

results contains the final per-fit metrics, 48-cell summary, LaTeX tables,
dataset/task seed maps, and compact validation evidence. Full CSV datasets,
predictions, training logs, and attempt histories are retained in the server
archive. release_registry.json preserves original server paths for provenance;
they are not portable input paths.

Earlier 50-replication code and results remain in the repository's Git history.
Do not combine them with the revised results.

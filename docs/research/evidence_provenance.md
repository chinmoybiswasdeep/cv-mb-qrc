# Evidence and provenance

Each run is an append-safe evidence directory. A raw job becomes complete only
after its finite JSON record is atomically replaced and a second atomic marker
binds the job ID to the result SHA-256. The manifest is ordered and retry-safe.
Feature caches are content-addressed by method, seeds, task, split inputs, and
the complete resolved configuration.

At terminal completion, `raw/run_index.json` records expected/completed job IDs,
their exact paths, every evidence-file SHA-256, and a Merkle-style root over the
sorted file/hash pairs. The index covers resolved configuration, environment,
datasets, splits, raw jobs, completion markers, caches, runtime records, summary
tables, report, figure provenance, and SVG/PDF/PNG files. Provenance separately
records commits, dirty state, source hashes, dependency versions, CPU/memory,
BLAS/LAPACK, seed roles, command, worker policy, and UTC timestamps.

Verification fails closed on a missing or extra file; duplicate, missing,
malformed, non-finite, truncated, or changed job; invalid completion marker;
nonterminal run; source/config/dataset/split mismatch; failed job; replaced
figure; or absent figure provenance. It reconstructs every main and derived
table, including paired comparisons, directly from registered raw jobs. Stored
aggregates are never trusted as primary evidence.

Run:

```sh
cvmbqrc verify-run path/to/run
# equivalent
python -m cv_mb_qrc.cli verify-run path/to/run
```

Exit status is nonzero for invalid evidence. Integrity-valid development or CI
runs remain publication-invalid. Publication validity additionally requires a
publication study class, clean source worktrees, all required seeds/methods and
null repetitions, and completed report generation.

This is tamper-evident evidence, not a cryptographic signature: a party able to
modify both data and source can rebuild all hashes. Archival releases should
therefore sign the final index externally and deposit it with the immutable
source release.

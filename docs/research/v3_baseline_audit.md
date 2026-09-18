# V3 baseline audit

An earlier partial audit began at
`2d227067881351ed64baf455028dfc83594e5040`. The implementation continuation
documented here began on `fix/research-v3` at
`e7af3f14d69b76ffa82def9fea4e232deb177398` with a clean worktree. The earlier
dependency-limited observations below are retained as historical context, not
as the final verification result.

The active interpreter is Python 3.14.4. `pip list --format=freeze` reported
only `pip==26.0.1`; NumPy, PhotoGraphiQ, Piquasso, Graphix, MentPy, pytest,
Ruff, mypy, and build are unavailable. No PhotoGraphiQ checkout is installed,
so its commit and dirty status cannot be recorded.

| Command | Result | Classification |
| --- | --- | --- |
| `python -m pytest -q` | `No module named pytest` | dependency unavailable |
| `python -m pytest --cov=...` | `No module named pytest` | dependency unavailable |
| `python -m ruff check ...` | `No module named ruff` | dependency unavailable |
| `python -m ruff format --check ...` | `No module named ruff` | dependency unavailable |
| `python -m mypy` | `No module named mypy` | dependency unavailable |
| `python -m build` | `No module named build` | dependency unavailable |
| `python -m pip wheel --no-deps .` | passed | wheel construction only |
| CI experiment | `No module named numpy` | dependency unavailable |

The existing `experiments/measurement_based_reservoir/results` tree was
created under the old `photographiqml` namespace. It is legacy evidence and is
not a validation of this checkout. The previous v2 attempt also introduced a
joint q/p Gaussian sampler labelled physical readout. V3 removes that claim:
the physical mode now fails explicitly at construction until a public-API
tap/homodyne channel is implemented and independently tested.

## Continuation environment

The later V3 continuation used the locally installed Astral Python 3.11.16
runtime and the clean PhotoGraphiQ checkout at
`6d49da06fa6ede78bf06ea1df07ccfb3b4f5250e`. This made NumPy, Piquasso,
Graphix, MentPy 0.1.0a15, pytest, Ruff, mypy and build available.

The final full suite passed 57 tests with no failures or skips; dedicated core,
Graphix and MentPy selections passed 48, 8 and 1 tests respectively. Statement
coverage was 91%. Ruff, formatting, mypy, sdist/wheel build, clean-wheel import,
the six-job CI smoke run, report regeneration and provenance verification all
passed. The smoke output is a CI study and the source worktree was dirty while
implementing V3, so it is correctly not publication-valid. A copied configuration
tamper failed verification with `configuration hash mismatch`.

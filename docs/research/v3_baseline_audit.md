# V3 baseline audit

The prompt's audited commit is the checked-out commit:
`2d227067881351ed64baf455028dfc83594e5040`. Work began on branch
`fix/research-v3`; the worktree was already dirty with the in-progress v2
namespace migration and audit files. These changes are treated as local work,
not as baseline evidence.

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

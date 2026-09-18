# Hardware-readout boundary

The implemented Tier A and Tier B features use exact simulator access to the
Gaussian mean and covariance. They are oracle observables and are not claimed
as simultaneously accessible hardware measurements. In particular, the code
does not substitute joint q/p samples for a physical tap-homodyne instrument.

`TapHomodyneReadout` raises an explicit capability error. The audited public
backend does not expose the complete operation required here: couple a fresh
vacuum probe through a tap beam splitter, homodyne a selected quadrature with
finite detector efficiency, sample an outcome, and return the conditioned
surviving memory state for subsequent temporal evolution.

The conservative measurement-budget helper assumes means share the diagonal
variance settings and that each off-diagonal covariance needs an additional
rotated setting. For `m` memory modes and `S` shots per setting:

| Tier | Simulator features | Distinct settings | State copies |
|---|---:|---:|---:|
| A | `4m` | `2m` | `2mS` |
| B | `2m + (2m)(2m+1)/2` | `(2m)(2m+1)/2` | settings × `S` |

For the default two modes and 1,000 shots, this is at least 4,000 state copies
for Tier A and 10,000 for Tier B per temporal observation, before calibration,
loss mitigation, or repeated conditional trajectories.

A hardware implementation must add a validated Gaussian instrument, explicit
tap strength, inefficiency/loss channels, backaction tests against analytic
cases, and a protocol for rebuilding a temporal state after destructive
measurements. Until then, claims are restricted to simulator-derived feature
maps and a hardware-readout roadmap.

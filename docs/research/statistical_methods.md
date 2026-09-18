# Statistical methods

## Experimental units and splits

The independent replication hierarchy is dataset seed, then reservoir seed within
dataset. Time points from one trajectory are not independent replicates. All
methods in a paired comparison use the same dataset seed, reservoir seed,
chronological split, washout, embargo, input-access policy, regularization grid,
and train-only normalization. The test partition is evaluated only after
validation selects the ridge penalty.

NARMA inputs are deterministically mapped from the IID driver to `[0, 0.2]`.
This resolved scale prevents numerical divergence of the declared NARMA10 and
NARMA20 recurrences and is recorded in every run configuration.

Forecast metrics are RMSE, MAE, raw R² (negative values are retained), Pearson
correlation, and NRMSE = RMSE divided by the population standard deviation of
the test target. Closed-loop valid prediction time is the first rollout index at
which normalized absolute error exceeds the configured threshold, or the full
rollout length if that never occurs.

## Capacity estimator

For target component `j`, the observed capacity score is the untouched test R²,
`s_j`. For `B` dataset-level null transformations generated before splitting,
the null scores are `s*_{b,j}`. The signed bias-corrected estimate is

`c_j = s_j - median_b(s*_{b,j})`.

It is not clipped. The empirical one-sided p-value is

`p_j = (1 + count_b[s*_{b,j} >= s_j]) / (B + 1)`.

Benjamini–Hochberg correction is applied across the declared family containing
all linear-delay, quadratic-self, and cross-delay components. The report keeps
these three subfamilies separate and defines total significant capacity as the
sum of signed corrected components whose q-value is at most alpha. Raw signed
and legacy clipped aggregates are retained only as diagnostics. The theoretical
upper bound is the smaller of target count and the effective design rank.

The implemented nulls are:

- block permutation (default): permutes complete target blocks before splitting;
- circular shift: displacement must exceed both maximum delay and exclusion window;
- independent surrogate: constructs an independent IID driver and all targets
  from it.

Every null seed and transformation descriptor is stored. Development calibration
uses 10 independent synthetic datasets, 99 null transformations per method, and
a predeclared acceptable null discovery interval of 0–10%. The default was
selected from calibration controls, never from which reservoir performed best.

## Uncertainty and model comparisons

Summary intervals use a dataset-then-reservoir hierarchical bootstrap. Paired
model effects are formed only from matching dataset/reservoir keys and report
the mean and median difference, paired standardized effect, and hierarchical
95% interval. A sign-flip test operates on dataset-cluster means; BH q-values
are computed across the declared model comparisons. With few dataset clusters,
these tests are necessarily low resolution and nonsignificance is described as
“no evidence of difference,” never equality.

CI smoke runs are engineering checks and intentionally report no scientific
uncertainty. Development results are exploratory. Only the locked publication
configuration is confirmatory.

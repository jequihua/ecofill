# Methodology

How ecofill reconstructs missing values and what each step assumes. The API
is in [docs/guide.md](docs/guide.md); the full bibliography is in
[docs/references.md](docs/references.md).

## Data, masks and time

Input is a table on an explicit, regular, timezone-aware grid with declared
cadence, variables and units. ecofill does not sort, round, deduplicate or
reindex timestamps. A missing row must be inserted explicitly and is marked
`inserted`. The caller supplies a QC-passed `observed` mask. Optional masks
mark QC rejection, structural absence (for example an uninstalled sensor) and
artificial hiding for evaluation. Hiding a variable
also hides its declared derived descendants. Only observed, unhidden cells are donors.

## Gaps, plans and abstention

A gap is a run of missing slots. Its duration is slots × cadence. Profiling
records each gap's position, geometry (bounded, leading, trailing or full
channel) and which variables are missing together. A plan names one method, an
information mode and a positive whole-gap limit. It marks each gap eligible or
abstaining with a reason, such as `gap_exceeds_limit`, `structural_absence` or
missing context. Gaps are never partly filled to satisfy a limit. `offline`
mode may use observations after a gap; `causal` mode refuses methods that
would. Execution re-derives the plan from the data and policy and rejects an
edited or stale plan. It writes only missing cells.

## Methods

**Classical baselines.** Persistence, next observation, linear interpolation,
a seasonal analog (median of same-phase observations) and a scalar local-level
Kalman filter or smoother with caller-supplied variances. Parameters are not
fitted.

**Driver regression.** Ordinary least squares, random forest or histogram
gradient boosting predicts a target from an explicit list of predictors. Fits
use observed training labels only. Derived variables are recomputed from
reconstructed parents with declared formulas rather than being regressed.

**Marginal distribution sampling.** `mds_hesseflux` wraps an unchanged copy of
the hesseflux gap-filling module. Its documentation attributes the algorithm to
Reichstein et al. (2005): fill a flux from observations taken under similar
radiation, temperature and vapour-pressure deficit within widening time
windows. REddyProc (Wutzler et al., 2018) is an established eddy-covariance
post-processing tool that also provides gap filling. ecofill claims agreement
with the pinned code only, not with REddyProc or other implementations. Its quality tiers describe the search, not uncertainty.

**Forecasting across a gap.** A forecaster pretrained for prediction, such as
TimesFM (Das et al., 2024), is not a native gap-filler. ecofill therefore makes
the adaptation explicit. It forecasts forward from contiguous observations
before the gap and backward from time-reversed observations after it. It then
blends the two with backward weight *i*/(*L*−1) at gap position *i* of *L*
slots (0.5 for one slot). Sartipi et al. (2025) used the same
forward/backward/weighted-interpolation construction to benchmark forecasting
models on smart-meter gaps. Reversal is a modeling assumption. Covariates
whose meaning depends on direction or availability time must be declared
reversible and available, or they are refused.

**Deep imputation.** BRITS (Cao et al., 2018) and SAITS (Du et al., 2023)
are trained on nonoverlapping windows with training-only scaling and an
explicit training hiding mask. ecofill uses the vendored PyPOTS numerical cores
with a small local training loop.

**Time-indexed regression.** Following the time-indexed view of imputation
(Le Naour et al., 2025), a tabular regressor predicts each missing value from
fixed features of its timestamp: normalized elapsed time and daily and weekly
sine/cosine phases. The regressor is fitted on a frozen, capped bank of observed
reference rows, possibly on both sides of the gap. The optional backend is
TabPFN (Hollmann et al., 2025; here version 3.5). This is again an adaptation.
The model sees time features, not a learned model of missingness, and cannot
anticipate changes that its reference bank does not contain.

## Identity and ensembles

Every fitted component has an identity: backend, checkpoint, settings, fitted
parameters and, for time-indexed methods, a digest of the reference bank. A fixed-weight ensemble recipe hashes its
members, weights and training split. `blend` rejects a member whose
configuration differs from the recipe. A cell is blended only if every member
predicted it; otherwise it abstains. Weights are never searched.

## Evaluation, calibration and draws

Benchmarks hide QC-passed observations in fresh copies of the data, so natural
gaps are never scored. They report hidden, predicted, abstained and failed
counts beside error metrics, so abstention cannot improve a score unseen.
Scales come from the training portion only.

Uncertainty uses caller-declared blocks. Blocks carry a source group,
physical identity, split and context. Calibration, evaluation and training
contexts must be separated by a declared purge within a source group, and
aliased or duplicated blocks are rejected. Each complete calibration block
gives one score: its largest absolute residual divided by a training scale.
With *n* blocks, the threshold is the *k*-th smallest score with
*k* = ⌈(*n*+1)(1−α)⌉, the split-conformal rank (Angelopoulos and Bates,
2021). Intervals are point ± threshold × scale for a new block of the same
support stratum. Their nominal simultaneous coverage assumes that calibration
and new blocks are exchangeable. Temporal dependence can break that (Barber
and Pananjady, 2025), and ecofill does not test it. Completed draws add one
whole calibration residual block to the point reconstruction. This keeps the
residual pattern within the block and across channels, and derived variables
are recomputed within the same draw. Draws are alternative reconstructions,
not observations.

## What is not established

Tests use synthetic data and prove contracts, not accuracy. The small
comparisons behind this release used single synthetic or development series;
they do not rank methods, set a default or show field coverage. No GPU run has
been made.

## References

- Angelopoulos, A. N., and Bates, S. (2021). A gentle introduction to conformal
  prediction and distribution-free uncertainty quantification. arXiv:2107.07511.
  <https://arxiv.org/abs/2107.07511>
- Barber, R. F., and Pananjady, A. (2025). Predictive inference for time series:
  why is split conformal effective despite temporal dependence? arXiv:2510.02471.
  <https://arxiv.org/abs/2510.02471>
- Cao, W., Wang, D., Li, J., Zhou, H., Li, L., and Li, Y. (2018). BRITS:
  Bidirectional recurrent imputation for time series. *Advances in Neural
  Information Processing Systems 31*.
  <https://papers.nips.cc/paper_files/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html>
- Das, A., Kong, W., Sen, R., and Zhou, Y. (2024). A decoder-only foundation
  model for time-series forecasting. *Proceedings of the 41st International
  Conference on Machine Learning*, PMLR 235, 10148–10167.
  <https://proceedings.mlr.press/v235/das24c.html>
- Du, W., Côté, D., and Liu, Y. (2023). SAITS: Self-attention-based imputation
  for time series. *Expert Systems with Applications*, 219, 119619.
  <https://doi.org/10.1016/j.eswa.2023.119619>
- Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo,
  S. B., Schirrmeister, R. T., and Hutter, F. (2025). Accurate predictions on
  small data with a tabular foundation model. *Nature*, 637, 319–326.
  <https://doi.org/10.1038/s41586-024-08328-6>
- Le Naour, E., Nabil, T., Petralia, A., and Agoua, G. (2025). Are
  time-indexed foundation models the future of time series imputation?
  arXiv:2511.05980. <https://arxiv.org/abs/2511.05980>
- Reichstein, M., et al. (2005). On the separation of net ecosystem exchange
  into assimilation and ecosystem respiration: review and improved algorithm.
  *Global Change Biology*, 11(9), 1424–1439.
  <https://doi.org/10.1111/j.1365-2486.2005.001002.x>
- Sartipi, A., Delgado Fernández, J., Potenciano Menci, S., and Magitteri, A.
  (2025). Bridging smart meter gaps: A benchmark of statistical, machine
  learning and time series foundation models for data imputation. *2025 IEEE
  Kiel PowerTech*. <https://doi.org/10.1109/PowerTech59965.2025.11180477>
- Wutzler, T., et al. (2018). Basic and extensible post-processing of eddy
  covariance flux data with REddyProc. *Biogeosciences*, 15, 5015–5030.
  <https://doi.org/10.5194/bg-15-5015-2018>

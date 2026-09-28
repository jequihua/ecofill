# References

Every entry below was checked against a primary record: the publisher or
proceedings page, the DOI registration record or the arXiv listing. Records
were accessed on 2026-09-28. Preprints are cited as preprints. A venue is named
only where a publisher or proceedings record confirmed it.

## Methods ecofill implements or adapts

- **Angelopoulos, A. N., and Bates, S. (2021).** A gentle introduction to
  conformal prediction and distribution-free uncertainty quantification.
  arXiv:2107.07511 (v6, 2022). <https://arxiv.org/abs/2107.07511>.
  A revised version appeared as *Conformal Prediction: A Gentle Introduction*,
  Foundations and Trends in Machine Learning 16(4), 494–591 (2023),
  <https://doi.org/10.1561/2200000101>.
  Used for: the finite-sample rank ⌈(*n*+1)(1−α)⌉ of the block calibration.
- **Cao, W., Wang, D., Li, J., Zhou, H., Li, L., and Li, Y. (2018).** BRITS:
  Bidirectional recurrent imputation for time series. *Advances in Neural
  Information Processing Systems 31*.
  <https://papers.nips.cc/paper_files/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html>.
  Used for: the `brits_pypots` method (vendored PyPOTS core).
- **Che, Z., Purushotham, S., Cho, K., Sontag, D., and Liu, Y. (2018).**
  Recurrent neural networks for multivariate time series with missing values.
  *Scientific Reports* 8, 6085. <https://doi.org/10.1038/s41598-018-24271-9>.
  Used for: the temporal-decay layer that the vendored BRITS core imports.
- **Das, A., Kong, W., Sen, R., and Zhou, Y. (2024).** A decoder-only
  foundation model for time-series forecasting. *Proceedings of the 41st
  International Conference on Machine Learning*, PMLR 235, 10148–10167.
  <https://proceedings.mlr.press/v235/das24c.html>.
  Used for: the TimesFM forecasting backends.
- **Du, W., Côté, D., and Liu, Y. (2023).** SAITS: Self-attention-based
  imputation for time series. *Expert Systems with Applications* 219, 119619.
  <https://doi.org/10.1016/j.eswa.2023.119619> (preprint arXiv:2202.08516).
  Used for: the `saits_pypots` method (vendored PyPOTS core).
- **Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M.,
  Hoo, S. B., Schirrmeister, R. T., and Hutter, F. (2025).** Accurate
  predictions on small data with a tabular foundation model. *Nature* 637,
  319–326. <https://doi.org/10.1038/s41586-024-08328-6>.
  Used for: TabPFN, the regressor behind `TabPFNBackend`.
- **Jäger, B., Erickson, N., Grinsztajn, L., et al. (2026).** TabPFN-3.5:
  Technical report. arXiv:2609.17895. <https://arxiv.org/abs/2609.17895>.
  Used for: the TabPFN 3.5 checkpoint version that `TabPFNBackend` accepts.
- **Le Naour, E., Nabil, T., Petralia, A., and Agoua, G. (2025).** Are
  time-indexed foundation models the future of time series imputation?
  arXiv:2511.05980 (v2, 2026). <https://arxiv.org/abs/2511.05980>.
  Used for: the time-indexed formulation of imputation.
- **Reichstein, M., Falge, E., Baldocchi, D., Papale, D., Aubinet, M., et al.
  (2005).** On the separation of net ecosystem exchange into assimilation and
  ecosystem respiration: review and improved algorithm. *Global Change Biology*
  11(9), 1424–1439. <https://doi.org/10.1111/j.1365-2486.2005.001002.x>.
  Used for: marginal distribution sampling, as attributed by the documentation
  of the vendored hesseflux module that `mds_hesseflux` wraps.
- **Sartipi, A., Delgado Fernández, J., Potenciano Menci, S., and Magitteri,
  A. (2025).** Bridging smart meter gaps: A benchmark of statistical, machine
  learning and time series foundation models for data imputation. *2025 IEEE
  Kiel PowerTech*, 1–7. <https://doi.org/10.1109/PowerTech59965.2025.11180477>
  (preprint arXiv:2501.07276).
  Used for: precedent for combining forward and backward forecasts with
  position-weighted interpolation across a gap.
- **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
  A. N., Kaiser, Ł., and Polosukhin, I. (2017).** Attention is all you need.
  *Advances in Neural Information Processing Systems 30*.
  <https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html>.
  Used for: the attention layers of the vendored SAITS core.

## Related work (not implemented in ecofill)

- **Barber, R. F., and Pananjady, A. (2025).** Predictive inference for time
  series: why is split conformal effective despite temporal dependence?
  arXiv:2510.02471 (v2, 2026). <https://arxiv.org/abs/2510.02471>.
  Relevance: conditions under which temporal dependence weakens
  exchangeability-based coverage; ecofill does not test them.
- **Hoo, S. B., Müller, S., Salinas, D., and Hutter, F. (2025).** From tables to
  time: Extending TabPFN-v2 to time series forecasting. arXiv:2501.02945
  (v4, 2026). <https://arxiv.org/abs/2501.02945>.
  Relevance: time-feature engineering for a tabular foundation model
  (TabPFN-TS).
- **Moffat, A. M., Papale, D., Reichstein, M., Hollinger, D. Y., Richardson,
  A. D., et al. (2007).** Comprehensive comparison of gap-filling techniques
  for eddy covariance net carbon fluxes. *Agricultural and Forest Meteorology*
  147(3–4), 209–232. <https://doi.org/10.1016/j.agrformet.2007.08.011>.
  Relevance: comparative background for eddy-covariance gap filling.
- **Montero-Manso, P., Athanasopoulos, G., Hyndman, R. J., and Talagala,
  T. S. (2020).** FFORMA: Feature-based forecast model averaging.
  *International Journal of Forecasting* 36(1), 86–92.
  <https://doi.org/10.1016/j.ijforecast.2019.02.011>.
  Relevance: learned forecast-combination weights; ecofill's ensembles use
  fixed weights only.
- **Pastorello, G., Trotta, C., Canfora, E., Chu, H., Christianson, D., et al.
  (2020).** The FLUXNET2015 dataset and the ONEFlux processing pipeline for
  eddy covariance data. *Scientific Data* 7(1).
  <https://doi.org/10.1038/s41597-020-0534-3>.
  Relevance: an established eddy-covariance processing pipeline; ecofill
  claims no equivalence with it.
- **Richardson, A. D., and Hollinger, D. Y. (2007).** A method to estimate the
  additional uncertainty in gap-filled NEE resulting from long gaps in the CO2
  flux record. *Agricultural and Forest Meteorology* 147(3–4), 199–208.
  <https://doi.org/10.1016/j.agrformet.2007.06.004>.
  Relevance: long gaps add uncertainty beyond short-gap estimates.
- **Wutzler, T., Lucas-Moffat, A., Migliavacca, M., Knauer, J., Sickel, K.,
  Šigut, L., Menzer, O., and Reichstein, M. (2018).** Basic and extensible
  post-processing of eddy covariance flux data with REddyProc.
  *Biogeosciences* 15, 5015–5030. <https://doi.org/10.5194/bg-15-5015-2018>.
  Relevance: an established eddy-covariance post-processing tool with gap
  filling; ecofill claims no equivalence with it.

## Software and model sources

- PyPOTS 1.5 numerical cores (BSD-3-Clause), vendored in `src/ecofill/_pypots/`.
- hesseflux `gapfill.py`, revision `53d1f9a75ba388d400e0cff65530f0da33720d8f`
  (MIT), <https://github.com/mcuntz/hesseflux>.
- TimesFM source and checkpoints, <https://github.com/google-research/timesfm>;
  the TimesFM 3 announcement, Google Research blog, 31 August 2026,
  <https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/>.
- TabPFN 9.0.0 source, <https://github.com/PriorLabs/TabPFN>; TabPFN 3.5
  checkpoint, <https://huggingface.co/Prior-Labs/tabpfn_3_5>.

Exact checkpoint revisions, hashes and license terms are in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

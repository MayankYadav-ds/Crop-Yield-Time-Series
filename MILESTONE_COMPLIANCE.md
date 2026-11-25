# Milestone compliance matrix

## Verification boundary

Official Milestone 1, Milestone 3 and final-evaluation requirement PDFs were unavailable. The reviewed report PDF is an edited research report, not a faculty requirement specification. This matrix verifies observable implementation against the available project brief. It cannot certify exact institutional compliance or marks.

| Milestone | Requirement or evidence area | Status | Evidence and limitation |
|---|---|---|---|
| 1 | Dataset dimensions, fields, dtypes, missingness and duplicates | VERIFIED | dataset_summary.json; feature_descriptions.json; raw CSV digest |
| 1 | Validity, seasonal overlap and annual aggregation | VERIFIED | aggregation_audit.csv; preprocessing_summary.json; weighted ratio identity tests |
| 1 | Missing production policy and outliers | VERIFIED | No imputation; invalid-component groups excluded; IQR flags retained without automatic deletion |
| 1 | Time series construction and continuity | VERIFIED WITH LIMITS | Annual district/crop panel; state/national-by-crop exports; continuity report; name-based geography lacks boundary crosswalk |
| 1 | Trends, distributions, missingness, rolling mean/SD, changes and decomposition | VERIFIED | Actual 29 notebook graph outputs; descriptive centered smooth without unjustified seasonality |
| 1 | Source, license and units | UNVERIFIED | Exact source lineage/license and physical units absent; contextual official portal is not proof of local file origin |
| 1 | Original milestone period | REVISED SCOPE | Archived first-milestone slides described 2013–2022; the confirmed replacement CSV ends in 2015. Final analysis explicitly uses 2005–2014; no false exact reproduction claim |
| 2 | ADF/KPSS and ACF/PACF | VERIFIED | 769 longer pre-test histories; representative level and AR1 residual diagnostics; warnings recorded |
| 2 | Naive, mean, MA3 and drift baselines | VERIFIED | Actual validation and both test protocols on common targets |
| 2 | SES, Holt, damped Holt, autoregression | VERIFIED | Explicit custom conditional least-squares smoothing and unconstrained OLS AR(1); saved coefficients and optimizer statuses |
| 2 | Established ARIMA / likelihood ETS | NOT SUCCESSFULLY EXECUTED | Local library import blocked; custom LS models and AR1_OLS are not renamed as library ARIMA/ETS |
| 2 | Seasonality and multivariate/volatility applicability | DOCUMENTED EXCLUSIONS | No justified periodic structure; short residual histories and unvalidated multivariate design; diagnostics are exploratory |
| 2 | Chronological validation and error metrics | VERIFIED WITH LIMITS | Origins/targets verified; seven metrics independently recalculated; future-feature checks passed; retrospective cohort/test reuse remains |
| 3 | Global XGBoost | VERIFIED | Twelve chronological candidates; full tuning scores/settings; selected-model refit and forecasts |
| 3 | Global LSTM and GRU | VERIFIED | Real PyTorch training, three-year sequences, training-only normalization, inner stopping, checkpoints, losses, parameter counts and both test protocols |
| 3 | Lightweight Transformer | VERIFIED | Real patching, embeddings, positions, two heads, encoder and forecast head; tiny inspired architecture, not exact PatchTST |
| 3 | Prophet | IMPLEMENTED; RUNTIME-EXCLUDED | Four longer crop aggregates specified; native optimizer failed; no metric or successful-fit claim |
| 3 | Common observations, horizons and model comparison | VERIFIED | 3,336 targets per district model/protocol; aggregate Prophet excluded from district leaderboard |
| 3 | Future forecasts and uncertainty | PARTIAL | Selected-model 2015–2017 point forecasts exported; no calibrated intervals/coverage claimed |
| All | Report, presentation, notebook consistency | VERIFIED AT PUBLISHED PRECISION | All metric tables and native chart values checked against canonical CSVs; old working version preserved |
| All | Reproducibility and tests | VERIFIED LOCALLY | Real-kernel sequential execution; exact numeric repeated-run match; saved model checkpoints and input digests |
| All | Hosted Google Colab run | UNVERIFIED EXTERNALLY | Upload/installation branches tested locally; browser signed out of Google |
| All | Individual team contribution | UNVERIFIED | Names/IDs supplied; no fabricated task allocation or contribution percentages |

## Overall status

The conceptual and executable scope of all three milestones is represented, including genuinely trained advanced models. An unconditional claim that all three official milestones are compliant is not supported without the actual faculty documents. If a faculty requirement makes ARIMA, Prophet or calibrated intervals mandatory without an exclusion option, that item remains unmet rather than passed.

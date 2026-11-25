# Model execution and exclusion evidence

| Model/family | Actual status | Evidence-based explanation |
|---|---|---|
| XGBoost | Trained and evaluated | Twelve completed chronological candidates; selected trial 1; test and forecast files present |
| LSTM | Trained and evaluated | 1,233 parameters, 49 selected epochs; saved losses/checkpoints and predictions |
| GRU | Trained and evaluated | 929 parameters, 52 selected epochs; saved losses/checkpoints and predictions |
| PatchTransformerTiny | Trained and evaluated | 657 parameters, 31 selected epochs; two overlapping patches and one encoder; not exact PatchTST |
| Prophet | Implemented, native optimization failed | Package imported, longer stable crop aggregates prepared, daily/weekly/yearly seasonal terms disabled; first native optimizer call exited with an error. Full traceback and native-process log retained. No successful fit, tuning completion or metric is claimed |
| Statsmodels ARIMA / smoothing imports | Failed locally | Repeated import test raises application-control block on state-space _inversions extension. No security-policy change attempted. Custom LS smoothing and AR1_OLS remain distinct estimators |
| SARIMA / seasonal Holt–Winters | Excluded | No defensible repeated period established in the annual observations. Annual frequency alone does not rule out every multi-year cycle; no such cycle was verified here |
| ARCH/GARCH | No fitted volatility model | One longer AR1 diagnostic has only 14 annual residuals and is exploratory. Its p-value is not panel-wide evidence, and no robust volatility forecasting design was validated |
| VAR | Outside validated experiment scope | No validated jointly aligned stationary multivariate design; current comparison is univariate yield history. This is not a claim that VAR is mathematically impossible |
| Calibrated probabilistic forecasts | Not implemented | Only two outer validation and three test years; no validated multi-horizon calibration procedure or interval coverage is claimed |

Relevant evidence: model_training_status.json, applicability_diagnostics.json, dependency_capabilities.json, arima_capability.json, execution_log.txt, hyperparameter_results.csv, *_loss.csv, models/ and saved prediction files.

No excluded model is given a zero error, invented score or successful-training label. Prophet could be retried in a supported environment; any resulting aggregate-target scores must remain separate from the district-level leaderboard and require corresponding report updates.

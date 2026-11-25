# Report accuracy audit

The detailed final evaluation audit is EVALUATION_READINESS_AUDIT.md. Machine-readable checks are in REPORT_EVIDENCE/audits/independent_verification.json.

- Dataset counts and input identity were independently checked; the original CSV hash is unchanged.
- All 36 model/protocol metric rows and actual-target alignment were independently recalculated.
- Two real-kernel prediction runs match exactly numerically for validation, both test protocols and future forecasts.
- Validation RMSE freezes model selection; no test-based reselection is presented.
- Neural normalization and training-target years are checked; parameter updates and checkpoint inference replay passed.
- All published report/presentation metric cells agree at displayed precision; nine native chart datasets match source CSVs to published precision.
- Seven A4 report pages, 167 abstract words, ten cited references, eight report figures and four tables.
- Sixteen PowerPoint slides opened in PowerPoint; nine editable charts and seven editable tables. All native slide renders and report pages visually inspected.
- Exactly fifty Hinglish questions and an eleven-minute demonstration script.
- Working-version backup integrity and unchanged original files verified.

No full faculty-PDF compliance, hosted Colab run, successful Prophet fit, guaranteed forecast accuracy, causal effect or calibrated interval is asserted. See model exclusions and milestone matrix for unresolved items.

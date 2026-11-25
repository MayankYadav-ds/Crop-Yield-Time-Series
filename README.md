# Agricultural Crop Yield Time Series Forecasting

**Mayank Yadav** · Bennett University · CSET 369

Research notebook, report, presentation and reproducibility evidence. This repository preserves the executed experiments from 18 November 2026; presentation credits were updated on 25 November 2026.

- [Executed notebook](Complete_Time_Series_Project_EXECUTED.ipynb)
- [Research report](Final_Time_Series_Research_Report.pdf)
- [Presentation](Final_Presentation.pptx)
- [50 Hinglish viva answers](VIVA_PREPARATION_HINGLISH.md)
- [11-minute demonstration](LIVE_DEMONSTRATION_SCRIPT.md)

# Final evaluation submission

Use this folder for the revised study. The earlier outputs/FINAL_SUBMISSION remains unchanged and has a verified backup. This edition uses the confirmed seven-column crop_production.csv, not the earlier 16-column dataset.

## Main files

- Complete_Time_Series_Project_EXECUTED.ipynb: real Jupyter-kernel execution with 39 cells, 19 executed code cells, zero error outputs and 29 actual graphs.
- Complete_Time_Series_Project.py: consolidated executable source.
- Final_Time_Series_Research_Report.pdf / .docx: seven-page report, 167-word abstract and ten references.
- Final_Presentation.pptx / .pdf: sixteen slides; editable native charts/tables; tested in PowerPoint.
- VIVA_PREPARATION_HINGLISH.md: exactly fifty short evaluation answers.
- LIVE_DEMONSTRATION_SCRIPT.md: eleven-minute presentation and live check.
- EVALUATION_READINESS_AUDIT.md, MILESTONE_COMPLIANCE.md, MODEL_EXCLUSIONS.md, COLAB_COMPATIBILITY.md and REPORT_ACCURACY_AUDIT.md: exact verification scope and limitations.
- evidence/: input copy, metrics/predictions, trial settings, losses, checkpoints, figures, logs, source and machine-readable audits.

## Reproduce locally

Use Python 3.12 (tested: 3.12.14). In a fresh environment:

    python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -r requirements.txt
    python Complete_Time_Series_Project.py --data evidence/raw/crop_production.csv --output results

The input is included at evidence/raw/crop_production.csv. Results go to a separate directory.

The code uses local CPU with seed 42 and two PyTorch threads. Package versions, input digest and fit statuses are recorded. A different platform/version can alter numeric results. Local application control blocks ARIMA imports and Prophet's native optimizer fails; those failures do not become fabricated model scores. A supported environment may produce Prophet results, requiring a separate aggregate-target interpretation and report update.

## Google Colab

Upload the executed notebook to Colab and Run all in a fresh runtime, then upload crop_production.csv when prompted. The dependency cell uses compatible APIs and an official CPU PyTorch fallback. Save the newly executed notebook and evidence. Hosted execution was not certified because the browser was signed out; see COLAB_COMPATIBILITY.md for exactly what was tested.

## Results

Validation selects XGBoost_Tuned (RMSE 0.5294). Its one-step test RMSE is 0.5617. LSTM has the lowest retrospective test RMSE, 0.5457; the patch Transformer the lowest MAE; Naive the lowest MAPE. Test ranking does not replace validation selection. Expanded XGBoost tuning has twelve candidates and keeps the earlier winning setting; no extra improvement is claimed.

## Essential limitations

Physical units, exact source/license and district boundary histories remain unverified. Complete-history selection creates survivor selection, and the test period was previously examined. Only seven pre-test annual observations per history and two outer validation years are available. Prophet failed locally, no calibrated intervals exist, official faculty PDFs were unavailable and no individual team contributions are fabricated. This package is not a guarantee of marks, future accuracy or full institutional compliance.

All computation used existing local hardware and free software; paid-service expenditure ₹0. The experimental run used no cloud purchase or deployment. This repository was subsequently published privately at the owner's request.


## Repository notes

The expanded `evidence/` directory replaces the original evidence ZIP. Data provenance and redistribution terms remain unverified; this repository is private and no new dataset license is asserted. Research citations and course coordinator details are retained. Previous audit records describe the numerical experiments; the credit-only edits did not rerun or alter models.

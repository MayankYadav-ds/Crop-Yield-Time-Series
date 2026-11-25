# 11-minute live demonstration script

The timed demonstration uses the final presentation plus the executed notebook. Speaker allocation is a rehearsal choice, not evidence of individual contribution. Use the seven-column crop_production.csv. Keep the original working backup separate.

## Before class

- Open Final_Presentation.pptx and Complete_Time_Series_Project_EXECUTED.ipynb.
- Run all notebook cells beforehand in a fresh runtime, upload crop_production.csv, and save the outputs. The delivered notebook was executed locally in a real Jupyter kernel; hosted Colab was not authenticated during the audit.
- Keep the evidence ZIP available offline. Inspect model_training_status.json: Prophet failed locally; never claim it trained.
- For the live code step, keep the notebook session state ready. Do not begin dependency downloads during the timed presentation.
- If only the saved notebook is available, say clearly that you are showing a verified saved run; do not pretend all training is occurring live.

## Timed presentation and spoken cues

| Time | Action | Short Hinglish cue |
|---|---|---|
| 0:00–0:35 | Title and objective | “Ham annual district-crop yield ratio forecast kar rahe hain. Objective complex model ko best declare karna nahi, fair chronological comparison karna hai.” |
| 0:35–1:25 | Dataset and quality | “Final file mein 246,091 rows aur seven fields hain. 3,730 production values missing hain. Units encoded nahi hain, isliye ratio ka physical-unit claim conditional hai.” |
| 1:25–2:15 | Cleaning and aggregation | “Hum total accepted production ko total area se divide karte hain. Whole Year overlap aur missing components ko audit kiya; outliers blindly remove nahi kiye.” |
| 2:15–3:00 | Time series EDA | “Matched cohort mein Rice aur Maize ratios increase hue, lekin Wheat ka 2014 decline visible hai. In observations se weather ya disease cause establish nahi hota.” |
| 3:00–4:00 | Experimental design | “1,112 complete histories, window 2005–2014. Outer validation 2010–2011; test 2012–2014. Neural inner stopping 2009 hai. Complete-history selection ka survivor-bias limitation hai.” |
| 4:00–5:05 | Tuning and architectures | “XGBoost ke 12 chronological trials run hue. LSTM, GRU aur tiny patch Transformer genuinely train hue. Same three past yields aur training-only normalization use hui.” |
| 5:05–6:10 | Results and honest selection | “Validation XGBoost select karti hai. Test mein LSTM ka RMSE 0.5457 lowest hai, patch model ka MAE lowest hai aur Naive ka MAPE lowest hai. Test winner dekhkar selected model switch nahi karte.” |
| 6:10–6:45 | Horizon distinction | “One-step mein new observed years next origin par available hote hain. Fixed-origin mein 2011 ke baad observations use nahi hote; selected XGBoost ka fixed-origin RMSE Naive se worse hai.” |
| 6:45–9:20 | Live notebook check | Use the detailed live steps below. |
| 9:20–10:15 | Forecasts and exclusions | “Selected XGBoost 2014 tak refit hua aur 2015–2017 point forecasts diye. Prophet optimizer fail hua; ARIMA import blocked tha. Humne inke fake scores report nahi kiye.” |
| 10:15–11:00 | Limits, conclusion and brief questions | “No 100% accuracy claim. Short annual histories, retrospective test reuse, source/units aur geographic-boundary limits remain. Next work new years, weather/soil inputs aur calibrated uncertainty hoga.” |

## Live notebook steps — 6:45 to 9:20

1. Show the dataset audit output and selected-series count. Say: “Yeh actual CSV statistics hain, report se match karte hain.”
2. Open the chronological validation cell and frozen selection JSON. Point out the origins before discussing test results.
3. Show one neural loss curve. Explain that the stopping curve is 2009, not test loss.
4. Re-run **Quick live demonstration check**. It refits the selected XGBoost setting at origin 2011 and verifies one 2012 prediction against the saved one-step output. Wait for the actual PASS message; do not announce it before execution finishes.
5. Show the one-step and fixed-origin tables separately. Point to XGBoost’s 0.5617 one-step RMSE and 0.6283 fixed-origin RMSE.
6. Show model_training_status.json or its displayed table. Explain Prophet’s failed optimizer and the exact patch-model name.
7. Show future_forecasts.csv and state its modeled cutoff. Download the actual run’s ZIP if time permits.

## If asked to run the entire notebook

Use Runtime → Run all and upload the CSV. Local audited runtime is recorded in the evidence; free Colab runtime and installation time are not guaranteed. Tell the teacher the full run is in progress, show actual progress, and wait for completion. A saved output or a caught model failure must not be described as a newly completed live training run.

## Useful short answers

- “Why not select LSTM now?” — “Outer validation ne XGBoost select kiya. Test dekhkar switch karna test selection hoga.”
- “Did tuning improve XGBoost?” — “12 trials ke baad bhi pehle wala setting best validation par raha; extra improvement claim nahi hai.”
- “Is this exact PatchTST?” — “Nahi, lightweight patch-based Transformer inspired design hai.”
- “Where is Prophet’s metric?” — “Native fit fail hua, isliye metric absent hai; failure log available hai.”
- “Can farmers rely on this directly?” — “Domain validation, verified units aur calibrated uncertainty ke bina operational guarantee nahi.”

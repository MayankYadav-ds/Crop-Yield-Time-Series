# Teacher evaluation preparation

50 short Hinglish answers based on actual implementation and measured results. Yeh rehearsal material hai; individual team contributions apne real records se explain karein.

## 1. Project ka main objective kya hai?

Annual district-crop production/area ratio forecast karna aur classical, boosting aur neural models ko chronological evaluation par compare karna.

## 2. Final dataset kaunsa hai?

crop_production.csv: 246,091 rows aur 7 columns. Purane 455,359-row dataset ke scores is experiment mein mix nahi kiye.

## 3. Raw aur modeled years alag kyun hain?

Raw labels 1997–2015 hain. 2015 mein sirf 562 records hain, isliye matched experiment 2005–2014 tak rakha.

## 4. Series aur observations mein kya difference hai?

Ek state-district-crop history ek series hai. 1,112 complete series ke 10 years milkar 11,120 modeled observations bante hain.

## 5. Kaunse crops select hue?

Rice 319, Maize 278, Wheat 267 aur Rapeseed &Mustard 248 histories. Yeh four-crop cohort hai, all 124 crops ka result nahi.

## 6. Yield ki physical unit confirmed hai?

CSV mein unit columns nahi hain. Isliye result ko production/area ratio bolte hain; verified tonnes/hectare claim nahi karte.

## 7. Missing production kaise handle ki?

3,730 raw production values missing hain. Imputation nahi ki; invalid seasonal component wale annual group ko incomplete maan kar exclude kiya.

## 8. Duplicates aur invalid values kitne hain?

Exact aur seasonal business-key duplicate rows zero hain. Nonpositive area aur negative production bhi zero hain; missing components ka separate audit hai.

## 9. Whole Year overlap kyun risky hai?

Whole Year aggregate ho sakta hai. Usko named seasons ke saath add karne se double counting ho sakti hai; ambiguous groups conservatively exclude kiye.

## 10. Weighted annual yield formula kya hai?

Total production divided by total accepted cultivated area. Individual yields ka simple mean nahi lete.

## 11. Simple average yield kyun galat ho sakta hai?

Chhote aur bade area ko equal weight milta hai. Production/area aggregation area weights preserve karta hai.

## 12. Outliers remove kiye?

Nahi. IQR flags descriptive audit ke liye hain; unusual observation ko bina domain evidence delete nahi kiya.

## 13. Stable geography ki limitation kya hai?

State aur district labels series identify karte hain, lekin boundary changes verify nahi hote. CSV mein geographic codes/crosswalk nahi hai.

## 14. Wheat ka 2014 decline weather se explain kar sakte hain?

Nahi. Matched aggregate ratio 2013 se 2014 mein 29.60% gira, par weather, disease ya fertilizer variables available nahi hain.

## 15. Data annual hai to seasonal model kyun nahi?

Annual observations se sub-annual seasonality observe nahi hoti. Koi defensible multi-year period establish nahi hua, isliye seasonal variants fit nahi kiye.

## 16. ADF ka null hypothesis kya hai?

ADF ka null unit root hai. Reject karna chosen specification ke under unit-root evidence ke against hota hai; perfect stationarity proof nahi.

## 17. KPSS ka null kya hai?

Constant specification mein level stationarity null hai. ADF aur KPSS ke nulls alag hain, isliye complementary diagnostics hain.

## 18. Actual stationarity finding kya thi?

769 complete 1997–2011 histories par ADF 5% level rejection 41.6% tha; first differences par 95.8%. Short samples aur multiple testing limitations hain.

## 19. ACF aur PACF mein difference kya hai?

ACF lagged correlation dikhata hai. PACF intermediate lags control karke lag-specific association dikhata hai.

## 20. Differencing se forecast guaranteed improve hota hai?

Nahi. Differencing stationarity mein help kar sakti hai, lekin useful level information bhi lose ho sakti hai. Out-of-sample comparison zaroori hai.

## 21. Naive baseline kya predict karta hai?

Latest available yield ratio repeat karta hai. Fixed-origin three-step mein woh same latest observation teen horizons par use karta hai.

## 22. Moving average aur ARIMA MA same hain?

Nahi. MA3 last three observations average karta hai; ARIMA ka MA component past forecast innovations/errors se related hota hai.

## 23. Drift forecast kya karta hai?

First aur latest training observation ke average annual change ko future mein extrapolate karta hai. Short noisy histories mein yeh unstable ho sakta hai.

## 24. SES, Holt aur damped Holt mein kya fark hai?

SES level smooth karta hai; Holt trend add karta hai; damped Holt future trend contribution ko gradually reduce karta hai.

## 25. AR1_OLS ko ARIMA bol sakte hain?

Is project mein nahi. AR1_OLS explicitly unconstrained least-squares intercept-plus-lag fit hai. Statsmodels ARIMA import fail hua; uska score report nahi kiya.

## 26. AR1_OLS ka error itna high kyun hai?

Short ya nearly constant histories par unconstrained coefficients explosive ho sakte hain. Extreme forecasts retain hue; scores ko artificially clip nahi kiya.

## 27. XGBoost features kya hain?

Yield lag1, lag2, lag3, unka trailing mean aur population standard deviation, aur year minus 2005. Current production/area target features nahi hain.

## 28. Hyperparameter tuning kitni expand hui?

Pehle 2 settings the; ab 12 seeded candidates hain. Har candidate 2010 aur 2011 targets ke expanding-origin validation RMSE par compare hua.

## 29. Selected XGBoost setting kya hai?

250 estimators, depth 3, learning rate 0.03, subsample 0.8, column sampling 0.9 aur L2 penalty 10. Expanded search ne better validation candidate nahi diya.

## 30. Final selected model kaunsa hai?

XGBoost_Tuned: outer validation RMSE 0.5294 sabse kam tha. Selection test evaluation se pehle freeze hui.

## 31. LSTM test mein better hai to usko selected kyun nahi bola?

Test set dekhkar model switch karna selection leakage hoga. LSTM ka retrospective test RMSE lowest hai, lekin validation-selected model XGBoost hi rehta hai.

## 32. LSTM architecture kya hai?

One recurrent layer, 16 hidden units, final-state linear head aur dropout 0.1. Actual parameter count 1,233 hai; input three past yields hain.

## 33. GRU LSTM se kaise different hai?

GRU mein reset aur update gates hote hain; separate LSTM-style cell state nahi hota. Is implementation mein GRU ke 929 parameters hain.

## 34. Transformer kaunsa implement kiya?

Tiny patch-based Transformer: three-year history se two overlapping length-two patches, 8-dimensional embedding, positional information, 2 heads aur 1 encoder layer.

## 35. Kya yeh exact PatchTST hai?

Nahi. Patching se inspired lightweight model hai, exact published PatchTST reproduction claim nahi hai. Actual parameter count 657 hai.

## 36. Lookback three years kyun?

Final pre-test history sirf seven years ki hai. Three lags se comparable global training windows milte hain; long-context superiority claim nahi karte.

## 37. Normalization leakage kaise avoid ki?

Har origin par mean aur standard deviation sirf us origin ke training observations se fit hue. Test years normalization fit mein use nahi hue.

## 38. Early stopping kaise hua?

Inner training 2008 tak aur stopping year 2009 tha; max 60 epochs, patience 8. Selected counts LSTM 49, GRU 52 aur Transformer 31 the, phir outer/test fits mein freeze rahe.

## 39. Training shuffle time leakage hai?

Chronological split ke baad sirf training rows shuffle hui. Validation/test rows training batches mein mix nahi hui; har input apne target se pehle ke years use karta hai.

## 40. One-step aur fixed-origin tests mein kya fark hai?

One-step mein 2011, 2012, 2013 origins par refit hota hai. Fixed-origin mein 2011 ki information se 2012–2014 ke teen horizons forecast hote hain.

## 41. MAE aur RMSE kya measure karte hain?

MAE absolute errors ka mean hai. RMSE squared errors ke mean ka square root hai, isliye large errors ko zyada penalty milti hai.

## 42. MAPE zero targets ko kaise handle karta hai?

Absolute target 1e-8 ya usse kam ho to MAPE se exclude karte hain. 3,336 test targets mein ek zero exclude hai; MAE/RMSE usko retain karte hain.

## 43. Selected model ke actual test scores kya hain?

XGBoost one-step MAE 0.3369, RMSE 0.5617, MAPE 24.77% hain. Yeh production/area numerical scale par measured errors hain.

## 44. Retrospective test winners kaun hain?

LSTM lowest RMSE 0.5457; patch Transformer lowest MAE 0.3253; Naive lowest MAPE 20.66%. Ek model har metric par best nahi hai.

## 45. Prophet ka actual status kya hai?

Longer annual crop aggregates ke liye code implement hua, seasonality disabled thi, lekin native optimizer fail hua. Successful fit ya accuracy invent nahi ki.

## 46. GARCH aur VAR exclude kyun hain?

Representative longer history mein sirf 14 pre-test residuals hain; robust volatility inference aur validated stationary multivariate design establish nahi hua. Isse methods impossible hone ka claim nahi hota.

## 47. Reproducibility evidence kya hai?

Seed 42, CPU execution, exact environment versions, saved parameters/checkpoints, actual predictions aur independent metric recalculation available hain. Checkpoint replay bhi test hua.

## 48. Future forecasts kaunse years ke hain?

Modeled cutoff 2014 ke baad 2015–2017. Raw 2015 partial hai; forecasts ko verified observations ya aaj ke relative future values nahi bolte.

## 49. Farmers/policymakers in outputs ko kaise use karein?

Investigated planning baseline ki tarah, domain validation ke baad. Calibrated uncertainty aur verified units ke bina operational guarantee ya insurance decision recommend nahi karte.

## 50. Kya project 100% evaluation-ready ya accurate hai?

Guarantee nahi. Survivor selection, previously examined test years, short histories, unverified source/units, unavailable faculty PDFs aur live Colab sign-in gap clearly documented hain.


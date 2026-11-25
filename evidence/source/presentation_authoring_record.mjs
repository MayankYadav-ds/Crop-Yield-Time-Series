import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const base=process.cwd(), build=path.join(base,'work/deck_build');
const data=JSON.parse(await fs.readFile(path.join(build,'data.json'),'utf8'));
const p=Presentation.create({slideSize:{width:1280,height:720}}), navy='#142D46',ink='#26333F',blue='#245E82',teal='#20816D',orange='#B66B2C',gray='#656F79',light='#EBEFF2';
const metrics=data.test_leaderboard, lookup=Object.fromEntries(metrics.map(r=>[r.model,r]));
const selected=data.best_model_summary.best_by_validation_RMSE;
function box(s,text,x,y,w,h,size=22,color=ink,bold=false){let a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=text;a.text.style={fontFamily:'Arial',fontSize:`${size}pt`,color,bold};return a;}
function slide(title,notes=''){let s=p.slides.add();s.background.fill='#FFFFFF';if(title)box(s,title,70,42,1140,82,32,navy,true);box(s,String(p.slides.items.length),1190,674,40,25,12,gray);s.speakerNotes.text=notes;return s;}
function table(s,values,{x=80,y=150,w=1120,h=420,widths,size=18}={}){let t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});t.borders.assign({style:'solid',fill:'#D2D9DF',width:1});for(let i=0;i<values.length;i++){for(let j=0;j<values[0].length;j++){let c=t.getCell(i,j);c.fill=i===0?navy:(i%2===0?'#F3F5F7':'#FFFFFF');c.text.style={fontFamily:'Arial',fontSize:`${size}pt`,color:i===0?'#FFFFFF':ink,bold:i===0};}}return t;}
function chart(s,type,categories,series,{x=80,y=160,w=1120,h=420,title='',horizontal=false,legend=true,yTitle='',xTitle='',max,min=0,labels=false,numberFormat='0.0000'}={}){let c=s.charts.add(type,{position:{left:x,top:y,width:w,height:h},title:title||undefined,titleTextStyle:{typeface:'Arial',fontSize:23,fill:navy},categories,series:series.map((r,i)=>({...r,values:r.values.map(v=>Number(Number(v).toPrecision(12))),fill:r.fill||[blue,orange,teal,'#755A8A',gray][i%5],line:{style:'solid',fill:r.fill||[blue,orange,teal,'#755A8A',gray][i%5],width:2}})),lineOptions:type==='line'?{smooth:false,grouping:'standard'}:undefined,hasLegend:legend,legend:{position:'bottom',overlay:false,textStyle:{typeface:'Arial',fontSize:16,fill:ink}},barOptions:type==='bar'?{direction:horizontal?'bar':'column',grouping:'clustered',gapWidth:60}:undefined,xAxis:{title:xTitle?{text:xTitle,textStyle:{typeface:'Arial',fontSize:17}}:undefined,textStyle:{typeface:'Arial',fontSize:17,fill:ink}},yAxis:{title:yTitle?{text:yTitle,textStyle:{typeface:'Arial',fontSize:17}}:undefined,textStyle:{typeface:'Arial',fontSize:17,fill:ink},numberFormatCode:numberFormat,min,max,majorGridlines:{style:'solid',fill:'#DCE1E6',width:1}},dataLabels:labels?{showValue:true,position:'outEnd',numberFormatCode:'0.0000',textStyle:{typeface:'Arial',fontSize:16,fill:ink}}:undefined});return c;}
// 1
let s=slide('', 'Project by Mayank Yadav. Course details retained from the project brief.');
box(s,'Agricultural crop yield forecasting',70,130,1130,145,42,navy,true);box(s,'Classical and advanced annual time series models',70,300,1130,60,26,ink);box(s,'Bennett University   CSET 369',70,402,1130,40,22,blue,true);box(s,'Course Coordinator: Anurag Choubey',70,450,1130,40,20);
box(s,'Mayank Yadav — S24CSEU1494',70,510,1130,125,18);
// 2
s=slide('Forecasting objective and scope','Sources: methodology and experimental_design.json. Yield is a production/area ratio because physical units are absent in the CSV. No weather or disease causality is inferred.');
box(s,'Annual district–crop production / area',80,160,1100,65,30,blue,true);
box(s,'The comparison asks whether historical patterns improve on persistence under matched chronological tests.',80,270,1090,110,25);
box(s,'The study covers four crops and 1,112 complete histories. Forecasts support investigation after domain validation; they are not guaranteed operational recommendations.',80,435,1080,135,23);
// 3
s=slide('Dataset and reporting coverage','Sources: dataset_summary.json, preprocessing_summary.json, input_identity.json. Raw 2015 has 562 records; 2014 has 10,973. Exact public source/license and units remain unverified. Official APY context portal: https://data.desagri.gov.in/website/apy-query-report-web');
table(s,[['Verified property','Value'],['Raw records / fields','246,091 / 7'],['Crop-year labels','1997–2015'],['State / crop labels','33 / 124'],['Missing production','3,730'],['Exact duplicates','0']],{x:75,y:155,w:510,h:405,widths:[350,160],size:19});
const years=Object.keys(data.reporting).sort();chart(s,'line',years,[{name:'Raw records',values:years.map(y=>data.reporting[y])}],{x:625,y:155,w:575,h:405,title:'Annual reporting',legend:false,yTitle:'Raw records',xTitle:'Crop year',min:0,numberFormat:'0'});box(s,'2015 reporting is partial; the modeled cutoff is 2014.',640,590,540,70,18,gray);
// 4
s=slide('Annual aggregation and data quality','Source: aggregation_audit.csv and preprocessing_summary.json. Counts: included 241,302; invalid or missing component 3,997; Whole Year overlap 792; total 246,091. No outliers automatically removed.');
box(s,'Annual ratio = Σ accepted production / Σ accepted area',80,152,1120,70,28,blue,true);
table(s,[['Annual-group decision','Raw rows','Annual groups'],['Included','241,302','210,131'],['Invalid / missing component','3,997','3,713'],['Whole Year overlap','792','362']],{x:80,y:265,w:1120,h:265,widths:[640,230,250],size:22});
box(s,'No production imputation or automatic outlier deletion. Unit labels and stable geographic codes are absent.',80,585,1100,70,20,gray);
// 5
s=slide('Observed crop trends in the matched cohort','Source: weighted_crop_series.csv. Curves use summed production divided by summed area within each crop. Rice +26.23%, Maize +37.11%, Wheat −11.21%, Rapeseed &Mustard −5.08%, 2005 to 2014. These are not causal effects.');
const cy=Array.from({length:10},(_,i)=>String(2005+i));const crops=['Rice','Maize','Wheat','Rapeseed &Mustard'];chart(s,'line',cy,crops.map(c=>({name:c,values:cy.map(y=>data.weighted_crop_series.find(r=>r.Crop===c&&r.Crop_Year===+y).yield_ratio)})),{x:80,y:145,w:1120,h:435,title:'Matched crop ratios',yTitle:'Production / area',xTitle:'Crop year'});
box(s,'Wheat falls 29.60% from 2013 to 2014. Area weights and reporting quality remain relevant; causes are unobserved.',80,615,1120,60,18,gray);
// 6
s=slide('Chronological validation and forecast origins','Source: experimental_design.json and neural checkpoint metadata. Complete-history sample construction uses future availability and creates survivor selection. Test years were examined in the archived earlier study.');
table(s,[['Stage','Observed through','Targets'],['Neural inner stopping','2008 training','2009 stopping'],['Outer validation','2009 / 2010 origins','2010 / 2011'],['Final pre-test fit','2011','2012–2014'],['Expanding one-step test','2011 / 2012 / 2013','2012 / 2013 / 2014'],['Fixed-origin test','2011 only','2012–2014'],['Selected-model refit','2014','2015–2017']],{y:150,h:405,widths:[430,345,345],size:20});
box(s,'Retrospective matched cohort: test-era availability and previously examined test years limit prospective claims.',80,600,1120,65,19,gray);
// 7
s=slide('Classical models and diagnostic evidence','Sources: fitted_parameters.csv, stationarity_results.csv and applicability_diagnostics.json. ADF/KPSS diagnostic histories 1997–2011; 769 identities. ARIMA import failed under local Windows application-control policy; AR1_OLS is a distinct estimator.');
table(s,[['Model','Forecast principle'],['Naive / Mean','Latest observation / historical mean'],['MA3 / Drift','Latest three-year mean / average change'],['SES_LS','Smoothed level'],['Holt_LS / DampedHolt_LS','Level and trend / damped trend'],['AR1_OLS','Unconstrained intercept plus lag one']],{y:150,h:365,widths:[430,690],size:21});
box(s,'ADF rejects a unit root in 41.6% of levels and 95.8% of differences. Short annual histories limit interpretation.',80,555,1120,70,20,gray);
box(s,'LS denotes custom conditional least squares. ARIMA was not successfully fitted.',80,640,1120,35,17,gray);
// 8
s=slide('Twelve chronological XGBoost trials','Sources: hyperparameter_results.csv and xgboost_selection.json. Two earlier candidates plus ten seeded candidates. Selection on expanding 2010–2011 validation RMSE; no test tuning. Trial 1 remains best, with 250 estimators, depth 3, learning rate .03, row subsample .8, column sampling .9 and L2 penalty 10.');
chart(s,'bar',data.hyperparameter_results.map(r=>String(r.trial)),[{name:'Validation RMSE',values:data.hyperparameter_results.map(r=>r.RMSE)}],{x:80,y:155,w:750,h:430,title:'Validation RMSE by trial',legend:false,labels:true,xTitle:'Candidate index',yTitle:'Validation RMSE'});
box(s,'Winning setting unchanged',870,175,330,80,25,blue,true);box(s,'250 trees\nDepth 3\nLearning rate 0.03\nL2 penalty 10',870,285,330,195,23);box(s,'No extra improvement is claimed merely because more trials ran.',870,535,330,100,19,gray);
// 9
s=slide('Neural models and inner stopping','Sources: neural_models.py, model_training_status.json and *_loss.csv. LSTM 1,233 parameters/49 epochs; GRU 929/52; patch Transformer 657/31. Three-year input, normalization on each training origin only. Training through 2008; stopping year 2009; patience 8, max 60 epochs. Not an exact PatchTST reproduction. References: https://www.bioinf.jku.at/publications/older/2604.pdf; https://aclanthology.org/D14-1179/; https://arxiv.org/abs/2211.14730');
table(s,[['Model','Parameters','Selected epochs'],['LSTM','1,233','49'],['GRU','929','52'],['PatchTransformerTiny','657','31']],{x:80,y:145,w:1120,h:205,widths:[630,230,260],size:20});
const epochs=Array.from({length:60},(_,i)=>i+1);const kinds=['LSTM','GRU','PatchTransformerTiny'];
// Curves are shown at common available checkpoints, avoiding fabricated continuation after stopping.
const maxCommon=Math.min(...kinds.map(k=>data[k+'_loss'].length));const showEpoch=Array.from({length:maxCommon},(_,i)=>i+1).filter(e=>e===1||e%3===0||e===maxCommon);
chart(s,'line',showEpoch.map(String),kinds.map(k=>({name:k,values:showEpoch.map(e=>data[k+'_loss'].find(r=>r.epoch===e).validation_normalized_MSE)})),{x:80,y:385,w:1120,h:240,title:'Inner stopping year 2009',yTitle:'2009 normalized MSE',xTitle:'Epoch'});box(s,'Common epochs 1–39 shown; full loss curves and checkpoints are in the evidence package.',80,642,1120,35,17,gray);
// 10
s=slide('One-step test results on common targets','Source: test_leaderboard.csv. Each model has 3,336 district-year targets. MAPE excludes one zero target; MAE/RMSE retain it. Physical units unverified. Negative and extreme forecasts are retained, not clipped.');
table(s,[['Model','Type','MAE','RMSE','MAPE %'],...metrics.map(r=>[r.model,['LSTM','GRU','PatchTransformerTiny'].includes(r.model)?'DL':r.model==='XGBoost_Tuned'?'ML':['Naive','Mean','MA3','Drift'].includes(r.model)?'Base':'Stat',r.MAE.toFixed(4),r.RMSE.toFixed(4),r.MAPE.toFixed(2)])],{x:80,y:140,w:1120,h:475,widths:[410,120,195,195,200],size:17});
box(s,'Production / area scale; validation selection remains frozen. Full metric definitions and additional metrics are exported.',80,640,1120,40,17,gray);
// 11
s=slide('Validation selection and retrospective test ranking','Sources: validation_leaderboard.csv and test_leaderboard.csv. Validation selects XGBoost_Tuned (RMSE .529400). LSTM has lowest retrospective test RMSE (.545658); patch model has lowest MAE; Naive has lowest MAPE. No test-based reselection.');
const compare=['XGBoost_Tuned','LSTM','GRU','PatchTransformerTiny','Naive'];chart(s,'bar',compare,[{name:'Validation',values:compare.map(n=>data.validation_leaderboard.find(r=>r.model===n).RMSE)}],{x:65,y:160,w:570,h:390,title:'Outer validation RMSE',horizontal:true,legend:false,max:.75,labels:true});chart(s,'bar',compare,[{name:'Test',values:compare.map(n=>lookup[n].RMSE)}],{x:650,y:160,w:570,h:390,title:'One-step test RMSE',horizontal:true,legend:false,max:.75,labels:true});
box(s,'Select on validation. Describe the test winner retrospectively; do not switch models after inspecting test results.',80,605,1120,70,22,blue,true);
// 12
s=slide('Observed values, predictions and forecast horizons','Sources: test_predictions.csv and fixed_origin_error_by_horizon.csv. Prediction curves average the matched district/crop cohort, not national yield. Fixed-origin forecasts use no observations beyond 2011. Selected XGBoost fixed-origin RMSE .6283 exceeds Naive .6191.');
const yy=['2012','2013','2014'];const act=data.actual_predicted[selected];const lm=data.actual_predicted['LSTM'];chart(s,'line',yy,[{name:'Actual',values:act.map(r=>r.actual)},{name:'Selected XGBoost',values:act.map(r=>r.prediction)},{name:'LSTM',values:lm.map(r=>r.prediction)}],{x:65,y:160,w:565,h:395,title:'Cohort mean: one-step',yTitle:'Production / area',min:1.7});
chart(s,'line',['1','2','3'],['Naive','XGBoost_Tuned','LSTM','GRU','PatchTransformerTiny'].map(k=>({name:k,values:[1,2,3].map(h=>data.fixed_origin_error_by_horizon.find(r=>r.model===k&&r.horizon===h).mean)})),{x:650,y:160,w:565,h:395,title:'Fixed-origin MAE',yTitle:'Production / area',xTitle:'Horizon (years)'});
box(s,'One-step and fixed-origin scores answer different forecasting questions.',80,615,1120,55,22,gray);
// 13
s=slide('Selected-model forecasts beyond the modeled cutoff','Source: future_forecasts.csv. Model is validation-selected XGBoost_Tuned, refit through 2014. Forecast years 2015–2017 are beyond the modeled cutoff; raw 2015 exists only partially. No calibrated intervals or verified future observations.');
const fy=Object.keys(data.future_means).sort();chart(s,'line',fy,[{name:'Mean forecast',values:fy.map(y=>data.future_means[y])}],{x:80,y:155,w:735,h:430,title:'2015–2017 point forecasts',legend:false,yTitle:'Mean production / area',xTitle:'Forecast crop year'});
box(s,'Refit through 2014',860,180,340,60,27,blue,true);box(s,'2015–2017 point forecasts\n\nUnweighted cohort mean\n\nNo calibrated intervals',860,285,340,270,22);box(s,'These forecasts are not verified observations or national crop yield.',80,630,1120,45,19,gray);
// 14
s=slide('Excluded models and external verification gaps','Sources: model_training_status.json, arima_capability.json, applicability_diagnostics.json and evaluation-readiness audit. Prophet imported but native optimizer failed; no metric. ARIMA extension blocked by Windows application-control policy. No security policy changes were attempted. Official milestone PDFs unavailable; browser signed out of Google.');
table(s,[['Item','Observed status','Interpretation'],['Prophet','Native optimization failed','No successful fit or accuracy claim'],['ARIMA','Library import blocked locally','AR1_OLS is a separate estimator'],['Seasonal models','No justified periodic structure','No seasonal fit claimed'],['GARCH / VAR','Short histories; unvalidated design','Exploratory diagnostics only'],['Faculty PDFs / hosted Colab','Unavailable / signed out','Exact compliance / cloud run unverified']],{x:70,y:145,w:1140,h:420,widths:[240,400,500],size:18});
box(s,'Preserve the failure evidence. A completed notebook does not mean every requested model trained.',80,615,1120,60,20,gray);
// 15
s=slide('Conclusions and agricultural interpretation','Sources: best_model_summary.json, final research report and limitations. No causal or guaranteed operational conclusions. The study is retrospective with survivor selection, few validation years and unknown physical units.');
box(s,'Model ranking depends on metric and horizon',80,155,1100,90,31,blue,true);
box(s,'Validation selects XGBoost. LSTM has the lowest observed one-step RMSE; Naive has the lowest MAPE.',80,290,1100,100,25);
box(s,'Planning use requires verified units, geographic consistency, domain validation and calibrated uncertainty.',80,435,1100,100,24);
box(s,'Next evidence: genuinely new years, weather/soil data, stable district crosswalks and prospective evaluation.',80,585,1100,75,21,gray);
// 16
s=slide('References','Full verified bibliography and links are in the research report and REPORT_EVIDENCE/references.md. References: '+data.references.map((r,i)=>`[${i+1}] ${r[0]} ${r[1]}`).join('\n'));
const refShort=['[1] Directorate of Economics and Statistics. APY query portal. Accessed 2026. Exact CSV lineage unverified.','[2] Hyndman & Athanasopoulos (2021). Forecasting: Principles and Practice, 3rd ed. OTexts.','[3] Box et al. (2015). Time Series Analysis: Forecasting and Control, 5th ed. Wiley.','[4] Hyndman et al. (2002). Exponential smoothing state-space framework. IJF 18, 439–454.','[5] van Klompenburg et al. (2020). Crop-yield ML review. Computers and Electronics in Agriculture 177, 105709.','[6] Chen & Guestrin (2016). XGBoost. KDD, 785–794. DOI 10.1145/2939672.2939785.','[7] Hochreiter & Schmidhuber (1997). Long short-term memory. Neural Computation 9, 1735–1780.','[8] Cho et al. (2014). RNN encoder–decoder. EMNLP, 1724–1734. DOI 10.3115/v1/D14-1179.','[9] Taylor & Letham (2018). Forecasting at scale. The American Statistician 72, 37–45.','[10] Nie et al. (2023). A time series is worth 64 words. ICLR. arXiv:2211.14730.'];
refShort.forEach((text,i)=>box(s,text,i<5?75:665,150+(i%5)*98,540,90,16));
await fs.mkdir(path.join(build,'renders'),{recursive:true});
const candidate=path.join(build,'expanded_candidate.pptx');await (await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<p.slides.items.length;i++){let img=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(build,'renders',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await img.arrayBuffer()));}
const SKILL='C:/Users/mayan/.codex/plugins/cache/openai-primary-runtime/presentations/26.1117.11355/skills/presentations';const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
await fs.mkdir(path.join(build,'finalized'),{recursive:true});let finalPath=path.join(build,'finalized/Final_Presentation_v3.pptx');
await finalizePresentation({workspaceDir:base,candidatePath:candidate,finalPath,pythonExecutable:'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1117-835-0_26.1117.835.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe',integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[3,4,6,7,9,10,14].flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:16,requiredNativeTableOwnerSlides:[3,4,6,7,9,10,14],requiredNativeChartOwnerSlides:[3,5,8,9,11,12,13],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,receiptPath:path.join(build,'final_presentation_v3.validation.json')});
console.log('FINAL PRESENTATION: 16 editable slides with verified data.');






"""Additional descriptive EDA, generated only from exported observations."""
from pathlib import Path
import json,os
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

def additional_eda(root):
 root=Path(root);figdir=root/'figures';panel=pd.read_csv(root/'annual_panel.csv');raw=pd.read_csv(os.environ.get('APY_CSV','raw/crop_production.csv'));selected=pd.read_csv(root/'selected_series.csv');ids=set(selected.sid);subset=panel[panel.sid.isin(ids)&panel.Crop_Year.between(2005,2014)]
 def save(name):
  plt.tight_layout();p=figdir/(name+'.png');plt.savefig(p,dpi=300,bbox_inches='tight');plt.close()
  try:
   from IPython.display import display,Image
   display(Image(filename=str(p)))
  except ImportError:print('Saved',p.name)
 # National and state totals remain separated by crop; mixed commodity unit sums are avoided.
 for label,keys in [('state_crop',['State_Name','Crop','Crop_Year']),('national_crop',['Crop','Crop_Year'])]:
  a=panel.groupby(keys)[['area','production']].sum().reset_index();a['yield_ratio']=a.production/a.area;a.to_csv(root/(label+'_annual_series.csv'),index=False)
 miss=raw.isna().groupby(raw.Crop_Year).mean();fig,ax=plt.subplots(figsize=(6,3));im=ax.imshow(miss.T.values,aspect='auto',vmin=0,vmax=1,cmap='Reds');ax.set_xticks(range(len(miss)),miss.index,rotation=90);ax.set_yticks(range(len(miss.columns)),miss.columns);ax.set(title='Raw missing-value proportions by crop year',xlabel='Crop year',ylabel='Field');fig.colorbar(im,ax=ax,label='Missing proportion');save('missing_heatmap')
 fig,ax=plt.subplots(figsize=(3.5,2.4));subset.groupby('Crop').apply(lambda g:g.production.sum()/g.area.sum(),include_groups=False).sort_values().plot.barh(ax=ax);ax.set(xlabel='Production / area',title='Four-crop weighted ratios, 2005–2014');save('crop_comparison')
 rice=subset[subset.Crop.eq('Rice')];state=rice.groupby('State_Name')[['area','production']].sum();state['ratio']=state.production/state.area;state.to_csv(root/'rice_state_comparison.csv');fig,ax=plt.subplots(figsize=(4,4));state.ratio.sort_values().plot.barh(ax=ax);ax.set(xlabel='Production / area',title='Rice weighted ratios by state, matched cohort');save('state_comparison')
 fig,ax=plt.subplots(figsize=(4.2,2.5));groups=[g.y.values for _,g in subset.groupby('Crop')];labels=[c for c,_ in subset.groupby('Crop')];ax.boxplot(groups,tick_labels=labels,showfliers=True);ax.tick_params(axis='x',labelsize=6);ax.set(ylabel='Production / area',title='Yield-ratio distributions; outliers retained');save('yield_distributions')
 corr=subset[['area','production','y']].corr();corr.to_csv(root/'descriptive_correlations.csv');fig,ax=plt.subplots(figsize=(3.5,2.7));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(3),['Area','Production','Ratio']);ax.set_yticks(range(3),['Area','Production','Ratio']);ax.set_title('Pooled numerical correlations; not causal')
 for i in range(3):
  for j in range(3):ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center')
 fig.colorbar(im,ax=ax);save('correlation_heatmap')
 crop=pd.read_csv(root/'weighted_crop_series.csv');y=crop[crop.Crop.eq('Rice')].set_index('Crop_Year').yield_ratio;trend=y.rolling(3,center=True,min_periods=2).mean();changes=y.pct_change(fill_method=None)*100
 pd.DataFrame({'observed':y,'centered_3yr_smooth':trend,'residual':y-trend,'year_over_year_pct':changes}).to_csv(root/'rice_descriptive_components.csv')
 fig,axes=plt.subplots(2,1,figsize=(4,4));axes[0].plot(y.index,y,label='Observed');axes[0].plot(trend.index,trend,label='Centered smooth');axes[0].plot(y.index,y-trend,label='Residual');axes[0].legend(fontsize=6);axes[0].set(ylabel='Production / area',title='Rice descriptive trend and residual (no seasonality)');changes.plot.bar(ax=axes[1]);axes[1].set(ylabel='Year-over-year change (%)',xlabel='Crop year');save('trend_residual_growth')
 out=[]
 for crop,g in subset.groupby('Crop'):
  q1,q3=g.y.quantile([.25,.75]);iqr=q3-q1;out.append({'crop':crop,'q1':q1,'q3':q3,'IQR':iqr,'flagged':int(((g.y<q1-1.5*iqr)|(g.y>q3+1.5*iqr)).sum()),'retained_rows':len(g)})
 pd.DataFrame(out).to_csv(root/'outlier_audit.csv',index=False)
 horizon=pd.read_csv(root/'fixed_origin_error_by_horizon.csv');fig,ax=plt.subplots(figsize=(4.3,2.6))
 for name in ['Naive','XGBoost_Tuned','LSTM','GRU','PatchTransformerTiny']:
  a=horizon[horizon.model.eq(name)];ax.plot(a.horizon,a['mean'],marker='o',label=name)
 ax.set(xlabel='Fixed-origin horizon (years)',ylabel='MAE (production / area)',title='Error by forecast horizon');ax.legend(fontsize=6);save('horizon_errors')
 fi=root/'feature_importance.csv'
 if fi.exists():
  a=pd.read_csv(fi,index_col=0).iloc[:,0];fig,ax=plt.subplots(figsize=(3.5,2.2));a.sort_values().plot.barh(ax=ax);ax.set(xlabel='XGBoost gain-based importance',title='Selected boosting feature importance');save('feature_importance')
 desc={c:{'dtype':str(raw[c].dtype),'missing':int(raw[c].isna().sum())} for c in raw};(root/'feature_descriptions.json').write_text(json.dumps(desc,indent=2))
 # Explicit rolling SD, raw distributions and residual diagnostics required for the EDA audit.
 fig,ax=plt.subplots(figsize=(4,2.5));ax.plot(y.index,y,label='Observed');ax.plot(y.index,y.rolling(3).mean(),label='3-year trailing mean');ax.plot(y.index,y.rolling(3).std(),label='3-year sample SD');ax.set(xlabel='Crop year',ylabel='Production / area',title='Rice descriptive rolling statistics');ax.legend(fontsize=6);save('rolling_mean_std')
 pd.DataFrame({'observed':y,'rolling_mean_3':y.rolling(3).mean(),'rolling_sample_std_3':y.rolling(3).std()}).to_csv(root/'rice_rolling_statistics.csv')
 cc=raw.Crop.str.strip().value_counts();cc.to_csv(root/'raw_crop_counts.csv');fig,ax=plt.subplots(figsize=(4.5,3.5));cc.head(12).sort_values().plot.barh(ax=ax);ax.set(xlabel='Raw observations',title='Twelve most reported crop labels');save('raw_crop_distribution')
 sc=raw.State_Name.str.strip().value_counts();sc.to_csv(root/'raw_state_counts.csv');fig,ax=plt.subplots(figsize=(5,6));sc.sort_values().plot.barh(ax=ax);ax.set(xlabel='Raw observations',title='Raw reporting volume by state label');ax.tick_params(axis='y',labelsize=7);save('raw_state_distribution')
 summary=json.loads((root/'best_model_summary.json').read_text());chosen=summary['best_by_validation_RMSE'];pred=pd.read_csv(root/'test_predictions.csv');sel=pred[pred.model.eq(chosen)].copy();sel['error']=sel.actual-sel.prediction;fig,ax=plt.subplots(figsize=(3.5,2.5));ax.hist(sel.error,bins=50);ax.set(xlabel='Actual minus prediction (production / area)',ylabel='District-year targets',title='Selected model: test error distribution');save('selected_test_residual_distribution')
 diagnostic=json.loads((root/'applicability_diagnostics.json').read_text());sid=diagnostic.get('representative_sid')
 if sid is not None:
  from statsmodels.graphics.tsaplots import plot_acf
  hist=panel[(panel.sid==sid)&panel.Crop_Year.between(1997,2011)].sort_values('Crop_Year');vals=hist.y.to_numpy();xx=np.column_stack([np.ones(len(vals)-1),vals[:-1]]);coef=np.linalg.lstsq(xx,vals[1:],rcond=None)[0];res=vals[1:]-xx@coef;pd.DataFrame({'year':hist.Crop_Year.values[1:],'residual':res}).to_csv(root/'representative_AR1_training_residuals.csv',index=False)
  fig,axes=plt.subplots(1,2,figsize=(6,2.5));axes[0].hist(res,bins=6);axes[0].set(xlabel='Production / area residual',title='Representative pre-test AR1 residuals');plot_acf(res,lags=5,ax=axes[1]);axes[1].set_title('AR1 residual ACF; n=14');save('representative_AR1_residual_diagnostics')
 print('Additional EDA generated; no forecast or model-selection values changed.')
if __name__=='__main__':additional_eda(os.environ.get('APY_OUTPUT','work/final_audit/results'))

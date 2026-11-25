from pathlib import Path
import os
import json,pandas as pd,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(os.environ.get('APY_OUTPUT','results'));tl=pd.read_csv(R/'test_leaderboard.csv');fig=R/'figures';colors=['#24557C','#A95C20','#238060','#754C91','#6B6B6B']
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
a=tl[tl.model.isin(['LSTM','GRU','PatchTransformerTiny','XGBoost_Tuned','Naive'])].sort_values('RMSE')
f,ax=plt.subplots(figsize=(3.5,2.5));bars=ax.barh(a.model,a.RMSE,color=colors);ax.invert_yaxis();ax.set(xlabel='RMSE (production / area)',title='One-step global models and Naive',xlim=(0,.7))
for bar,val in zip(bars,a.RMSE):ax.text(val+.008,bar.get_y()+bar.get_height()/2,f'{val:.4f}',va='center',fontsize=7)
f.tight_layout();f.savefig(fig/'competitive_rmse.png',dpi=300,bbox_inches='tight');plt.close(f)
f,axes=plt.subplots(3,1,figsize=(3.5,5.2))
for ax,kind in zip(axes,['LSTM','GRU','PatchTransformerTiny']):
 d=pd.read_csv(R/(kind+'_loss.csv'));ax.plot(d.epoch,d.train_normalized_MSE,label='Train through 2008');ax.plot(d.epoch,d.validation_normalized_MSE,label='Inner stop 2009');ax.set(title=kind,ylabel='Normalized MSE');ax.legend(fontsize=5)
axes[-1].set_xlabel('Epoch');f.tight_layout();f.savefig(fig/'neural_losses_panel.png',dpi=300,bbox_inches='tight');plt.close(f)
f,ax=plt.subplots(figsize=(3.5,2.6));ax.axis('off');steps=['CSV validation and annual aggregation','EDA and pre-test diagnostics','Chronological tuning and inner stopping','Frozen test and common targets','Selected-model refit and forecasts','Agricultural interpretation and limitations']
for i,s in enumerate(steps):
 y=.94-i*.17;ax.text(.5,y,s,ha='center',va='center',fontsize=7,bbox=dict(boxstyle='round,pad=.35',facecolor='white',edgecolor='#555555'),transform=ax.transAxes)
 if i<5:ax.annotate('',xy=(.5,y-.115),xytext=(.5,y-.035),xycoords='axes fraction',arrowprops={'arrowstyle':'->','lw':.7})
f.tight_layout();f.savefig(fig/'final_workflow.png',dpi=300,bbox_inches='tight');plt.close(f)
print('Publication figures generated from experiment CSVs.')

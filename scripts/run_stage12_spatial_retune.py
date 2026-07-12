"""Stage 12 isolated spatial cleanup and limited CatBoost retune."""
from __future__ import annotations
import json,time
from datetime import UTC,datetime
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostClassifier,Pool
ROOT=Path(__file__).resolve().parents[1]; IN=ROOT/'data/processed/stage11'; OUT=ROOT/'reports/stage12'; MODELS=ROOT/'models/stage12'; SEEDS=(20260711,20260712,20260713); SP=['segment_neighbors_accidents_prev_24h','segment_neighbors_accidents_prev_7d','graph_degree','graph_betweenness_centrality']
def prep(d,f,c):
 x=d[f].copy()
 for z in c:x[z]=x[z].astype('string').fillna('__MISSING__').astype(str)
 return x
def pr(y,p):
 g=pd.DataFrame({'p':p,'y':y}).groupby('p').y.agg(['count','sum']).sort_index(ascending=False); r=g['sum'].cumsum()/y.sum(); q=g['sum'].cumsum()/g['count'].cumsum();return float((q*r.diff().fillna(r)).sum())
def metrics(y,p):
 top=np.argsort(-p)[:int(np.ceil(.1*len(y)))]; b=np.clip(np.digitize(p,np.linspace(0,1,11),right=True)-1,0,9); e=sum(abs(p[b==i].mean()-y[b==i].mean())*(b==i).sum() for i in range(10) if (b==i).any())/len(y);return {'pr_auc':pr(y,p),'recall_at_top_10pct':float(y[top].sum()/y.sum()),'lift_at_top_10pct':float(y[top].mean()/y.mean()),'brier_score':float(np.mean((p-y)**2)),'expected_calibration_error_10_bins':float(e)}
def fit(x,y,v,vy,cats,params,seed):
 m=CatBoostClassifier(**(params|{'random_seed':seed,'task_type':'GPU','devices':'0','allow_writing_files':False,'loss_function':'Logloss','eval_metric':'PRAUC'}));m.fit(x,y,cat_features=cats,eval_set=(v,vy),early_stopping_rounds=100,verbose=False);return m
def main():
 OUT.mkdir(parents=True,exist_ok=True); d={s:pd.read_parquet(IN/f'training_dataset_1h_{s}.parquet') for s in ('train','validation','test')}; cfg=json.loads((ROOT/'reports/stage11/stage11_feature_config.json').read_text()); basef=cfg['numerical_features']+cfg['categorical_features']; cats=cfg['categorical_features']; ids=[basef.index(c) for c in cats]; m=CatBoostClassifier();m.load_model(ROOT/'models/stage11/catboost_1h_stage11_candidate.cbm'); sample=prep(d['test'].iloc[:5000],basef,cats); val=np.abs(m.get_feature_importance(Pool(sample,cat_features=ids),type='ShapValues')[:,:-1]).mean(0); shap=dict(zip(basef,map(float,val))); total=float(val.sum());
 for s in d.values():
  denom=s['graph_degree'].clip(lower=1);s['segment_neighbors_accidents_prev_24h_normalized']=s['segment_neighbors_accidents_prev_24h']/denom;s['segment_neighbors_accidents_prev_7d_normalized']=s['segment_neighbors_accidents_prev_7d']/denom
 keep=[z for z in SP if z!='graph_betweenness_centrality' or shap[z]/total>=.01]; nums=[z for z in cfg['numerical_features'] if z in keep or z not in SP]+['segment_neighbors_accidents_prev_24h_normalized','segment_neighbors_accidents_prev_7d_normalized']; features=nums+cats; catidx=[features.index(c) for c in cats]; outcfg={'target_column':'target_1h','numerical_features':nums,'categorical_features':cats,'excluded_from_model_features':sorted(set(cfg['excluded_from_model_features'])|{'road_segment_id'}),'spatial_features':{'kept':keep+['segment_neighbors_accidents_prev_24h_normalized','segment_neighbors_accidents_prev_7d_normalized'],'static':[z for z in keep if z.startswith('graph_')]},'stage11_individual_mean_abs_shap':shap,'betweenness_share_percent':shap['graph_betweenness_centrality']/total*100};(OUT/'stage12_feature_config.json').write_text(json.dumps(outcfg,ensure_ascii=False,indent=2))
 x={s:prep(d[s],features,cats) for s in d};y={s:d[s].target_1h.to_numpy(np.int8) for s in d}; trials=[{'depth':a,'learning_rate':b,'l2_leaf_reg':c,'border_count':q,'iterations':1500} for a,b,c,q in [(5,.03,2,64),(5,.05,5,128),(6,.03,5,128),(6,.05,3,64),(7,.04,5,128),(7,.06,2,32)]]; rows=[]; models=[]
 for p in trials:
  z=fit(x['train'],y['train'],x['validation'],y['validation'],catidx,p,SEEDS[0]); pv=z.predict_proba(x['validation'])[:,1]; rows.append({**p,'validation_pr_auc':pr(y['validation'],pv),'validation_recall_at_top_10pct':metrics(y['validation'],pv)['recall_at_top_10pct'],'gap':pr(y['train'],z.predict_proba(x['train'])[:,1])-pr(y['validation'],pv),'best_iteration':int(z.get_best_iteration())});models.append(z)
 winner_i=max(range(len(rows)),key=lambda i:(rows[i]['validation_pr_auc'],rows[i]['validation_recall_at_top_10pct'],-rows[i]['gap'])); winner=rows[winner_i]; model=models[winner_i]; stable=[]
 for seed in SEEDS:
  z=model if seed==SEEDS[0] else fit(x['train'],y['train'],x['validation'],y['validation'],catidx,trials[winner_i],seed);stable.append({'seed':seed,'validation_pr_auc':pr(y['validation'],z.predict_proba(x['validation'])[:,1]),'best_iteration':int(z.get_best_iteration())})
 pv=model.predict_proba(x['validation'])[:,1];pt=model.predict_proba(x['test'])[:,1];mm={'validation':metrics(y['validation'],pv),'test':metrics(y['test'],pt)};base=json.loads((ROOT/'reports/stage7d/1h/stage7d_weather_experiment_report.json').read_text())['experimental_metrics'];std=float(np.std([a['validation_pr_auc'] for a in stable]));accept=mm['validation']['pr_auc']>=base['validation']['pr_auc']*1.03 and mm['test']['pr_auc']>=base['test']['pr_auc'] and std<=.01 and all(mm[s][k]<=base[s]['calibration'][k] for s in ('validation','test') for k in ('brier_score','expected_calibration_error_10_bins'));MODELS.mkdir(parents=True,exist_ok=True);path=MODELS/'catboost_1h_stage12_candidate.cbm';model.save_model(path)
 rep={'generated_at_utc':datetime.now(UTC).isoformat(),'stage':'12','individual_spatial_shap':{z:{'mean_abs':shap[z],'share_percent':shap[z]/total*100} for z in SP},'feature_config':str((OUT/'stage12_feature_config.json').resolve()),'search':{'trials':rows,'selection_rule':'max validation PR-AUC, Recall@10%, min gap'},'winner':winner,'stability_three_seeds':{'runs':stable,'pr_auc_std':std},'metrics':mm,'comparison_with_stage7d':base,'acceptance_rule':'validation PR-AUC +3%, test PR-AUC not worse, std<=0.01, Brier/ECE not worse','accepted_as_candidate':accept,'decision_ru':'Кандидат принят для ручного рассмотрения.' if accept else 'Spatial-направление закрыто: очистка и retune не дали требуемого +3% PR-AUC; Stage 7D остаётся финальной.','model_path':str(path.resolve())};(OUT/'stage12_candidate_report.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

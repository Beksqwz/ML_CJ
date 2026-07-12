"""Train only after Stage 11 leakage audit has zero mismatches."""
from __future__ import annotations
import json,time
from datetime import UTC,datetime
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostClassifier,Pool
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/stage11'; OUT=ROOT/'reports/stage11'; MODELS=ROOT/'models/stage11'; SEEDS=(20260711,20260712,20260713)
def prep(d,f,c):
 x=d[f].copy()
 for z in c:x[z]=x[z].astype('string').fillna('__MISSING__').astype(str)
 return x
def pr(y,p):
 g=pd.DataFrame({'p':p,'y':y}).groupby('p').y.agg(['count','sum']).sort_index(ascending=False); r=g['sum'].cumsum()/y.sum(); q=g['sum'].cumsum()/g['count'].cumsum(); return float((q*r.diff().fillna(r)).sum())
def met(y,p):
 top=np.argsort(-p,kind='stable')[:max(1,int(np.ceil(.1*len(y))))]; bins=np.clip(np.digitize(p,np.linspace(0,1,11),right=True)-1,0,9); e=sum(abs(float(p[bins==i].mean())-float(y[bins==i].mean()))*(bins==i).sum() for i in range(10) if (bins==i).any())/len(y)
 return {'pr_auc':pr(y,p),'recall_at_top_10pct':float(y[top].sum()/y.sum()),'lift_at_top_10pct':float(y[top].mean()/y.mean()),'brier_score':float(np.mean((p-y)**2)),'expected_calibration_error_10_bins':float(e)}
def main():
 audit=json.loads((OUT/'leakage_audit.json').read_text()); assert audit['mismatch_count']==0
 cfg=json.loads((OUT/'stage11_feature_config.json').read_text()); f=cfg['numerical_features']+cfg['categorical_features']; c=cfg['categorical_features']; d={s:pd.read_parquet(DATA/f'training_dataset_1h_{s}.parquet') for s in ('train','validation','test')}; x={s:prep(d[s],f,c) for s in d}; y={s:d[s].target_1h.to_numpy(np.int8) for s in d}; idx=[f.index(z) for z in c]; params={'iterations':1500,'learning_rate':.05,'depth':7,'l2_leaf_reg':5.,'loss_function':'Logloss','eval_metric':'PRAUC','allow_writing_files':False}
 runs=[]; model=None
 for seed in SEEDS:
  m=CatBoostClassifier(**(params|{'random_seed':seed,'task_type':'GPU','devices':'0'})); m.fit(x['train'],y['train'],cat_features=idx,eval_set=(x['validation'],y['validation']),early_stopping_rounds=100,verbose=False); p=m.predict_proba(x['validation'])[:,1]; runs.append({'seed':seed,'best_iteration':int(m.get_best_iteration()),'validation_pr_auc':pr(y['validation'],p)}); model=m if seed==SEEDS[0] else model
 pv=model.predict_proba(x['validation'])[:,1]; pt=model.predict_proba(x['test'])[:,1]; mm={'validation':met(y['validation'],pv),'test':met(y['test'],pt)}; base=json.loads((ROOT/'reports/stage7d/1h/stage7d_weather_experiment_report.json').read_text())['experimental_metrics']; std=float(np.std([r['validation_pr_auc'] for r in runs])); imp=mm['validation']['pr_auc']/base['validation']['pr_auc']-1; accepted=imp>=.03 and mm['test']['pr_auc']>=base['test']['pr_auc'] and std<=.01 and all(mm[s][k]<=base[s]['calibration'][k] for s in ('validation','test') for k in ('brier_score','expected_calibration_error_10_bins'))
 vals=model.get_feature_importance(Pool(x['test'].iloc[:min(5000,len(x['test']))],cat_features=idx),type='ShapValues')[:,:-1]; groups={'spatial':[i for i,z in enumerate(f) if z in cfg['spatial_features']['temporal']+cfg['spatial_features']['static']],'historical':[i for i,z in enumerate(f) if z.startswith(('segment_accidents_','city_accidents_','segment_hours_','segment_has_history'))]}; total=np.abs(vals).mean(0).sum(); share={g:float(np.abs(vals).mean(0)[i].sum()/total*100) for g,i in groups.items()}
 MODELS.mkdir(parents=True,exist_ok=True); path=MODELS/'catboost_1h_stage11_candidate.cbm'; model.save_model(path)
 report={'generated_at_utc':datetime.now(UTC).isoformat(),'stage':'11','winner':{'params':params,'best_iteration':int(model.get_best_iteration())},'stability_three_seeds':{'runs':runs,'pr_auc_std':std,'threshold_max':.01},'metrics':mm,'shap_feature_group_contributions_percent':share,'acceptance_rule':'validation PR-AUC >=3%, test PR-AUC not worse, std<=0.01, Brier and ECE not worse than Stage7D','comparison_stage7d':base,'accepted_as_candidate':accepted,'decision_ru':'Кандидат принят для ручного рассмотрения; Stage 7D автоматически не заменяется.' if accepted else 'Кандидат отклонён; Stage 7D остаётся финальной моделью.','model_path':str(path.resolve())}; (OUT/'stage11_candidate_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

"""Direct-filter causality audit for Stage 11 neighbor 24h values."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/stage11'; OUT=ROOT/'reports/stage11/leakage_audit.json'
def main():
 d=pd.read_parquet(DATA/'training_dataset_1h_train.parquet'); adj=pd.read_parquet(DATA/'segment_adjacency.parquet').set_index('road_segment_id').neighbor_segment_ids.to_dict(); e=pd.read_parquet(ROOT/'data/processed/accidents_with_roads_ml_ready.parquet')[['road_segment_id','accident_datetime']]; e.road_segment_id=e.road_segment_id.astype(str); e.accident_datetime=pd.to_datetime(e.accident_datetime)
 sample=d.sample(min(100,len(d)),random_state=20260711); mismatches=[]
 for r in sample.itertuples(index=False):
  t=pd.Timestamp(r.datetime_hour); direct=int(((e.accident_datetime<t)&(e.accident_datetime>=t-pd.Timedelta(hours=24))&e.road_segment_id.isin(adj[str(r.road_segment_id)])).sum())
  if direct!=int(r.segment_neighbors_accidents_prev_24h): mismatches.append({'road_segment_id':str(r.road_segment_id),'datetime_hour':str(t),'stored':int(r.segment_neighbors_accidents_prev_24h),'direct':direct})
 OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps({'sample_rows':len(sample),'window':'[T-24h,T), accident_datetime < T','mismatch_count':len(mismatches),'mismatches':mismatches},ensure_ascii=False,indent=2))
 if mismatches: raise SystemExit('Leakage audit mismatch')
if __name__=='__main__': main()

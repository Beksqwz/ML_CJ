"""Read existing Stage 8C demo outputs and update only report threshold provenance."""
from __future__ import annotations
import json, sys
from datetime import UTC, datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from inference.risk_thresholds import PATH, load, level

def main() -> None:
    folder=ROOT/'reports'/'stage8c'/'demo_20220908T150000'; config=load(); detail={}
    for h in ('1h','24h'):
        rows=json.loads((folder/f'predictions_current_{h}.json').read_text(encoding='utf8'))
        counts={name:sum(row['risk_level']==name for row in rows) for name in ('LOW','MEDIUM','HIGH','CRITICAL')}
        detail[h]={"segments":len(rows),"independent_level_recalculation_passed":all(row['risk_level']==level(float(row['risk_probability']),config) for row in rows),"counts":counts}
    common={"risk_threshold_config_path":str(PATH.resolve()),"risk_threshold_config_version":config['version'],"risk_thresholds":config['levels'],"risk_threshold_purpose":config['purpose']}
    build={"generated_at_utc":datetime.now(UTC).isoformat(),**common,"builds":{h:{"segments":v['segments']} for h,v in detail.items()}}
    validation={"generated_at_utc":datetime.now(UTC).isoformat(),**common,"validation":detail}
    summary={"generated_at_utc":datetime.now(UTC).isoformat(),"datetime_hour":"2022-09-08 15:00:00",**common,"horizons":detail}
    (folder/'build_report.json').write_text(json.dumps(build,ensure_ascii=False,indent=2),encoding='utf8')
    (folder/'validation_report.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf8')
    (folder/'prediction_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(detail,ensure_ascii=False))
if __name__=='__main__':main()

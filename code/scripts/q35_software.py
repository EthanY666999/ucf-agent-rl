"""Actual production package metadata without importing/allocating model weights."""
import datetime,json,sys
from importlib.metadata import version
from pathlib import Path

def software_versions():
 return {name:version(name) for name in ['torch','vllm','transformers']}

def is_q35_2b(cfg):
 model=str(cfg.get('model',''))
 return ('Qwen3.5-2B' in model or '/S02_SFT_seed42/' in model or '/F03_SFT_seed42/' in model) and cfg.get('model_family','Qwen3.5')=='Qwen3.5'

def annotate_hardware(out,cfg,backfill=False):
 if not is_q35_2b(cfg):return None
 from scripts.q35_run_receipt import atomic
 p=Path(out)/'hardware.json';data=json.loads(p.read_text()) if p.exists() else dict(model_family='Qwen3.5',model=cfg['model'])
 data['software_versions']=software_versions()
 data['software_versions_provenance']=dict(recorded_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),python=sys.executable,source='importlib.metadata of actual production interpreter',backfilled=backfill,historical_loaded_versions_proven=not backfill)
 atomic(p,data);return data

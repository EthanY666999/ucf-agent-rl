"""UCF entry point using family-specific, independently regenerated gates."""
import argparse,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT),str(HERE)]

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--variant',choices=['ucf','noIS','uniform'],default='ucf');p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--fork_point',choices=['first_fence'],default=None);a=p.parse_args()
 cfg=json.loads(a.config.read_text());is_q35='qwen3.5' in cfg['model'].lower() or cfg.get('model_family')=='Qwen3.5'
 manifest=ROOT/'artifacts'/('frozen_q35.json' if is_q35 else 'frozen.json')
 frozen=json.loads(manifest.read_text())
 for name,h in frozen['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
 if not a.smoke:
  gate=ROOT/'artifacts'/('q35_gates' if is_q35 else '')
  for name in ['cpu_acceptance','gpu_acceptance','calibration','phase1_gate']:
   record=json.loads((gate/(name+'.json')).read_text());assert record['passed'],name
   if is_q35:assert record['model_family']=='Qwen3.5',name
 assert cfg['ucf_beta']==.25 and cfg['ucf_variant']==a.variant
 import train
 sys.argv=['train.py','--method','ucf','--config',str(a.config),'--output-dir',str(a.output_dir)]+(['--resume'] if a.resume else [])+(['--smoke'] if a.smoke else [])
 if a.fork_point is not None:sys.argv+=['--fork_point',a.fork_point]
 train.main()
if __name__=='__main__':main()

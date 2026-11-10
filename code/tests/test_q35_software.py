import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.q35_software import annotate_hardware,is_q35_2b
class SoftwareMetadata(unittest.TestCase):
 def test_only_2b_scope(self):
  self.assertFalse(is_q35_2b(dict(model='/data/models/Qwen2.5-0.5B')))
  self.assertFalse(is_q35_2b(dict(model='/data/models/Qwen3.5-9B')))
  self.assertTrue(is_q35_2b(dict(model='/data/models/Qwen3.5-2B')))
 def test_preserves_original_hardware_fields(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'hardware.json';p.write_text('{"code_commit":"original","gpu":"80GB"}')
   with patch('scripts.q35_software.software_versions',return_value={'torch':'a','vllm':'b','transformers':'5.22.2'}):annotate_hardware(Path(td),dict(model='/data/models/Qwen3.5-2B'),backfill=True)
   d=json.loads(p.read_text());self.assertEqual(d['code_commit'],'original');self.assertFalse(d['software_versions_provenance']['historical_loaded_versions_proven']);self.assertEqual(d['software_versions']['transformers'],'5.22.2')

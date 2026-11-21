"""Explicit finalized numeric projection; no field IO at import."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
from tools.study202.stream202 import Reader,Writer,file_sha
PREREG='0528579a3678f2367a514920ef625f85f9b3fa10'
SOURCE203_SCORE='dbbc5837f2747024539a395f2c3f239848475fe711dc7cfc38aacc7998928263'
SOURCE203_ZIP='ef84315ba8611c7685c899714d38e9d35c43838825ae4de1a6f027bf119d59dd'
def sealed(root,expected):
 root=Path(root);assert file_sha(root/'manifest.json')==expected and(root/'SEALED.sha256').read_text().strip()==expected
 manifest=json.loads((root/'manifest.json').read_text())
 for name,digest in manifest.items():
  path=(root/name).resolve();assert path.is_relative_to(root.resolve())and file_sha(path)==digest
 return manifest

def array_field(path,field):
 with Path(path).open(encoding='utf-8')as stream:
  reader=Reader(stream);found=False
  for key in reader.object():
   if key!=field:reader.skip_value();continue
   assert not found;found=True
   for _ in reader.array():yield reader.value()
  assert found and not reader.peek()

def project_query(query):
 return {k:v for k,v in query.items()if k not in {'raw_run_indices','run_source_keys'}}

def numeric_components(score_root,writer):
 score_root=Path(score_root)
 for unit in array_field(score_root/'numeric_inputs.json','units'):writer.write('numeric_unit',unit['unit'],unit)
 for name in ('input_receipt','summary','bounds','greedy','native_scores','selected_original_edges','physical_receipts'):
  writer.write(name,'all',json.loads((score_root/(name+'.json')).read_text()))

def coverage_projection(coverage):
 out={k:v for k,v in coverage.items()if k!='runs'}
 out['probes']=[row for row in coverage['runs']if row['probe']is not None]
 out['source_run_census_count']=len(coverage['runs']);return out

def new_physical_leaf_calls(calls):
 seen=set()
 for call in calls:
  if call['origin']=='historical':continue
  path=call['path'];assert not Path(path).is_absolute()and '..'not in Path(path).parts
  if path not in seen:seen.add(path);yield call

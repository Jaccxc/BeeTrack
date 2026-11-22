"""Public release entry binding the numeric bundle and all-disjoint summary."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from replay210 import replay


def run(bundle_path,summary_path,repo):
    with gzip.open(bundle_path,'rt',encoding='utf-8') as stream:bundle=json.load(stream)
    summary=json.loads(Path(summary_path).read_text(encoding='utf-8'))
    assert summary['schema']=='interleave210.all-disjoint-summary.v1'
    assert hashlib.sha256(Path(bundle_path).read_bytes()).hexdigest()==summary['numeric_bundle_sha256']
    result=replay(bundle,repo)
    assert result['all_disjoint_pooled']==summary['pooled']
    for key,value in result['clips'].items():
        assert value['all_disjoint']==summary['clips'][key]
    head=result['clips']['35/c01']
    budgets=head['all_disjoint']['by_budget']
    best=min((v['min_witness'] for v in budgets.values()),key=lambda v:v['numerator'])
    assert sorted(best['selected'])==sorted(head['selection']['selected']['ALL'])
    assert best['numerator']==head['scores']['ALL']['window_error_sum']
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle',type=Path);p.add_argument('all_disjoint_summary',type=Path)
    p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();value=run(a.bundle,a.all_disjoint_summary,a.repo.resolve())
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(passed=True,investment_pass=value['preregistered_gates']['investment_pass'],
        attribution_pass=value['preregistered_gates']['attribution_pass'],
        global_count_extrema=value['all_disjoint_pooled'])))

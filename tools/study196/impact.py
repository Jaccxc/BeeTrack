"""Private numeric-only T-minus-C complete-pair impact decomposition.

No images,tracks,GT files or scorers. decompose accepts already-scored rows:
{key,hive,C_swap,T_swap,truth_swap(usually0),identity:{N,nGT,nPred,TP},cross:{...}}.
Bits refer to ORIGINAL true-run ordering,after blind-order/fallback conversion.
N is the SINGLE-flow integer30-phase L1 numerator (no extra factor3).
"""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path


def _integer(value,name):
    if isinstance(value,bool) or int(value)!=value or value<0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


def _score(record):
    value={key:_integer(record[key],key) for key in ('N','nGT','nPred','TP')}
    if value['TP']>min(value['nGT'],value['nPred']):
        raise ValueError('TP exceeds GT/prediction count')
    value['FP']=value['nPred']-value['TP'];value['FN']=value['nGT']-value['TP']
    return value


def _metrics(total):
    denominator=total['nPred']+total['nGT']
    return dict(**total,W_percent=100*total['N']/(30*total['nGT']) if total['nGT'] else None,
                F1=2*total['TP']/denominator if denominator else None)


def decompose(rows):
    """Exact additive N/TP/Pred decomposition; F1 is computed only AFTER sums.

    corrected/harmed use effective hard choices,including blind fallback for
    zero votes. Abstentions are NOT magically recoded as correct identity.
    Potential nonzeroN means cross-versus-identity difference,notT-minus-C.
    Trials reusing bees/frames remain repeated observational units,not iid.
    """
    keys=set();by_hive=defaultdict(list)
    for row in rows:
        if row['key'] in keys:
            raise ValueError('duplicate trial key')
        keys.add(row['key']);by_hive[row['hive']].append(row)
    output={}
    for hive,trials in by_hive.items():
        categories={};totals={model:{k:0 for k in ('N','nGT','nPred','TP','FP','FN')} for model in ('identity','C','T')}
        zero={model:{k:0 for k in ('N','nPred','TP','FP')} for model in ('C','T')}
        identity_checks={k:0 for k in ('N','TP','nPred')};trial_effects=[]
        for row in trials:
            c,t,truth=(row.get(k,0) for k in ('C_swap','T_swap','truth_swap'))
            if any(v not in (0,1) for v in (c,t,truth)):
                raise ValueError('effective choices must be0/1')
            original=_score(row['identity']);cross=_score(row['cross'])
            if original['nGT']!=cross['nGT']:
                raise ValueError('both alternatives require the same fixed GT inventory')
            C=cross if c else original;T=cross if t else original
            category='same' if c==t else 'corrected' if t==truth else 'harmed'
            record=categories.setdefault(category,dict(trials=0,potential_nonzero_N=0,actual_nonzero_N=0,
                potential_nonzero_TP=0,swap_minus_identity_N=0,swap_minus_identity_TP=0,
                swap_minus_identity_Pred=0,T_minus_C_N=0,T_minus_C_TP=0,T_minus_C_Pred=0,keys=[]))
            record['trials']+=1;record['keys'].append(row['key'])
            record['potential_nonzero_N']+=cross['N']!=original['N']
            record['actual_nonzero_N']+=T['N']!=C['N']
            record['potential_nonzero_TP']+=cross['TP']!=original['TP']
            effect={}
            for scorekey,label in (('N','N'),('TP','TP'),('nPred','Pred')):
                contrast=cross[scorekey]-original[scorekey]
                change=T[scorekey]-C[scorekey]
                assert change==(t-c)*contrast
                record['swap_minus_identity_'+label]+=contrast
                record['T_minus_C_'+label]+=change
                identity_checks[scorekey]+=(t-c)*contrast
                effect[label]=change
            for model,chosen in (('identity',original),('C',C),('T',T)):
                for key,value in chosen.items():
                    totals[model][key]+=value
            if original['nGT']==0:
                for model,chosen in (('C',C),('T',T)):
                    for key in zero[model]:
                        zero[model][key]+=chosen[key]
            trial_effects.append(dict(key=row['key'],category=category,C_swap=int(c),T_swap=int(t),
                zero_GT=original['nGT']==0,Nswap_minus_Nidentity=cross['N']-original['N'],**effect))
        for key in identity_checks:
            assert totals['T'][key]-totals['C'][key]==identity_checks[key]
        for category in ('corrected','harmed','same'):
            categories.setdefault(category,dict(trials=0,potential_nonzero_N=0,actual_nonzero_N=0,
                potential_nonzero_TP=0,swap_minus_identity_N=0,swap_minus_identity_TP=0,
                swap_minus_identity_Pred=0,T_minus_C_N=0,T_minus_C_TP=0,T_minus_C_Pred=0,keys=[]))
        output[hive]=dict(trials=len(trials),categories=categories,totals={m:_metrics(v) for m,v in totals.items()},
            zero_GT_totals=zero,zero_GT_trials=sum(t['zero_GT'] for t in trial_effects),
            identity_verified=True,delta_numerators=identity_checks,trial_effects=trial_effects,
            caveat='SumN/TP/Pred first,then compute F1;F1 differences are not additive. Controlled complete-GT pair task,not naturaltracking evidence.')
    return output


def effective_swap(order_bit,physical_choice):
    """given is the blindinput order;cross reverses it. Zero vote usesgiven."""
    if order_bit not in (0,1) or physical_choice not in ('given','cross'):
        raise ValueError('expected blindorderbit andgiven/cross choice')
    return int(bool(order_bit)^(physical_choice=='cross'))


def completed_impact(predictions,physical,variant='R'):
    """In-memory adapter for fit185/runner185 sealed numeric schemas.

    Caller verifies BOTH seals before loading. This function does no I/O.
    Separate .5/.3 TP reports share identical N/Pred/GT and effective choices.
    Identity votes andabstentions are NOT this physical-hard-choice partition.
    """
    rows=[];rows03=[];complete03=True
    for pair in predictions['pairs']:
        key=pair['key'];scored=physical['results'][key];selected=pair['variants'][variant]
        row=dict(key=key,hive=pair['hive'],truth_swap=0,
            C_swap=effective_swap(pair['order_bit'],selected['C']['physical_choice']),
            T_swap=effective_swap(pair['order_bit'],selected['T']['physical_choice']),
            identity=scored['identity'],cross=scored['cross'])
        rows.append(row)
        if all('0.3' in scored[arm].get('iou',{}) for arm in ('identity','cross')):
            rows03.append(dict(row,**{arm:dict(scored[arm],TP=scored[arm]['iou']['0.3']['tp'])
                                      for arm in ('identity','cross')}))
        else:
            complete03=False
    return dict(variant=variant,iou05=decompose(rows),iou03=decompose(rows03) if complete03 else None)


def selfcheck():
    rows=[]
    for i in range(3):
        rows.append(dict(key=f'fix{i}',hive='W',C_swap=1,T_swap=0,
            identity=dict(N=0,nGT=1,nPred=1,TP=1),cross=dict(N=2,nGT=1,nPred=1,TP=0)))
    rows.append(dict(key='expensive_wrong_flip',hive='W',C_swap=0,T_swap=1,
        identity=dict(N=0,nGT=2,nPred=2,TP=2),cross=dict(N=12,nGT=2,nPred=2,TP=0)))
    result=decompose(rows)['W']
    assert result['categories']['corrected']['trials']==3 and result['categories']['harmed']['trials']==1
    assert result['delta_numerators']==dict(N=6,TP=1,nPred=0)
    assert result['totals']['C']['W_percent']==4 and result['totals']['T']['W_percent']==8
    assert result['totals']['C']['F1']==.4 and result['totals']['T']['F1']==.6
    predictions=dict(pairs=[]);physical=dict(results={})
    for i,row in enumerate(rows):
        order=i%2
        predictions['pairs'].append(dict(key=row['key'],hive=row['hive'],order_bit=order,variants=dict(R={
            model:dict(physical_choice='cross' if row[model+'_swap']^order else 'given') for model in ('C','T')})))
        physical['results'][row['key']]={arm:dict(row[arm],iou={'0.3':dict(tp=row[arm]['TP'])})
                                        for arm in ('identity','cross')}
    adapted=completed_impact(predictions,physical)
    assert adapted['iou05']['W']==adapted['iou03']['W']==result
    zero=decompose([dict(key='zero',hive='G',C_swap=0,T_swap=1,
        identity=dict(N=0,nGT=0,nPred=0,TP=0),cross=dict(N=60,nGT=0,nPred=2,TP=0))])['G']
    assert zero['zero_GT_totals']['T']['FP']==2 and zero['zero_GT_totals']['T']['N']==60
    assert zero['totals']['T']['W_percent'] is None
    assert effective_swap(0,'given')==0 and effective_swap(1,'given')==1
    assert effective_swap(0,'cross')==1 and effective_swap(1,'cross')==0
    print(json.dumps(dict(corrected=3,harmed=1,C_W=4,T_W=8,C_F1=.4,T_F1=.6,
                          zero_GT_T_false_positives=2,all_integer_identities_passed=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--prediction-seal');parser.add_argument('--physical-seal');args=parser.parse_args()
    selfcheck()
    if args.root is not None:
        assert args.out and args.prediction_seal and args.physical_seal
        sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        inputs={}
        for folder,filename,expected in (('predictions','predictions.json',args.prediction_seal),
                                         ('physical','physical.json',args.physical_seal)):
            directory=args.root/folder;manifest=directory/'manifest.json'
            assert sha(manifest)==expected==(directory/'SEALED.sha256').read_text().strip()
            hashes=json.loads(manifest.read_text());assert sha(directory/filename)==hashes[filename]
            inputs[folder]=json.loads((directory/filename).read_text())
        result=completed_impact(inputs['predictions'],inputs['physical'])
        result['input_seals']=dict(predictions=args.prediction_seal,physical=args.physical_seal)
        result['source_sha256']=sha(__file__)
        with args.out.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,sort_keys=True,separators=(',',':'));stream.write('\n')
        print(json.dumps({h:dict(categories={k:{f:v[f] for f in ('trials','potential_nonzero_N','actual_nonzero_N',
             'T_minus_C_N','T_minus_C_TP','T_minus_C_Pred')} for k,v in r['categories'].items()},
             totals=r['totals'],zero_GT_trials=r['zero_GT_trials'],zero_GT_totals=r['zero_GT_totals'])
             for h,r in result['iou05'].items()},indent=2))
        print('output SHA256',sha(args.out))

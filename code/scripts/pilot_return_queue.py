"""Resume original runtime after pilots, preserving the untouched queue suffix."""
import argparse
import copy
import json
from pathlib import Path
import sys


def revised_plan(plan, terminal_results=None):
    result=copy.deepcopy(plan)
    entries=result['entries'];by_id={e['id']:e for e in entries}
    original_ids=[e['id'] for e in entries]
    moved=[by_id['S07'],by_id['S10']]
    entries=[e for e in entries if e['id'] not in ('S07','S10')]
    pos=next(i for i,e in enumerate(entries) if e['id']=='S06')+1
    entries[pos:pos]=moved;result['entries']=entries
    for previous,current in [('S06','S07'),('S07','S10'),('S10','H03')]:
        by_id[current]['strict_predecessor']=by_id[previous]['output']
    s10=by_id['S10'];cfg=s10['config']
    cfg['ucf_controller']='v2'
    for name in ('ucf_corrections','fork_point'):
        cfg.pop(name,None);s10.get('config_difference',{}).pop(name,None)
    s10['implementation_version']='UCF-v2, paired seed43 repeat of historical S05 by explicit user instruction'
    if terminal_results is not None:
        for ident in ('S07','S10'):
            if terminal_results.get(ident) is False:
                assert (Path(by_id[ident]['output'])/'needs_manual.json').exists(), 'Failed entry must be fenced before handback'
                by_id[ident].pop('strict_predecessor',None)
                by_id[ident]['pilot_terminal_failure_preserved']=True
        # These independent SFT-start jobs are terminal; preserve their evidence,
        # skip failures and let the original unrelated H03 suffix continue.
        previous=next(i for i in ('S10','S07','S06') if i=='S06' or terminal_results.get(i))
        by_id['H03']['strict_predecessor']=by_id[previous]['output']
    assert [e['id'] for e in entries if e['id'] not in ('S07','S10')]==[i for i in original_ids if i not in ('S07','S10')]
    result['revision']='user_20261109_after_P1_P2_S07_S10_v2_original_suffix'
    result['pilot_insertion_receipt']='pilot_p1p2_20261109/entry_results.json'
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--state',type=Path,required=True)
    args=parser.parse_args();plan=json.loads((args.state/'pilot_plan.json').read_text())
    terminal=json.loads((args.state/'entry_results.json').read_text())
    # Import the original runtime, including its existing source-specific gates.
    original=Path(plan['original_root'])
    sys.path[:0]=[str(original/'scripts'),str(original),str(original/'ucf')]
    import hard_queue_plan
    import hard_queue_supervisor
    compile_original=hard_queue_plan.compile_plan
    def compile_revised(base,cost):return revised_plan(compile_original(base,cost),terminal)
    hard_queue_supervisor.compile_plan=compile_revised
    hard_queue_supervisor.main()


if __name__=='__main__':main()

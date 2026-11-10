"""Explicitly versioned corrections shared by future SFT and no-SFT UCF.

The historical v2 collector/select/weights modules stay byte-identical.
"""
import json
import math
import os
import random
from pathlib import Path
from cert_select import stable_seed
from cert_entropy import consequence_classes
from ucf_select import calibrate, feature, RHO_MIN

def complement_stratum(node, token, mapping):
    classes = consequence_classes(node['branch'].env)
    action = mapping.get(token, 0)
    original = classes[action]
    p = node['pi']
    po = sum(p[a] for a in range(5) if classes[a] == original)
    # STAY contains ALL residual tokens, not just invalid action words.
    other_actions = [a for a in range(1,5) if classes[a] != original]
    allow = sorted(t for t,a in mapping.items() if classes[a] != original)
    ban = sorted(t for t,a in mapping.items() if classes[a] == original)
    pc = 1. - po
    assert 0 <= po <= 1 and 0 <= pc <= 1
    return dict(first_action=action, original_class=original, pi_o=po, pi_cf=pc,
                unsampled_mass=0., allowed_actions=other_actions,
                allowed_tokens=allow if original == classes[0] else None,
                banned_tokens=ban if original != classes[0] else None,
                sampling_support='full vocabulary minus original consequence class',
                cf_fallback=pc <= 0 or (original == classes[0] and not allow),
                parse_mismatch=False)

def sampling_restriction(stratum):
    assert not stratum['cf_fallback']
    if stratum['allowed_tokens'] is not None:
        return dict(allowed_token_ids=stratum['allowed_tokens'], n=1)
    # User v2.2: unrestricted ordered n=256 proposal batch, no logit bias.
    return dict(n=256)

def first_complement_output(outputs, stratum, mapping, classes):
    ordered=sorted(outputs,key=lambda x:x.index)
    expected=1 if stratum['allowed_tokens'] is not None else 256
    assert len(ordered)==expected and [x.index for x in ordered]==list(range(expected))
    audit=[];chosen=None
    for out in ordered:
        assert len(out.token_ids)==1
        token=int(out.token_ids[0]);accepted=token_in_complement(token,stratum,mapping,classes)
        audit.append(dict(index=out.index,token=token,logprob=float(out.logprobs[0][token].logprob),in_original_class=not accepted))
        if chosen is None and accepted:chosen=out
    if stratum['allowed_tokens'] is not None:assert chosen is not None
    return chosen,audit

def token_in_complement(token, stratum, mapping, classes):
    return classes[mapping.get(token,0)] != stratum['original_class']

def leave_question_out_constants(task_weights, target=2.):
    assert len(task_weights) >= 2, 'Leave-one-question calibration requires >=2 questions'
    return {ti:calibrate([ws for tj,ws in task_weights.items() if tj != ti], target)
            for ti in task_weights}

def select_corrected(candidates, first, tasks, constants, seed, step, path, variant='ucf', pilot=False):
    rows=[]; picked={ti:[] for ti in candidates}
    for ti, nodes in candidates.items():
        for node in nodes:
            w,ph,base=feature(node,first,variant)
            c=constants[ti]; rho=min(1.,max(RHO_MIN,c*w))
            if pilot:
                from ucf_solvability import pilot_probability
                rho=pilot_probability(node,c,w)
            u=random.Random(stable_seed(seed,tasks[ti]['uid'],node['j'],node['turn'],step,'select')).random()
            node.update(w=w,rho=rho,u=u,selected=u<rho,p_hat_minus_j=ph,baseline=base,c_step=c,
                        calibration='leave_one_question_out')
            rows.append(dict(task_id=ti,uid=tasks[ti]['uid'],j=node['j'],turn=node['turn'],step=step,
                             w=w,rho=rho,u=u,selected=u<rho,p_hat_minus_j=ph,baseline=base,c_step=c,
                             parse_mismatch=False,calibration='leave_one_question_out'))
            if pilot:
                from ucf_solvability import selection_log_fields
                rows[-1].update(selection_log_fields(node))
                rows[-1]['calibration']='leave_one_question_out_with_node_floors'
                node['calibration']=rows[-1]['calibration']
            if u<rho: picked[ti].append(node)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
        f.flush();os.fsync(f.fileno())
    return picked,rows

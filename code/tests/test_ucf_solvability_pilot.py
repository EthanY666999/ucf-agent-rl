"""CPU acceptance of P1/P2. Tests do not claim full PPO+KL unbiasedness."""
import copy
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE), str(CODE/'ucf')]
from task_envs import SokobanEnv
from ucf_select import feature, calibrate
from ucf_corrected import select_corrected
from ucf_solvability import (pilot_enabled, exact_distance, annotate_candidate,
    calibrate_floors, prepare_pilot, pilot_probability, pilot_statistics, controller_bootstrap_c)


def task(player=(2, 1), box=(2, 2), goal=(2, 4)):
    walls = [[r,c] for r in range(6) for c in range(6) if r in (0,5) or c in (0,5)]
    return dict(environment='sokoban',uid='fixture',size=6,walls=walls,
                player=list(player),box=list(box),goal=list(goal))


def fixture():
    first = [NS(task_id=ti,lineage=str(j),reward=0.,terminal='turn_limit')
             for ti in range(3) for j in range(4)]
    candidates = {ti:[dict(branch=NS(task_id=ti,turns=0,env=SokobanEnv(task())),
                j=j,turn=1,S=0.,H_cons=.4,state=13,boundary=0) for j in range(4)]
                for ti in range(3)}
    cfg = dict(solvability_gate=True,early_exploration_floor=False,max_turns=10,
               ucf_controller='v2',ucf_variant='ucf',is_slippery=False)
    return first,candidates,cfg


def test_flags_explicit_and_default_off():
    assert not pilot_enabled({})
    assert not pilot_enabled(dict(solvability_gate=False,early_exploration_floor=False))
    with pytest.raises(TypeError):pilot_enabled({'solvability_gate':'false'})
    with pytest.raises(ValueError):pilot_enabled({'early_exploration_floor':True})


@pytest.mark.parametrize('t,remaining,dead,structural',[
    (task((1,2),(1,1),(3,3)),10,True,True),
    (task((2,2),(1,2),(2,1)),10,True,True),
    (task(),1,True,False),
    (task(),2,False,False),
    (task((2,3),(2,4),(2,4)),0,False,False),
])
def test_corner_wall_budget_solvable_and_solved(t,remaining,dead,structural):
    first,cs,cfg=fixture();n=cs[0][0]
    if t['box']==t['goal']:
        # Solved states can occur after stepping, not at environment creation.
        env=SokobanEnv(task());env.player=tuple(t['player']);env.box=tuple(t['box'])
        n['branch'].env=env
    else:n['branch'].env=SokobanEnv(t)
    n['branch'].turns=10-remaining;n['turn']=11-remaining
    annotate_candidate(n,first,cfg);m=n['solvability']
    assert m['solvability_gated']==dead and m['structural_deadlock']==structural
    assert m['remaining_turns']==remaining


def test_deterministic_frozenlake_and_flh_distance():
    env=NS(s=0,nrow=3,ncol=3,desc=np.array([list('SFF'),list('HHF'),list('FFG')],dtype='S1'),is_slippery=False)
    assert exact_distance(env)==4
    env.s=3;assert exact_distance(env) is None
    env.s=8;assert exact_distance(env)==0
    env.is_slippery=True
    with pytest.raises(ValueError):exact_distance(env)


def test_fatal_action_not_gated_before_it_is_taken():
    first,cs,cfg=fixture();n=cs[0][0];n['branch'].env=SokobanEnv(task((3,2),(2,2),(2,1)))
    annotate_candidate(n,first,cfg)
    assert n['solvability']['within_budget']
    old_classes=n['branch'].env.consequence_classes()
    assert old_classes[4].startswith('STATE:')  # Do NOT change FAIL classes in P1/P2.
    n['branch'].env.step(4);n['branch'].turns=1;n['turn']=2
    annotate_candidate(n,first,cfg)
    assert n['solvability']['solvability_gated']
    assert not n['branch'].env.deadlock()  # Wall death missed by old corner rule.


@pytest.mark.parametrize('successes,expected',[(0,.08),(1,.03+.05*2/3),(2,.03+.05/3),(3,.03)])
def test_p2_exact_leave_trajectory_floor(successes,expected):
    first,cs,cfg=fixture();cfg['early_exploration_floor']=True
    for b in first:
        if b.task_id==0 and 1<=int(b.lineage)<=successes:b.reward=1.;b.terminal='success'
    annotate_candidate(cs[0][0],first,cfg)
    assert cs[0][0]['solvability']['rho_floor']==pytest.approx(expected,abs=1e-16)
    for x in (0.,1.):
        first[0].reward=x;first[0].terminal='success' if x else 'turn_limit'
        annotate_candidate(cs[0][0],first,cfg)
        assert cs[0][0]['solvability']['rho_floor']==pytest.approx(expected,abs=1e-16)


def test_early_only_zero_weight_receives_floor_and_late_does_not():
    first,cs,cfg=fixture();cfg['early_exploration_floor']=True
    n=cs[0][0];w,_=annotate_candidate(n,first,cfg)
    assert w==0 and pilot_probability(n,0.,w)==.08
    n['branch'].turns=3;n['turn']=4
    w,_=annotate_candidate(n,first,cfg)
    assert pilot_probability(n,10000.,w)==.03


def test_node_floor_calibration_and_fixed_task_denominator():
    c,m=calibrate_floors([[(.4,.08)]*8,[(.8,.03)]*8])
    assert m['calibrated_expected_forks']==pytest.approx(2.)
    assert not m['target_unreachable']
    # Empty questions are retained: one positive node over two tasks cannot hit 2.
    c,m=calibrate_floors([[],[(.4,.08)]])
    assert m['maximum_expected_forks']==.5 and m['target_unreachable']
    assert c>=2.5 and m['calibrated_expected_forks']==.5
    c,m=calibrate_floors([[],[]]);assert c==0 and m['maximum_expected_forks']==0
    c,m=calibrate_floors([[(0.,.08)]*4,[(0.,.03)]*4])
    assert c==0 and m['calibrated_expected_forks']==pytest.approx(.22)
    with pytest.raises(ValueError):calibrate_floors([[(0.,.08)]*30])


def test_zero_support_batch_can_restore_controller_then_recalibrate():
    from ucf_select import Controller
    old=Controller(controller_bootstrap_c(0.),[0.])
    assert old.c==1. and old.history==[0.]
    first,cs,cfg=fixture()
    constants,_=prepare_pilot(cs,first,cfg)
    # Real candidates have zero w: the bootstrap value never replaces their c.
    assert set(constants.values())=={0.}
    assert controller_bootstrap_c(3.)==3.
    with pytest.raises(ValueError):controller_bootstrap_c(float('nan'))
    with pytest.raises(ValueError):controller_bootstrap_c(-1.)


def test_ordinary_floor_calibration_preserves_v22_arithmetic():
    weights=[[.0,.1,.3,.8]*3,[.2,.8,.1,.3]*3]
    c,_=calibrate_floors([[(w,.03) for w in ns] for ns in weights])
    assert c==calibrate(weights)


def test_own_suffix_added_candidates_and_outcome_do_not_change_own_rho(tmp_path):
    first,cs,cfg=fixture();cfg['early_exploration_floor']=True
    results=[]
    for mutate in (False,True):
        ff,cc=copy.deepcopy(first),copy.deepcopy(cs)
        if mutate:
            ff[0].reward=1.;ff[0].terminal='success'
            for turn in range(2,9):
                extra=copy.deepcopy(cc[0][0]);extra['turn']=turn;extra['branch'].turns=turn-1
                extra['S']=.9;cc[0].append(extra)
        constants,_=prepare_pilot(cc,ff,cfg)
        _,rows=select_corrected(cc,ff,[{'uid':str(i)} for i in cc],constants,42,3,tmp_path/str(mutate),pilot=True)
        results.append((constants[0],rows[0]))
    assert results[0]==results[1]


def test_gated_nodes_are_logged_but_never_selected_or_divided_by_rho(tmp_path):
    first,cs,cfg=fixture()
    for nodes in cs.values():
        for n in nodes:n['branch'].env=SokobanEnv(task((1,2),(1,1),(3,3)))
    constants,budget=prepare_pilot(cs,first,cfg)
    picked,rows=select_corrected(cs,first,[{'uid':str(i)} for i in cs],constants,42,1,tmp_path/'s',pilot=True)
    assert all(not ns for ns in picked.values())
    assert len(rows)==12 and all(r['rho']==0 and not r['selected'] for r in rows)
    assert all(r['solvability_gated'] and r['bfs_distance'] is None for r in rows)
    assert budget['expected_forks']==0 and budget['target_unreachable']
    stats=pilot_statistics(cs,first,{},budget)
    assert stats['all_failure']['early_count']==12
    assert stats['all_failure']['early_mean_rho']==0
    assert stats['all_failure']['rescued_tasks']==0 and stats['all_failure']['task_count']==3


def test_rescue_rate_denominator_is_all_initial_failed_tasks(tmp_path):
    first,cs,cfg=fixture();constants,budget=prepare_pilot(cs,first,cfg)
    select_corrected(cs,first,[{'uid':str(i)} for i in cs],constants,42,1,tmp_path/'s',pilot=True)
    children={(0,0,1):NS(task_id=0,reward=1.),(0,1,1):NS(task_id=0,reward=1.)}
    stats=pilot_statistics(cs,first,children,budget)
    assert stats['all_failure']['rescued_tasks']==1
    assert stats['all_failure']['rescue_rate']==1/3


def test_default_selector_log_bytes_match_frozen_v22(tmp_path):
    baseline=Path(os.environ.get('PILOT_BASELINE_CODE',str(CODE.parents[1]/'baseline/code')))
    oldpath=baseline/'ucf/ucf_corrected.py'
    spec=importlib.util.spec_from_file_location('old_corrected',oldpath)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    first,cs,cfg=fixture();constants={i:3. for i in cs};tasks=[{'uid':str(i)} for i in cs]
    _,a=old.select_corrected(copy.deepcopy(cs),first,tasks,constants,7,10,tmp_path/'old')
    _,b=select_corrected(copy.deepcopy(cs),first,tasks,constants,7,10,tmp_path/'new',pilot=False)
    assert a==b and (tmp_path/'old').read_bytes()==(tmp_path/'new').read_bytes()


def test_complete_sokoban_state_graph_matches_independent_forward_bfs():
    from collections import deque
    env=SokobanEnv(task());floor=[(r,c) for r in range(6) for c in range(6) if not env.blocked((r,c))]
    for player in floor:
        for box in floor:
            if player==box:continue
            env.player,env.box=player,box
            q=deque([((player,box),0)]);seen={(player,box)};expected=None
            while q:
                state,d=q.popleft()
                if state[1]==env.goal:expected=d;break
                for a in (1,2,3,4):
                    nxt=env.transition(state,a)
                    if nxt not in seen:seen.add(nxt);q.append((nxt,d+1))
            assert exact_distance(env)==expected


def test_ideal_dead_state_stratified_reward_score_expectation_zero():
    # Only the outcome score-function term, fixed action-independent baseline,
    # fixed length and exact complementary sampling. NOT clipped PPO or KL.
    p=np.array([.08,.12,.2,.3,.3]);classes=np.array([0,0,1,2,3]);scores=np.eye(5)-p
    for baseline in (0.,.4):
        expected=np.zeros(5)
        for a in range(5):
            same=classes==classes[a];po=p[same].sum();pc=1-po
            for b in np.where(~same)[0]:
                value=-baseline*(2*po*scores[a]+2*pc*scores[b])
                expected+=p[a]*(p[b]/pc)*value
        assert np.max(np.abs(expected))<1e-15


def test_ideal_ht_simulation_with_gate_and_residual_mass_z_below_four():
    # Alive states retain positive rho and inverse actual-rho weights; certified
    # zero-reward states are analytically removed. Tests the ideal model only.
    p=np.array([.08,.12,.2,.3,.3]);classes=np.array([0,0,1,2,3]);scores=np.eye(5)-p
    rewards=np.array([0.,0.,0.,1.,1.]);rng=np.random.default_rng(20261109)
    target=2*np.sum((p*rewards)[:,None]*scores,axis=0)
    n=300000
    for rho in (.03,.08):
        a=rng.choice(5,size=n,p=p);b=np.empty(n,dtype=int)
        po=np.empty(n)
        for x in range(5):
            mask=a==x;same=classes==classes[x];po[mask]=p[same].sum()
            options=np.where(~same)[0];b[mask]=rng.choice(options,size=int(mask.sum()),p=p[options]/p[options].sum())
        sample=2*(po[:,None]*(rewards[a]-.4)[:,None]*scores[a]+(1-po)[:,None]*(rewards[b]-.4)[:,None]*scores[b])
        selected=rng.random(n)<rho;sample*=selected[:,None]/rho
        # Equal mixture: half alive, half gated dead; no teacher/model calls.
        gated=rng.random(n)<.5;sample[gated]=0
        se=sample.std(axis=0,ddof=1)/math.sqrt(n)
        z=np.abs(sample.mean(axis=0)-target*.5)/se
        assert np.max(z)<4

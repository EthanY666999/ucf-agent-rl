import unittest,sys,copy,hashlib
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from task_envs import *
from patr_core import make_env,score_branch
from hard_metrics import group_counts,leaf_group_counts
class HardEnvs(unittest.TestCase):
    def task(self,player=(2,1),box=(2,2),goal=(2,3)):
        return dict(environment='sokoban',walls=[(r,c) for r in range(6) for c in range(6) if r in [0,5] or c in [0,5]],player=player,box=box,goal=goal)
    def test_push_goal_and_reward(self):
        e=SokobanEnv(self.task());self.assertEqual(e.consequence_classes()[3],'GOAL')
        self.assertEqual(e.solution(),[3]);_,r,done,_=e.step(3)
        self.assertEqual(r,1);self.assertTrue(done);self.assertTrue(e.success());self.assertIn('*',e.render())
    def test_wall_deadlock_and_state_classes(self):
        e=SokobanEnv(self.task(player=(1,3),box=(1,2),goal=(3,3)))
        classes=e.consequence_classes();self.assertEqual(classes[4],'STAY');self.assertEqual(classes[1],'FAIL');self.assertTrue(classes[2].startswith('STATE:'))
        _,r,done,_=e.step(1);self.assertTrue(e.deadlock());self.assertEqual(r,0);self.assertFalse(done)
    def test_scorer_and_anchors(self):
        e=SokobanEnv(self.task(goal=(2,4)));b=SimpleNamespace(env=e,initial_distance=2,reward=0)
        cfg=dict(scorer_reward_weight=1,scorer_progress_weight=1)
        old=score_branch(b,cfg);e.step(3);self.assertGreater(score_branch(b,cfg),old)
        self.assertEqual(e.anchor(),(14,15))
    def test_flh_bfs_and_determinism(self):
        from rllm.environments.frozenlake import frozenlake as legacy
        for size in range(9,13):
            old=legacy.MAX_STEPS
            legacy.MAX_STEPS=1
            rows,meta=generate_flh_map(size,.65,1010000+size)
            self.assertEqual(legacy.MAX_STEPS,1)
            legacy.MAX_STEPS=old
            self.assertTrue(7<=meta['shortest_path_length']<=16)
            self.assertEqual((rows,meta),generate_flh_map(size,.65,1010000+size))
            t=dict(environment='flh',seed=1010000+size,size=size,p=.65,map_sha256=meta['map_sha256'])
            import tempfile,json
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'maps.json';p.write_text(json.dumps({'test':rows}));t['uid']='test'
                e=make_env(t,dict(max_turns=20,is_slippery=False,frozen_maps_path=str(p),frozen_maps_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            for action in meta['solution']:e.step(action)
            self.assertTrue(e.success())
    def test_flat_counts(self):
        bs=[SimpleNamespace(task_id=i,reward=r) for i,rs in enumerate([[1]*4,[0]*4,[0,1,0,1]]) for r in rs]
        self.assertEqual(group_counts(bs,3)['initial_group_counts'],dict(all_success=1,all_failure=1,mixed=1))
    def test_ucf_and_random_all_leaf_counts(self):
        initial=[SimpleNamespace(task_id=i,reward=r) for i,rs in enumerate([[1]*4,[0]*4,[0]*4]) for r in rs]
        cf=[SimpleNamespace(task_id=0,reward=0),SimpleNamespace(task_id=1,reward=1)]
        stats=group_counts(initial,3)
        stats.update(leaf_group_counts(initial+cf,3,initial_not_applicable=False))
        self.assertEqual(stats['initial_group_counts'],dict(all_success=1,all_failure=2,mixed=0))
        self.assertEqual(stats['leaf_group_counts'],dict(all_success=0,all_failure=1,mixed=2))
        self.assertEqual(stats['leaf_group_total'],14)
        random=leaf_group_counts(initial+cf,3)
        self.assertIsNone(random['initial_group_counts'])
        self.assertEqual(random['leaf_group_counts'],stats['leaf_group_counts'])
        with self.assertRaises(AssertionError):leaf_group_counts(initial+cf,4)
    def test_ucf_evaluation_delegates_without_training_state(self):
        from unittest.mock import Mock
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ucf'))
        from ucf_collect import make_collect
        original=Mock(return_value='evaluation_ok')
        wrapped=make_collect(None,original)
        args=([],None,None,{},'grpo',None)
        self.assertEqual(wrapped(*args,evaluation=True),'evaluation_ok')
        original.assert_called_once_with(*args,evaluation=True)
    def test_queue_order_and_caps(self):
        import json
        from hard_queue_plan import compile_plan,ORDER
        root=Path(__file__).resolve().parents[1]
        base=json.loads((root/'configs/frozenlake_q35_2b.json').read_text())
        entries=compile_plan(base,dict(passed=True,steps=5,trajectory_length_p95=10240,nosft_max_response_tokens=4096))['entries']
        self.assertEqual([e['id'] for e in entries],[x[0] for x in ORDER])
        ids=[e['id'] for e in entries];self.assertLess(ids.index('F10'),ids.index('F09'));self.assertLess(ids.index('S05'),ids.index('F04'))
        for e in entries:
            c=e['config'];self.assertEqual(c['steps'],400);self.assertEqual(c['learning_rate'],1e-6)
            if e['environment']=='flh':self.assertEqual(c['max_turns'],20)
            if e['environment']=='sokoban':self.assertEqual(c['max_turns'],10)
            if e['id'] in ['F10','F09']:self.assertEqual(c['max_response_tokens'],4096)
    def test_tau_caps_only_part_b(self):
        from ucf.ucf_weights import training_weights_ucf
        # Test effective importance factor without changing all non-IS fields.
        from unittest.mock import patch
        initial=[SimpleNamespace(task_id=0) for _ in range(4)]
        row=SimpleNamespace(ucf=dict(variant='ucf',rho=.03,beta=.25,original_tokens=10,adv=2,ht_weight_cap=10))
        with patch('patr_core.training_weights',return_value=[('unchanged',1,2)]):
            result=training_weights_ucf(initial+[row],1)
        self.assertEqual(result[0],('unchanged',1,2));self.assertEqual(result[1][2],.25*10/40)
    def test_retry_without_checkpoint_preserves_attempt(self):
        import tempfile,json
        from unittest.mock import patch
        import hard_queue_supervisor as sup
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);state=root/'queue';state.mkdir();out=root/'run';child=root/'child.py'
            child.write_text("import sys,json\nfrom pathlib import Path\np=Path(sys.argv[1]);marker=p.parent/'called';first=not marker.exists();marker.write_text('yes');(p/'evidence').write_text('original');(p/'status.json').write_text(json.dumps({'status':'failed' if first else 'complete'}));raise SystemExit(1 if first else 0)\n")
            with patch.object(sup,'ROOT',root),patch.object(sup,'RUNS',root),patch.object(sup,'STATE',state),(root/'lock').open('a') as lock:
                self.assertTrue(sup.execute([sys.executable,'-u',str(child),str(out)],out,'complete',lock))
            self.assertEqual(len(list(root.glob('run_attempt1_*'))),1)
    def test_sft_output_not_precreated(self):
        import tempfile
        from unittest.mock import patch
        import hard_queue_supervisor as sup
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);state=root/'queue';state.mkdir();out=root/'sft';child=root/'sft.py'
            child.write_text("import sys,json\nfrom pathlib import Path\np=Path(sys.argv[1]);assert not p.exists();p.mkdir();(p/'status.json').write_text(json.dumps({'status':'complete'}))\n")
            with patch.object(sup,'ROOT',root),patch.object(sup,'RUNS',root),patch.object(sup,'STATE',state),(root/'lock').open('a') as lock:
                self.assertTrue(sup.execute([sys.executable,'-u',str(child),str(out)],out,'complete',lock))

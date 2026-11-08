import unittest,sys,copy,hashlib
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/scripts'))
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
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/ucf'))
        from ucf_collect import make_collect
        original=Mock(return_value='evaluation_ok')
        wrapped=make_collect(None,original)
        args=([],None,None,{},'grpo',None)
        self.assertEqual(wrapped(*args,evaluation=True),'evaluation_ok')
        original.assert_called_once_with(*args,evaluation=True)
    def test_tau_caps_only_part_b(self):
        from ucf.ucf_weights import training_weights_ucf
        # Test effective importance factor without changing all non-IS fields.
        from unittest.mock import patch
        initial=[SimpleNamespace(task_id=0) for _ in range(4)]
        row=SimpleNamespace(ucf=dict(variant='ucf',rho=.03,beta=.25,original_tokens=10,adv=2,ht_weight_cap=10))
        with patch('patr_core.training_weights',return_value=[('unchanged',1,2)]):
            result=training_weights_ucf(initial+[row],1)
        self.assertEqual(result[0],('unchanged',1,2));self.assertEqual(result[1][2],.25*10/40)

import unittest
from scripts.paired_report import bootstrap,task_scores,shortest_path


class PairedReportTest(unittest.TestCase):
    def test_constant_paired_difference_has_exact_interval(self):
        left={u:dict(task=dict(uid=u),success=.75,pass4=1.) for u in ['a','b','c']}
        right={u:dict(task=dict(uid=u),success=.25,pass4=0.) for u in left}
        result=bootstrap(left,right,['c','a','b'])
        self.assertEqual(result['success']['percentile_95_ci_percentage_points'],[50.,50.])
        self.assertEqual(result['pass4']['difference_percentage_points'],100.)
        self.assertEqual(result,bootstrap(left,right,['a','b','c']))
        right['a']['task']['uid']='wrong'
        with self.assertRaises(AssertionError):bootstrap(left,right,['a','b','c'])
    def test_requires_four_distinct_trajectories_per_task(self):
        rows=[dict(task=dict(uid='a'),lineage=i,reward=i%2) for i in range(4)]
        self.assertEqual(task_scores(rows)['a']['success'],.5)
        with self.assertRaises(AssertionError):task_scores(rows[:3])
        rows[-1]['lineage']=0
        with self.assertRaises(AssertionError):task_scores(rows)
    def test_shortest_path_avoids_holes(self):
        self.assertEqual(shortest_path([[b'S',b'H',b'G'],[b'F',b'F',b'F']]),4)
        with self.assertRaises(ValueError):shortest_path([['S','H','G']])

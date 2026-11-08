"""A passing regression here means the known unbiasedness FAILURE is detected."""
import importlib.util
from pathlib import Path


def test_residual_support_counterexample_remains_visible():
    path=Path(__file__).resolve().parents[1]/'code/ucf/tests/UCF_无偏性模拟_htsim.py'
    spec=importlib.util.spec_from_file_location('v21_math_counterexample',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    complete=module.run(0.)
    residual=module.run(.08)
    assert complete['passed'] and complete['max_analytic_bias']==0
    assert not residual['passed'] and residual['max_abs_z']>4
    assert residual['max_z_against_analytic_estimator']<4
    assert residual['max_analytic_bias']>0

import numpy as np
import pytest

from waveforge.lab.epyc_tasks import sample_epyc_task
from waveforge.lab.mcl.baselines import uniform_parallel
from waveforge.lab.mcl.locked_channels import canonical_frame
from waveforge.lab.mcl.solver_free_tabular import candidate_features
from waveforge.lab.mcl.solver_free_v2 import fast_features


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_features_are_identical_to_reference(seed):
    rng = np.random.default_rng(seed)
    task = canonical_frame(sample_epyc_task(rng)).task
    designs = np.concatenate((rng.random((8, 64, 64)) < 0.25, uniform_parallel()[None]))
    x1, f1 = candidate_features(task, designs)
    x2, f2 = fast_features(task, designs)
    assert x1.shape == x2.shape == (9, 519)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(f1, f2)

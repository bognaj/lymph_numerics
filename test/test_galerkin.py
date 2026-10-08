import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from galerkin import Galerkin


class TestGalerkinExtrapolation(unittest.TestCase):
    def test_first_step_uses_predictor_corrector(self):
        xs = np.linspace(0.0, 1.0, 5)

        def v(x):
            return 0.1 + 0.9 * np.sin(np.pi * x)

        galerkin = Galerkin(
            xs=xs,
            N=len(xs),
            t0=0.0,
            dt=0.01,
            t_final=1.0,
            initial_condition=v,
        )

        first_state = 0.05 + 0.01 * np.arange(galerkin.num_dofs, dtype=float)
        u_history = np.vstack([
            first_state,
            first_state,
        ])
        lifting_prev = np.full_like(xs, 0.1)
        lifting_next = np.full_like(xs, 0.1)

        result = galerkin.crank_nicolson_galerkin_step(
            n=1,
            u_history=u_history,
            lifting_prev=lifting_prev,
            lifting_next=lifting_next,
        )

        self.assertEqual(result.shape, u_history[0, :].shape)
        self.assertTrue(np.isfinite(result).all())

    def test_extrapolate_u_for_second_step_uses_two_point_formula(self):
        xs = np.linspace(0.0, 1.0, 5)

        def v(x):
            return 0.1 + 0.9 * np.sin(np.pi * x)

        galerkin = Galerkin(
            xs=xs,
            N=len(xs),
            t0=0.0,
            dt=0.01,
            t_final=1.0,
            initial_condition=v,
        )
        galerkin.n = 2

        u_history = np.vstack([
            v(xs),
            1.1 * v(xs),
            0.9 * v(xs),
        ])

        extrapolated = galerkin.extrapolate_u(u_history)

        expected = 1.5 * u_history[1, :] - 0.5 * u_history[0, :]
        np.testing.assert_allclose(extrapolated, expected)


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runge_kutta import RungeKutta


class TestRungeKuttaShear(unittest.TestCase):
    def test_shear_magnitude_is_nonnegative(self):
        xs = np.linspace(0.0, 1.0, 5)
        rk = RungeKutta(
            x0=np.array([0.5, 0.1]),
            t0=0.0,
            dt=0.01,
            t_final=1.0,
            xs=xs,
        )
        radii = np.array([0.6, 0.8, 0.6, 0.8, 0.6])

        shear_stress = rk.get_shear_stress(radii)
        shear_magnitude = rk.get_shear_magnitude(radii)

        self.assertTrue(np.any(shear_stress < 0.0))
        self.assertTrue(np.any(shear_stress > 0.0))
        self.assertTrue(np.all(shear_magnitude >= 0.0))
        np.testing.assert_allclose(shear_magnitude, np.abs(shear_stress))

    def test_single_step_uses_spatial_mean_of_shear_magnitude(self):
        xs = np.linspace(0.0, 1.0, 5)
        rk = RungeKutta(
            x0=np.array([0.5, 0.1]),
            t0=0.0,
            dt=0.01,
            t_final=1.0,
            xs=xs,
        )
        radii = np.array([0.6, 0.8, 0.6, 0.8, 0.6])
        expected_mean_magnitude = rk.mean_value_in_x(
            rk.get_shear_magnitude(radii)
        )

        with patch.object(rk, "get_zeta", return_value=0.0) as get_zeta:
            rk.single_step(t=0.0, NC_s=np.array([0.5, 0.1]), R_vector=radii)

        get_zeta.assert_called_once_with(expected_mean_magnitude)


if __name__ == "__main__":
    unittest.main()

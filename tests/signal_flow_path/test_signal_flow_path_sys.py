import numpy as np
from pandas import DataFrame

from dysys import SignalFlowPathSys, ScalarLinearDySys


class TestSignalFlowPathSys:
    sys = SignalFlowPathSys(
        [
            ScalarLinearDySys(1, 0.04, lambda *_: 0.01, 0.5),
            ScalarLinearDySys(1, 0.05, lambda _, __, ___, ____, y: y / 25, 0.5),
        ]
    )
    eqm = np.array([0.25, 0.2])

    def test_equilibrium(self):
        """The exact solution is (0.25, 0.2)."""
        np.testing.assert_almost_equal(self.sys.equilibrium(), self.eqm)

    def test_march(self):
        trajectory = DataFrame(
            {t: x for t, x, _ in self.sys.march_till(100, 1, self.sys.zero)}
        ).T
        t = trajectory.index
        exact = np.array(
            [
                self.eqm[0] * (1 - np.exp(-t / 25)),
                (1 - np.exp(-t / 25)) - 4 / 5 * (1 - np.exp(-t / 20)),
            ]
        ).T
        np.testing.assert_allclose(trajectory, exact, atol=1e-3)

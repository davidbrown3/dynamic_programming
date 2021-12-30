import pytest
from collections import namedtuple

import jax
import jax.numpy as np

from ddpy.solvers.cost_functions import quadratic_cost_function

states = [
    np.array([[-1.0], [0.0], [0.0], [0.0]]),
    np.array([[-1.0], [-1.0], [0.0], [0.0]]),
    np.array([[-1.0], [-1.0], [-1.0], [0.0]]),
    np.array([[-1.0], [-1.0], [-1.0], [-1.0]]),
    np.array([[-2.0], [-3.0], [-4.0], [-5.0]]),
]

controls = [
    np.array([[0.0]]),
    np.array([[-1.0]]),
    np.array([[-2.0]]),
]

Case = namedtuple('Case', 'states controls')

cases = []
for state in states:
    for control in controls:
        cases.append(Case(state, control,))


@pytest.fixture(scope="session")
def cost_function():
    g_xx = np.diag(np.array([0.1, 0.0, 2.0, 0.0]))
    g_xu = np.zeros([4, 1])
    g_ux = np.zeros([1, 4])
    g_uu = np.array([[0.1]])
    g_x = np.array([[0.0, 0.0, 0.0, 0.0]])
    g_u = np.array([[0.0]])
    return quadratic_cost_function(g_xx=g_xx, g_xu=g_xu, g_ux=g_ux, g_uu=g_uu, g_x=g_x, g_u=g_u)


@pytest.mark.parametrize("states, controls", cases)
def test_jacobian(cost_function, states, controls):

    auto_jacobian = jax.jacfwd(cost_function.calculate_cost, [0, 1])(states[:, 0], controls[:, 0])

    g_u = cost_function.calculate_g_u(x=states, u=controls)
    g_x = cost_function.calculate_g_x(x=states, u=controls)

    assert np.all(np.abs(auto_jacobian[1] - g_u) < 1e-4)
    assert np.all(np.abs(auto_jacobian[0] - g_x) < 1e-4)

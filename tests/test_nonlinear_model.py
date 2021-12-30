from collections import namedtuple

import jax.numpy as np
import pytest

from .cart_pole import CartPole


states_0s = [
    np.array([0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, np.pi/4, 0.0])
]

controls_0s = [
    np.array([0.0]),
    np.array([1.0])
]

states_deltas = [
    np.array([0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 1e-3, 0.0]),
    np.array([0.0, 0.0, 0.0, 1e-3]),
    np.array([0.0, 0.0, 1e-3, 1e-3]),
]

controls_deltas = [
    np.array([0.0]),
    np.array([1e-3]),
    np.array([1.0]),
]

Case = namedtuple('Case', 'states_0 controls_0 states_delta controls_delta')

cases = []
for states_0 in states_0s:
    for controls_0 in controls_0s:
        for states_delta in states_deltas:
            for controls_delta in controls_deltas:
                cases.append(Case(states_0, controls_0, states_delta, controls_delta))


@pytest.fixture(scope="session")
def problem():
    return CartPole(dt=0.01)


def compare_jacobian(problem, A, B, states_0, controls_0, states_delta, controls_delta, derivatives_0):

    linear = np.matmul(A, states_delta) + np.matmul(B, controls_delta) + derivatives_0
    nonlinear = problem.derivatives(states_delta + states_0, controls_delta + controls_0)

    assert np.all(np.abs(linear - nonlinear) < 1e-4)


def compare_hessian(problem, A, B, states_0, controls_0, states_delta, controls_delta, hessian):

    A_nonlinear, B_nonlinear = problem.calculate_statespace(states_delta + states_0, controls_delta + controls_0)
    dA_nonlinear = A_nonlinear - A
    dB_nonlinear = B_nonlinear - B

    dA_linear = np.tensordot(hessian[0][0], np.expand_dims(states_delta, 0).T, 1).squeeze() + \
        np.tensordot(hessian[0][1], np.expand_dims(controls_delta, 0).T, 1).squeeze()

    dB_linear = np.tensordot(hessian[1][0], np.expand_dims(states_delta, 0).T, 1).squeeze() + \
        np.tensordot(hessian[1][1], np.expand_dims(controls_delta, 0).T, 1).squeeze()

    assert np.allclose(dA_linear, dA_nonlinear, atol=5e-5)
    assert np.allclose(dB_linear, dB_nonlinear, atol=5e-5)


@pytest.mark.parametrize("states_0, controls_0, states_delta, controls_delta", cases)
def test_gradients(problem, states_0, controls_0, states_delta, controls_delta):

    # Process
    derivatives_0 = problem.derivatives(states_0, controls_0)
    A, B = problem.calculate_statespace(states_0, controls_0)
    hessian = problem.calculate_hessian(states_0, controls_0)

    # Tests
    compare_jacobian(problem, A, B, states_0, controls_0, states_delta, controls_delta, derivatives_0)
    compare_hessian(problem, A, B, states_0, controls_0, states_delta, controls_delta, hessian)

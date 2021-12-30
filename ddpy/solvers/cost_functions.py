import jax
import jax.numpy as np


class cost_function():

    def calculate_cost(self, x, u):
        pass

    def calculate_cost_hessian(self, x, u):
        return jax.hessian(self.calculate_cost, (x, u))

    def calculate_cost_jacobian(self, x, u):
        # For quadratic cost function, hessian is just the matrix
        return jax.jacfwd(self.calculate_cost, (x, u))


class quadratic_cost_function(cost_function):

    def __init__(self, g_xx, g_xu, g_ux, g_uu, g_x, g_u):

        if g_xx.shape[0] != g_xx.shape[1]:
            raise ValueError('g_xx must have shape (N_x, N_x)')
        if g_uu.shape[0] != g_uu.shape[1]:
            raise ValueError('g_uu must have  shape (N_u, N_u)')
        if g_xu.shape != (g_xx.shape[0], g_uu.shape[0]):
            raise ValueError('g_xu must have  shape (N_x, N_u)')
        if g_ux.shape != (g_uu.shape[0], g_xx.shape[0]):
            raise ValueError('g_ux must have  shape (N_u, N_x)')
        if g_x.shape != (1, g_xx.shape[0]):
            raise ValueError('g_x must have  shape (N_x, 1)')
        if g_u.shape != (1, g_uu.shape[0]):
            raise ValueError('g_u must have  shape (N_u, 1)')

        self.g_xx = g_xx
        self.g_xu = g_xu
        self.g_ux = g_ux
        self.g_uu = g_uu
        self.g_x = g_x
        self.g_u = g_u

        self.calculate_g_u = jax.jit(self._calculate_g_u)
        self.calculate_g_x = jax.jit(self._calculate_g_x)
        self.calculate_quadratic_cost = jax.jit(self._calculate_quadratic_cost)
        self.calculate_linear_cost = jax.jit(self._calculate_linear_cost)
        self.calculate_cost = jax.jit(self._calculate_cost)
        self.calculate_g_u_batch = jax.vmap(self.calculate_g_u, in_axes=(0, 0))
        self.calculate_g_x_batch = jax.vmap(self.calculate_g_x, in_axes=(0, 0))
        self.calculate_cost_batch = jax.vmap(self.calculate_cost, in_axes=(0, 0))

    def _calculate_quadratic_cost(self, x, u):

        return (np.matmul(x.T, np.matmul(self.g_xx, x)) +
                2 * np.matmul(x.T, np.matmul(self.g_xu, u)) +
                np.matmul(u.T, np.matmul(self.g_uu, u))) / 2

    def _calculate_linear_cost(self, x, u):
        return np.matmul(self.g_x, x) + np.matmul(self.g_u, u)

    def _calculate_cost(self, x, u):
        return self.calculate_quadratic_cost(x, u) + self.calculate_linear_cost(x, u)

    def _calculate_g_x(self, x, u):
        return np.matmul(self.g_xx, x).T + np.matmul(u.T, self.g_ux) + self.g_x

    def _calculate_g_u(self, x, u):
        return np.matmul(self.g_ux, x) + np.matmul(self.g_uu, u) + self.g_u

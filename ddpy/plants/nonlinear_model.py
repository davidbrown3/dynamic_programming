import jax


class NonLinearModel():

    def __init__(self, dt, random_i):
        self.dt = dt
        self._random_key = jax.random.PRNGKey(random_i)
        self._derivatives = jax.jit(self.derivatives)
        self.step = jax.jit(self._step)
        self.calculate_statespace = jax.jacfwd(self._derivatives, [0, 1])
        self.calculate_statespace_discrete = jax.jacfwd(self.step, [0, 1])
        self.calculate_statespace_discrete_batch = jax.vmap(self.calculate_statespace_discrete, in_axes=(0, 0))
        self._calculate_hessian_discrete = jax.hessian(self.step, [0, 1])
        self.calculate_hessian_discrete = jax.jit(self._calculate_hessian_discrete)
        self.calculate_hessian_discrete_batch = jax.vmap(self.calculate_hessian_discrete, in_axes=(0, 0))
        self._calculate_hessian = jax.hessian(self._derivatives, [0, 1])
        self.calculate_hessian = jax.jit(self._calculate_hessian)
        self.calculate_hessian_batch = jax.vmap(self.calculate_hessian, in_axes=(0, 0))

    @ property
    def N_x(self):
        return 0

    @ property
    def N_u(self):
        return 0

    def _step(self, x, u):
        dx_dt = self._derivatives(x, u)
        return x + dx_dt * self.dt

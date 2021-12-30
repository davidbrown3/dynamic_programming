import jax
import jax.numpy as np
import tqdm
from jax.config import config

config.update("jax_enable_x64", True)


class DifferentialDynamicProgramming:

    def __init__(self, plant, running_cost_function, terminal_cost_function, order=1, control_constraints=False, debug=False):

        self.debug = debug
        self.plant = plant
        self.order = order
        self.control_constraints = control_constraints
        self.running_cost_function = running_cost_function
        self.terminal_cost_function = terminal_cost_function

        self._line_alpha = [10.0**x for x in np.linspace(0, -15, 16)]
        self._regularisation_min = 1e-9
        self._regularisation = 1.0
        self._regularisation_delta_min = 2.0

        # Dispatching
        if self.debug:

            self._forward_pass_inner = self.forward_pass_inner
            self._forward_pass_prescribed = self.forward_pass_prescribed
            self._forward_pass_prescribed_inner = self.forward_pass_prescribed_inner
            self._forward_pass = self.forward_pass
            self._backward_pass = self.backward_pass
            self._check_positive_definite = self.check_positive_definite
            self._calculate_R_partials = self.calculate_R_partials
            self._calculate_Q_partials_1st = self.calculate_Q_partials_1st
            self._calculate_Q_partials_2nd = self.calculate_Q_partials_2nd
            if self.control_constraints:
                pass
            else:
                self._calculate_control_gains = self.calculate_control_gains
            self._calculate_cost = self.calculate_cost
            if self.order == 1:
                self._backward_pass_loop = self.backward_pass_loop
                self._backward_pass_loop_jit_inner = None
            else:
                self._backward_pass_loop = self.backward_pass_loop
                self._backward_pass_loop_jit_inner = None
        else:
            self._forward_pass_inner = jax.jit(self.forward_pass_inner)
            self._forward_pass_prescribed = jax.jit(self.forward_pass_prescribed)
            self._forward_pass_prescribed_inner = jax.jit(self.forward_pass_prescribed_inner)
            self._forward_pass = jax.jit(self.forward_pass)
            self._backward_pass = jax.jit(self.backward_pass)
            self._check_positive_definite = jax.jit(self.check_positive_definite)
            self._calculate_R_partials = jax.jit(self.calculate_R_partials)
            self._calculate_Q_partials_1st = jax.jit(self.calculate_Q_partials_1st)
            self._calculate_Q_partials_2nd = jax.jit(self.calculate_Q_partials_2nd)
            if self.control_constraints:
                pass
            else:
                self._calculate_control_gains = jax.jit(self.calculate_control_gains)
            self._calculate_cost = jax.jit(self.calculate_cost)
            if self.order == 1:
                self._backward_pass_loop = jax.jit(self.backward_pass_loop_1st_jit)
                self._backward_pass_loop_jit_inner = jax.jit(self.backward_pass_loop_1st_jit_inner)
            else:
                self._backward_pass_loop = jax.jit(self.backward_pass_loop_2nd_jit)
                self._backward_pass_loop_jit_inner = jax.jit(self.backward_pass_loop_2nd_jit_inner)

    def check_positive_definite(self, X):
        return np.any(np.isnan(np.linalg.cholesky(X)))

    def increase_regularisation(self, regularisation, regularisation_delta):
        regularisation_delta_ = np.maximum(self._regularisation_delta_min, self._regularisation_delta_min * regularisation_delta)
        regularisation_ = np.maximum(self._regularisation_min, regularisation * regularisation_delta)
        return regularisation_, regularisation_delta_

    def decrease_regularisation(self, regularisation, regularisation_delta):
        regularisation_delta_ = np.minimum(1.0 / self._regularisation_delta_min, regularisation_delta / self._regularisation_delta_min)

        if regularisation * regularisation_delta > regularisation_delta:
            regularisation_ = regularisation * regularisation_delta
        else:
            regularisation_ = 0.0

        return regularisation_, regularisation_delta_

    def solve(self, states_initial, time_total, controls_initial=None, convergence=1e-4, iterations=300):

        N_steps = int(time_total / self.plant.dt)
        xs = np.zeros([N_steps, self.plant.N_x])
        us = np.zeros([N_steps, self.plant.N_u])

        if controls_initial is None:

            beta = np.zeros([self.plant.N_u, self.plant.N_x])
            betas = np.tile(beta, (N_steps, 1, 1))
            alphas = np.zeros([N_steps, self.plant.N_u])

            xs, us = self._forward_pass(np.squeeze(states_initial), xs, us, betas, alphas, line_alpha=1.0)

        else:
            if callable(controls_initial):
                # NOTE: Can not JIT this function since passing in a function
                xs, us = self.forward_pass_law(xi=np.squeeze(states_initial), xs=xs, us=us, control_law=controls_initial)
            else:
                xs, us = self._forward_pass_prescribed(xi=np.squeeze(states_initial), xs=xs, us=controls_initial)

        regularisation = self._regularisation
        regularisation_delta = self._regularisation_delta_min

        cost = self._calculate_cost(xs, us)

        costs = []
        with tqdm.trange(iterations) as t:
            for _ in t:
                while True:
                    betas, alphas, bNotPositiveDefinite, dcost_expected_d_line_alpha, d2cost_expected_d_line_alpha_2 = \
                        self._backward_pass(xs=xs, us=us, regularisation=regularisation)

                    if bNotPositiveDefinite:
                        regularisation, regularisation_delta = self.increase_regularisation(regularisation, regularisation_delta)
                    else:
                        regularisation_success = regularisation
                        regularisation, regularisation_delta = self.decrease_regularisation(regularisation, regularisation_delta)
                        break

                for i, line_alpha in enumerate(self._line_alpha):
                    dcost_expected = self.calculate_expected_dcost(line_alpha, dcost_expected_d_line_alpha, d2cost_expected_d_line_alpha_2)
                    _xs, _us = self._forward_pass(xi=np.squeeze(states_initial), xs=xs, us=us, betas=betas, alphas=alphas, line_alpha=line_alpha)
                    cost_new = self._calculate_cost(_xs, _us)

                    if (cost_new - cost) / dcost_expected >= 1e-1:
                        cost = cost_new
                        bcost_reduction = True
                        break
                    else:
                        bcost_reduction = False

                if bcost_reduction:
                    xs = _xs
                    us = _us
                    costs.append(cost)
                    t.set_postfix(
                        cost=f'{cost: .3f}',
                        regularisation=f'{regularisation_success: .6f}',
                        line_alpha=f'{line_alpha: .2f}'
                    )
                else:
                    break

        return xs, us, costs

    @staticmethod
    @jax.jit
    def calculate_expected_dcost(line_alpha, dcost_expected_d_line_alpha, d2cost_expected_d_line_alpha_2):
        return line_alpha * (dcost_expected_d_line_alpha + line_alpha * d2cost_expected_d_line_alpha_2)

    def calculate_cost(self, xs, us):
        return np.sum(self.running_cost_function.calculate_cost_batch(xs, us)) + \
            np.sum(self.terminal_cost_function.calculate_cost(xs[-1], us[-1]))

    def estimated_delta_cost_coefficients(self, alpha, Q_u, Q_uu):
        dcost_expected_d_line_alpha = np.matmul(Q_u, alpha)
        d2cost_expected_d_line_alpha_2 = 0.5 * np.matmul(alpha.T, np.matmul(Q_uu, alpha))
        return np.squeeze(dcost_expected_d_line_alpha), np.squeeze(d2cost_expected_d_line_alpha_2)

    def backward_pass(self, xs, us, regularisation):

        xs = np.flip(xs, axis=0)
        us = np.flip(us, axis=0)

        T_xs, T_us = self.plant.calculate_statespace_discrete_batch(xs, us)

        g_xs = self.running_cost_function.calculate_g_x_batch(xs, us)
        g_us = self.running_cost_function.calculate_g_u_batch(xs, us)
        R_xx = self.terminal_cost_function.g_xx
        R_x = self.terminal_cost_function.calculate_g_x(xs[0], us[0])

        betas, alphas, bNotPositiveDefinite, expected_delta_cost_1st, expected_delta_cost_2nd = \
            self._backward_pass_loop(xs, us, R_x, R_xx, T_xs, T_us, g_xs, g_us, regularisation)

        return betas, alphas, bNotPositiveDefinite, expected_delta_cost_1st, expected_delta_cost_2nd

    def backward_pass_loop(self, xs, us, R_x, R_xx, T_xs, T_us, g_xs, g_us, regularisation):

        if self.order == 2:
            T_hessians = self.plant.calculate_hessian_discrete_batch(xs, us)
            T_xxs = T_hessians[0][0]
            T_xus = T_hessians[0][1]
            T_uxs = T_hessians[1][0]
            T_uus = T_hessians[1][1]

        betas = np.empty([xs.shape[0], us.shape[1], xs.shape[1]])
        alphas = np.empty([us.shape[0], us.shape[1]])

        bNotPositiveDefinite = False
        dcost_expected_d_line_alpha = 0.0
        d2cost_expected_d_line_alpha_2 = 0.0

        for i in range(xs.shape[0]):

            if not bNotPositiveDefinite:

                Q_xx, Q_uu, Q_xu, Q_ux, Q_x, Q_u, Q_uu_reg, Q_ux_reg = self._calculate_Q_partials_1st(
                    R_x=R_x,
                    R_xx=R_xx,
                    T_x=T_xs[jax.ops.index[i, :, :]],
                    T_u=T_us[jax.ops.index[i, :, :]],
                    g_x=g_xs[jax.ops.index[i, :]],
                    g_u=g_us[jax.ops.index[i, :]],
                    g_xx=self.running_cost_function.g_xx,
                    g_uu=self.running_cost_function.g_uu,
                    g_xu=self.running_cost_function.g_xu,
                    g_ux=self.running_cost_function.g_ux,
                    regularisation=regularisation
                )

                if self.order == 2:

                    dQ_xx, dQ_uu, dQ_xu, dQ_ux = self._calculate_Q_partials_2nd(
                        R_x=R_x,
                        T_xx=T_xxs[jax.ops.index[i, :, :, :]],
                        T_xu=T_xus[jax.ops.index[i, :, :, :]],
                        T_ux=T_uxs[jax.ops.index[i, :, :, :]],
                        T_uu=T_uus[jax.ops.index[i, :, :, :]]
                    )

                    Q_xx += dQ_xx
                    Q_uu += dQ_uu
                    Q_uu_reg += dQ_uu
                    Q_xu += dQ_xu
                    Q_ux += dQ_ux
                    Q_ux_reg += dQ_ux

                beta, alpha = self._calculate_control_gains(Q_ux_reg, Q_uu_reg, Q_u)

                # Generates a nan if positive definite
                bNotPositiveDefinite = jax.lax.bitwise_or(bNotPositiveDefinite, self._check_positive_definite(Q_uu_reg))

                # TODO: Replace control gain solution with 2 stage cholesky
                # Since we are already using a cholesky factorisation to check positive definiteness
                # https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.cho_factor.html#jax.scipy.linalg.cho_factor
                # https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.cho_solve.html#jax.scipy.linalg.cho_solve

                R_x, R_xx = self._calculate_R_partials(beta, alpha, Q_xx, Q_uu, Q_xu, Q_ux, Q_x, Q_u)

                betas = jax.ops.index_update(betas, jax.ops.index[i, :], beta)
                alphas = jax.ops.index_update(alphas, jax.ops.index[i, :], alpha.squeeze())

                dcost_expected_d_line_alpha_i, d2cost_expected_d_line_alpha_2_i = self.estimated_delta_cost_coefficients(alpha, Q_u, Q_uu)
                dcost_expected_d_line_alpha += dcost_expected_d_line_alpha_i
                d2cost_expected_d_line_alpha_2 += d2cost_expected_d_line_alpha_2_i

        return np.flip(betas, axis=0), np.flip(alphas, axis=0), bNotPositiveDefinite, dcost_expected_d_line_alpha, d2cost_expected_d_line_alpha_2

    def backward_pass_loop_1st_jit(self, xs, us, R_x, R_xx, T_xs, T_us, g_xs, g_us, regularisation):

        betas = np.empty([xs.shape[0], us.shape[1], xs.shape[1]])
        alphas = np.empty([us.shape[0], us.shape[1]])
        bNotPositiveDefinite = False
        dcost_expected_d_line_alpha = 0.0
        d2cost_expected_d_line_alpha_2 = 0.0

        data = (R_x, R_xx, T_xs, T_us, g_xs, g_us, betas, alphas, bNotPositiveDefinite, regularisation,
                dcost_expected_d_line_alpha, d2cost_expected_d_line_alpha_2)
        data = jax.lax.fori_loop(0, xs.shape[0], self._backward_pass_loop_jit_inner, data)

        return np.flip(data[6], axis=0), np.flip(data[7], axis=0), data[8], data[10], data[11]

    def backward_pass_loop_2nd_jit(self, xs, us, R_x, R_xx, T_xs, T_us, g_xs, g_us, regularisation):

        T_hessians = self.plant.calculate_hessian_discrete_batch(xs, us)
        T_xxs = T_hessians[0][0]
        T_xus = T_hessians[0][1]
        T_uxs = T_hessians[1][0]
        T_uus = T_hessians[1][1]

        betas = np.empty([xs.shape[0], us.shape[1], xs.shape[1]])
        alphas = np.empty([us.shape[0], us.shape[1]])
        bNotPositiveDefinite = False
        dcost_expected_d_line_alpha = 0.0
        d2cost_expected_d_line_alpha_2 = 0.0

        data = (R_x, R_xx, T_xs, T_us, T_xxs, T_xus, T_uxs, T_uus, g_xs, g_us, betas, alphas, bNotPositiveDefinite,
                regularisation, dcost_expected_d_line_alpha, d2cost_expected_d_line_alpha_2)
        data = jax.lax.fori_loop(0, xs.shape[0], self._backward_pass_loop_jit_inner, data)

        return np.flip(data[10], axis=0), np.flip(data[11], axis=0), data[12], data[14], data[15]

    def backward_pass_loop_1st_jit_inner(self, i, data):

        R_x, R_xx, T_xs, T_us, g_xs, g_us, betas, alphas, bNotPositiveDefinite, regularisation, \
            dcost_expected_d_line_alpha, d2cost_expected_d_line_alpha_2 = data

        # TODO: Skip execution of code when bNotPositiveDefinite True

        Q_xx, Q_uu, Q_xu, Q_ux, Q_x, Q_u, Q_uu_reg, Q_ux_reg = self._calculate_Q_partials_1st(
            R_x=R_x,
            R_xx=R_xx,
            T_x=T_xs[jax.ops.index[i, :, :]],
            T_u=T_us[jax.ops.index[i, :, :]],
            g_x=g_xs[jax.ops.index[i, :]],
            g_u=g_us[jax.ops.index[i, :]],
            g_xx=self.running_cost_function.g_xx,
            g_uu=self.running_cost_function.g_uu,
            g_xu=self.running_cost_function.g_xu,
            g_ux=self.running_cost_function.g_ux,
            regularisation=regularisation
        )

        beta, alpha = self._calculate_control_gains(Q_ux_reg, Q_uu_reg, Q_u)

        # Generates a nan if positive definite
        bNotPositiveDefinite = jax.lax.bitwise_or(bNotPositiveDefinite, self._check_positive_definite(Q_uu_reg))

        R_x, R_xx = self._calculate_R_partials(beta, alpha, Q_xx, Q_uu, Q_xu, Q_ux, Q_x, Q_u)

        betas = jax.ops.index_update(betas, jax.ops.index[i, :], beta)
        alphas = jax.ops.index_update(alphas, jax.ops.index[i, :], alpha.squeeze())

        dcost_expected_d_line_alpha_i, d2cost_expected_d_line_alpha_2_i = self.estimated_delta_cost_coefficients(alpha, Q_u, Q_uu)
        dcost_expected_d_line_alpha += dcost_expected_d_line_alpha_i
        d2cost_expected_d_line_alpha_2 += d2cost_expected_d_line_alpha_2_i

        return (R_x, R_xx, T_xs, T_us, g_xs, g_us, betas, alphas, bNotPositiveDefinite, regularisation,
                dcost_expected_d_line_alpha, d2cost_expected_d_line_alpha_2)

    def backward_pass_loop_2nd_jit_inner(self, i, data):

        R_x, R_xx, T_xs, T_us, T_xxs, T_xus, T_uxs, T_uus, g_xs, g_us, betas, alphas, bNotPositiveDefinite, regularisation, \
            dcost_expected_d_line_alpha, d2cost_expected_d_line_alpha_2 = data

        # TODO: Skip execution of code when bNotPositiveDefinite True

        Q_xx_1st, Q_uu_1st, Q_xu_1st, Q_ux_1st, Q_x, Q_u, Q_uu_1st_reg, Q_ux_1st_reg = self._calculate_Q_partials_1st(
            R_x=R_x,
            R_xx=R_xx,
            T_x=T_xs[jax.ops.index[i, :, :]],
            T_u=T_us[jax.ops.index[i, :, :]],
            g_x=g_xs[jax.ops.index[i, :]],
            g_u=g_us[jax.ops.index[i, :]],
            g_xx=self.running_cost_function.g_xx,
            g_uu=self.running_cost_function.g_uu,
            g_xu=self.running_cost_function.g_xu,
            g_ux=self.running_cost_function.g_ux,
            regularisation=regularisation
        )

        Q_xx_2nd, Q_uu_2nd, Q_xu_2nd, Q_ux_2nd = self._calculate_Q_partials_2nd(
            R_x=R_x,
            T_xx=T_xxs[jax.ops.index[i, :, :, :]],
            T_xu=T_xus[jax.ops.index[i, :, :, :]],
            T_ux=T_uxs[jax.ops.index[i, :, :, :]],
            T_uu=T_uus[jax.ops.index[i, :, :, :]]
        )

        Q_xx = Q_xx_1st + Q_xx_2nd
        Q_xu = Q_xu_1st + Q_xu_2nd
        Q_ux = Q_ux_1st + Q_ux_2nd
        Q_ux_reg = Q_ux_1st_reg + Q_ux_2nd
        Q_uu = Q_uu_1st + Q_uu_2nd
        Q_uu_reg = Q_uu_1st_reg + Q_uu_2nd

        beta, alpha = self._calculate_control_gains(Q_ux_reg, Q_uu_reg, Q_u)

        # Generates a nan if positive definite
        bNotPositiveDefinite = jax.lax.bitwise_or(bNotPositiveDefinite, self._check_positive_definite(Q_uu_reg))

        R_x, R_xx = self._calculate_R_partials(beta, alpha, Q_xx, Q_uu, Q_xu, Q_ux, Q_x, Q_u)

        betas = jax.ops.index_update(betas, jax.ops.index[i, :], beta)
        alphas = jax.ops.index_update(alphas, jax.ops.index[i, :], alpha.squeeze())

        dcost_expected_d_line_alpha_i, d2cost_expected_d_line_alpha_2_i = self.estimated_delta_cost_coefficients(alpha, Q_u, Q_uu)
        dcost_expected_d_line_alpha += dcost_expected_d_line_alpha_i
        d2cost_expected_d_line_alpha_2 += d2cost_expected_d_line_alpha_2_i

        return (R_x, R_xx, T_xs, T_us, T_xxs, T_xus, T_uxs, T_uus, g_xs, g_us, betas, alphas, bNotPositiveDefinite, regularisation,
                dcost_expected_d_line_alpha, d2cost_expected_d_line_alpha_2)

    @ staticmethod
    def calculate_R_partials(beta, alpha, Q_xx, Q_uu, Q_xu, Q_ux, Q_x, Q_u):

        R_xx = Q_xx + np.matmul(Q_xu, beta) + np.matmul(Q_xu, beta) + np.matmul(beta.T, np.matmul(Q_uu, beta))
        R_x = Q_x + np.matmul(alpha.T, Q_ux) + np.matmul(Q_u, beta) + np.matmul(alpha.T, np.matmul(Q_uu, beta))

        return R_x, R_xx

    @ staticmethod
    def calculate_Q_partials_1st(R_x, R_xx, T_x, T_u, g_x, g_u, g_xx, g_uu, g_xu, g_ux, regularisation):

        reg_states = regularisation * np.eye(len(R_xx))
        reg_control = regularisation * np.eye(len(g_uu))

        Q_xx = g_xx + np.matmul(T_x.T, np.matmul(R_xx, T_x))
        Q_uu_reg = g_uu + np.matmul(T_u.T, np.matmul(R_xx+reg_states, T_u)) + reg_control
        Q_ux_reg = g_ux + np.matmul(T_u.T, np.matmul(R_xx+reg_states, T_x))
        Q_uu = g_uu + np.matmul(T_u.T, np.matmul(R_xx, T_u))
        Q_xu = g_xu + np.matmul(T_x.T, np.matmul(R_xx, T_u))
        Q_ux = g_ux + np.matmul(T_u.T, np.matmul(R_xx, T_x))
        Q_x = g_x + np.matmul(R_x, T_x)
        Q_u = g_u + np.matmul(R_x, T_u)

        return Q_xx, Q_uu, Q_xu, Q_ux, Q_x, Q_u, Q_uu_reg, Q_ux_reg

    @ staticmethod
    def calculate_Q_partials_2nd(R_x, T_xx, T_xu, T_ux, T_uu):

        dQ_xx = np.tensordot(R_x, T_xx, 1)[0, :, :]
        dQ_uu = np.tensordot(R_x, T_uu, 1)[0, :, :]
        dQ_xu = np.tensordot(R_x, T_xu, 1)[0, :, :]
        dQ_ux = np.tensordot(R_x, T_ux, 1)[0, :, :]

        return dQ_xx, dQ_uu, dQ_xu, dQ_ux

    @ staticmethod
    def calculate_control_gains(Q_ux, Q_uu, Q_u):

        beta = np.linalg.solve(-Q_uu, Q_ux)
        alpha = np.linalg.solve(-Q_uu, Q_u.T)

        return beta, alpha

    def forward_pass_law(self, xi, xs, us, control_law):
        x_ = xi
        data = (xs, us, x_)
        def fun(i, data): return self.forward_pass_law_inner(i, data, control_law)
        data = jax.lax.fori_loop(0, xs.shape[0], fun, data)

        return data[0], data[1]

    def forward_pass_law_inner(self, i, data, control_law):

        xs, us, x_ = data
        xs = jax.ops.index_update(xs, jax.ops.index[i, :], x_)
        u_ = control_law(x_)
        us = jax.ops.index_update(us, jax.ops.index[i, :], u_)
        x_ = self.plant.step(x_, u_)

        return (xs, us, x_)

    def forward_pass_prescribed(self, xi, xs, us):
        x_ = xi
        data = (xs, us, x_)
        data = jax.lax.fori_loop(0, xs.shape[0], self._forward_pass_prescribed_inner, data)

        return data[0], data[1]

    def forward_pass_prescribed_inner(self, i, data):

        xs, us, x_ = data
        xs = jax.ops.index_update(xs, jax.ops.index[i, :], x_)
        u_ = us[i]
        x_ = self.plant.step(x_, u_)

        return (xs, us, x_)

    def forward_pass(self, xi, xs, us, betas, alphas, line_alpha=1e-2):
        x_ = xi
        data = (xs, us, betas, alphas, x_, line_alpha)
        data = jax.lax.fori_loop(0, xs.shape[0], self._forward_pass_inner, data)

        return data[0], data[1]

    def forward_pass_inner(self, i, data):

        xs, us, betas, alphas, x_, line_alpha = data
        dx = x_ - xs[i]
        xs = jax.ops.index_update(xs, jax.ops.index[i, :], x_)
        du = np.matmul(betas[i], dx) + line_alpha * alphas[i]
        u_ = us[i] + du
        us = jax.ops.index_update(us, jax.ops.index[i, :], u_)
        x_ = self.plant.step(x_, u_)

        return (xs, us, betas, alphas, x_, line_alpha)

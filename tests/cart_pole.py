import jax.numpy as np
import jax
from jax_ddp.plants import NonLinearModel


class CartPole(NonLinearModel):
    '''
    Example documented at:
    https://coneural.org/florian/papers/05_cart_pole.pdf
    '''

    # u = force
    # States = position, velocity, angular_position, angular_velocity

    def __init__(self, dt, random_i=0):

        self.mass_pendulum = 5
        self.mass_cart = 20
        self.length = 1
        self.gravity = 9.81
        self.angular_friction = 0.1
        self.random_key = jax.random.PRNGKey(random_i)

        super().__init__(dt, random_i)

    @property
    def N_x(self):
        return 4

    @property
    def N_u(self):
        return 1

    @property
    def mass_total(self):
        return self.mass_pendulum + self.mass_cart

    def random_control_law(self, x):
        force = jax.random.normal(self._random_key) * 0.01
        _, self._random_key = jax.random.split(self._random_key)
        return force,

    def derivatives(self, x, u):

        angular_position = x[2]
        angular_velocity = x[3]
        velocity = x[1]
        force = u[0]

        angular_position_sin = np.sin(angular_position)
        angular_position_cos = np.cos(angular_position)

        angular_acceleration = (
            self.gravity * angular_position_sin + angular_position_cos * (
                (-force - self.mass_pendulum * self.length * angular_velocity**2 * angular_position_sin) / self.mass_total
            )
        ) / (
            self.length * (
                4/3 - self.mass_pendulum * angular_position_cos**2 / self.mass_total
            )
        ) - angular_velocity * self.angular_friction

        acceleration = (
            force + self.mass_pendulum * self.length * (
                angular_velocity**2 * angular_position_sin - angular_acceleration * angular_position_cos
            )
        ) / self.mass_total

        derivatives = (
            velocity,
            acceleration,
            angular_velocity,
            angular_acceleration
        )

        return np.stack(derivatives)

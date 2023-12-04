import math

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from scipy.signal import convolve2d

NUM_ITERS = 40
NEIGHBORS = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
NEIGHBORS_U = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
NEIGHBORS_V = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
DIFFUSION = 0.1
FRICTION = 0.8

U_FEILD = 0
V_FEILD = 1
S_FEILD = 2


class Fluid:
    def __init__(self, x_size, y_size) -> None:
        self.h = 1 / 100
        self.x_size = x_size
        self.y_size = y_size
        self.size = x_size * y_size

        self.u = np.zeros(shape=(self.x_size, self.y_size), dtype=np.float32)
        self.v = np.zeros(shape=(self.x_size, self.y_size), dtype=np.float32)
        self.smoke = np.zeros(shape=(self.x_size, self.y_size), dtype=np.float32)
        self.pressure = np.zeros(shape=(self.x_size, self.y_size), dtype=np.float32)
        self.div = np.zeros(shape=(self.x_size, self.y_size), dtype=np.float32)

        self.normal_u = np.zeros(shape=(self.x_size, self.y_size), dtype=np.float32)
        self.normal_v = np.zeros(shape=(self.x_size, self.y_size), dtype=np.float32)

        self.normal_u[:2, :] = 1
        self.normal_u[:, :2] = 0
        self.normal_u[-2:, :] = -1
        self.normal_u[:, -2:] = 0

        self.normal_v[:2, :] = 0
        self.normal_v[:, :2] = -1
        self.normal_v[-2:, :] = 0
        self.normal_v[:, -2:] = 1

        self.normal_u[40, 40:60] = -1
        self.normal_u[50, 40:60] = 1
        self.normal_u[40:50, 40] = 0
        self.normal_u[40:50, 60] = 0

        self.normal_v[40, 40:60] = 0
        self.normal_v[50, 40:60] = 0
        self.normal_v[40:50, 40] = 1
        self.normal_v[40:50, 60] = -1

        self.frame = 0

    def step(self, dt: float):
        dt = 0.01

        self.u[20, 10:60] = 2.0

        if self.frame <= 500:
            self.smoke[20, 30:70] = 1.0

        self.projection(dt)
        self.diffusion(dt)
        advect_vel(
            self.u,
            self.v,
            self.u.copy(),
            self.v.copy(),
            self.smoke,
            dt,
            self.h,
            self.x,
            self.y,
        )

        calc_borders(
            self.u, self.v, self.normal_u, self.normal_v, self.x_size, self.y_size
        )

        advect_smoke(
            self.smoke,
            self.smoke.copy(),
            self.u,
            self.v,
            dt,
            self.h,
            self.x,
            self.y,
        )

        self.frame += 1

    @property
    def x(self) -> int:
        return self.x_size

    @property
    def y(self) -> int:
        return self.y_size

    @property
    def data_to_draw(self):
        return np.rot90(self.smoke)  # self.normal_u + self.normal_v

    def projection(self, dt: float):
        div = np.gradient(self.u, axis=0) + np.gradient(self.v, axis=1)

        pressure = np.zeros_like(self.smoke)
        for _ in range(NUM_ITERS):
            sum_pressure = convolve2d(pressure, NEIGHBORS, mode="same", boundary="fill")
            pressure = (sum_pressure - div) / 4

        self.pressure = pressure  # * 1.9

        self.u -= np.gradient(pressure, axis=0)
        self.v -= np.gradient(pressure, axis=1)

        self.u[0, :] = 0.0
        self.u[-1, :] = 0.0
        self.u[:, 0] = 0.0
        self.u[:, -1] = 0.0

        self.v[0, :] = 0.0
        self.v[-1, :] = 0.0
        self.v[:, 0] = 0.0
        self.v[:, -1] = 0.0

        self.u[40:50, 40:60] = 0
        self.v[40:50, 40:60] = 0

    def diffusion(self, dt: float):
        for _ in range(NUM_ITERS):
            sum_smoke = convolve2d(
                self.smoke, NEIGHBORS, mode="same", boundary="fill"
            ).astype(np.float32)
            self.smoke = (self.smoke + DIFFUSION * dt * sum_smoke) / (
                1 + 4 * DIFFUSION * dt
            )


@nb.njit(nb.float32(nb.float32[:, :], nb.int32, nb.int32, nb.int32))
def avg_v(v: np.ndarray, i: int, j: int, y_size: int) -> np.float32:
    return (v[(i - 1), j] + v[i, j] + v[(i - 1), j + 1] + v[i, j + 1]) * 0.25


@nb.njit(nb.float32(nb.float32[:, :], nb.int32, nb.int32, nb.int32))
def avg_u(u: np.ndarray, i: int, j: int, y_size: int) -> np.float32:
    return (u[i, j - 1] + u[i, j] + u[(i + 1), j - 1] + u[(i + 1), j]) * 0.25


@nb.njit(
    nb.float32(
        nb.float32[:, :],
        nb.float32[:, :],
        nb.float32[:, :],
        nb.float32,
        nb.float32,
        nb.int32,
        nb.float32,
        nb.int32,
        nb.int32,
    )
)
def sample_field(
    u: np.ndarray,
    v: np.ndarray,
    smoke: np.ndarray,
    x: float,
    y: float,
    field: int,
    h: float,
    x_size: int,
    y_size: int,
) -> np.float32:
    h1 = 1.0 / h
    h2 = 0.5 * h

    x = max(min(x, x_size * h), h)
    y = max(min(y, y_size * h), h)

    dx = dy = 0.0
    f = None

    if field == U_FEILD:
        f = u
        dy = h2
    elif field == V_FEILD:
        f = v
        dx = h2
    else:
        f = smoke
        dx = h2
        dy = h2

    x0 = min(math.floor((x - dx) * h1), x_size - 1)
    tx = ((x - dx) - x0 * h) * h1
    x1 = min(x0 + 1, x_size - 1)

    y0 = min(math.floor((y - dy) * h1), y_size - 1)
    ty = ((y - dy) - y0 * h) * h1
    y1 = min(y0 + 1, y_size - 1)

    sx = 1.0 - tx
    sy = 1.0 - ty

    return (
        sx * sy * f[x0, y0]
        + tx * sy * f[x1, y0]
        + tx * ty * f[x1, y1]
        + sx * ty * f[x0, y1]
    )


@nb.njit(
    (
        nb.float32[:, :],
        nb.float32[:, :],
        nb.float32[:, :],
        nb.float32[:, :],
        nb.float32[:, :],
        nb.float32,
        nb.float32,
        nb.int32,
        nb.int32,
    )
)
def advect_vel(
    u: np.ndarray,
    v: np.ndarray,
    new_u: np.ndarray,
    new_v: np.ndarray,
    smoke: np.ndarray,
    dt: float,
    h: float,
    x_size: int,
    y_size: int,
):
    h2 = 0.5 * h

    for i in range(1, x_size - 1):
        for j in range(1, y_size - 1):
            x = i * h
            y = j * h + h2

            u_scalar = u[i, j]
            v_scalar = avg_v(v, i, j, y_size)
            x -= dt * u_scalar
            y -= dt * v_scalar

            new_u[i, j] = sample_field(u, v, smoke, x, y, U_FEILD, h, x_size, y_size)

            x = i * h + h2
            y = j * h
            u_scalar = avg_u(u, i, j, y_size)
            v_scalar = v[i, j]
            x -= dt * u_scalar
            y -= dt * v_scalar

            new_v[i, j] = sample_field(u, v, smoke, x, y, V_FEILD, h, x_size, y_size)

    u[:] = new_u.copy()
    v[:] = new_v.copy()


@nb.njit(
    (
        nb.float32[:, :],
        nb.float32[:, :],
        nb.float32[:, :],
        nb.float32[:, :],
        nb.float32,
        nb.float32,
        nb.int32,
        nb.int32,
    )
)
def advect_smoke(
    smoke: np.ndarray,
    new_mass: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dt: float,
    h: float,
    x_size: int,
    y_size: int,
):
    h2 = 0.5 * h

    for i in range(1, x_size - 1):
        for j in range(1, y_size - 1):
            u_scalar = (u[i, j] + u[(i + 1), j]) * 0.5
            v_scalar = (v[i, j] + v[i, j + 1]) * 0.5
            x = i * h + h2 - dt * u_scalar
            y = j * h + h2 - dt * v_scalar

            val = sample_field(u, v, smoke, x, y, S_FEILD, h, x_size, y_size)
            new_mass[i, j] = val

    smoke[:] = new_mass.copy()


@nb.njit(nb.float32[:](nb.float32[:]))
def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


@nb.njit(nb.float32[:](nb.float32[:], nb.float32[:]))
def flip_vector(original_vector: np.ndarray, normal: np.ndarray):
    normalized_normal = normalize(normal)
    dot_product = np.dot(original_vector, normalized_normal)
    flipped_vector = original_vector - 2 * dot_product * normalized_normal
    return flipped_vector.astype(np.float32)


@nb.njit(
    (
        nb.float32[:, :],
        nb.float32[:, :],
        nb.float32[:, :],
        nb.float32[:, :],
        nb.int32,
        nb.int32,
    ),
)
def calc_borders(
    u: np.ndarray,
    v: np.ndarray,
    normal_u: np.ndarray,
    normal_v: np.ndarray,
    x_size: int,
    y_size: int,
):
    for i in range(0, x_size):
        for j in range(0, y_size):
            if normal_u[i, j] == 0 and normal_v[i, j] == 0:
                continue

            u_scalar, v_scalar = flip_vector(
                np.asarray([u[i, j], v[i, j]]),
                np.asarray([normal_u[i, j], normal_v[i, j]]),
            )
            u[i, j] = u_scalar * FRICTION
            v[i, j] = v_scalar * FRICTION

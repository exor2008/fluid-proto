import math
from enum import Enum

import numba as nb
import numpy as np
from matplotlib import pyplot as plt

GRAVITY = -9.81
NUM_ITERS = 20
DENSITY = 1000.0

U_FEILD = 0
V_FEILD = 1
S_FEILD = 2


class Fluid:
    def __init__(self, size_x, size_y) -> None:
        self.size_x = size_x
        self.size_y = size_y
        self.size = size_x * size_y
        self.h = 1 / 100

        self.pressure = np.zeros(shape=(self.size), dtype=np.float32)
        self.smoke = np.ones(shape=(self.size), dtype=np.float32)
        self.u = np.zeros(shape=(self.size), dtype=np.float32)
        self.v = np.zeros(shape=(self.size), dtype=np.float32)
        self.mass = np.ones(shape=(self.size), dtype=np.float32)
        self.obstacle_x = 0.0
        self.obstacle_y = 0.0

        self.smoke = self.smoke.reshape([self.size_x, self.size_y])
        self.smoke[0, :] = 0
        self.smoke[:, 0] = 0
        self.smoke[:, size_y - 1] = 0
        self.smoke = self.smoke.ravel().astype(np.float32)

        self.mass = self.mass.reshape([self.size_x, self.size_y])
        self.mass[0, 25:55] = 0.0
        self.mass = self.mass.ravel().astype(np.float32)

        self.u = self.u.reshape([self.size_x, self.size_y])
        self.u[1, :] = 2.0
        self.u = self.u.ravel().astype(np.float32)

        self.frame = 0

        self.set_obstacle(0.8, 0.5, True)

    def step(self, dt: float):
        # print(dt)

        if self.frame == 200:
            self.mass = self.mass.reshape([self.size_x, self.size_y])
            self.mass[0, 25:55] = 1.0
            self.mass = self.mass.ravel().astype(np.float32)

        dt = 0.01
        # gravity(self.smoke, self.v, dt, GRAVITY, self.x, self.y)
        self.pressure = np.zeros_like(self.pressure, dtype=np.float32)

        projection(
            self.smoke,
            self.u,
            self.v,
            self.pressure,
            NUM_ITERS,
            dt,
            self.h,
            self.x,
            self.y,
        )

        self.extrapolate()

        advect_vel(
            self.u,
            self.v,
            self.u.copy(),
            self.v.copy(),
            self.smoke,
            self.mass,
            dt,
            self.h,
            self.x,
            self.y,
        )

        advect_smoke(
            self.smoke,
            self.mass,
            self.mass.copy(),
            self.u,
            self.v,
            dt,
            self.h,
            self.x,
            self.y,
        )

        self.frame += 1

    def set_obstacle(self, x, y, reset):
        vx = vy = 0.0
        if not reset:
            vx = (x - self.obstacle_x) / (1 / 100)
            vy = (y - self.obstacle_y) / (1 / 100)

        self.obstacle_x = x
        self.obstacle_y = y

        r = 0.15

        for i in range(1, self.x - 2):
            for j in range(1, self.y - 2):
                self.smoke[i * self.y + j] = 1

                dx = (i + 0.5) * self.h - x
                dy = (j + 0.5) * self.h - y

                if dx * dx + dy * dy < r * r:
                    self.smoke[i * self.y + j] = 0
                    self.mass[i * self.y + j] = 1
                    self.u[i * self.y + j] = vx
                    self.u[(i + 1) * self.y + j] = vx
                    self.v[i * self.y + j] = vy
                    self.u[i * self.y + (j + 1)] = vy

    def extrapolate(self):
        self.u = self.u.reshape([self.x, self.y])
        self.v = self.v.reshape([self.x, self.y])

        self.u[:, 0] = self.u[:, 1]
        self.u[:, -1] = self.u[:, -2]

        self.v[0, :] = self.v[1, :]
        self.v[-1, :] = self.v[-2, :]

        self.u = self.u.ravel()
        self.v = self.v.ravel()

    @property
    def x(self) -> int:
        return self.size_x

    @property
    def y(self) -> int:
        return self.size_y

    @property
    def data_to_draw(self):
        # print(self.u.max(), self.v.min())
        return np.rot90(self.mass.reshape([self.x, self.y]))


@nb.njit((nb.float32[:], nb.float32[:], nb.float32, nb.float32, nb.int32, nb.int32))
def gravity(
    smoke: np.ndarray,
    v: np.ndarray,
    dt: float,
    gravity: float,
    x_size: int,
    y_size: int,
):
    for x in range(x_size):
        for y in range(y_size - 1):
            if smoke[x * y_size + y] != 0.0 and smoke[x * y_size + (y - 1)] != 0.0:
                v[x * y_size + y] += gravity * dt


@nb.njit(
    (
        nb.float32[:],
        nb.float32[:],
        nb.float32[:],
        nb.float32[:],
        nb.int32,
        nb.float32,
        nb.float32,
        nb.int32,
        nb.int32,
    )
)
def projection(
    smoke: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    pressure: np.ndarray,
    num_iters: int,
    dt: float,
    h: float,
    x_size: int,
    y_size: int,
):
    cp = DENSITY * h / dt
    # cp /= 10

    for _ in range(num_iters):
        for x in range(1, x_size - 1):
            for y in range(1, y_size - 1):
                if smoke[x * y_size + y] == 0.0:
                    continue

                sx0 = smoke[(x - 1) * y_size + y]
                sx1 = smoke[(x + 1) * y_size + y]
                sy0 = smoke[x * y_size + (y - 1)]
                sy1 = smoke[x * y_size + (y + 1)]

                s = sum([sx0, sx1, sy0, sy1])

                if s == 0.0:
                    continue

                div = (
                    u[(x + 1) * y_size + y]
                    - u[x * y_size + y]
                    + v[x * y_size + (y + 1)]
                    - v[x * y_size + y]
                )

                pressure_scalar = -div / s
                pressure_scalar *= 1.9
                pressure[x * y_size + y] += cp * pressure_scalar

                u[x * y_size + y] -= sx0 * pressure_scalar
                u[(x + 1) * y_size + y] += sx1 * pressure_scalar
                v[x * y_size + y] -= sy0 * pressure_scalar
                v[x * y_size + (y + 1)] += sy1 * pressure_scalar
                # print(self.pressure[x, y])


@nb.njit(nb.float32(nb.float32[:], nb.int32, nb.int32, nb.int32))
def avg_v(v: np.ndarray, i: int, j: int, y_size: int) -> np.float32:
    return (
        v[(i - 1) * y_size + j]
        + v[i * y_size + j]
        + v[(i - 1) * y_size + j + 1]
        + v[i * y_size + j + 1]
    ) * 0.25


@nb.njit(nb.float32(nb.float32[:], nb.int32, nb.int32, nb.int32))
def avg_u(u: np.ndarray, i: int, j: int, y_size: int) -> np.float32:
    return (
        u[i * y_size + j - 1]
        + u[i * y_size + j]
        + u[(i + 1) * y_size + j - 1]
        + u[(i + 1) * y_size + j]
    ) * 0.25


@nb.njit(
    nb.float32(
        nb.float32[:],
        nb.float32[:],
        nb.float32[:],
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
    mass: np.ndarray,
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
        f = mass
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
        sx * sy * f[x0 * y_size + y0]
        + tx * sy * f[x1 * y_size + y0]
        + tx * ty * f[x1 * y_size + y1]
        + sx * ty * f[x0 * y_size + y1]
    )


@nb.njit(
    (
        nb.float32[:],
        nb.float32[:],
        nb.float32[:],
        nb.float32[:],
        nb.float32[:],
        nb.float32[:],
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
    mass: np.ndarray,
    dt: float,
    h: float,
    x_size: int,
    y_size: int,
):
    h2 = 0.5 * h

    for i in range(1, x_size):
        for j in range(1, y_size):
            if (
                smoke[i * y_size + j] != 0.0
                and smoke[(i - 1) * y_size + j] != 0.0
                and j < y_size - 1
            ):
                x = i * h
                y = j * h + h2
                u_scalar = u[i * y_size + j]
                v_scalar = avg_v(v, i, j, y_size)
                x -= dt * u_scalar
                y -= dt * v_scalar
                new_u[i * y_size + j] = sample_field(
                    u, v, mass, x, y, U_FEILD, h, x_size, y_size
                )

            if (
                smoke[i * y_size + j] != 0.0
                and smoke[i * y_size + j - 1] != 0.0
                and i < x_size - 1
            ):
                x = i * h + h2
                y = j * h
                u_scalar = avg_u(u, i, j, y_size)
                v_scalar = v[i * y_size + j]
                x -= dt * u_scalar
                y -= dt * v_scalar
                new_v[i * y_size + j] = sample_field(
                    u, v, mass, x, y, V_FEILD, h, x_size, y_size
                )

    u[:] = new_u.copy()
    v[:] = new_v.copy()


@nb.njit(
    (
        nb.float32[:],
        nb.float32[:],
        nb.float32[:],
        nb.float32[:],
        nb.float32[:],
        nb.float32,
        nb.float32,
        nb.int32,
        nb.int32,
    )
)
def advect_smoke(
    smoke: np.ndarray,
    mass: np.ndarray,
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
            if smoke[i * y_size + j] != 0.0:
                u_scalar = (u[i * y_size + j] + u[(i + 1) * y_size + j]) * 0.5
                v_scalar = (v[i * y_size + j] + v[i * y_size + j + 1]) * 0.5
                x = i * h + h2 - dt * u_scalar
                y = j * h + h2 - dt * v_scalar
                val = sample_field(
                    u,
                    v,
                    mass,
                    x,
                    y,
                    S_FEILD,
                    h,
                    x_size,
                    y_size,
                )
                new_mass[i * y_size + j] = val

    mass[:] = new_mass.copy()


def print_arr(arr):
    print(arr.min(), arr.mean(), arr.max())

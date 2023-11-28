import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

NUM_ITERS = 40
NEIGHBORS = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
NEIGHBORS_U = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
NEIGHBORS_V = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
DIFFUSION = 0.01


class Fluid:
    def __init__(self, x_size, y_size) -> None:
        self.x_size = x_size
        self.y_size = y_size
        self.size = x_size * y_size

        self.u = np.zeros(shape=(self.x_size, self.y_size), dtype=np.float32)
        self.v = np.zeros(shape=(self.x_size, self.y_size), dtype=np.float32)
        self.smoke = np.zeros(shape=(self.x_size, self.y_size), dtype=np.float32)
        self.pressure = np.zeros(shape=(self.x_size, self.y_size), dtype=np.float32)
        self.div = np.zeros(shape=(self.x_size, self.y_size), dtype=np.float32)

        self.smoke[30:70, 0] = 1.0
        self.smoke[30:70, -1] = 1.0

        self.frame = 0

    def step(self, dt: float):
        dt = 0.01

        self.u[:, :4] = 2.0
        self.smoke[30:70, 0] = 1.0
        if self.frame < 100:
            self.u[:, -4:] = -2.0
            self.smoke[30:70, -1] = 1.0

        self.projection(dt)
        # self.diffusion(dt)
        self.advect_UV(dt)

        self.frame += 1

    @property
    def x(self) -> int:
        return self.x_size

    @property
    def y(self) -> int:
        return self.y_size

    @property
    def data_to_draw(self):
        return self.smoke

    def projection(self, dt: float):
        div = np.gradient(self.u, axis=1) + np.gradient(self.v, axis=0)

        pressure = np.zeros_like(self.smoke)
        for _ in range(NUM_ITERS):
            sum_pressure = convolve2d(pressure, NEIGHBORS, mode="same", boundary="fill")
            pressure = (sum_pressure - div) / 4

        self.pressure = pressure

        self.u -= np.gradient(pressure, axis=1)
        self.v -= np.gradient(pressure, axis=0)

    def diffusion(self, dt: float):
        for _ in range(NUM_ITERS):
            sum_smoke = convolve2d(self.smoke, NEIGHBORS, mode="same", boundary="fill")
            self.smoke = (self.smoke + DIFFUSION * dt * sum_smoke) / (
                1 + 4 * DIFFUSION * dt
            )

    def advect_UV(self, dt):
        y, x = np.mgrid[0 : self.x_size, 0 : self.y_size]
        x = x.astype(np.float32) - self.u
        y = y.astype(np.float32) - self.v

        x = np.clip(x, 0, self.y_size - 1).round().astype(int)
        y = np.clip(y, 0, self.x_size - 1).round().astype(int)

        u = convolve2d(self.u, NEIGHBORS_U, mode="same", boundary="fill") / 3
        v = convolve2d(self.v, NEIGHBORS_V, mode="same", boundary="fill") / 3
        smoke = convolve2d(self.smoke, NEIGHBORS, mode="same", boundary="fill") / 4

        self.u = u[y, x]
        self.v = v[y, x]
        self.smoke = smoke[y, x]

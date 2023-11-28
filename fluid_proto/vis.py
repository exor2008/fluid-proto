import numpy as np
from matplotlib import pyplot as plt
from vispy import app, gloo, scene, visuals
from vispy.visuals import BoxVisual, GridLinesVisual, ImageVisual, transforms

from fluid_proto.fluid import Fluid

TIME_STEP = 0.001


class Canvas(app.Canvas):
    def __init__(self, fluid: Fluid):
        app.Canvas.__init__(self, "Cube", keys="interactive", size=(400, 400))

        self.fluid = fluid
        self.image = ImageVisual(
            fluid.data_to_draw,
            # np.random.random(size=(100, 100)),
            cmap="coolwarm",
            clim=(-1, 1),
        )
        self.time = 0

        self.image.transform = transforms.STTransform(scale=(1, 1, 1))
        self._timer = app.Timer(TIME_STEP, connect=self.on_timer, start=True)
        app.Timer()

        self.show()

    def on_resize(self, event):
        vp = (0, 0, *self.physical_size)
        self.context.set_viewport(*vp)
        self.image.transforms.configure(canvas=self, viewport=vp)

        # Update the scale to fit the canvas
        canvas_size = self.size
        grid_size = self.image.size
        scale_x = canvas_size[0] / grid_size[0]
        scale_y = canvas_size[1] / grid_size[1]

        # Apply the new scale
        self.image.transform.scale = (scale_x, scale_y, 1)
        self.update()

    def on_draw(self, event):
        gloo.set_viewport(0, 0, *self.physical_size)
        gloo.clear("white", depth=True)

        self.image.draw()

    def on_timer(self, event):
        # if self.time > 5:
        self.fluid.step(event.dt)
        self.image.set_data(self.fluid.data_to_draw)
        # self.image.set_data(np.random.random(size=(100, 100)))
        self.update()
        self.time += event.dt

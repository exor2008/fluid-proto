import sys

from matplotlib import pyplot as plt

from fluid_proto.fluid_parallel import Fluid
from fluid_proto.vis import Canvas

# from fluid_proto.fluid import Fluid


if __name__ == "__main__":
    f = Fluid(100, 100)
    # while True:
    #     f.step(0.01)
    # plt.imshow(f.pressure)
    # plt.show()

    win = Canvas(f)
    win.show()
    if sys.flags.interactive != 1:
        win.app.run()

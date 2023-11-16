import sys

from fluid_proto.fluid import Fluid
from fluid_proto.vis import Canvas

if __name__ == "__main__":
    f = Fluid(500, 102)
    win = Canvas(f)
    win.show()
    if sys.flags.interactive != 1:
        win.app.run()

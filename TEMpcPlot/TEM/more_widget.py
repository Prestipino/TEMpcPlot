from matplotlib.widgets import AxesWidget

import numpy as np
# from . import _api, cbook, colors, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import math


rpd = math.pi / 180.0


class LineBuilder(AxesWidget):
    """
    class defined to trace lines on an existing figure
    the class one time defined calculate few attributes
    self.origin = origin of the line
    self.vect = vector represented
    self.mod = lenght of the line
    self.fline = line object passing grom the two point


    """

    def __init__(self, ax, xy, callback=None, useblit=True,
                 color='r', stay=False):
        super().__init__(ax)

        self.useblit = useblit and self.canvas.supports_blit
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.stay = stay
        x, y = xy

        self.verts = [(x, y)]
        self.line = Line2D([x], [y], linestyle='-', color=color, lw=2)
        self.ax.add_line(self.line)
        self.callback = callback

        self.connect_event('button_release_event', self.onrelease)
        self.connect_event('motion_notify_event', self.onmove)

        return

    def onrelease(self, event):
        if self.ignore(event):
            return
        if self.verts is not None:
            self.verts.append((event.xdata, event.ydata))
            if len(self.verts) > 1:
                self.callback(self.verts)
        if not(self.stay):
            self.ax.lines.remove(self.line)
        self.verts = None
        self.disconnect_events()

    def onmove(self, event):
        if self.ignore(event):
            return
        if self.verts is None:
            return
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        data = self.verts + [(event.xdata, event.ydata)]
        self.line.set_data(list(zip(*data)))

        if self.useblit:
            self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.line)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()


class RectangleBuilder(AxesWidget):
    """
    class defined to trace lines on an existing figure
    the class one time defined calculate few attributes
    self.origin = origin of the line
    self.vect = vector represented
    self.mod = lenght of the line
    self.fline = line object passing grom the two point
    """

    def __init__(self, ax, xy, callback=None, useblit=False):
        super().__init__(ax)

        self.useblit = useblit and self.canvas.supports_blit
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        self.line = LineBuilder(ax, xy, callback=self.line_callback,
                                useblit=useblit)
        self.callback = callback
        # self.canvas.widgetlock(self.line)
        # self.__xtl = []
        return

    def line_callback(self, verts):
        x0, y0 = verts[0]
        x1, y1 = verts[1]
        self.line.origin = np.array([x0, y0])
        self.line.vect = np.array([x1 - x0, y1 - y0])
        self.line.mod = np.sqrt(self.line.vect @ self.line.vect)
        self.line.angle = -np.arctan2(*self.line.vect) / rpd
        self.width = 0.0
        self.Rleft = Rectangle(self.line.origin, self.width, self.line.mod,
                               self.line.angle, color='r', alpha=0.3)
        self.Rright = Rectangle(self.line.origin, -self.width, self.line.mod,
                                self.line.angle, color='r', alpha=0.3)
        self.ax.add_patch(self.Rleft)
        self.ax.add_patch(self.Rright)
        self.connect_event('button_press_event', self.onrelease)
        self.connect_event('motion_notify_event', self.onmove)
        if self.useblit:
            self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.Rleft)
            self.ax.draw_artist(self.Rright)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()


    def onrelease(self, event):
        print("non capisco")
        if self.ignore(event):
            return
        if self.width:
            self.callback(self.line.origin, self.line.vect,
                          self.width)
            self.Rleft.remove()
            self.Rright.remove()
            print("non capisco2")
        self.canvas.draw_idle()
        self.disconnect_events()


    def onmove(self, event):
        if self.ignore(event):
            return
        if event.inaxes != self.ax:
            return
        # if event.button != 1:
        #    return

        coor = np.array([event.xdata, event.ydata])
        dist = np.abs(np.cross(self.line.vect, coor - self.line.origin))

        self.width = dist / self.line.mod

        self.Rleft.set_width(self.width)
        self.Rright.set_width(-self.width)
        if self.useblit:
            self.canvas.restore_region(self.background)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

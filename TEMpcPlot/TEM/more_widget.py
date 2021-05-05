from matplotlib.widgets import AxesWidget

import numpy as np
# from . import _api, cbook, colors, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.pyplot import axline

import math


rpd = math.pi / 180.0


class Picker(AxesWidget):
    def __init__(self, ax, picked, callback=None):
        super().__init__(ax)

        self.picked = picked
        self.callback = callback
        self.connect_event('pick_event', self.onpress)
        self.canvas.widgetlock(self)
        return

    def onpick(self, event):
        if event.artist != self.picked:
            return
        if event.button == 1:
            self.callback(event.ind[0])
        if event.button == 3:

            return

    def endpick(self, event):
        if event.button != 3:
            return
        self.disconnect_events()
        self.canvas.widgetlock.release(self)
        return


class LineBuilder(AxesWidget):
    """
    class defined to trace lines on an existing figure
    the class one time defined calculate few attributes
    self.origin = origin of the line
    self.vect = vector represented
    self.mod = lenght of the line
    self.fline = line object passing grom the two point


    """

    def __init__(self, ax, callback=None, useblit=True,
                 stay=False, linekargs={}):
        super().__init__(ax)

        self.useblit = useblit and self.canvas.supports_blit
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.stay = stay

        self.callback = callback
        self.linekargs = linekargs
        self.connect_event('button_press_event', self.onpress)
        return

    def onrelease(self, event):
        if self.ignore(event):
            return
        if self.verts is not None:
            self.verts.append((event.xdata, event.ydata))
            if self.callback is not None:
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
        data = np.array(data, dtype=float).T
        self.line.set_data(*data)

        if self.useblit:
            self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.line)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

    def onpress(self, event):
        if self.canvas.widgetlock.locked():
            return
        if event.inaxes is None:
            return
        # acquire a lock on the widget drawing
        if self.ignore(event):
            return
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        self.verts = [(event.xdata, event.ydata)]

        self.line = Line2D([event.xdata], [event.ydata],
                           linestyle='-', lw=2, **self.linekargs)
        self.ax.add_line(self.line)
        if self.useblit:
            self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.line)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()
        self.connect_event('button_release_event', self.onrelease)
        self.connect_event('motion_notify_event', self.onmove)


###########################################################################################################


class LineAxes(AxesWidget):

    def __init__(self, ax, m, callback=None, useblit=True,
                 linekargs={}):
        super().__init__(ax)

        self.useblit = useblit and self.canvas.supports_blit
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.m = m
        self.callback = callback
        self.linekargs = linekargs
        self.connect_event('button_press_event', self.onpress)

    def onpress(self, event):
        self.line = Line2D([0], [0], linestyle='+-', lw=2, **self.linekargs)
        self.text = self.ax.text(0, 0, '')
        self.ax.add_line(self.line)

        lim = 1.5 * max([self.ax.get_xlim(), self.ax.get_xlim()])
        self.p_line = Line2D([lim * event.datay, -lim * event.datay],
                             [lim * event.datax, lim * event.datax],
                             linestyle='--', lw=2, color='grey')
        self.ax.add_line(self.p_line)
        if self.useblit:
            self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.line)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()
        self.connect_event('button_release_event', self.onrelease)
        self.connect_event('motion_notify_event', self.onmove)
        return

    def onmove(self, event):
        if self.ignore(event):
            return
        if self.verts is None:
            return
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return

        lim = 1.5 * max([self.ax.get_xlim(), self.ax.get_xlim()])
        self.p_line.set_data([lim * event.datay, -lim * event.datay],
                             [-lim * event.datax, lim * event.datax])

        datax = np.linspace(0, event.xdata, self.m + 1)
        datay = np.linspace(0, event.ydata, self.m + 1)

        inv_mod = round(self.m / np.sqrt(event.xdata**2 + event.ydata**2), 2)
        self.line.set_data(datax, datay)
        self.ax.text.set_position((event.xdata, event.ydata))
        self.ax.text.set_text(f'{inv_mod:3.2f} ')
        if self.useblit:
            self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.line)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

    def onrelease(self, event):
        if self.ignore(event):
            return
        self.text.remove()
        self.p_line.remove()
        if self.callback is not None:
            self.callback(self.event.xdata,
                          self.event.xdata)
        self.disconnect_events()


class RectangleBuilder(AxesWidget):
    """
    class defined to trace lines on an existing figure
    the class one time defined calculate few attributes
    self.origin = origin of the line
    self.vect = vector represented
    self.mod = lenght of the line
    self.fline = line object passing grom the two point
    """

    def __init__(self, ax, callback=None, useblit=False):
        super().__init__(ax)

        self.useblit = useblit and self.canvas.supports_blit
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        self.line = LineBuilder(ax, callback=self.line_callback,
                                useblit=useblit, linekargs={'color': 'red'})
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

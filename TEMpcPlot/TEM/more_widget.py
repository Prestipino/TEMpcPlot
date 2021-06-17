
import os
import numpy as np
# from . import _api, cbook, colors, ticker
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.widgets import AxesWidget

from matplotlib.backends.qt_compat import QtGui
import matplotlib.pyplot as plt

import mplcursors

from packaging import version
if version.parse(matplotlib.__version__) > version.parse("3.3.1"):
    matplotlib_old = False
    from matplotlib.backend_bases import _Mode
else:
    matplotlib_old = True

import math
from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv

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
        return

    def onpress(self, event):
        self.line = Line2D([0], [0], linestyle='-', marker='+',
                           lw=2, **self.linekargs)
        self.ax.add_line(self.line)

        self.p_line = [Line2D([0], [0], linestyle='--',
                              color='grey', lw=1) for i in range(self.m)]
        for pline_i in self.p_line:
            self.ax.add_line(pline_i)

        self.text = self.ax.text(0, 0, '')

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
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return

        lim = 1.5 * max(self.ax.get_xlim() + self.ax.get_xlim())
        pdatax = lim * np.array((-event.ydata, event.ydata))
        pdatay = lim * np.array((event.xdata, -event.xdata)) 
        for i, pline_i in enumerate(self.p_line):
            pline_i.set_data(pdatax + (event.xdata * (i + 1) / self.m),
                             pdatay + (event.ydata * (i + 1) / self.m))
        # (event.xdata * (i + 1) / self.m)
        # (event.ydata * (i + 1) / self.m)

        datax = np.linspace(0, event.xdata, self.m + 1)
        datay = np.linspace(0, event.ydata, self.m + 1)
        inv_mod = round(self.m / np.sqrt(event.xdata**2 + event.ydata**2), 2)
        self.line.set_data(datax, datay)
        self.text.set_position((event.xdata, event.ydata))
        self.text.set_text(f'{inv_mod:3.2f} ')

        if self.useblit:
            self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.line)
            self.ax.draw_artist(self.text)
            for pline_i in self.p_line:            
                self.ax.draw_artist(pline_i)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

    def onrelease(self, event):
        if self.ignore(event):
            return
        self.text.remove()
        for pline_i in self.p_line:
            pline_i.remove()
        self.canvas.draw()
        if self.callback is not None:
            print('callback')
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
        if self.ignore(event):
            return
        if self.width:
            self.callback(self.line.origin, self.line.vect,
                          self.width)
            self.Rleft.remove()
            self.Rright.remove()
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

class ToolbarPlus():

    def __init__(self, selfi, log=False, fig=None, ax=None, tool_b=None, *args, **kwds):
        index = 0
        lun = len(selfi)
        self.Peak_plot = True
        self.args = args
        self.kwds = kwds

        def UP_DO(up):
            nonlocal index
            index += up
            index -= up * lun * (abs(index) // lun)
            selfi.ima = selfi[index]
            plt.sca(ax)
            selfi.ima.plot(new=0, log=log, peaks=self.Peak_plot, *self.args, **self.kwds)
            ax.set_axis_off()
            ax.set_frame_on(False)
            #self.canvas.draw_idle()
            fig.canvas.draw()

        def Plot_p():
            self.Peak_plot = not(self.Peak_plot)
            if self.Peak_plot:
                selfi.ima.Peaks.plot()
            else:
                selfi.ima.Peaks.deplot()
            fig.canvas.draw()

        def Del_p():
            selfPL = selfi.ima.Peaks
            if not hasattr(selfi.ima.Peaks, 'lp'):
                return
            if fig.canvas.widgetlock.locked():
                return

            def onpick(event):
                if event.artist != selfi.ima.Peaks.lp:
                    return
                selfi.ima.Peaks.del_peak(event.ind[0])
                return

            def endpick(event):
                if event is None:
                    pass
                elif event.button != 3:
                    return
                fig.canvas.mpl_disconnect(selfPL._cid)
                fig.canvas.mpl_disconnect(selfPL._mid)
                fig.canvas.widgetlock.release(tool_b._actions['del_p'])
                tool_b._actions['del_p'].setChecked(False)
                return

            selfPL._cid = fig.canvas.mpl_connect('pick_event', onpick)
            selfPL._mid = fig.canvas.mpl_connect('button_press_event', endpick)
            # fig.canvas.widgetlock(self)
            # fig.canvas.widgetlock(tool_b._actions['del_p'])

            #tool_b._actions['pan'].setChecked(tool_b._active == 'PAN')
            #tool_b._actions['zoom'].setChecked(tool_b._active == 'ZOOM')

        def DelR_p():
            if not hasattr(selfi.ima.Peaks, 'lp'):
                return
            if matplotlib_old:
                if tool_b._active == 'DelR P':
                    tool_b._active = None
                else:
                    tool_b._active = 'DelR P'

                if tool_b._idPress is not None:
                    tool_b._idPress = fig.canvas.mpl_disconnect(
                        tool_b._idPress)
                    tool_b.mode = ''

                if tool_b._idRelease is not None:
                    tool_b._idRelease = fig.canvas.mpl_disconnect(
                        tool_b._idRelease)
                    tool_b.mode = ''
            else:
                if tool_b.mode == _Mode.ZOOM:
                    tool_b.mode = _Mode.NONE
                    tool_b._actions['zoom'].setChecked(False)
                if tool_b.mode == _Mode.PAN:
                    tool_b.mode = _Mode.NONE
                    tool_b._actions['pan'].setChecked(False)
            selfi.ima.Peaks.del_PlotRange()

        def lenght():
            if hasattr(selfi.ima, 'line'):
                del selfi.ima.line
            selfi.ima.profile_Line(plot=True)
            while not(hasattr(selfi.ima.line, 'fline')):
                plt.pause(0.3)
            at = '\nlengh of the vector'
            le = selfi.ima.line.mod * selfi.ima.scale
            print(f'{at} {10*le: 4.2f} 1/Ang.')
            print(f'and {0.1/le: 4.2f} Ang. in direct space')
            at = 'component of the vector'
            le = selfi.ima.line.vect * selfi.ima.scale
            print(f'{at} {le[0]: 4.2f} {le[1]: 4.2f} 1/nm')
            print('\n\n')

        def angle():
            if hasattr(selfi.ima, 'line'):
                del selfi.ima.line
            angle = selfi.ima.angle()
            at = 'angle between the vectors'
            print(f'{at} {angle: 4.2f} degrees')
            print('\n\n')

        def press(event):
            if event.key == 'f4':
                DelR_p()

        def _icon(name):
            direct = os.path.dirname(__file__)
            name = os.path.join(direct, name)
            pm = QtGui.QPixmap(name)
            if hasattr(pm, 'setDevicePixelRatio'):
                pm.setDevicePixelRatio(fig.canvas._dpi_ratio)
            return QtGui.QIcon(pm)



        fig.canvas.toolbar.addSeparator()
        a = tool_b.addAction(_icon('down.png'), 'back', lambda: UP_DO(-1))
        a.setToolTip('Previous image')
        a = tool_b.addAction(_icon('up.png'), 'foward', lambda: UP_DO(1))
        a.setToolTip('Next image')

        tool_b.addSeparator()
        a = tool_b.addAction(_icon('PlotP.png'), 'Peaks', Plot_p)
        a.setToolTip('Peaks On/Off')

        a = tool_b.addAction(_icon('RemP.png'), 'Del_P', Del_p)
        a.setCheckable(True)
        a.setToolTip('Delete Peaks')
        tool_b._actions['del_p'] = a

        a = tool_b.addAction(_icon('RanP.png'), 'DelR P', DelR_p)
        a.setToolTip('Delete Peaks in range (F4)')

        tool_b.addSeparator()
        a = tool_b.addAction(_icon('lenght.png'), 'len', lenght)
        a.setToolTip('calculate lenght of a line and plot profile')
        a = tool_b.addAction(_icon('angle.png'), 'angle', angle)
        a.setToolTip('calculate angle between two lines')


class ToolbarPlusCal():
    def __init__(self, selfi, axes, log=False, fig=None, ax=None, tool_b=None, *args, **kwds):
        self.index = 0
        lun = len(selfi)
        self.args = args
        self.kwds = kwds
        self.round = 0


        # paasage to axes base
        P = inv(axes)
        Rot = R.from_rotvec([0, 0, -selfi.zangles[self.index]])
        Rot = R.from_rotvec(selfi.rot_vect * selfi.angles[self.index]) * Rot

        def HKL_integer(appr):
            if appr:
                self.round = 0
            else:
                self.round = 2


        def format_coord(x, y):
            d2 = np.round(1 / np.sqrt(x**2 + y**2), 3)
            xy3d = Rot.apply([y, x, 0])
            hkl = np.round(P @ xy3d, 2)
            # return f'{z:s} d={dist2:2.4f} [{dist1:2.4f} pixel]'
            z = f'x={y:4.1f}, y={x:4.1f}, hkl={hkl},  d_sp={d2:4.4f}nm'
            return f'{z:s}             '

        def label(x, y):
            d2 = np.round(1 / np.sqrt(x**2 + y**2), 3)
            xy3d = Rot.apply([y, x, 0])
            hkl = np.round(P @ xy3d, 2)
            # return f'{z:s} d={dist2:2.4f} [{dist1:2.4f} pixel]'
            z = f'x={y:4.1f}, y={x:4.1f}, hkl={hkl},  d_sp={d2:4.4f}nm'
            return f'{z:s}             '

        def UP_DO(up):
            nonlocal Rot
            self.index += up
            self.index -= up * lun * (abs(self.index) // lun)
            self.ima = selfi[self.index]
            centro = np.array(self.ima.center) * self.ima.scale
            forma = np.array(self.ima.ima.shape) * self.ima.scale
            estensione = [-centro[1], forma[0] - centro[1],
                          forma[1] - centro[0], -centro[0], ]
            plt.sca(ax)
            ax.clear()
            if log:
                self.pltim = plt.imshow(np.log(np.abs(self.ima.ima)),
                                        extent=estensione,
                                        cmap='binary', *self.args, **self.kwds)
            else:
                self.pltim = plt.imshow(self.ima.ima, extent=estensione,
                                        cmap='binary', *args, **kwds)
            plt.title(f'Image {self.ima.info.filename}')
            #ax.set_frame_on(False)
            ax.set_xlabel(r'$nm^{-1}$')
            ax.set_ylabel(r'$nm^{-1}$')
            Rot = R.from_rotvec([0, 0, -selfi.zangles[self.index]])
            Rot = R.from_rotvec(selfi.rot_vect * selfi.angles[self.index]) * Rot
            fig.canvas.draw()
            ax.format_coord = format_coord

            c1 = mplcursors.cursor(self.pltim, multiple=True)

            @c1.connect("add")
            def _(sel):
                centro = np.array(self.ima.center)
                x, y = sel.target.index
                xc = (x - centro[0]) * selfi.scale
                yc = (y - centro[1]) * selfi.scale
                xy3d = Rot.apply([xc, yc, 0])
                hkl = P @ xy3d
                if self.round == 0:
                    hkl = np.round(P @ xy3d, self.round).astype('int')
                else:
                    hkl = np.round(P @ xy3d, self.round)
                sel.annotation.get_bbox_patch().set(fc="white", alpha=.5)
                sel.annotation.set(text=str(tuple(hkl)))
                sel.annotation.arrow_patch.set(arrowstyle="simple", fc="white", alpha=.5)



        def _icon(name):
            direct = os.path.dirname(__file__)
            name = os.path.join(direct, name)
            pm = QtGui.QPixmap(name)
            if hasattr(pm, 'setDevicePixelRatio'):
                pm.setDevicePixelRatio(fig.canvas._dpi_ratio)
            return QtGui.QIcon(pm)


        fig.canvas.toolbar.addSeparator()
        a = tool_b.addAction(_icon(' '), 'int', lambda: HKL_integer(True))
        a.setToolTip('Integer HKL')
        a = tool_b.addAction(_icon(' '), 'float', lambda: HKL_integer(False))
        a.setToolTip('Float HKL')

        a = tool_b.addAction(_icon('down.png'), 'back', lambda: UP_DO(-1))
        a.setToolTip('Previous image')
        a = tool_b.addAction(_icon('up.png'), 'foward', lambda: UP_DO(1))
        a.setToolTip('Next image')


        UP_DO(0)








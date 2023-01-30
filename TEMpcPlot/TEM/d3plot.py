import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv
from . import more_widget as mw
from . import math_tools as mt
plt.ion()

"""
   pos_0   initial position
   pos_i   position after the set of rotation and scale.
"""


class D3plot(object):
    """
    Class used to plot a set of 3D peaks

    """

    def __init__(self, EwPePos, size='o', Fig=None):
        self.__size = size
        self._EwPePos = EwPePos
        # list in whic pos are sotred for each image
        self.pos_i = EwPePos.pos[:]  # positions after rotation
        self.r0 = R.from_rotvec([0, 0, 0])  # rotation from the start
        if hasattr(EwPePos, '_rot_vect'):
            self.__angle = np.arccos(np.dot(EwPePos._rot_vect[0],
                                            [0, 1, 0]))
            self.__angle *= 180 / np.pi      # angle to start good

        # -----------define axis if already present
        self.axes = {}
        if hasattr(EwPePos, 'axes'):
            for i, abc in enumerate('abc'):
                self.axes[abc] = LineAxes(abc, 1, axis=EwPePos.axes.T[i])
        # -----------------------------------------------

        self.fig = plt.figure()

        gs = self.fig.add_gridspec(5, 5)
        self.ax_x = self.fig.add_subplot(gs[4, 0:4])
        self.ax_y = self.fig.add_subplot(gs[0:4, 4])
        self.ax_y.tick_params(axis='y', left=False, right=True,
                              labelleft=False, labelright=True)
        self.ax = self.fig.add_subplot(gs[0:4, 0:4])
        plt.figtext(0.8, 0.22, '- a', color='green',
                    size='x-large', weight='bold')
        plt.figtext(0.8, 0.17, '- b', color='blue',
                    size='x-large', weight='bold')
        plt.figtext(0.8, 0.12, '- c', color='black',
                    size='x-large', weight='bold')

        self.plot_ax()
        # ---------------------------------------------
        # ----------------------------------------------

        self.__cid = self.fig.canvas.mpl_connect('button_press_event',
                                                 self.main_click)

        def rezize_g():
            self.plot_ax()
            self.plot_hist()
        self.__res = self.fig.canvas.mpl_connect('resize_event',
                                                 lambda event: rezize_g())

    def filter_int(self, operator=None, lim=None):
        """conserve only peaks respecting an intensity condition
        conserve only peaks respecting an intensity condition, to 
        determine the most usefull values use Exp1.EwP.plot_int()
        Example:
            >>> Exp1.EwP.graph.filter_int('>', 1000)
            >>> Exp1.EwP.graph.filter_int('<', 1000)            
        """
        if operator == '>':
            def lcond(x):
                return x > lim
        elif operator == '<':
            def lcond(x):
                return x < lim
        pos_i = []
        for i_pos, i_inte in zip(self._EwPePos.pos, self._EwPePos.int):
            pos_i.append(self.r0.apply(i_pos[lcond(i_inte)]))
        self.pos_i = pos_i
        self.plot_ax()
        self.plot_hist()

    def filter_layer(self, listn):
        """conserve only the layers in list

        Examples:
            >>> Exp1.EwP.graph.filter_layer([0,1,2])
        """
        pos_i = []
        for j, i_pos in enumerate(self._EwPePos.pos):
            if j in listn:
                pos_i.append(self.r0.apply(i_pos))
        self.pos_i = pos_i
        self.plot_ax()
        self.plot_hist()

    def plot_ax(self):
        self.ax.cla()
        self.ax.set_aspect(aspect='equal', adjustable='datalim')
        self.fig.canvas.draw()
        self.bkg_cache = self.fig.canvas.copy_from_bbox(self.ax.bbox.padded(0))
        # list with the graph
        self.m_planes = list()
        for j, i in enumerate(self.pos_i):
            self.m_planes.extend(self.ax.plot(*i.T[:2], self.__size,
                                              label=str(j), picker=5))
        self.zer, = self.ax.plot([0], [0], 'xr', markersize=20)
        for axis in self.axes.values():
            axis.plot()
        self.ax.set_axis_off()        # ###############
        self.ax.set_frame_on(False)      # ###############

        # legend
        plt.rcParams['toolbar'] = 'toolbar2'

        x = abs(np.array(self.ax.get_xlim())).max()
        y = abs(np.array(self.ax.get_ylim())).max()
        self.__axxlim = (-x, x)
        self.__axylim = (-y, y)
        self.ax.set_xlim(*self.__axxlim)
        self.ax.set_ylim(*self.__axylim)

    def plot_hist(self):
        self.ax_x.cla()
        self.ax_y.cla()
        self._xh = self.ax_x.hist(np.concatenate(self.pos_i)[:, 0],
                                  bins=100, rwidth=4)
        self._yh = self.ax_y.hist(np.concatenate(self.pos_i)[:, 1],
                                  bins=100, rwidth=4, orientation='horizontal')
        self.ax_x.set_autoscalex_on(False)
        self.ax_y.set_autoscalex_on(True)
        self.ax_y.invert_xaxis()
        self.ax_x.set_xlim(*self.ax.get_xlim())
        self.ax_y.set_ylim(*self.ax.get_ylim())
        plt.draw()

    def zoom(self, event):
        if event is not None:
            zoom_m = np.sqrt(np.sum(np.array([self.__c_x0, self.__c_y0])**2))
            zoom_m /= np.sqrt(np.sum(np.array([event.xdata, event.ydata])**2))
        else:
            zoom_m = 1
        self.fig.canvas.restore_region(self.bkg_cache)
        self.__axxlim = (self.ax.get_xlim())
        self.__axylim = (self.ax.get_ylim())
        self.ax.set_xlim(self.__axxlim[0] * zoom_m, self.__axxlim[1] * zoom_m)
        self.ax.set_ylim(self.__axylim[0] * zoom_m, self.__axylim[1] * zoom_m)

        # for n_pos, i_pos in enumerate(self.pos_i):
        #    self.ax.draw_artist(self.m_planes[n_pos])
        # if len(self.axes_i) > 0:
        #    for i in self.axes_i.keys():
        #        self.ax.draw_artist(self.l_axes[i])
        # self.fig.canvas.blit(self.ax.bbox)
        plt.draw()

    def __find_rot(self, event):
        rot_vec = np.array([self.__c_y0 - event.ydata,
                            event.xdata - self.__c_x0, 0])
        rot_vec = np.pi * rot_vec / abs(self.ax.get_xlim()[0] * 4)
        if event.key == 'control':
            z = np.sqrt(rot_vec @ rot_vec.T) * np.array([0, 0, -1])
            rot_vec = z if rot_vec[0] > rot_vec[1] else -z
        return R.from_rotvec(rot_vec)

    def __rotate(self, r):
        """rotate graphically
        """
        self.fig.canvas.restore_region(self.bkg_cache)
        for n_pos, i_pos in enumerate(self.pos_i):
            z = r.apply(i_pos)
            self.m_planes[n_pos].set_data(z[:, 0], z[:, 1])
            self.ax.draw_artist(self.m_planes[n_pos])

        for axis in self.axes.values():
            axis.rotate(r, store=False)

        self.ax.draw_artist(self.zer)
        self.fig.canvas.blit(self.ax.bbox)

    def __stab_rot(self, r):
        """track all rotation 
        track the rotatyion when mouse is released...
        """
        self.r0 = r * self.r0
        for i, j in enumerate(self.pos_i):
            self.pos_i[i] = r.apply(j)
        for axis in self.axes.values():
            axis.rotate(r, store=True)

    def __m_rotate(self, event):
        r = self.__find_rot(event)
        self.__rotate(r)

    def _c_rotate(self, xyz, deg):
        """command rotate
        """
        rot_vec = np.array([np.radians(deg) if i in xyz else 0 for i in 'xyz'])
        r = R.from_rotvec(rot_vec)
        self.__rotate(r)
        self.__stab_rot(r)
        self.plot_hist()

    def _set__axes(self, axes):
        for i, abc in enumerate('abc'):
            self.axes[abc] = LineAxes(abc, 1,
                                      axis=axes.T[i],
                                      rot=self.r0)

    def rotatex(self, deg=90):
        """
        rotate along the x axis default value 90
        same command for y and z

        Args:
            deg (float): angle in degree to rotate
        Examples:
            >>> Exp1.EwP.graph.rotatex(30)
            >>> Exp1.EwP.graph.rotatex(-30)

        """
        self._c_rotate('x', deg)

    def rotatey(self, deg=90):
        self._c_rotate('y', deg)

    def rotatez(self, deg=90):
        self._c_rotate('z', deg)

    def rotate_0(self, allign = False):
        """
        rotate to first orientation

        Example:
            >>> Exp1.EwP.graph.rotate_0()
        """
        r = self.r0.inv()
        self.__rotate(r)
        self.__stab_rot(r)
        if allign:
            self.rotatez(self.__angle)  #  why ??

    def _c_allign(self, abc):
        """command allign to an axis
        """
        try:
            rot_vec = np.cross(self.axes[abc].pos_i - self.axes[abc].ori,
                               np.array([0, 0, 1]))
        except AttributeError:
            rot_vec = np.cross(self.axes[abc].pos_i,
                               np.array([0, 0, 1]))
        r_mod = np.sqrt(rot_vec @ rot_vec)
        rot_vec_n = rot_vec / r_mod
        rot_vec_n *= -np.arcsin(r_mod / self.axes[abc].mod)
        r = R.from_rotvec(rot_vec_n)
        self.__rotate(r)
        self.__stab_rot(r)

    def allign_a(self):
        """
        rotate the peaks in order to allign to a* axis to z
        same command for b* and c*

        Example:
            >>> Exp1.EwP.graph.allign_a()
        """
        self._c_allign('a')

    def allign_b(self):
        self._c_allign('b')

    def allign_c(self):
        self._c_allign('c')

    def main_click(self, event):
        if event.inaxes != self.ax:
            return
        self.fig.canvas.mpl_disconnect(self.__cid)
        canv = self.fig.canvas

        def endpick(event):
            if event.button == 1:
                r = self.__find_rot(event)
                self.r0 = r * self.r0
                for i, j in enumerate(self.pos_i):
                    self.pos_i[i] = r.apply(j)
                for axis in self.axes.values():
                    axis.rotate(r, store=True)
            self.plot_hist()
            canv.mpl_disconnect(self.__mid)
            canv.mpl_disconnect(self.__rid)
            canv.mpl_disconnect(self.__rid2)
            self.__cid = canv.mpl_connect('button_press_event',
                                          self.main_click)
            return
        self.__c_x0 = event.xdata
        self.__c_y0 = event.ydata

        if event.button == 1:
            self.__mid = canv.mpl_connect('motion_notify_event',
                                          self.__m_rotate)
        if event.button == 3:
            self.__mid = canv.mpl_connect('motion_notify_event', self.zoom)
        self.__rid = canv.mpl_connect('button_release_event', endpick)
        self.__rid2 = canv.mpl_connect('axes_leave_event', endpick)

    def define_axis(self, abc, m):
        """define axis
        define axis graphically tracing a line

        Args:
            abc (str): name of the axis
            m (int): multiple that will be traCED
        Example:
            >>>Exp1.EwP.graph.define_axis('a', 4)
        """
        assert abc in 'abc', 'only three axis a, b, c '
        self.fig.canvas.mpl_disconnect(self.__cid)
        self.axes[abc] = LineAxes(abc, m)
        self.axes[abc]._graph_init_()
        plt.waitforbuttonpress(65)
        self.axes[abc].calc_axis(self.r0)
        self.__cid = self.fig.canvas.mpl_connect('button_press_event',
                                                 self.main_click)

    def legend(self):
        self.ax.legend()


class LineAxes:
    def __init__(self, abc, m, axis=None, rot=None):
        self.m = m
        self.abc = abc
        fmt = {'a': '+-g', 'b': '+-b', 'c': '+-k'}
        if axis is not None:
            self.axis = axis
            self.mod = np.sqrt(self.axis @ self.axis.T)
            self.inv_mod = 1 / self.mod
            if rot is not None:
                self.pos_i = rot.apply(self.axis)
            else:
                self.pos_i = self.axis
        else:
            self.line, = plt.plot([0], [0], fmt[abc])

    def plot(self):
        fmt = {'a': '+-g', 'b': '+-b', 'c': '+-k'}
        self.line, = plt.plot([0], [0], fmt[self.abc])
        datax = np.linspace(0, self.pos_i[0] * self.m, self.m + 1)
        datay = np.linspace(0, self.pos_i[1] * self.m, self.m + 1)
        self.line.set_data(datax, datay)
        self.line.axes.draw_artist(self.line)

    def rotate(self, r, store=False):
        z = r.apply(self.pos_i)
        self.line.set_data([0, z[0]], [0, z[1]])
        self.line.axes.draw_artist(self.line)
        if store:
            self.pos_i = z

    def calc_axis(self, r0):
        self.axis = r0.inv().apply(self.pos_i)
        self.mod = np.sqrt(self.axis @ self.axis.T)
        self.inv_mod = 1 / self.mod

    def _graph_init_(self):
        m = self.m
        text = plt.text(0, 0, '')
        canv = self.line.figure.canvas
        p_line = [Line2D([0], [0], linestyle='--',
                         color='grey', lw=1) for i in range(self.m + 1)]
        for pline_i in p_line:
            self.line.axes.add_line(pline_i)

        def endpick(event):
            self.pos_i = np.array([event.xdata / m,
                                   event.ydata / m, 0])
            # print(self.pos_i)
            self.inv_mod = m / np.sqrt(self.pos_i[0]**2 + self.pos_i[1]**2)
            canv.mpl_disconnect(self.__cid)
            canv.mpl_disconnect(self.__mid)
            text.remove()
            for pline_i in p_line:
                pline_i.remove()
            return

        def move_m(event):
            if event.inaxes != self.line.axes:
                return
            # canv.restore_region(self.bkg_cache)
            datax = np.linspace(0, event.xdata, m + 1)
            datay = np.linspace(0, event.ydata, m + 1)
            inv_mod = round(m / np.sqrt(event.xdata**2 + event.ydata**2), 2)
            self.line.set_data(datax, datay)
            self.line.axes.draw_artist(self.line)
            #
            lim = 1.5 * max(self.line.axes.get_xlim() +
                            self.line.axes.get_xlim())
            pdatax = lim * np.array((-event.ydata, event.ydata))
            pdatay = lim * np.array((event.xdata, -event.xdata))
            for i, pline_i in enumerate(p_line):
                pline_i.set_data(pdatax + (event.xdata * (i) / self.m),
                                 pdatay + (event.ydata * (i) / self.m))
            for pline_i in p_line:
                self.line.axes.draw_artist(pline_i)
            #
            text.set_position((event.xdata, event.ydata))
            text.set_text(f'{inv_mod:3.2f} ')
            self.line.axes.draw_artist(text)
            # canv.blit(self.line.axes.bbox)
            plt.draw()
            return

        # self.bkg_cache = canv.copy_from_bbox(self.line.axes.bbox.padded(0))
        self.__mid = canv.mpl_connect('motion_notify_event', move_m)
        self.__cid = canv.mpl_connect('button_press_event', endpick)
        return

    def __del__(self):
        if hasattr(self, '__mid'):
            self.line.figure.canvas.mpl_disconnect(self.mid)
        if hasattr(self, '__rid'):
            self.line.figure.canvas.mpl_disconnect(self.rid)
        if hasattr(self, 'line'):
            self.line.remove()
        # plt.legend()
        return

# collapsed 3D view r is for reduced)


class D3plotr(D3plot):
    def __init__(self, EwPePos, origin, size='o'):
        self._D3plot__size = size
        self._D3plot__EwPePos = EwPePos
        self.origin = origin
        # list in whic pos are sotred for each image
        self.pos_i = EwPePos.pos[:]  # positions after rotation
        self.r0 = R.from_rotvec([0, 0, 0])  # rotation from the start
        if hasattr(EwPePos, '_rot_vect'):
            __angle = np.arccos(np.dot(EwPePos._rot_vect[0], [0, 1, 0]))
            self._D3plot__angle = __angle * 180 / np.pi      # angle to start good

        # -----------define axis if already present
        self.axes = {}
        if hasattr(EwPePos, 'axes'):
            for i, abc in enumerate('abc'):
                self.axes[abc] = LineAxesr(abc, 1,
                                           axis=EwPePos.axes.T[i],
                                           origin=origin)

                self.axes['g' + abc] = LineAxesr('g', 1,
                                                 axis=-EwPePos.axes.T[i],
                                                 origin=-origin)

                shift = np.delete(EwPePos.axes, [i], axis=1).T
                self.axes['g1' + abc] = LineAxesr('g', 1,
                                                  axis=EwPePos.axes.T[i],
                                                  origin=origin + shift[0])
                self.axes['g2' + abc] = LineAxesr('g', 1,
                                                  axis=EwPePos.axes.T[i],
                                                  origin=origin + shift[1])
        # -----------------------------------------------

        plt.rcParams['toolbar'] = 'None'
        self.fig = plt.figure()

        gs = self.fig.add_gridspec(4, 4)
        self.ax_x = self.fig.add_subplot(gs[0:1, 3])
        self.ax_y = self.fig.add_subplot(gs[1:2, 3])
        self.ax_z = self.fig.add_subplot(gs[2:3, 3])
        self.ax = self.fig.add_subplot(gs[0:4, 0:3])
        plt.figtext(0.8, 0.22, '- a', color='green',
                    size='x-large', weight='bold')
        plt.figtext(0.8, 0.17, '- b', color='blue',
                    size='x-large', weight='bold')
        plt.figtext(0.8, 0.12, '- c', color='black',
                    size='x-large', weight='bold')

        self.plot_ax()
        # ---------------------------------------------
        # ----------------------------------------------

        self._D3plot__cid = self.fig.canvas.mpl_connect('button_press_event',
                                                        self.main_click)

        def rezize_g():
            self.plot_ax()
            self.plot_hist()
        self.__res = self.fig.canvas.mpl_connect('resize_event',
                                                 lambda event: rezize_g())

    def plot_hist(self):
        self.ax_x.cla()
        self.ax_y.cla()
        self.ax_z.cla()
        self.ax_x.set_title('x', x=0.5, y=.7)
        self.ax_y.set_title('y', x=0.5, y=.7)
        self.ax_z.set_title('z', x=0.5, y=.7)
        self.ax_x.hist(np.concatenate(self.pos_i)[:, 0], bins=50, rwidth=4)
        self.ax_y.hist(np.concatenate(self.pos_i)[:, 1], bins=50, rwidth=4)
        self.ax_z.hist(np.concatenate(self.pos_i)[:, 2], bins=50, rwidth=4)
        plt.draw()

    def define_axis(self, abc, m, origin=[0, 0, 0]):
        self.fig.canvas.mpl_disconnect(self._D3plot__cid)
        origin = np.where(np.array(origin) == 1, -self.origin, self.origin)
        self.axes[abc] = LineAxesr(abc, m, origin=origin, rot=self.r0)
        self.axes[abc]._graph_init_()
        plt.waitforbuttonpress(65)
        self.axes[abc].calc_axis(self.r0)
        self.axes[abc].vect = inv(
            self._D3plot__EwPePos.axes) @ self.axes[abc].axis
        self._D3plotr__cid = self.fig.canvas.mpl_connect('button_press_event',
                                                         self.main_click)


class LineAxesr(LineAxes):
    def __init__(self, abc, m, origin=[-0.5, -0.5, -0.5], axis=None, rot=None):
        """origin format:
           vectorial position of the origin
        """
        self.m = m
        self.abc = abc
        self.ori = np.array(origin)
        if rot is not None:
            self.ori = rot.apply(self.ori)
        if axis is not None:
            self.axis = axis
            self.mod = np.sqrt(self.axis @ self.axis.T)
            self.inv_mod = 1 / self.mod
            self.pos_i = self.axis + self.ori
            if rot is not None:
                self.pos_i = rot.apply(self.pos_i) - self.ori
        else:
            fmt = {'a': '+-g', 'b': '+-b', 'c': '+-k',
                   'd': '*--r', 'e': '*--m'}
            self.line, = plt.plot([self.ori[0]], [self.ori[1]], fmt[self.abc])

    def _graph_init_(self):
        m = self.m
        text = plt.text(0, 0, '')
        canv = self.line.figure.canvas

        def endpick(event):
            evdat = np.array([event.xdata, event.ydata, self.ori[2]])
            self.pos_i = (evdat - self.ori) / m + self.ori
            self.inv_mod = m / np.sqrt(self.pos_i[0]**2 + self.pos_i[1]**2)
            canv.mpl_disconnect(self.__cid)
            canv.mpl_disconnect(self.__mid)
            text.remove()
            return

        def move_m(event):
            if event.inaxes != self.line.axes:
                return
            # canv.restore_region(self.bkg_cache)
            datax = np.linspace(self.ori[0], event.xdata, m + 1)
            datay = np.linspace(self.ori[1], event.ydata, m + 1)
            inv_mod = round(m / np.sqrt(event.xdata**2 + event.ydata**2), 2)
            self.line.set_data(datax, datay)
            self.line.axes.draw_artist(self.line)
            text.set_position((event.xdata, event.ydata))
            text.set_text(f'{inv_mod:3.2f} ')
            self.line.axes.draw_artist(text)
            # canv.blit(self.line.axes.bbox)
            plt.draw()
            return

        # self.bkg_cache = canv.copy_from_bbox(self.line.axes.bbox.padded(0))
        self.__mid = canv.mpl_connect('motion_notify_event', move_m)
        self.__cid = canv.mpl_connect('button_press_event', endpick)
        return

    def plot(self):
        fmt = {'a': '+-g', 'b': '+-b', 'c': '+-k',
               'd': '*--r', 'e': '*--m', 'g': '--k'}
        if self.abc in fmt.keys():
            self.line, = plt.plot([0], [0], fmt[self.abc])
        else:
            self.line, = plt.plot([0], [0], color=(0.3, 0.3, 0.3))
        datax = np.linspace(self.ori[0], self.pos_i[0] * self.m, self.m + 1)
        datay = np.linspace(self.ori[1], self.pos_i[1] * self.m, self.m + 1)
        self.line.set_data(datax, datay)
        self.line.axes.draw_artist(self.line)

    def rotate(self, r, store=False):
        p = r.apply(self.pos_i)
        z = r.apply(self.ori)
        evdat = (p - z) * self.m + z
        datax = np.linspace(z[0], evdat[0], self.m + 1)
        datay = np.linspace(z[1], evdat[1], self.m + 1)
        self.line.set_data(datax, datay)
        self.line.axes.draw_artist(self.line)
        if store:
            self.ori, self.pos_i = z, p

    def calc_axis(self, r0):
        self.axis = self.pos_i - self.ori
        self.axis = r0.inv().apply(self.axis)
        self.mod = np.sqrt(self.axis @ self.axis.T)
        self.inv_mod = 1 / self.mod

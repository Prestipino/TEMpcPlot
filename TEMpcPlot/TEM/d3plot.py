import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

plt.ion()

"""
   pos_0   initial position
   pos_i   position after the set of rotation and scale.
"""


class D3plot(object):
    """
    Class used to stored a set of 3D peaks
    """

    def __init__(self, EwPePos, size='o'):
        self.__size = size
        # list in whic pos are sotred for each image
        self.pos_i = EwPePos.pos[:]  # positions after rotation
        self.r0 = R.from_rotvec([0, 0, 0])  # rotation from the start
        self.axes = {}

        plt.rcParams['toolbar'] = 'None'
        self.fig = plt.figure()

        gs = self.fig.add_gridspec(5, 5)
        self.ax_x = self.fig.add_subplot(gs[4, 0:4])
        self.ax_y = self.fig.add_subplot(gs[0:4, 4])
        self.ax = self.fig.add_subplot(gs[0:4, 0:4])
        plt.figtext(0.8, 0.22, '- a', color='green',
                    size='x-large', weight='bold')
        plt.figtext(0.8, 0.17, '- b', color='blue',
                    size='x-large', weight='bold')
        plt.figtext(0.8, 0.12, '- c', color='black',
                    size='x-large', weight='bold')

        self.plot_ax()
        x = abs(np.array(self.ax.get_xlim())).argmax()
        y = abs(np.array(self.ax.get_ylim())).argmax()
        self.__axxlim = (-x, x)
        self.__axylim = (-y, y)
        self.ax.set_xlim(self.__axxlim)
        self.ax.set_ylim(self.__axylim)
        plt.draw()
        self.plot_hist()

        if hasattr(EwPePos, '_rot_vect'):
            self.__angle = np.arccos(np.dot(EwPePos._rot_vect[0],
                                            [0, 1, 0]))
            self.__angle *= 180 / np.pi
        self.__cid = self.fig.canvas.mpl_connect('button_press_event',
                                                 self.main_click)

        def rezize_g():
            self.plot_ax()
            self.plot_hist()
        self.__res = self.fig.canvas.mpl_connect('resize_event',
                                                 lambda event: rezize_g())

    def plot_ax(self):
        self.ax.cla()
        self.ax.set_aspect(aspect='equal', adjustable='datalim')
        self.fig.canvas.draw()
        self.bkg_cache = self.fig.canvas.copy_from_bbox(self.ax.bbox.padded(0))
        # list with the graph
        self.m_planes = list()
        for j, i in enumerate(self.pos_i):
            self.m_planes.extend(self.ax.plot(*i.T[:2], self.__size, label=str(j)))
        self.zer, = self.ax.plot([0], [0], 'xr', markersize=20)
        fmt = {'a': '+-g', 'b': '+-b', 'c': '+-k', }
        for abc, i in self.axes.items():
            i.line, = self.ax.plot([0, i.pos_i[0]], [0, i.pos_i[1]], fmt[abc])
        self.ax.set_axis_off()
        self.ax.set_frame_on(False)

        # legend
        plt.rcParams['toolbar'] = 'toolbar2'
        self.__axxlim = self.ax.get_xlim()
        self.__axylim = self.ax.get_ylim()

    def plot_hist(self):
        self.ax_x.cla()
        self.ax_y.cla()
        self._xh = self.ax_x.hist(np.concatenate(self.pos_i)[:, 0],
                                  bins=100, rwidth=4)
        self._yh = self.ax_y.hist(np.concatenate(self.pos_i)[:, 1],
                                  bins=100, rwidth=4, orientation='horizontal')
        self.ax_x.set_autoscalex_on(False)
        self.ax_y.set_autoscalex_on(False)
        self.ax_x.set_xlim(self.__axxlim)
        self.ax_y.set_ylim(self.__axylim)
        plt.draw()

    def zoom(self, event):
        self.fig.canvas.restore_region(self.bkg_cache)
        zoom_m = np.sqrt(np.sum(np.array([self.__c_x0, self.__c_y0])**2))
        zoom_m /= np.sqrt(np.sum(np.array([event.xdata, event.ydata])**2))
        self.__axxlim = (self.__axxlim[0] * zoom_m, self.__axxlim[1] * zoom_m)
        self.__axylim = (self.__axylim[0] * zoom_m, self.__axylim[1] * zoom_m)
        self.ax.set_xlim(self.__axxlim)
        self.ax.set_ylim(self.__axylim)
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
        if len(self.axes) > 0:
            for i in self.axes.keys():
                z = r.apply(self.axes[i].pos_i)
                self.axes[i].line.set_data([0, z[0]], [0, z[1]])
                self.ax.draw_artist(self.axes[i].line)
        self.ax.draw_artist(self.zer)
        self.fig.canvas.blit(self.ax.bbox)

    def __stab_rot(self, r):
        """track all rotation ...
        """
        self.r0 = r * self.r0
        for i, j in enumerate(self.pos_i):
            self.pos_i[i] = r.apply(j)
        for i in self.axes.keys():
            self.axes[i].pos_i = r.apply(self.axes[i].pos_i)

    def __m_rotate(self, event):
        r = self.__find_rot(event)
        self.__rotate(r)

    def rotatex(self, deg=90):
        rot_vec = np.array((np.radians(deg), 0, 0))
        r = R.from_rotvec(rot_vec)
        self.__rotate(r)
        self.__stab_rot(r)

    def rotatey(self, deg=90):
        rot_vec = np.array((0, np.radians(deg), 0))
        r = R.from_rotvec(rot_vec)
        self.__rotate(r)
        self.__stab_rot(r)

    def rotatez(self, deg=90):
        rot_vec = np.array((0, 0, np.radians(deg)))
        r = R.from_rotvec(rot_vec)
        self.__rotate(r)
        self.__stab_rot(r)

    def rotate_0(self):
        r = self.r0.inv()
        self.__rotate(r)
        self.__stab_rot(r)
        self.rotatez(self.__angle)

    def allign_a(self):
        rot_vec = np.cross(self.axes['a'].pos_i, np.array([0, 0, 1]))
        r_mod = np.sqrt(rot_vec @ rot_vec)
        a_mod = np.sqrt(self.axes['a'].pos_i @ self.axes['a'].pos_i)
        rot_vec_n = rot_vec / r_mod
        rot_vec_n *= -np.arcsin(r_mod / a_mod)
        r = R.from_rotvec(rot_vec_n)
        self.__rotate(r)
        self.__stab_rot(r)

    def allign_b(self):
        rot_vec = np.cross(self.axes['b'].pos_i, np.array([0, 0, 1]))
        r_mod = np.sqrt(rot_vec @ rot_vec)
        a_mod = np.sqrt(self.axes['b'].pos_i @ self.axes['b'].pos_i)
        rot_vec_n = rot_vec / r_mod
        rot_vec_n *= -np.arcsin(r_mod / a_mod)
        r = R.from_rotvec(rot_vec_n)
        self.__rotate(r)
        self.__stab_rot(r)

    def allign_c(self):
        rot_vec = np.cross(self.axes['c'].pos_i, np.array([0, 0, 1]))
        r_mod = np.sqrt(rot_vec @ rot_vec)
        a_mod = np.sqrt(self.axes['c'].pos_i @ self.axes['c'].pos_i)
        rot_vec_n = rot_vec / r_mod
        rot_vec_n *= -np.arcsin(r_mod / a_mod)
        r = R.from_rotvec(rot_vec_n)
        self.__rotate(r)
        self.__stab_rot(r)

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
                for i in self.axes.keys():
                    self.axes[i].pos_i = r.apply(self.axes[i].pos_i)
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
        """
        assert abc in 'abc', 'only three axis a, b, c '
        self.fig.canvas.mpl_disconnect(self.__cid)
        self.axes[abc] = LineAxes(abc, m)
        plt.waitforbuttonpress(65)
        self.axes[abc].pos = self.r0.inv().apply(self.axes[abc].pos_i)
        print(self.axes[abc].pos)
        self.__cid = self.fig.canvas.mpl_connect('button_press_event',
                                                 self.main_click)

    def legend(self):
        self.ax.legend()

class LineAxes:
    def __init__(self, abc, m):
        fmt = {'a': '+-g', 'b': '+-b', 'c': '+-k', }
        self.line, = plt.plot([0], [0], fmt[abc])
        text = plt.text(0, 0, '')
        canv = self.line.figure.canvas

        def endpick(event):
            self.pos_i = np.array([event.xdata / m,
                                   event.ydata / m, 0])
            #print(self.pos_i)
            self.inv_mod = m / np.sqrt(self.pos_i[0]**2 + self.pos_i[1]**2)
            canv.mpl_disconnect(self.__cid)
            canv.mpl_disconnect(self.__mid)
            text.remove()
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
        self.line.remove()
        # plt.legend()
        return

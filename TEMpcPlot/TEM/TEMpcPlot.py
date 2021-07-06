import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, Button, CheckButtons
# from matplotlib.backend_tools import ToolBase
# plt.rcParams['toolbar'] = 'toolmanager'
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import inv
import scipy.ndimage as ndimage
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import pickle
import glob

from importlib import reload
import os

from .. import dm3_lib as dm3
from .. import Symmetry
from . import plt_p
from .profileline import profile_line
from . import d3plot
from . import more_widget as mw
from . import math_tools as mt
from . import Index as ind
# import  scipy.optimize  as opt
plt.ion()

# from skimage.measure import LineModelND, ransa


rpd = np.pi / 180.0
RSQ2PI = 1. / np.sqrt(2. * np.pi)
SQ2 = np.sqrt(2.)
RSQPI = 1. / np.sqrt(np.pi)
R2pisq = 1. / (2. * np.pi**2)


def sind(x):
    return np.sin(x * rpd)


def asind(x):
    return np.arcsin(x) / rpd


def tand(x):
    return np.tan(x * rpd)


def cosd(x):
    return np.cos(x * rpd)


def acosd(x):
    return np.arccos(x) / rpd


def rdsq2d(x, p):
    return round(1.0 / np.sqrt(x), p)


class Object(object):
    pass


def cir_foot(c_rad):
    b = np.zeros((2 * c_rad + 1, 2 * c_rad + 1))

    b[c_rad, :] = 1
    b[:, c_rad] = 1

    z = 0
    for i in range(1, c_rad):  # lines
        for j in range(z, c_rad - i + 1):
            if ((c_rad - i)**2 + (j)**2) <= c_rad**2:

                b[i: 2 * c_rad - i + 1, j + c_rad] = 1
                b[c_rad - j:c_rad + j + 1, i] = 1

                b[i: 2 * c_rad - i + 1, c_rad - j] = 1
                b[c_rad - j:c_rad + j + 1, 2 * c_rad - i] = 1

            else:
                z = j
                break

    return b


class LineBuilder:
    """
    class defined to trace lines on an existing figure
    the class one time defined calculate few attributes
    self.origin = origin of the line
    self.vect = vector represented
    self.mod = lenght of the line
    self.fline = line object passing grom the two point


    """

    def __init__(self):
        self.__xtl = []
        return

    def defFplot(self, ax, plot, ima):
        """graphical definition of line
        define the line parameter by clicking
        Args:
           plot (bool) : if must plot the profile
           ima (TEMpcPlot.Mimage): image for profile calulation 
           args : argument for the plot 
           a graphical approach
        """
        if plot:
            def callb(event):
                self.calc(event)
                self.plot_profile(ima)
            callback = callb
        else:
            callback = self.calc

        self.line = mw.LineBuilder(ax, callback=callback, useblit=True,
                                   stay=True)

    def __del__(self):
        for i in self.__xtl:
            i.remove()
        return

    def calc(self, data=None):
        """calc
        format data [[x0,x1][y0, y1]]
        """
        if not(data is None):
            y0, x0, y1, x1 = *data[0], *data[-1]
        self.origin = np.array([x0, y0])
        self.vect = np.array([x1 - x0, y1 - y0])
        self.mod = np.sqrt(self.vect.dot(self.vect))
        self.fline = np.poly1d(np.polyfit([x0, x1], [y0, y1], 1))
        return

    def dist_p(self, coor):
        """return the distances of a set of point from the line
           the position of the point if given by an iterable of shape 2xN
        """
        coor = np.array(coor)
        assert 2 in coor.shape
        if coor.shape[0] == 2:
            coor = coor.T
        return np.abs((np.cross(self.vect, coor - self.origin)) / self.mod)

    def ref_near_p(self, coor, max_dist, plot_p=True):
        """refine the position of the line forcing the line
           to pass near a set of point
           input
           coor: tuple(array, array) position of a set of point
           max_dist: float max_distance from the line to use a point to
                      refine the line
           plot_p  : boolean if peak used should be crossed
        """
        coor = np.array(coor)
        dist = self.dist_p(coor)
        rcoor = np.compress(dist < max_dist,
                            np.array(coor).T,
                            axis=0).T
        self.fline = np.poly1d(np.polyfit(rcoor[0], rcoor[1], 1))
        self.y0, self.y1 = self.fline([self.x0, self.x1])
        self.line.set_data([self.y0, self.y1], [self.x0, self.x1])
        self.line.figure.canvas.draw()
        self.calc()
        self.rcoor = rcoor
        if plot_p:
            self.__xtl += plt.plot(rcoor[1],
                                   rcoor[0],
                                   'x', markersize=max_dist)
        return

    def plot_profile(self, ima, lw=1, order=1, plot=True):
        """plot profile along the line, nned to define the image before as ima
        """
        self.profile = profile_line(ima, self.origin,
                                    self.origin + self.vect,
                                    linewidth=lw,
                                    order=order)
        if plot:
            plt.figure()
            plt.plot(self.profile)


class PeakL(list):
    """
    this class manage a 2D peak list it is a tuple with
    two array that contains the x and y  of positions of the peaks

    attributes
    ----------------------------------------------------------
    self.int  array with the intensities of the peaks
    ----------------------------------------------------------

    methods
    ----------------------------------------------------------
    self.del_PlotPeak()
    self.del_peak(n)
    self.deplot()
    self.plot()
    ----------------------------------------------------------
    """

    def __init__(self, inlist, min_dis=15, threshold=300, dist=None, symf=None,
                 circle=False, comass=True):
        """
        input
        inlist : source of the list of peak. Could be a tuple of peaks or an
                 image if is it an image the search is performed automatically
        threshold :total range coefficent the minimal intensity of the peak
                 should be at list tr_c*self.ima.max()

        min_dis: coefficent in respect of the center radious
                  peaks should be separate from at list self.rad*rad_c

        circle : boolean if min_dis use a square of a circle
                 square is less precise but faster

        comass : boolean if center of mass opton refines the position using the
                 center of mass, repeat 5 time the refinement or stop if any
                  coordiname moves more than 2 pixel

        dist : float or integer maximum distance from the center of the image

        """
        if isinstance(inlist, tuple):
            super().__init__(inlist)
        elif isinstance(inlist, Mimage):
            pos, intent = self.findpeaks(inlist.ima, int(min_dis),
                                         threshold, dist, symf, circle, comass,
                                         np.array(inlist.center))
            super().__init__(pos)
            self.int = intent
            self.ps_in = {'min_dis': int(min_dis), 'threshold': threshold,
                          'dist': dist, 'circle': circle, 'comass': comass}
        else:
            pass

    def del_peak(self, n):
        self[0] = np.delete(self[0], n)
        self[1] = np.delete(self[1], n)
        if hasattr(self, 'int'):
            self.int = np.delete(self.int, n)
        if hasattr(self, 'lp'):
            self.lp.set_data(self[1], self[0])
            self.lp.figure.canvas.draw()

    @classmethod
    def findpeaks(cls, ima, min_dis=15, threshold=300, dist=None, symf=None,
                  circle=True, comass=True, center=None):

        if circle:
            foot = cir_foot(min_dis)
        else:
            foot = np.ones((2 * min_dis + 1, 2 * min_dis + 1))

        def subselect(center):
            try:
                xx = ima[center[0] - min_dis:center[0] + min_dis + 1,
                         center[1] - min_dis:center[1] + min_dis + 1]
            except IndexError:
                return np.array(())
            if xx.shape == (min_dis * 2 + 1, min_dis * 2 + 1):
                return xx
            else:
                return np.array(())

        def center_m():
            out = []
            for c in zip(*coor.round().astype(int)):
                regio = subselect(c)
                if regio.size > 0:
                    val = (ima[c[0], c[1]] - regio.min()) / 2.0
                    mc = ndimage.center_of_mass(
                        (regio > val + regio.min()) * foot)
                    out.append([c[0] + mc[0] - min_dis,
                                c[1] + mc[1] - min_dis])
                else:
                    out.append(c)
            return np.array(out).T

        ima_max = ndimage.maximum_filter(ima, footprint=foot)
        ima_min = ndimage.minimum_filter(ima, footprint=foot)
        coor = np.where((ima == ima_max) * ((ima - ima_min) > threshold))

        # if center of mass opton refines the position using the center of mass
        # repeat 5 time the refinement or stop if any coordiname moves
        # more than 2 pixel
        if comass:
            coor = np.asarray(coor, dtype=float)
            for i in range(5):
                newcoor = center_m()
                coor = newcoor.round(2)
                if np.max(np.abs(coor - newcoor)) < 2:
                    break
                else:
                    print(np.max(np.abs(coor - newcoor)) < 2)

        # remove peaks around the center
        coor_d = coor.T - center
        coor_d = np.sqrt(np.sum(coor_d**2, axis=1))
        cond = coor_d > min_dis
        # distance parameter
        if dist:
            cond = cond &  (coor_d < dist)
        coor = coor[:, cond]
        coor = np.insert(coor, 0, center, axis=1)


        # symmetry parameter
        if symf:
            coor = coor.T
            symf = symf**2
            c_coor = coor - center
            index = list(range(len(c_coor)))
            sym_coor = []
            for i in index[:-1]:
                if len(c_coor) < i + 2:
                    break
                z = c_coor[i + 1:] + c_coor[i]
                z1 = np.sum(z**2, 1)
                zam = z1.argmin()
                if z1[zam] < symf:
                    sym_coor.append(coor[i])
                    sym_coor.append(c_coor[i + 1 + zam] + center)
                    c_coor = np.delete(c_coor, i + 1 + zam, 0)
                    del index[i + 1 + zam]
            coor = np.array(sym_coor).T
        z = ((coor + .5).astype(int))
        # print('in peak', ima[z].shape, z.shape)
        return tuple(coor), ima[tuple(z)] - ima_min[tuple(z)]


    @property
    def r(self):
        return list(reversed(self))

    def plot(self):
        self.lp, = plt.plot(*self.r, 'r.', picker=5)

    def deplot(self):
        self.lp.remove()

    def __del__(self):
        try:
            self.deplot()
        except:
            pass

    def del_PlotPeak(self):
        """clic left button to apeak to delete
           to stop click righ peak
        """
        mw.Picker(self.lp.axes, self.lp, callback=self.del_peak)

    def del_PlotRange(self, ):
        """delete the peak inside a rectangle plotted on the axis
        """
        def del_inside(origin, vect, width):
            '''delete point inside the rrectangle
            '''
            # calc perp line
            p_vect = width * mt.perp_vect(vect)
            cen = origin + (vect / 2)
            # delete points
            coor = np.array(self)
            dist_1 = mt.dist_p2vect(np.flip(origin),
                                    np.flip(vect), coor)
            dist_2 = mt.dist_p2vect(np.flip(cen),
                                    np.flip(p_vect) * 2, coor)
            rcoor = np.where((dist_1 < width) & (dist_2 < (mt.mod(vect) / 2)))
            for i in np.flip(rcoor):
                self.del_peak(i)

        canv = self.lp.figure.canvas
        self._break_loop = False
        if not hasattr(self, 'lp'):
            return
        ax = self.lp.axes
        if canv.widgetlock.locked():
            return
        self.rect = mw.RectangleBuilder(ax, callback=del_inside)
        # self.rect.canvas.widgetlock(self.rect)

    def help(self):
        print(self.__doc__)


class Mimage():
    """
    this class manage a single image

    attributes
    ----------------------------------------------------------
    self.ima = matrix with data
    self.info gives info
    self.center = center of the image (pixel)
    self.rad  = radius of central peak > satur*max(image)
    self.scale = scale
    self.Peaks = PeakL object containing the peaks
    ----------------------------------------------------------

    methods
    ----------------------------------------------------------
    self.despike(satur=0.9)
    self.find_centralpeak(satur=0.8)
    self.find_peaks(rad_c=1.5, tr_c=0.02, dist=None)
    self.plot(new=True, log=False, peaks=True, *args, **kwds)
    self.angle()
    ----------------------------------------------------------

    -
    """

    def __init__(self, filename, peaks=None):
        """
        at the creation the image is
        1) read
        2) despiked
        3) center is found
        4) all peaks are searched

        ###########################################

        important attributes created
           self.ima  : numerical matrix with the data
           self.info : info on the data
           self.scale: scale of the data
           self.center: coordinaed of the center
           self.rad   : radius of of central peak at 80% of saturation
        """

        if filename[-3:].upper() in 'TIFF':
            self.ima = plt.imread(filename)
            self.info = Object()
            self.info.filename = filename
            self.scale = 1
        elif filename[-3:].upper() in 'DM3':
            self.info = dm3.DM3(filename)
            self.ima = self.info.imagedata
            self.scale = self.info.pxsize[0]
        self.despike()
        self.center, self.rad = self.find_centralpeak()
        if peaks is None:
            self.find_peaks()
        else:
            self.Peaks = peaks
        return

    def plot(self, new=True, log=False, peaks=True, *args, **kwds):
        """
        plot the image
        """
        if new:
            fig = plt.figure()
        else:
            fig = plt.gcf()
            ax = plt.gca()
            ax.cla()

        if log:
            self.pltim = plt.imshow(np.log(np.abs(self.ima)), *args, **kwds)
        else:
            self.pltim = plt.imshow(self.ima, *args, **kwds)
        pltcenter = plt.plot(*reversed(self.center), 'bx')
        ax = plt.gca()
        plt.title(f'Image {self.info.filename}')

        def format_coord(x, y):
            d1 = np.round(np.sqrt((x - self.center[0])**2 +
                                  (y - self.center[1])**2), 2)
            d2 = 1 / (d1 * self.scale)
            # return f'{z:s} d={dist2:2.4f} [{dist1:2.4f} pixel]'
            z = f'x={y:4.1f}, y={x:4.1f},     d_sp={d2:4.4f}nm {d1:4.2f}pix'
            return f'{z:s}             '
        ax.format_coord = format_coord

        if peaks:
            if hasattr(self, 'Peaks'):
                self.Peaks.plot()

    def despike(self, satur=0.90):
        '''
        satur = fraction of the maximum
        if the surface is 1 pixel the pixel is replaced
        '''
        self.satur = satur
        # np.bincount(a.flat)[1:]
        imlabm, nlab = ndimage.label(self.ima > satur * self.ima.max())
        surf = np.bincount(imlabm.flat)[1:]
        if max(surf) == 1:
            imlabm = ndimage.median_filter(self.ima,
                                           footprint=[[1, 1, 1],
                                                      [1, 0, 1],
                                                      [1, 1, 1]])
            self.ima = np.where(self.ima < satur * self.ima.max(),
                                self.ima, imlabm)
        return

    def find_centralpeak(self, satur=0.80):
        '''
        The function recognize the cetral peak as the
        largest surface of the peaks that are higher than the satur*max(image)
        returns the position and the radious
        '''
        self.satur = satur
        imlabm, nlab = ndimage.label(self.ima > satur * self.ima.max())
        surf = np.bincount(imlabm.flat)[1:]
        l_max = np.argmax(surf) + 1
        coor = ndimage.center_of_mass(self.ima, labels=imlabm, index=l_max)
        radius = np.sqrt(surf[l_max - 1] / np.pi)
        return coor, radius

    def find_peaks(self, rad_c=1.5, tr_c=0.02, dist=None, symf=None):
        '''
        finds peaks in the image

        inputs:
        tr_c :total range coefficent, the minimal intensity of the peak
                 should be at list tr_c*self.ima.max()

        rad_c: coefficent in respect of the center radious
                  peaks should be separate from at list self.rad*rad_c

        dist : maximal distance from the center (pixel)

        '''
        if hasattr(self, 'Peaks') and hasattr(self.Peaks, 'lp'):
            plot = True
        else:
            plot = False

        self.Peaks = PeakL(self, min_dis=round(self.rad * rad_c),
                           threshold=tr_c * self.ima.max(), dist=dist,
                           symf=symf)

        if plot:
            self.Peaks.plot()

    def profile_Line(self, data=None, lw=1, order=1, plot=True):
        """create a line object store in the attribute self.line
           and calculae its profile stored in self.line.profile
        """
        fig = plt.gcf()
        self.line = LineBuilder()
        if data is None:
            self.line.defFplot(fig.axes[0], plot=True, ima=self.ima)
        else:
            self.line.calc(data)

    def angle(self):
        fig = plt.gcf()
        self.line = [LineBuilder(), LineBuilder()]
        self.line[0].defFplot(fig.axes[0], plot=False, ima=self.ima)
        while not(hasattr(self.line[0], 'fline')):
            plt.pause(0.3)
        self.line[1].defFplot(fig.axes[0], plot=False, ima=self.ima)
        while not(hasattr(self.line[1], 'fline')):
            plt.pause(0.3)
        angle = acosd((self.line[0].vect @ self.line[1].vect) /
                      (self.line[0].mod * self.line[1].mod))
        return angle

    def help(self):
        print(self.__doc__)


class SeqIm(list):
    """sequence of images

    this class is supposed to use a sequence of image.
    each element of the class is an image

    Args:
        filenames (list): list string describing the exception.
        filesangle (str): Human readable file with angles


    Attributes:
        EwP  (TEMpcPlot.EwaldPeaks): Ewald peaks 3D set of peaks
        rot_vect (list): list of Rotation vector for each image
        scale  (list): scale(magnification) of the images
        ima  (TEMpcPlot.Mimage): current image of the sequence

    Note:
        | Methods to use:
        | def find_peaks(rad_c=1.5, tr_c=0.02, dist=None)
        | def D3_peaks(stollerance)
        | def plot(log=False)
        | def plot_cal(axes)    
        | def save(axes)
        | def load(axes)
        | def help()
    """

    def __init__(self, filenames, filesangle=None, *args, **kwords):
        """
        when the class is created all the image are read and stored in
        the iterator as Mimage class. At the creation of Mimage a search
        of the peaks is done
        attribute angle it rapresent the angle of rotation

        """
        def red_sqi(filena):
            filelist = []
            gon_angles = []
            with open(filena, 'r') as sqi:
                for line in sqi.readlines():
                    sline = line.strip()
                    if sline == '':
                        continue
                    elif sline[0] == '#':
                        continue
                    f, xa, xb = sline.split()[:3]
                    filelist.append(f)
                    gon_angles.append([float(xa), float(xb)])
            gon_angles = np.radians(np.array(gon_angles))
            return filelist, gon_angles

        if isinstance(filenames, str):
            if filenames[-3:].lower() == 'sqi':
                self.filenames, gon_angles = red_sqi(filenames)
            else:
                self.filenames = glob.glob(filenames)
                assert len(filenames) > 0, 'no image found'
        elif isinstance(filenames, list):
            self.filenames = filenames
        else:
            raise TypeError('filenames type not not available')

        assert isinstance(self.filenames, list), 'list of filenames please'

        if filesangle:
            if isinstance(filenames, str):
                gon_angles = np.radians(np.loadtxt(filesangle)[:, :2])
            if isinstance(filenames, list):
                gon_angles = np.array(filesangle)

        for i in self.filenames:
            assert os.path.isfile(i), f'No such file: \'{i}\''

        super().__init__([Mimage(i) for i in self.filenames])
        for i, im in enumerate(self):
            setattr(im.info, 'gon_angles', gon_angles[i])

        if len(set([i.scale for i in self])) == 1:
            self.scale = self[0].scale
        else:
            raise ValueError('images with different scales')
        self.__rot__ = gon_angles
        g_ang = gon_angles - gon_angles[0]
        ssign = 0 if np.argmax(np.abs(g_ang), axis=0).mean() <= 0.5 else 1
        self.angles = np.arccos(
            np.cos(g_ang[:, 0]) * np.cos(g_ang[:, 1])) * np.sign(g_ang[:, ssign])
         

    def help(self):
        """
        print class help
        """
        print(self.__doc__)

    def find_peaks(self, rad_c=1.5, tr_c=0.02, dist=None, symf=None):
        """ findf the peak
        allows to search again the peaks in all the image witht the same
        parameter

        Args:
            tr_c (float): total range coefficent the minimal intensity of the peak should be at list tr_c*self.ima.max()
            rad_c (float): coefficent in respect of the center radious peaks should be separate from at list self.rad*rad_c
            dist: (float): maximum distance in pixel

        Examples:
        >>> Exp1.find_peaks()
        """
        for i in self:
            if hasattr(i, 'Peaks'):
                del i.Peaks
            i.find_peaks(rad_c, tr_c, dist, symf)
        if hasattr(self, 'ima'):
            self.ima.Peaks.plot()

    def D3_peaks(self, tollerance=15, tol_ang=5, tol_in=0.10):
        """sum and correct the peaks of all images
        Args:
            tollerance () = pixel tollerance to determine if a peak
                        in two images is the same peak.
        """
        # shape of one element of all  peaks n_p *2
        all_peaks = [np.array(i.Peaks).T - np.array(i.center) for i in self]
        out = mt.find_common_peaks(tollerance, all_peaks)

        print('found %d common peaks' % out.shape[1])

        # find possible rotation on the plane of the rotation axes
        # evaluated between the fitted line that pass for the common peaks
        # return the angle of cortrection and a vector passing from all points
        self.zangles, self.rot_vect = mt.find_zrot_correction(out, tollerance)
        print('angle correction', np.round(np.degrees(self.zangles), 2))


        # calibration for rotation of the image i the plane
        # and correct the center on the basis of average difference of out
        for i, peaks in enumerate(all_peaks):
            if i != 0:
                peaks = peaks @ mt.zrotm(-self.zangles[i])
                shift = out[0] - (out[i] @ mt.zrotm(-self.zangles[i]))
                shift = shift.sum(axis=0) / len(out[i])
                all_peaks[i] = peaks + shift
        all_peaks = [np.column_stack((i, np.zeros(len(i)))) for i in all_peaks]
      

        # absolute rotation
        self.__rot__ = np.array([im.info.gon_angles for im in self])
        self.z0 = mt.find_z_rotation(self.__rot__, self.rot_vect)[0]

        axis = mt.creaxex(self.__rot__, self.z0)
        sign = np.where(axis @ self.rot_vect > 0, -1, 1)
        angles = np.array([mt.mod(i) for i in axis])
        self.angles = np.insert(sign * angles, 0, 0.0)

        intensity = []
        position = []
        for i, peaks in enumerate(all_peaks):
            if i == 0:
                position.append(peaks)
            else:
                r = R.from_rotvec(self.rot_vect * self.angles[i])
                position.append(r.apply(peaks))
        
        intensity = [i.Peaks.int for i in self]
        position = [i * self.scale for i in position]

        self.EwP = EwaldPeaks(position, intensity, rot_vect=self.rot_vect,
                              angles=self.angles, r0=self.__rot__, z0=self.z0)
        # abs_rotz
        return 

    def plot_cal(self, axes=None, log=False, *args, **kwds):
        '''
        plot the images of the sequences with peaks
        if axes is not defined and an EwP.axes is defined such bases will be used

        Args:
            axes base for reciprocal space as 
                            one defined in EwP
            log (Bool): plot logaritm of intyensity
            aargs anf keyworg directly of matplotlib plot

        Examples:
            >>> Exp1.plot(Exp1.EwP.axes, log=True)
            >>> Exp1.plot(Exp1.EwP.axes)

        left button define an annotation 
        right button remoce an anotation
        left button + move drag ana annotation
        '''
        reload(plt)
        fig = plt.figure()
        ax = plt.axes([0.05, 0.10, 0.75, 0.80])
        if axes is None:
            try:
                axes = self.EwP.axes
            except AttributeError as A:
                raise A

        tool_b = fig.canvas.manager.toolbar

        tbarplus = mw.ToolbarPlusCal(self, axes, log=log, fig=fig,
                                     ax=ax, tool_b=tool_b, *args, **kwds)

        axvmax = plt.axes([0.80, 0.10, 0.04, 0.80])
        vmaxb = Slider(ax=axvmax, label='max', valmin=0.01, valmax=100.0, valinit=100,
                       orientation="vertical")

        def refresh(val):
            i = tbarplus.index
            mini = np.abs(self[i].ima).min()
            maxi = self[i].ima.max()
            vmax = mini + vmaxb.val**3 * ((maxi - mini) / 1000000)
            if log:
                vmax = np.log(vmax)
            plt.sca(ax)
            nonlocal kwds
            if kwds:
                kwds.update({'vmax': vmax})
            else:
                kwds = {'vmax': vmax}
            tbarplus.pltim.set_clim(vmax=vmax)
            tbarplus.kwds = kwds

        vmaxb.on_changed(refresh)

    def plot(self, log=False, fig=None, ax=None, tool_b=None, *args, **kwds):
        '''
        plot the images of the sequences with peaks

        Args:
            log (Bool): plot logaritm of intyensity
            aargs anf keyworg directly of matplotlib plot

        Examples:
            >>> Exp1.plot(log=True)
            >>> Exp1.plot(True)
            >>> Exp1.plot(1)
            >>> Exp1.plot(0)
            >>> Exp1.plot(vmin = 10, )
        '''
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = plt.axes([0.0, 0.10, 0.80, 0.80])
        if tool_b is None:
            tool_b = fig.canvas.manager.toolbar

        self.ima = self[0]
        self.ima.plot(new=0, log=log, *args, **kwds)
        ax.set_axis_off()
        ax.set_frame_on(False)

        tbarplus = mw.ToolbarPlus(self, log=log, fig=fig,
                                  ax=ax, tool_b=tool_b, *args, **kwds)

        axcolor = 'lightgoldenrodyellow'
        axinte = plt.axes([0.75, 0.87, 0.20, 0.03], facecolor=axcolor)
        axdist = plt.axes([0.75, 0.82, 0.20, 0.03], facecolor=axcolor)
        axspac = plt.axes([0.75, 0.77, 0.20, 0.03], facecolor=axcolor)
        axsym = plt.axes([0.75, 0.72, 0.20, 0.03], facecolor=axcolor)

        inteb = Slider(ax=axinte, label='int', valmin=0.01, valmax=10.0, valinit=5)  # , valstep=delta_f
        distb = Slider(ax=axdist, label='dis', valmin=0, valmax=1, valinit=0.9)
        spacb = Slider(ax=axspac, label='rad', valmin=0.1, valmax=10.0, valinit=1)  # , valstep=delta_f
        symb = Slider(ax=axsym, label='sym', valmin=0.0, valmax=20.0, valinit=0)

        # Create a `matplotlib.widgets.Button` to apply to all
        axapal = plt.axes([0.75, 0.67, 0.15, 0.04])
        tbarplus.butpal = Button(axapal, 'apply to all', color=axcolor, hovercolor='0.975')

        axvmax = plt.axes([0.75, 0.17, 0.15, 0.04])
        vmaxb = Slider(ax=axvmax, label='max', valmin=0.01, valmax=100.0, valinit=100)

        def update(val):
            inte = inteb.val**2 / 101
            dist = distb.val
            spac = spacb.val
            symB = symb.val
            plt.sca(ax)
            try:
                self.ima.find_peaks(rad_c=spac, tr_c=inte,
                                    dist=dist * len(self.ima),
                                    symf=symB)
            except ValueError:
                pass

        inteb.on_changed(update)
        distb.on_changed(update)
        spacb.on_changed(update)
        symb.on_changed(update)

        def app_all(event):
            inte = inteb.val**2 / 101
            dist = distb.val
            spac = spacb.val
            symB = symb.val
            plt.sca(ax)
            self.find_peaks(rad_c=spac, tr_c=inte,
                            dist=dist * len(self.ima),
                            symf=symB)
        tbarplus.butpal.on_clicked(app_all)

        def refresh(val):
            mini = np.where(self.ima.ima > 0, self.ima.ima, np.inf).min()
            maxi = self.ima.ima.max()
            vmax = mini + vmaxb.val**3 * ((maxi - mini) / 1000000)
            if log:
                vmax = np.log(vmax)
            plt.sca(ax)
            nonlocal kwds
            if kwds:
                kwds.update({'vmax': vmax})
            else:
                kwds = {'vmax': vmax}
            self.ima.pltim.set_clim(vmax=vmax)
            tbarplus.kwds = kwds

        vmaxb.on_changed(refresh)
        # self._Rdal_peak = fig.canvas.mpl_connect('key_press_event', press)

    def save(self, filesave):
        """
            save the project to open later
            formats available: None: pickel format good for python

        Args:
            filename (str): filename to save

        Examples:
            >>> Exp1.save('exp1')
        """
        if '.' in filesave:
            filesave = filesave.split('.')[0]
        filesave += '.sqm'

        out = Object()
        out.peaks = [{'pos': tuple(i.Peaks),
                      'inte': i.Peaks.int,
                      'ps_in': i.Peaks.ps_in} for i in self]
        out.filename = self.filenames
        out.filesangle = [i.info.gon_angles for i in self]
        if hasattr(self, 'EwP'):
            out.EwP = {'positions': self.EwP.pos,
                       'intensity': self.EwP.int}
            if hasattr(self.EwP, 'axes'):
                out.EwP['axes'] = self.EwP.axes
            if hasattr(self.EwP, '_rot_vect'):
                out.EwP['rot_vect'] = self.EwP._rot_vect
            if hasattr(self.EwP, 'angles'):
                out.EwP['angles'] = self.EwP.angles
            if hasattr(self.EwP, '__rot__'):
                out.EwP['r0'] = self.EwP.__rot__
            if hasattr(self.EwP, '__rotz__'):
                out.EwP['z0'] = self.EwP.__rotz__
        with open(filesave, "wb") as file_save:
            pickle.dump(out, file_save)
        return

    @classmethod
    def load(cls, filename):
        """ load a saved project
        it is necessary that images remain in the same relative
        position
        Args:
            filename (str): filename to open

        Examples:
            >>> exp1 = SeqIm.load(exp1.pkl)

        """
        inn = pickle.load(open(filename, 'rb'))
        out = SeqIm(inn.filename, inn.filesangle)
        for i, Peaksi in enumerate(inn.peaks):
            out[i].Peaks = PeakL(Peaksi['pos'])
            out[i].Peaks.int = Peaksi['inte']
            out[i].Peaks.ps_in = Peaksi['ps_in']
        if hasattr(inn, 'EwP'):
            try:
                inn.EwP.update({'positions': inn.EwP['pos'],
                                'intensity': inn.EwP['int']})
            except:
                pass
            out.EwP = EwaldPeaks(**inn.EwP)
        return out


class EwaldPeaks(object):
    """Set of peaks position and intensity
    this class manages peaks position and intensity and the methods related to
    lattice indexing and refinement
    could be created as an attribute EwP of a SeqIm class by using methods D3_peaks
    or by sum with an another EwaldPeaks class with the same first image

    Example:
        >>>Exp1.D3_peaks(tollerance=5)
        >>>EWT= Exp1.EwP +  Exp2.EwP

    Args:
        positions (list): list containing the coordonates of peaks
        intensity (list): list containing the intensity of peaks

    Attributes:
        pos  (list): Ewald peaks 3D set of peaks
        int (list): list of Rotation vector for each image
        pos_cal (np.array): array witht he position in the new basis
        rMT     (np.array): reciprocal metric tensor
        axis    (np.array): reciprocal basis set, 3 coloums
        cell    (dict): a dictionary witht the value of
                         real space cell
        graph  (D3plot.D3plot):graph Ewald peaks 3D set of peaks used to index


    Note:
        | Examples:
        |     >>>Exp1.D3_peaks(tollerance=5)
        |     >>>Exp1.EwP is defined
        |     >>>EWT= Exp1.EwP +  Exp2.EwP
        |
        | Methods to use:
        | def D3_peaks(tollerance=15)
        | def plot_int()
        | def plot_proj_int()
        | def plot_reduce
        | def plot_reduce
        | def refine_axes
        | def set_cell
        | def create_layer
        | def save(name)
        | def load(name)
        | def help()
    """

    def __init__(self, positions, intensity,
                 rot_vect=None, angles=None,
                 r0=None, z0=None, pos0=None,
                 scale=None,
                 axes=None, set_cell=True, **kwds):
        # list in whic pos are sotred for each image
        self.pos = positions
        self.int = intensity
        if rot_vect is None:
            pass
        elif isinstance(rot_vect, list):
            self._rot_vect = rot_vect
        else:
            self._rot_vect = [rot_vect] * len(self.int)
        if angles is not None:
            self._angles = angles
        if r0 is not None:
            self.__rot__ = r0
        if z0 is not None:
            self.__rotz__ = z0
        if pos0 is not None:
            self.__pos0__ = pos0
        if scale is not None:
            self.__scale__ = pos0
        if axes is not None:
            if set_cell:
                self.set_cell(axes)
            else:
                self.axes = axes

    def __add__(self, other):
        """
        """
        pos = self.pos + other.pos
        inte = self.int + other.int
        cond = hasattr(self, '_rot_vect') and hasattr(other, '_rot_vect')
        if cond:
            rot_vect = self._rot_vect + other._rot_vect
        return EwaldPeaks(pos, inte, rot_vect=rot_vect)

    def merge(self, other, tollerance=0.61):
        # all_peaks = [self.__pos0__[0][:, :2], other.__pos0__[0][:, :2]]
        all_peaks = [self.pos[0][:, :2], other.pos[0][:, :2]]
        rot = np.array([self.__rot__[0], other.__rot__[0]])

        out = mt.find_common_peaks(tollerance, all_peaks)
        print('found %d common peaks' % out.shape[1])
        zangle, rot_vect = mt.find_zrot_correction(out, tollerance)

        zrot = R.from_rotvec([0, 0, -zangle[1]])
        print('angle correction', np.round(np.degrees(zangle), 2))
        otherpos = [zrot.apply(i) for i in other.pos]

        z0 = mt.find_z_rotation(rot, rot_vect)[0]
        print('z0 angle %4.2f' % np.degrees(z0))
        axis = mt.creaxex(rot, z0)
        sign = np.where(axis @ rot_vect > 0, -1, 1)
        angle = mt.mod(axis[0]) * sign

        position = []
        r = R.from_rotvec(rot_vect * -angle)
        for i, peaks in enumerate(otherpos):
            position.append(r.apply(peaks))

        position = self.pos + position
        inte = self.int + other.int
        r0 = np.vstack([self.__rot__, other.__rot__])

        # rot_vect = self._rot_vect + otherrotvect

        # out = EwaldPeaks(position, inte, rot_vect=rot_vect)
        out = EwaldPeaks(position, inte, r0=r0)
        if hasattr(self, 'axes'):
            out.set_cell(self.axes)
        return out

    def plot(self):
        """ open a D3plot graph
            Attributes:
                graph  (D3plot.D3plot): graph Ewald peaks 3D set of peaks used to index
        """
        self.graph = d3plot.D3plot(self)

    def plot_int(self):
        """Plot instogramm of intensity of the peaks
        """
        intens = np.hstack([i for i in self.int])
        plt.figure()
        plt.hist(sorted(intens), bins=100, rwidth=4)
        plt.xlabel('peaks intensity')
        plt.ylabel('n. of peaks')

    def plot_proj_int(self, cell=True):
        """plot peak presence instogramm as a function of the cell 
        """ 
        if cell:
            print([i.shape for i in self.pos_cal])
            pos = np.vstack([i for i in self.pos_cal])
            pos = np.fmod(abs(pos), 1)
            pos = np.where(pos < 0.5, pos, pos - 0.5)
            fig = plt.figure()
            gs = fig.add_gridspec(3, 1, hspace=0.50)
            pa = fig.add_subplot(gs[0, 0])
            pb = fig.add_subplot(gs[1, 0])
            pc = fig.add_subplot(gs[2, 0])
            pa.hist(pos[:, 0], range=(0, .5), bins=100)
            pa.set_title('a')
            pb.hist(pos[:, 1], range=(0, .5), bins=100)
            pb.set_title('b')
            pc.hist(pos[:, 2], range=(0, .5), bins=100)
            pc.set_title('c')
            plt.draw()
        else:
            fig = plt.figure()
            plt.hist(np.vstack(self.pos)[:, 2], bins=1000)
            plt.title('z')
            plt.xlabel('position')
            plt.ylabel('n. peaks')
            plt.draw()

    def find_cell(self, sort=0, cond=None, layers=None,
                  toll=0.1, toll_angle=5):
        """automatic find cell
        search the *cell in the present peaks 

        Args:
            sort (int): 0 or 1 small change in the algoritm should be indifferent
            cond (lambda function): fil;tering condition created by cr_cond
            layer (list): specifies the layer to be used if none all image are used 
            toll  (float): [0.1] indexing tollerance 
            toll_angle  (float): [5.0] angle in degree to determine coplanarity or colinearity 
        """


        if layers is None:
            layers = range(len(self.pos))
        if cond is None:
            cond = lambda pos, inte: pos
        pos=[cond(self.pos[i], self.int[i]) for i in layers]

        vectors = []
        for i in layers:
            vectors.extend(ind.Find_2D_uc(pos[i], toll_angle,
                                          toll))
        vectors = ind.check_colinearity(vectors, toll_angle)
        if sort:
            allpos = np.vstack(pos)
            vectors = ind.sort_Calib(allpos, vectors, toll) 
        else:
            vectors =  ind.check_3D_coplanarity(vectors, toll_angle)   
        vectors = ind.check_3Dlincomb(vectors)
        self.set_cell(vectors.T)
        return

    def plot_reduce(self, tollerance=0.1, condition=None):
        """plot collapsed reciprocal space
           plot the position of the peaks in cell coordinatete and all
           reduced to a single cell.
           it create a self.reduce attribute containingt he graph  
        """
        pos = np.vstack([i for i in self.pos_cal])

        pos = np.fmod(pos, 1)
        pos = np.where(pos < 0, pos + 1, pos)

        pos_c = np.where(pos > 0.5, pos - 1, pos)
        pos_c = np.where(pos_c < -0.5, pos_c + 1, pos_c)
        cond = abs(pos_c).max(axis=1) > tollerance

        pos = pos - 0.5
        pos = (pos @ self.axes.T)
        orig = (self.axes @ [-0.5, -0.5, -0.5]).T
        inte_o = np.hstack([i for i in self.int])
        self.reduce = d3plot.D3plotr(EwaldPeaks([pos[cond], pos[~cond]],
                                                [inte_o[cond], inte_o[~cond]],
                                                rot_vect=self._rot_vect,
                                                axes=self.axes,
                                                set_cell=False),
                                     origin=orig)
        self.reduce.rotate_0()
        return

    def create_layer(self, hkl, n, size=0.25, toll=0.15, mir=0, spg=None):
        """create a specific layer
        create a reciprocal space layer

        Args:
            hkl (str): constant index for the hkl plane to plot, format('k')
            n (float, int): value of hkl
            size (float): intensity scaling
                * if positive, scale intensity of each peaks respect the max
                * if negative, scale a common value for all peaks
            mir (bool):  mirror in respect of n meaning =/-n
            tollerance (float): exclude from the plot peaks at higher distance
            spg (str): allows to index the peaks, and check if they are extinted
        """
        hkl = hkl.lower()
        o1, o2 = [i for i in [0, 1, 2] if i != 'hkl'.find(hkl)]

        if mir:
            def app_cond(pos):
                return np.where(abs(abs(pos['hkl'.find(hkl)]) - abs(n)) < toll)
        else:
            def app_cond(pos):
                return np.where(abs(pos['hkl'.find(hkl)] - n) < toll)

        # creation of a bidimensional orthogonalization matrix
        mod = np.sqrt(np.sum(self.axes**2, axis=0))
        cos_a = (self.axes[:, o1]  @ self.axes[:, o2]) / (mod[o1] * mod[o2])
        print('degree between the axis',
              np.round(np.arccos(cos_a) / rpd, 2), '\n')
        sin_a = np.sin(np.arccos(cos_a))
        Ort_mat = np.array([[mod[o1], mod[o2] * cos_a],
                            [0, mod[o2] * sin_a]])
        inv_Ort_mat = inv(Ort_mat)

        # print(Ort_mat @ np.array([1, 0]))
        # print(Ort_mat @ np.array([0, 1]))
        # print(Ort_mat @ np.array([1, 1]))

        # create a set of theoretical reflection
        if spg:
            # filter the reflection on the plane
            spgo = Symmetry.Spacegroup(spg)
            layer_pos = np.vstack([i for i in self.pos_cal])
            cond = app_cond(layer_pos.T)
            layer_pos = layer_pos[cond].T
            inte_o = np.hstack([i for i in self.int])[cond]

            # ortogonalize
            pos_o = Ort_mat @ layer_pos[[o1, o2]]

            # find limit of the layer
            o12maxmin = [layer_pos[o1].min(), layer_pos[o2].min(),
                         layer_pos[o1].max() + 1, layer_pos[o2].max() + 1]

            o12maxmin = [int(np.rint(i)) for i in o12maxmin]

            # create the HKL grid
            refx, refy = np.mgrid[o12maxmin[0]: o12maxmin[2],
                                  o12maxmin[1]: o12maxmin[3]]
            ref = [refx.flat, refy.flat]
            ref = np.vstack(ref).T  # transform in nline 3 colum format

            # create extinction
            ref3ind = np.insert(ref, 'hkl'.find(hkl),
                                np.ones_like(refx.flat) * n,
                                axis=1)
            ext_c = spgo.search_exti(ref3ind[:, 0].flat,
                                     ref3ind[:, 1].flat,
                                     ref3ind[:, 2].flat)

            if np.any(ext_c):
                print(ref[ext_c].T.shape)
                ref_ext = Ort_mat @ ref[ext_c].T
            if np.any(~ext_c):
                ref_act = Ort_mat @ ref[~ext_c].T

            plt.figure()
            if size > 0:
                c_size = inte_o / inte_o.max() * size
            else:
                c_size = inte_o.mean() / inte_o.max() * abs(size) * 2
            ax = plt.subplot(aspect='equal')
            plt_p.circles(pos_o[0], pos_o[1],
                          s=c_size,
                          color='b')
            if np.any(~ext_c):
                plt_p.rectangles(*ref_act,
                                 w=inte_o.mean() / inte_o.max() * abs(size * 1.5),
                                 color='r', alpha=0.5)
            if np.any(ext_c):
                plt_p.rectangles(*ref_ext,
                                 w=inte_o.mean() / inte_o.max() * abs(size * 1.5),
                                 color='r', fc='none')
            title = '(H K L)'.replace(hkl.upper(), str(n))
            title += '  %s (%d)' % (spgo.symbol.strip().replace(' ', ''), spgo.no)
            plt.title(title, weight='bold')
            # plt.scatter(*ref_act, s= 0.5 *inte_o.min() * size,
            #             marker="D",
            #             facecolors='r', edgecolors='r')
            # plt.scatter(*ref_ext, s= 0.5 *inte_o.min() * size,
            #             marker="D",
            #             facecolors='none', edgecolors='r')

        else:
            # filter the reflections in condition
            inte_o = []     # list with intensity  for image
            pos_o = []      # list with coordinate  for image
            max_inte = 0
            for pos_i, inte_i in zip(self.pos_cal, self.int):
                cond = app_cond(pos_i.T)
                inte_o.append(inte_i[cond])
                max_inte = max(max_inte, inte_i.max())
                pos_o.append(Ort_mat @ pos_i[cond].T[[o1, o2]])
            layer_pos = np.hstack(pos_o)
            plt.figure()
            ax = plt.subplot(aspect='equal')
            cmap = plt.get_cmap('brg')
            colors = [cmap(i) for i in np.linspace(0, 1, len(pos_o))]
            for i, pos_i in enumerate(pos_o):
                if size > 0:
                    c_size = inte_o[i] / max_inte * size
                else:
                    c_size = - size
                plt_p.circles(pos_i[0], pos_i[1], s=c_size,
                              color=colors[i])

            plt.title('(H K L)'.replace(hkl.upper(), str(n)), weight='bold')

        def format_coord(x, y):
            hkl_i = np.array([float(n)] * 3)
            hkl_i[o1], hkl_i[o2] = inv_Ort_mat @ np.array([x, y])

            d2 = 1 / np.sqrt(hkl_i @ self.rMT @ hkl_i)
            z = f'H={hkl_i[0]:4.1f}, K={hkl_i[1]:4.1f}, L={hkl_i[2]:4.1f}'
            return f'{z:s}    d_sp={d2:4.4f}nm'
        ax.format_coord = format_coord

        plt.xlabel('%s (1/nm)' % 'HKL'[o1], weight='bold')
        plt.ylabel('%s (1/nm)' % 'HKL'[o2], weight='bold')
        plt.draw()

    def cr_cond(self, operator=None, lim=None):
        """define filtering condition

        fuch function create a function that filter the data following the condition

        """
        if operator == '>':
            def lcond(pos, inte):
                return inte > lim

        elif operator == '<':
            def lcond(pos, inte):
                return inte < lim

        elif operator == 'tollerance':
            # to use with calibrated position
            def lcond(pos, inte):
                resx = abs(pos) % 1
                resx = np.where(resx > 0.5, 1 - resx, resx)
                return np.sum(resx < lim, 1) > 2
        return lcond

    def refine_axes(self, axes=None, tollerance=0.1):
        """refine reciprocal cell basis
        refine the reciprocal cell basis in respect to data that are
        indexed in the tollerance range.
        """
        if axes is None:
            axes = self.axes
        else:
            self.axes = axes
            self.__calibrate()
        filt = self.cr_cond('tollerance', tollerance)

        def setup():
            cond = filt(np.vstack(self.pos_cal), 0)
            data = np.compress(cond, np.vstack(self.pos), axis=0)
            n_peaks = sum(cond)
            return n_peaks, data

        def res(axesr):
            axes = np.array(axesr[:9]).reshape(3, 3)
            origin = np.array(axesr[9:])
            P = inv(axes)
            resx = abs(P @ (data - origin).T) % 1
            resx = np.where(resx > 0.5, 1 - resx, resx)
            return resx.sum(0)

        n_peaks, data = setup()
        axesr = np.append(axes.flatten(), [0, 0, 0])
        while True:
            res_1 = least_squares(res, axesr, verbose=1)
            self.axes = np.array(res_1.x[:9]).reshape(3, 3)
            self.pos = [i - res_1.x[9:] for i in self.pos]
            self.__calibrate()
            n, new_data = setup()
            if n == n_peaks:
                break
            else:
                n_peaks, data = n, new_data

        # chi square
        s_sq = (res(res_1.x)**2).sum() / (n_peaks - len(res_1.x))
        # fvariance covariance matrix
        pcov = inv(res_1.jac.T @ res_1.jac) * s_sq
        error = []
        for i in range(len(res_1.x)):
            try:
                error.append(np.absolute(pcov[i][i]) ** 0.5)
            except:
                error.append(0.00)

        self._axes_std = np.array(error[:9]).reshape(3, 3)
        self.set_cell(self.axes, self._axes_std)
        return res_1.x[9:]  # res_1.success, res_1.njev,

    def refine_axang(self, axes=None, tollerance=0.1, zero_tol=0.1):
        """refine reciprocal cell basis
        refine the reciprocal cell basis in respect to data that are
        indexed in the tollerance range.
        """
        """refine reciprocal cell basis
        refine the reciprocal cell basis in respect to data that are
        indexed in the tollerance range.
        """
        if axes is None:
            axes = self.axes
        else:
            self.axes = axes
            self.__calibrate()

        assert hasattr(self, '_rot_vect'), \
            'angle refinement impossible without rotation axes'
        # Filter position in tollerance
        pos_f = []
        n_peak = []
        for i, pos_i in enumerate(self.pos_cal):
            filt = self.cr_cond('tollerance', tollerance)
            cond = filt(pos_i, 0)
            cond2 = abs(self.pos[i][:, 2]) > zero_tol
            pos_f.append(np.compress(cond, self.pos[i], axis=0))
            n_peak.append(sum(cond * cond2))
        ref_planes, = np.where(np.array(n_peak) > 0)[:1]

        def res(axes_ang):
            # angles part
            origin = axes_ang[9:12]
            angles = axes_ang[12:]
            axesr = axes_ang[:9].reshape(3, 3)

            pos = pos_f[:]
            for i, i_pla in enumerate(ref_planes):
                r = R.from_rotvec(self._rot_vect[i_pla] * angles[i])
                pos[i_pla] = r.apply(pos_f[i_pla] - origin)
            data = np.vstack(pos)

            # axes part
            P = inv(axesr)
            resx = abs(P @ data.T) % 1
            resx = np.where(resx > 0.5, 1 - resx, resx)
            return resx.sum(0)

        axes_ang = list(axes.flatten()) + [0] * 3 + [0] * len(ref_planes)
        res_1 = least_squares(res, axes_ang, verbose=1)

        # chi square
        s_sq = (res(res_1.x)**2).sum() / (sum(n_peak) - len(res_1.x))
        pcov = inv(res_1.jac.T @ res_1.jac) * s_sq
        error = []
        for i in range(len(res_1.x)):
            try:
                error.append(np.absolute(pcov[i][i]) ** 0.5)
            except:
                error.append(75.00)

        self.axes = np.array(res_1.x[:9]).reshape(3, 3)
        self._axes_std = np.array(error)[:9].reshape(3, 3)

        # change origin 
        self.pos = [i - res_1.x[9:12] for i in self.pos]

        # change the angles
        for i, i_pla in enumerate(ref_planes):
            r = R.from_rotvec(self._rot_vect[i_pla] * res_1.x[12 + i])
            self.pos[i_pla] = r.apply(self.pos[i_pla])

        self.set_cell(self.axes, self._axes_std)

        return ref_planes, np.degrees(res_1.x), np.degrees(error)

    def refine_angles(self, axes=None, tollerance=0.1, zero_tol=0.1):
        """refine reciprocal cell basis
        refine the reciprocal cell basis in respect to data that are
        indexed in the tollerance range.
        """
        """refine reciprocal cell basis
        refine the reciprocal cell basis in respect to data that are
        indexed in the tollerance range.
        """
        if axes is None:
            axes = self.axes
        else:
            self.axes = axes
            self.__calibrate()

        assert hasattr(self, '_rot_vect'), \
            'angle refinement impossible without rotation axes'
        # Filter position in tollerance
        pos_f = []
        n_peak = []
        for i, pos_i in enumerate(self.pos_cal):
            filt = self.cr_cond('tollerance', tollerance)
            cond = filt(pos_i, 0)
            cond2 = abs(self.pos[i][:, 2]) > zero_tol
            pos_f.append(np.compress(cond, self.pos[i], axis=0))
            n_peak.append(sum(cond * cond2))
        ref_planes, = np.where(np.array(n_peak) > 0)[:1]
        n_peak = sum(n_peak) - n_peak[0]
        angles = [0.0] * len(ref_planes)

        P = inv(axes)

        def res(angles):
            pos = pos_f[:]
            for i, i_pla in enumerate(ref_planes):
                r = R.from_rotvec(self._rot_vect[i_pla] * angles[i])
                pos[i_pla] = r.apply(pos_f[i_pla])
            data = np.vstack(pos)
            resx = abs(P @ data.T) % 1
            resx = np.where(resx > 0.5, 1 - resx, resx)
            return resx.sum(0)

        res_1 = least_squares(res, angles, verbose=1)

        # chi square
        s_sq = (res(res_1.x)**2).sum() / (n_peak * 3 - len(res_1.x))
        pcov = inv(res_1.jac.T @ res_1.jac) * s_sq
        error = []
        for i in range(len(res_1.x)):
            try:
                error.append(np.absolute(pcov[i][i]) ** 0.5)
            except:
                error.append(75.00)

        # change the angles
        for i, i_pla in enumerate(ref_planes):
            r = R.from_rotvec(self._rot_vect[i_pla] * res_1.x[i])
            self.pos[i_pla] = r.apply(self.pos[i_pla])

        self.set_cell(self.axes)
        return ref_planes, np.degrees(res_1.x), np.degrees(error)

    def set_cell(self, axes=None, axes_std=None, tollerance=0.1, cond=None):
        ''' calculation of the cell
        effect the calculation to obtain the cell
        Args:
            axis (np.array 3,3): the new reciprocal basis to be used in the format
                         | np.array[[a1, b1, c1],
                         |         [a2, b2, c2],
                         |         [a3, b3, c3]]
                if axis is not inoput the programm seach if a new basis
                has been defined graphically

        return:
                nothing

        Attributes:
            self.rMT   (np.array) : reciprocal metric tensor
            self.cell   (dict)    : a dictionary witht the value of
                                        real space cell

        """
            self.rMT    : reciprocal metric tensor
           self.cell   : a dictionary witht the value of
                         real space cell
        '''
        if axes is None:
            try:
                assert len(self.graph.axes) == 3, 'not prperly defined axes'
                self.axes = np.array(
                    [self.graph.axes[i].axis for i in 'abc']).T
            except:
                print('\nusing old cell')
                pass
        else:
            self.axes = axes

        # reciprocal metric tensor
        self.rMT = self.axes.T @ self.axes
        # metric tensor
        self.MT = inv(self.rMT)

        # 
        self.__calibrate()

        from uncertainties import unumpy

        def calc_cell(axesflat):
            ax = axesflat.reshape(3, 3)
            MT = unumpy.ulinalg.inv(np.dot(ax.T, ax))
            a, b, c = unumpy.sqrt(np.diagonal(MT))
            al, bt, gm = unumpy.arccos([MT[2, 1] / (b * c),
                                        MT[2, 0] / (a * c),
                                        MT[0, 1] / (a * b)]) / rpd
            return a, b, c, al, bt, gm

        if axes_std is None:
            self.cell = dict(zip(['a', 'b', 'c', 'alpha', 'beta', 'gammma'],
                                 calc_cell(self.axes.flatten())))
        else:
            self._axes_std = axes_std
            axes = unumpy.uarray(self.axes, self._axes_std)

            class cell(dict):
                def __str__(self):
                    z = f''
                    for k, v in self.items():
                        z += f'{k} = {v}\n'
                    return z
            self.cell = cell(zip(['a', 'b', 'c', 'alpha', 'beta', 'gammma'],
                                 calc_cell(axes.flatten())))

        data = np.vstack([i for i in self.pos_cal])
        if cond is not None:
            data = np.compress(cond(data, np.hstack([i for i in self.int])),
                               data, axis=0)
        filt = self.cr_cond(operator='tollerance', lim=tollerance)
        indexed = sum(filt(data, 0)) / data.shape[0] * 100
        print(f'\n{sum(filt(data, 0))} indexed peaks, {round(indexed, 2)}%')
        print(f'with tollence {tollerance*100}%\n')
        for i, j in self.cell.items():
            print(i, ' = ', j)
        self.__check_cent__(tollerance=tollerance)

        if hasattr(self, 'graph'):
            num = self.graph.fig.number
            if plt.fignum_exists(num):
                self.graph._set__axes(self.axes)
                self.graph.plot_ax()
        return

    def __calibrate(self):
        """calibrate
        given a set of axis reindex the peaks with
        the new basis and calculate the cell
        calculated the attribute:
                self.pos_cal  (list): array witht the position in the new basis
        """
        # paasage to axes base
        P = inv(self.axes)
        self.pos_cal = []
        for i in self.pos:
            self.pos_cal.append(np.round(P @ i.T, 2).T)

    def __check_cent__(self, tollerance=0.1):
        center = ['I', 'F', 'A', 'B', 'C', 'Ro', 'Rr']
        # w is the total on all images
        self.__centering__ = {k: {'w': 0} for k in center}

        # create a list of indexed reflection
        indexed = []
        filt = self.cr_cond(operator='tollerance', lim=tollerance)
        for ima in self.pos_cal:
            cond = filt(ima, 0)
            indexed.append(np.round(ima[cond], 0))

        # try I centering
        for i, i_imai in enumerate(indexed):
            # I centering
            NoEx = (i_imai.sum(1) % 2).astype('bool')
            self.__centering__['I'][i] = NoEx
            self.__centering__['I']['w'] += NoEx.sum()
            # F centering
            NoEx = ((i_imai % 2).sum(1) % 3).astype('bool')
            self.__centering__['F'][i] = NoEx
            self.__centering__['F']['w'] += NoEx.sum()
            # A centering
            NoEx = ((i_imai.T[1] + i_imai.T[2]) % 2).astype('bool')
            self.__centering__['A'][i] = NoEx
            self.__centering__['A']['w'] += NoEx.sum()
            # B centering
            NoEx = ((i_imai.T[0] + i_imai.T[2]) % 2).astype('bool')
            self.__centering__['B'][i] = NoEx
            self.__centering__['B']['w'] += NoEx.sum()
            # C centering
            NoEx = ((i_imai.T[0] + i_imai.T[1]) % 2).astype('bool')
            self.__centering__['C'][i] = NoEx
            self.__centering__['C']['w'] += NoEx.sum()
            # 'R obverse
            NoEx = ((i_imai.sum(1) - (2 * i_imai.T[0])) % 3).astype('bool')
            self.__centering__['Ro'][i] = NoEx
            self.__centering__['Ro']['w'] += NoEx.sum()
            # 'R reverse
            NoEx = ((i_imai.sum(1) - (2 * i_imai.T[1])) % 3).astype('bool')
            self.__centering__['Rr'][i] = NoEx
            self.__centering__['Rr']['w'] += NoEx.sum()
        print('\n')
        for i in center:
            print(i, ' existing extincted peaks  ', self.__centering__[i]['w'])
        return

    def save(self, filename, dictionary=False):
        """ save EwP
            formats available:
               None: pickel format good for python
               Idx : for Ind_x
        """
        if filename[-3:].lower() == 'idx':
            pos = np.vstack([i for i in self.pos]) / 10
            header = '_NumberOfReflections\n     {:d}'.format(pos.shape[0])
            header += '\n_Reflections'
            footer = '_MinMaxVolumeOfPrimitiveCell\n     5.0  1000.0\n'
            footer += '_TheMainCriterion \n 0.25'
            np.savetxt(filename, pos, header=header,
                       footer=footer, fmt='%10.5f', comments='')
            return

        out = {'positions': self.EwP.pos, 'intensity': self.EwP.int}
        if hasattr(self, 'axes'):
            out['axes'] = self.axes
        if hasattr(self, '_rot_vect'):
            out['rot_vect'] = self._rot_vect
        if hasattr(self, 'angles'):
            out['angles'] = self.angles
        if hasattr(self, '__rot__'):
            out['r0'] = self.__rot__
        if hasattr(self, '__rotz__'):
            out['z0'] = self.__rotz__

        if dictionary:
            return out

        if '.' not in filename:
            filename += '.ewp'
        with open(filename, 'wb') as filesave:
            pickle.dump(out, filesave)
        return

    @classmethod
    def load(cls, filename):
        """load EwP in python format
            Example:
            >>>cr1 = EwaldPeaks.load('cr1.ewp')
        """
        dd = pickle.load(open(filename, 'rb'))
        z = EwaldPeaks(dd['pos'], dd['int'], dd['rot_vect'])
        if 'axes' in dd.keys():
            z.set_cell(dd['axes'])
            z.set_cell()
        return z

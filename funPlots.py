import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib import cm
import matplotlib.colors as pltcolors

from matplotlib.ticker import MultipleLocator
import matplotlib
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.legend_handler import HandlerPathCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.lines import Line2D


# ----------------------------------
# General axis formatting and styles
# ----------------------------------

def master_format(**kwargs):
    format_axes(**kwargs)
    format_text(**kwargs)
    format_legend(**kwargs)

# ----------------------------------
# Thanks to Victor for this part :)
# ----------------------------------


def format_axes(despined = False, tickpad = 1.2, capstyle = "round", axes_lw = 0.7,
                tick_maj_size = 3, tick_maj_width = 1.1, tick_min_size = 1.7,
                ax_ec = "#878787", ax_tc = "#4a4a4a", ax_lc = "#262626",
                min_maj_w_ratio = 0.65, **kwargs):
    """
    Function to set the format of the axes.

    Parameters:
    ----------
    despined: bool
        If true, the top and right axes are removed. Default is False.
    tickpad: float
        Padding between the tick and the label. Default is 1.2.
    capstyle: str
        Capstyle of the lines. Can be "round", "butt" or "projecting". Default is "round".
    axes_lw: float
        Width of the axes. Default is 0.7.
    tick_maj_size: float
        Size of the major ticks. Default is 3.
    tick_maj_width: float
        Width of the major ticks. Default is 1.1.
    tick_min_size: float
        Size of the minor ticks. Default is 1.7.
    ax_ec: str
        Color of the axes. Default is "#878787".
    ax_tc: str
        Color of the tick labels. Default is "#4a4a4a".
    ax_lc: str
        Color of the axis labels. Default is "#262626".
    min_maj_w_ratio: float
        Ratio between the minor and major tick width. Default is 0.65.
    **kwargs: dict
        Additional keyword arguments. Not used at the moment.
    """


    mpl.rcParams["axes.spines.right"] = not despined
    mpl.rcParams["axes.spines.top"] = not despined
    mpl.rcParams["lines.dash_capstyle"] = capstyle
    mpl.rcParams["lines.solid_capstyle"] = capstyle

    mpl.rcParams["axes.linewidth"] = axes_lw
    mpl.rcParams["xtick.major.width"] = tick_maj_width
    mpl.rcParams["ytick.major.width"] = tick_maj_width
    mpl.rcParams["xtick.minor.width"] = tick_maj_width * min_maj_w_ratio
    mpl.rcParams["ytick.minor.width"] = tick_maj_width * min_maj_w_ratio

    mpl.rcParams["xtick.major.size"] = tick_maj_size
    mpl.rcParams["ytick.major.size"] = tick_maj_size
    mpl.rcParams["xtick.minor.size"] = tick_min_size
    mpl.rcParams["ytick.minor.size"] = tick_min_size

    mpl.rcParams["xtick.major.pad"] = tickpad
    mpl.rcParams["ytick.major.pad"] = tickpad
    mpl.rcParams["xtick.minor.pad"] = tickpad * 0.75
    mpl.rcParams["ytick.minor.pad"] = tickpad * 0.75

    mpl.rcParams['axes.edgecolor'] = ax_ec
    mpl.rcParams['axes.facecolor'] = 'None'
    mpl.rcParams['axes.labelcolor'] = ax_lc
    mpl.rcParams['lines.solid_capstyle'] = 'round'
    mpl.rcParams['patch.edgecolor'] = 'w'
    mpl.rcParams['text.color'] = 'k'

    mpl.rcParams["xtick.color"] = ax_ec
    mpl.rcParams["ytick.color"] = ax_ec
    mpl.rcParams["xtick.labelcolor"] = ax_tc
    mpl.rcParams["ytick.labelcolor"] = ax_tc

def despine(axes, bottom=True, left=True):
    """
    Eliminate top and right lines from axes.

    Parameters:
    ----------
    axes: matplotlib axes
        Axes to be despined. Can be an iterable of axes.
    bottom: bool
        If true, the bottom line is removed. Default is True.
    left: bool
        If true, the left line is removed. Default is True.
    """

    if type(axes) is np.ndarray:
        for ax in np.ravel(axes):
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(bottom)
            ax.spines['left'].set_visible(left)

    elif type(axes) is list:
        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(bottom)
            ax.spines['left'].set_visible(left)
    else:
        axes.spines['right'].set_visible(False)
        axes.spines['top'].set_visible(False)
        axes.spines['bottom'].set_visible(bottom)
        axes.spines['left'].set_visible(left)


def label_axes(axes, textpos, uppercase=False, bracket=True, fontsize = 10):
    """
    Fast way to label all the diagrams using the alphabet.

    Parameters:
    ----------
    axes: matplotlib axes
        Axes to be labeled. Can be an iterable of axes.
    textpos: list
        Position of the label, in axes coordinates. Can be a pair of values or a list of pairs
    uppercase: bool
        If true, the labels are in uppercase. Default is False.
    bracket: bool
        If true, the labels are in brackets. Default is True.
    fontsize: float
        Size of the font. Default is 10.
    """

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if uppercase:
        alphabet = alphabet.upper()

    if bracket:
        axlabel = "({0})"
    else:
        axlabel = "{0}"

    #If we give just a pair of values, convert it into a list for compatibility
    if type(textpos[0]) is float:
        textpos = [[textpos[0], textpos[1]] for i in range(len(axes))]

    #Iterate over the axes and set the things
    if type(axes) is np.ndarray:
        for i,ax in enumerate(axes.flatten()):
            ax.text(textpos[i][0], textpos[i][1], axlabel.format(alphabet[i]),
                    color = "#404040", transform=ax.transAxes, weight="bold", size = fontsize)
    elif type(axes) is list:
        for i,ax in enumerate(axes):
            ax.text(textpos[i][0], textpos[i][1], axlabel.format(alphabet[i]),
                    color = "#404040", transform=ax.transAxes, weight="bold", size = fontsize)


# ----------------------------------
# Measures and sizes
# ----------------------------------

class Measures:
    fig_w_1col = 11.0 #cm
    fig_w_2col = 21.0 #cm

def to_inches(cm):
    """
    Convert cm to inches.

    Parameters:
    ----------
    cm: float
        Value in cm to be converted to inches
    
    Returns:
    -------
    float
        Value in inches
    """
    return cm/2.54

def one_col_size(ratio=1.618, height=None):
    """
    Sets the figure size to a single-column plot.

    Parameters:
    ----------
    ratio: float
        How large width is with respect to height. This parameter is ignored if height is not None
    height: float, cm
        Height of the figure. If different from None (default), ratio parameter is ignored.
    
    Returns:
    -------
    tuple
        Returns a tuple (w,h), where w is the width if a single-column graph.
        Height by default is the golden ratio of the width, but can be chosen (in cm).
    """
    width = to_inches(Measures.fig_w_1col)

    if height == None:
        height = width/ratio
    else:
        height = to_inches(height)
    return (width, height)


def two_col_size(ratio=1.618, height=None):
    """
    Sets the figure size to a two-column plot.

    Parameters:
    ----------
    ratio: float
        How large width is with respect to height. This parameter is ignored if height is not None
    height: float, cm
        Height of the figure. If different from None (default), ratio parameter is ignored.
    
    Returns:
    -------
    tuple
        Returns a tuple (w,h), where w is the width if a single-column graph.
        Height by default is the golden ratio of the width, but can be chosen (in cm).
    """

    width = to_inches(Measures.fig_w_2col)
    if height == None:
        height = width/ratio
    else:
        height = to_inches(height)

    return (width, height)


# ----------------------------------
# Text styles for figures
# ----------------------------------
def format_text(ncols = 1, nrows = 1, usetex=False, font="Avenir", label_fs=20, tick_fs=16, legend_fs = 19, pdffonttype=3, **kwargs):
    """
    Sets the text style for the figures.

    Parameters:
    ----------
    ncols: int
        Number of columns in the figure
    nrows: int
        Number of rows in the figure
    usetex: bool
        If true, use LaTeX to render the text. Default is False.
    font: str
        Font to be used. Default is Avenir.
    label_fs: int
        Font size for the labels. Default is 20.
    tick_fs: int
        Font size for the ticks labels. Default is 16.
    legend_fs: int
        Font size for the legend. Default is 19.
    pdffonttype: int
        Font type for the pdf. Default is 3.
    **kwargs: dict
        Additional arguments to be passed to matplotlib.rcParams
    """

    scale_factor = 1 - 0.05*(ncols*nrows - 1)
    if scale_factor < 0.5:
        scale_factor = 0.5

    #Font and sizes
    mpl.rcParams["font.family"] = font

    mpl.rcParams['text.latex.preamble']=r"\usepackage{amsmath, amstext}"

    mpl.rcParams["axes.labelsize"] = label_fs*scale_factor
    mpl.rcParams['axes.labelpad'] = 4
    mpl.rcParams["xtick.labelsize"] = tick_fs*scale_factor
    mpl.rcParams["ytick.labelsize"] = tick_fs*scale_factor
    mpl.rcParams["legend.fontsize"] = legend_fs*scale_factor

    #Set TeX mode to beautiful math, even if usetex if false
    mpl.rcParams["text.usetex"] = usetex
    mpl.rcParams["mathtext.fontset"] = "cm"

    #Type 3 is smaller, type 42 is more compliant and sometimes required
    mpl.rcParams["pdf.fonttype"] = pdffonttype


#Handy functions to globally change the fontsizes when needed
def set_label_size( fontsize):
    """
    Function to set the axes labels font size globally.

    Parameters:
    ----------
    fontsize: int
        Font size for the axes labels
    """

    mpl.rcParams["axes.labelsize"] = fontsize

def set_tick_size(fontsize):
    """
    Function to set the tick labels font size globally.

    Parameters:
    ----------
    fontsize: int
        Font size for the tick labels
    """

    mpl.rcParams["xtick.labelsize"] = fontsize
    mpl.rcParams["ytick.labelsize"] = fontsize

def set_legend_size(fontsize):
    """
    Function to set the legend font size globally.
    
    Parameters:
    ----------
    fontsize: int
        Font size for the legend
    """
    mpl.rcParams["legend.fontsize"] = fontsize


# ----------------------------------
# Setting up the legend
# ----------------------------------

def format_legend(frame=True, backcolor="#eeeeee", hdlLength=0.5, hdlText=0.2, hdlHeigth=0.6, labspacing=0.07, colspacing=0.4, **kwargs):
    """
    Apply all format we need to the legend
    """
    mpl.rcParams["legend.frameon"] = frame
    mpl.rcParams['legend.framealpha'] = 0.7
    mpl.rcParams["legend.handlelength"] = hdlLength
    mpl.rcParams["legend.handletextpad"] = hdlText
    mpl.rcParams["legend.handleheight"] = hdlHeigth
    mpl.rcParams["legend.labelspacing"] = labspacing
    mpl.rcParams["legend.columnspacing"] = colspacing
    mpl.rcParams["legend.facecolor"] = "white"


def right_ylabel(ax):
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")


# ----------------------------------
# Handy functions to plot lines & 
# multicolored lines, in both 2D and
# 3D. Possibly fill for the 2D case
# ----------------------------------
    

def MultiColorLine(ax, norm, cmap, x, y, ls = 'solid', lw = 1, zorder = 1, alpha = 1, cvals = None,
                   **kwargs):
    """
    Function to plot a multicolor line, by plotting a different color for each segment.

    Parameters:
    ----------
    ax: matplotlib.axes.Axes
        Axes to plot on
    norm: matplotlib.colors.Normalize
        Normalization to use for the colormap
    cmap: matplotlib.colors.Colormap
        Colormap to use
    x: array-like
        x coordinates of the points
    y: array-like
        y coordinates of the points
    ls: str, optional
        Linestyle to use. Default is 'solid'
    lw: float, optional
        Linewidth to use. Default is 1
    zorder: int, optional
        Zorder to use. Default is 1
    alpha: float, optional
        Transparency to use. Default is 1
    cvals: array-like, optional
        Values to use for the colormap. If None, y is used. Default is None
    **kwargs: dict
        Additional arguments to be passed to matplotlib.collections.LineCollection

    Returns:
    -------
    lc: matplotlib.collections.LineCollection
        LineCollection object
    """
    points = np.array([x, y]).transpose().reshape(-1,1,2)
    segs = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
    lc = LineCollection(segs, cmap = cmap, norm = norm, linestyles = ls, linewidths = lw,
                        alpha = alpha, zorder = zorder, **kwargs)
    if cvals is None:
        lc.set_array(y)
    else:
        lc.set_array(cvals)
    ax.add_collection(lc)

    # void plot to get correct axis limits
    ax.plot(x, y, lw = 0)

    return lc

def MultiColorLine3D(ax, norm, cmap, x, y, z, ls = 'solid', lw = 1, cvals = None, **kwargs):
    """
    Function to plot a multicolor line in 3D, by plotting a different color for each segment.

    Parameters:
    ----------
    ax: matplotlib.axes.Axes
        Axes to plot on
    norm: matplotlib.colors.Normalize
        Normalization to use for the colormap
    cmap: matplotlib.colors.Colormap
        Colormap to use
    x: array-like
        x coordinates of the points
    y: array-like
        y coordinates of the points
    z: array-like
        z coordinates of the points
    ls: str, optional
        Linestyle to use. Default is 'solid'
    lw: float, optional
        Linewidth to use. Default is 1
    cvals: array-like, optional
        Values to use for the colormap. If None, y is used. Default is None
    **kwargs: dict
        Additional arguments to be passed to matplotlib.collections.Line3DCollection

    Returns:
    -------
    lc: matplotlib.collections.Line3DCollection
        Line3DCollection object
    """
    points = np.array([x, y, z]).transpose().reshape(-1,1,3)
    segs = np.concatenate([points[:-1],points[1:]],axis=1)
    lc = Line3DCollection(segs, cmap = cmap, norm = norm, linestyles = ls, linewidths = lw, **kwargs)
    if cvals is None:
        lc.set_array(y)
    else:
        lc.set_array(cvals)
    ax.add_collection3d(lc)
    
    # void plot to get correct axis limits
    ax.plot(x, y, z, lw = 0)

    return lc


def plot_with_fill(ax, norm, cmap, x, y, y_err, cvals = None, alpha_fill = 0.05, nres = 200,
                   lw_fill = 0, shading = 'nearest', xoffset = np.zeros(2), yoffset = np.zeros(2),
                   zorder = 10, log_plot = False, force_ymin = None, lw = 3, cut_at_zero = False,
                   alpha_contour = 0.5, ls_contour = "--", lw_contour = 1, **kwargs):
    
    if cvals is None:
        cvals = y

    ylow = y - y_err
    yup =  y + y_err

    # void plot to get the axis limit automatically
    ax.plot(x, yup, lw = 0)
    ax.plot(x, ylow, lw = 0)
    xmin, xmax = ax.get_xlim() + xoffset
    ymin, ymax = ax.get_ylim() + yoffset

    # force the ymin if needed
    if force_ymin != None:
        ymin = force_ymin

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    if cut_at_zero:
        ylow[ylow < 0] = ymin

    line_data = MultiColorLine(ax, norm, cmap, x, y, lw = lw, zorder = zorder, cvals = cvals, **kwargs)

    # define the polygon and add the patches
    verts = np.vstack([np.stack([x[x < xmax], ylow[x < xmax]], 1),
                       np.stack([np.flip(x[x < xmax]), np.flip(yup[x < xmax])], 1)])
    
    path = matplotlib.path.Path(verts)
    patch = matplotlib.patches.PathPatch(path, facecolor='none', alpha = 0.0)
    ax.add_patch(patch)

    # create a dummy image
    nres = nres
    if log_plot:
        dummy_y = np.logspace(np.log10(ymin),np.log10(ymax),nres)
    else:
        dummy_y = np.linspace(ymin, ymax, nres)

    dummy_img = np.linspace(cvals.min(), cvals.max(), nres)
    dummy_img = np.repeat(dummy_img.reshape(dummy_img.size, 1), nres, axis = 1)

    # plot and clip the image
    im = ax.pcolormesh(np.linspace((xmin),(xmax), nres), dummy_y, dummy_img,
                       cmap = cmap, norm = norm, alpha = alpha_fill, shading = shading,
                       linewidth = lw_fill, zorder = zorder, antialiased = True)
    im.set_clip_path(patch)

    _ = MultiColorLine(ax, norm, cmap, x, yup, alpha = alpha_contour, lw = lw_contour, ls = ls_contour,
                       zorder = zorder, cvals = cvals, **kwargs)
    _ = MultiColorLine(ax, norm, cmap, x, ylow, alpha = alpha_contour, lw = lw_contour, ls = ls_contour, 
                       zorder = zorder, cvals = cvals, **kwargs)

    return line_data










class ThreePointsHandler(HandlerPathCollection):
    def create_artists(self, legend, artist,
                       x0, y0, width, height, fontsize, trans):

        new_width = width*0.7
        x0 = width-new_width
        l1 = Line2D([x0/2], [height/2], marker='o', color='w',
                    markerfacecolor=artist.cmap(0.1), markersize=10)
        l2 = Line2D([x0/2+1/2*new_width], [height/2], marker='o', color='w',
                    markerfacecolor=artist.cmap(0.5), markersize=10)
        l3 = Line2D([x0/2+new_width], [height/2], marker='o', color='w',
                    markerfacecolor=artist.cmap(0.9), markersize=10)

        return [l1, l2, l3]


class HandlerColorLineCollection(HandlerLineCollection):
    def create_artists(self, legend, artist ,xdescent, ydescent,
                       width, height, fontsize, trans):

        x = np.linspace(0, width, self.get_numpoints(legend)+1)
        y = np.zeros(self.get_numpoints(legend)+1) + height/2. - ydescent

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=artist.cmap,
                            transform=trans)
        lc.set_array(x)
        lc.set_linewidth(artist.get_linewidth()+1)
        return [lc]


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = pltcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_with_lines(x, y, color, s, lw, label, ax, ec = 'white', lw_ec = 0.1, alpha = 0.9,
                    alpha_scatter = 1, zorder_scatter = np.inf, zorder_line = 1):
    ax.scatter(x, y, s = s, color = color, label = label, ec = ec, lw = lw_ec, zorder = zorder_scatter, alpha = alpha_scatter)
    ax.plot(x, y, color = color, lw = lw, alpha = alpha, zorder = zorder_line)


def create_legend(ax, legend_elements, labels, handler_maps = None,
                  fontsize = 17, loc = 'upper right', handlelength = 1):

    if handler_maps == None:
        handler_maps = [matplotlib.legend_handler.HandlerLine2D()]*len(legend_elements)

    handler_dict = dict(zip(legend_elements,handler_maps))

    ll = ax.legend(legend_elements, labels, handler_map = handler_dict,
                   framealpha=1, fontsize = fontsize, loc = loc, handlelength = handlelength)
    return ll

def set_axis_prop3d(ax):
    [t.set_va('center') for t in ax.get_yticklabels()]
    [t.set_ha('left') for t in ax.get_yticklabels()]
    [t.set_va('center') for t in ax.get_xticklabels()]
    [t.set_ha('right') for t in ax.get_xticklabels()]
    [t.set_va('center') for t in ax.get_zticklabels()]
    [t.set_ha('left') for t in ax.get_zticklabels()]

    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4

    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.zaxis.set_major_locator(MultipleLocator(0.5))

    ax.view_init(elev=20, azim=135)
    return ax

class MidpointNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def restore_log_ticks(ax_axis, numticks):
    major = matplotlib.ticker.LogLocator(base = 10.0, numticks = numticks)
    ax_axis.set_major_locator(major)
    minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax_axis.set_minor_locator(minor)
    ax_axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    
def white_to_color(color, name = "cmap"):
    colors = ["w", color]
    nodes = np.linspace(0, 1, len(colors))
    return LinearSegmentedColormap.from_list("name", list(zip(nodes, colors)))


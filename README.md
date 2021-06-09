# TEMpcPlot

## Introduction
The library TEMpcPlot has as object the treatments of a Sequence of electropn diffraction cliches to obtain a three dimensional redcipriocal lattice. The idea behind is to find a way of work for TEM, available on pc, with graphycal approach (TEMpcPlot)

The library is divided in two main class
** TEMpcPlot.SeqIm **
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
        | def D3_peaks(tollerance=15)
        | def plot(log=False)
        | def plot_cal(axes)
    """

** EwaldPeaks **
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











The mode of use is relativelly simple :
create a SeqIm object

Ex1 = TEMpcPlot.SeqIm('cr2.sqi')
construct a unindixed reciprocal lattice

Ex1.plot()

Ex1.D3_peaks(tollerance=15)
Index manually the reciprocal space

Ex1.EwP.plot()
reconstruct reciprocal space

Ex1.Ewp.create_layer(hkl)


## Step by step instructions for installation
- install anaconda or miniconda(https://docs.conda.io/en/latest/miniconda.html)
- open an anaconda prompt on the start menu
- type```bash
conda install -c cprestip tempcplot
```
#### facultative but significant better if ipython is installed
```bash
conda install ipython
```

## in order to work
The library is divided in two main class





<div class="document">

<div class="documentwrapper">

<div class="bodywrapper">

<div class="body" role="main">

<div class="section" id="module-TEMpcPlot"><span id="welcome-to-tempcplot-s-documentation"></span>

# Welcome to TEMpcPLOT’s documentation![¶](#module-TEMpcPlot "Permalink to this headline")

The library TEMpcPlot has as object the treatments of a Sequence of electropn diffraction cliches to obtain a three dimensional redcipriocal lattice. The idea behind is to find a way of work for TEM, available on pc, with graphycal approach (TEMpcPlot)

<dl>

<dt>The mode of use is relativelly simple :</dt>

<dd>

1.  create a SeqIm object

<div class="doctest highlight-default notranslate">

<div class="highlight">

<pre><span></span><span class="gp">>>></span> <span class="n">Ex1</span> <span class="o">=</span> <span class="n">TEMpcPlot</span><span class="p">(</span><span class="n">filelist</span><span class="p">,</span> <span class="n">angles</span><span class="p">)</span>
</pre>

</div>

</div>

1.  construct a unindixed reciprocal lattice

<div class="doctest highlight-default notranslate">

<div class="highlight">

<pre><span></span><span class="gp">>>></span> <span class="n">Ex1</span><span class="o">.</span><span class="n">D3_peaks</span><span class="p">(</span><span class="n">tollerance</span><span class="o">=</span><span class="mi">15</span><span class="p">)</span>
</pre>

</div>

</div>

1.  Index manually the reciprocal space

<div class="doctest highlight-default notranslate">

<div class="highlight">

<pre><span></span><span class="gp">>>></span> <span class="n">Ex1</span><span class="o">.</span><span class="n">EwP</span><span class="o">.</span><span class="n">plot</span><span class="p">()</span>
</pre>

</div>

</div>

1.  reconstruct reciprocal space

<div class="doctest highlight-default notranslate">

<div class="highlight">

<pre><span></span><span class="gp">>>></span> <span class="n">Ex1</span><span class="o">.</span><span class="n">Ewp</span><span class="o">.</span><span class="n">create_layer</span><span class="p">(</span><span class="n">hkl</span><span class="p">)</span>
</pre>

</div>

</div>

</dd>

</dl>

* * *

<dl class="py class">

<dt id="TEMpcPlot.SeqIm">_class_ `TEMpcPlot.``SeqIm`<span class="sig-paren">(</span>_<span class="n">filenames</span>_, _<span class="n">filesangle</span><span class="o">=</span><span class="default_value">None</span>_, _<span class="o">*</span><span class="n">args</span>_, _<span class="o">**</span><span class="n">kwords</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.SeqIm "Permalink to this definition")</dt>

<dd>

sequence of images

this class is supposed to use a sequence of image. each element of the class is an image

<dl class="field-list simple">

<dt class="field-odd">Parameters</dt>

<dd class="field-odd">

*   **filenames** (_list_) – list string describing the exception.

*   **filesangle** (_str_) – Human readable file with angles

</dd>

<dt class="field-even">Variables</dt>

<dd class="field-even">

*   **EwP** ([_TEMpcPlot.EwaldPeaks_](#TEMpcPlot.EwaldPeaks "TEMpcPlot.EwaldPeaks")) – Ewald peaks 3D set of peaks

*   **rot_vect** (_list_) – list of Rotation vector for each image

*   **scale** (_list_) – scale(magnification) of the images

*   **ima** (_TEMpcPlot.Mimage_) – current image of the sequence

</dd>

</dl>

<div class="admonition note">

Note

<div class="line-block">

<div class="line">Methods to use:</div>

<div class="line">def D3_peaks(tollerance=15)</div>

<div class="line">def plot(log=False)</div>

<div class="line">def plot_cal(axes)</div>

<div class="line">def save(axes)</div>

<div class="line">def load(axes)</div>

<div class="line">def help()</div>

</div>

</div>

<dl class="py method">

<dt id="TEMpcPlot.SeqIm.D3_peaks">`D3_peaks`<span class="sig-paren">(</span>_<span class="n">tollerance</span><span class="o">=</span><span class="default_value">15</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.SeqIm.D3_peaks "Permalink to this definition")</dt>

<dd>

sum and correct the peaks of all images :param tollerance () = pixel tollerance to determine if a peak: in two images is the same peak.

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.SeqIm.find_peaks">`find_peaks`<span class="sig-paren">(</span>_<span class="n">rad_c</span><span class="o">=</span><span class="default_value">1.5</span>_, _<span class="n">tr_c</span><span class="o">=</span><span class="default_value">0.02</span>_, _<span class="n">dist</span><span class="o">=</span><span class="default_value">None</span>_, _<span class="n">symf</span><span class="o">=</span><span class="default_value">None</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.SeqIm.find_peaks "Permalink to this definition")</dt>

<dd>

findf the peak allows to search again the peaks in all the image witht the same parameter

<dl class="field-list simple">

<dt class="field-odd">Parameters</dt>

<dd class="field-odd">

*   **tr_c** (_float_) – total range coefficent the minimal intensity of the peak should be at list tr_c*self.ima.max()

*   **rad_c** (_float_) – coefficent in respect of the center radious peaks should be separate from at list self.rad*rad_c

*   **dist** – (float): maximum distance in pixel

</dd>

</dl>

Examples

<div class="doctest highlight-default notranslate">

<div class="highlight">

<pre><span></span><span class="gp">>>></span> <span class="n">Exp1</span><span class="o">.</span><span class="n">find_peaks</span><span class="p">()</span>
</pre>

</div>

</div>

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.SeqIm.help">`help`<span class="sig-paren">(</span><span class="sig-paren">)</span>[¶](#TEMpcPlot.SeqIm.help "Permalink to this definition")</dt>

<dd>

print class help

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.SeqIm.load">_classmethod_ `load`<span class="sig-paren">(</span>_<span class="n">filename</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.SeqIm.load "Permalink to this definition")</dt>

<dd>

load a saved project it is necessary that images remain in the same relative position :param filename: filename to open :type filename: str

Examples

<div class="doctest highlight-default notranslate">

<div class="highlight">

<pre><span></span><span class="gp">>>></span> <span class="n">exp1</span> <span class="o">=</span> <span class="n">SeqIm</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="s1">'exp1.sqm'</span><span class="p">)</span>
</pre>

</div>

</div>

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.SeqIm.plot">`plot`<span class="sig-paren">(</span>_<span class="n">log</span><span class="o">=</span><span class="default_value">False</span>_, _<span class="n">fig</span><span class="o">=</span><span class="default_value">None</span>_, _<span class="n">ax</span><span class="o">=</span><span class="default_value">None</span>_, _<span class="n">tool_b</span><span class="o">=</span><span class="default_value">None</span>_, _<span class="o">*</span><span class="n">args</span>_, _<span class="o">**</span><span class="n">kwds</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.SeqIm.plot "Permalink to this definition")</dt>

<dd>

plot the images of the sequences with peaks

<dl class="field-list simple">

<dt class="field-odd">Parameters</dt>

<dd class="field-odd">

*   **log** (_Bool_) – plot logaritm of intyensity

*   **anf keyworg directly of matplotlib plot** (_aargs_) –

</dd>

</dl>

Examples

<div class="doctest highlight-default notranslate">

<div class="highlight">

<pre><span></span><span class="gp">>>></span> <span class="n">Exp1</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">log</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="gp">>>></span> <span class="n">Exp1</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="kc">True</span><span class="p">)</span>
<span class="gp">>>></span> <span class="n">Exp1</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>
<span class="gp">>>></span> <span class="n">Exp1</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
<span class="gp">>>></span> <span class="n">Exp1</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">vmin</span> <span class="o">=</span> <span class="mi">10</span><span class="p">,</span> <span class="p">)</span>
</pre>

</div>

</div>

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.SeqIm.plot_cal">`plot_cal`<span class="sig-paren">(</span>_<span class="n">axes</span>_, _<span class="n">log</span><span class="o">=</span><span class="default_value">False</span>_, _<span class="o">*</span><span class="n">args</span>_, _<span class="o">**</span><span class="n">kwds</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.SeqIm.plot_cal "Permalink to this definition")</dt>

<dd>

plot the images of the sequences with peaks

<dl class="field-list simple">

<dt class="field-odd">Parameters</dt>

<dd class="field-odd">

*   **base for reciprocal space as** (_axes_) – one defined in EwP

*   **log** (_Bool_) – plot logaritm of intyensity

*   **anf keyworg directly of matplotlib plot** (_aargs_) –

</dd>

</dl>

Examples

<div class="doctest highlight-default notranslate">

<div class="highlight">

<pre><span></span><span class="gp">>>></span> <span class="n">Exp1</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">Exp1</span><span class="o">.</span><span class="n">EwP</span><span class="o">.</span><span class="n">axes</span><span class="p">,</span> <span class="n">log</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="gp">>>></span> <span class="n">Exp1</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">Exp1</span><span class="o">.</span><span class="n">EwP</span><span class="o">.</span><span class="n">axes</span><span class="p">)</span>
</pre>

</div>

</div>

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.SeqIm.save">`save`<span class="sig-paren">(</span>_<span class="n">filesave</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.SeqIm.save "Permalink to this definition")</dt>

<dd>

> <div>
> 
> save the project to open later formats available: None: pickel format good for python
> 
> </div>

<dl class="field-list simple">

<dt class="field-odd">Parameters</dt>

<dd class="field-odd">

**filename** (_str_) – filename to save

</dd>

</dl>

Examples

<div class="doctest highlight-default notranslate">

<div class="highlight">

<pre><span></span><span class="gp">>>></span> <span class="n">Exp1</span><span class="o">.</span><span class="n">save</span><span class="p">(</span><span class="s1">'exp1'</span><span class="p">)</span>
</pre>

</div>

</div>

</dd>

</dl>

</dd>

</dl>

<dl class="py class">

<dt id="TEMpcPlot.EwaldPeaks">_class_ `TEMpcPlot.``EwaldPeaks`<span class="sig-paren">(</span>_<span class="n">positions</span>_, _<span class="n">intensity</span>_, _<span class="n">rot_vect</span><span class="o">=</span><span class="default_value">None</span>_, _<span class="n">angles</span><span class="o">=</span><span class="default_value">None</span>_, _<span class="n">r0</span><span class="o">=</span><span class="default_value">None</span>_, _<span class="n">z0</span><span class="o">=</span><span class="default_value">None</span>_, _<span class="n">pos0</span><span class="o">=</span><span class="default_value">None</span>_, _<span class="n">scale</span><span class="o">=</span><span class="default_value">None</span>_, _<span class="n">axes</span><span class="o">=</span><span class="default_value">None</span>_, _<span class="n">set_cell</span><span class="o">=</span><span class="default_value">True</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.EwaldPeaks "Permalink to this definition")</dt>

<dd>

Set of peaks position and intensity this class manages peaks position and intensity and the methods related to lattice indexing and refinement could be created as an attribute EwP of a SeqIm class by using methods D3_peaks or by sum with an another EwaldPeaks class with the same first image

<dl class="field-list simple">

<dt class="field-odd">Parameters</dt>

<dd class="field-odd">

*   **positions** (_list_) – list containing the coordonates of peaks

*   **intensity** (_list_) – list containing the intensity of peaks

</dd>

<dt class="field-even">Variables</dt>

<dd class="field-even">

*   **pos** (_list_) – Ewald peaks 3D set of peaks

*   **int** (_list_) – list of Rotation vector for each image

*   **pos_cal** (_np.array_) – array witht he position in the new basis

*   **rMT** (_np.array_) – reciprocal metric tensor

*   **axis** (_np.array_) – reciprocal basis set, 3 coloums

*   **cell** (_dict_) – a dictionary witht the value of real space cell

*   **graph** (_D3plot.D3plot_) – graph Ewald peaks 3D set of peaks used to index

</dd>

</dl>

<div class="admonition note">

Note

<div class="line-block">

<div class="line">Examples:</div>

<div class="line-block">

<div class="line">>>>Exp1.D3_peaks(tollerance=5)</div>

<div class="line">>>>Exp1.EwP is defined</div>

<div class="line">>>>EWT= Exp1.EwP + Exp2.EwP</div>

</div>

<div class="line">Methods to use:</div>

<div class="line">def D3_peaks(tollerance=15)</div>

<div class="line">def plot_int()</div>

<div class="line">def plot_proj_int()</div>

<div class="line">def plot_reduce</div>

<div class="line">def plot_reduce</div>

<div class="line">def refine_axes</div>

<div class="line">def set_cell</div>

<div class="line">def create_layer</div>

<div class="line">def save(name)</div>

<div class="line">def load(name)</div>

<div class="line">def help()</div>

</div>

</div>

<dl class="py method">

<dt id="TEMpcPlot.EwaldPeaks.cr_cond">`cr_cond`<span class="sig-paren">(</span>_<span class="n">operator</span><span class="o">=</span><span class="default_value">None</span>_, _<span class="n">lim</span><span class="o">=</span><span class="default_value">None</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.EwaldPeaks.cr_cond "Permalink to this definition")</dt>

<dd>

define filtering condition

fuch function create a function that filter the data following the condition

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.EwaldPeaks.create_layer">`create_layer`<span class="sig-paren">(</span>_<span class="n">hkl</span>_, _<span class="n">n</span>_, _<span class="n">size</span><span class="o">=</span><span class="default_value">0.25</span>_, _<span class="n">toll</span><span class="o">=</span><span class="default_value">0.15</span>_, _<span class="n">mir</span><span class="o">=</span><span class="default_value">0</span>_, _<span class="n">spg</span><span class="o">=</span><span class="default_value">None</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.EwaldPeaks.create_layer "Permalink to this definition")</dt>

<dd>

create a specific layer create a reciprocal space layer

<dl class="field-list simple">

<dt class="field-odd">Parameters</dt>

<dd class="field-odd">

*   **hkl** (_str_) – constant index for the hkl plane to plot, format(‘k’)

*   **n** (_float__,_ _int_) – value of hkl

*   **size** (_float_) – intensity scaling * if positive, scale intensity of each peaks respect the max * if negative, scale a common value for all peaks

*   **mir** (_bool_) – mirror in respect of n meaning =/-n

*   **tollerance** (_float_) – exclude from the plot peaks at higher distance

*   **spg** (_str_) – allows to index the peaks, and check if they are extinted

</dd>

</dl>

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.EwaldPeaks.load">_classmethod_ `load`<span class="sig-paren">(</span>_<span class="n">filename</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.EwaldPeaks.load "Permalink to this definition")</dt>

<dd>

load EwP in python format Example: >>>cr1 = EwaldPeaks.load(‘cr1.ewp’)

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.EwaldPeaks.plot">`plot`<span class="sig-paren">(</span><span class="sig-paren">)</span>[¶](#TEMpcPlot.EwaldPeaks.plot "Permalink to this definition")</dt>

<dd>

open a D3plot graph :ivar ~EwaldPeaks.plot.graph: graph Ewald peaks 3D set of peaks used to index

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.EwaldPeaks.plot_int">`plot_int`<span class="sig-paren">(</span><span class="sig-paren">)</span>[¶](#TEMpcPlot.EwaldPeaks.plot_int "Permalink to this definition")</dt>

<dd>

Plot instogramm of intensity of the peaks

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.EwaldPeaks.plot_proj_int">`plot_proj_int`<span class="sig-paren">(</span>_<span class="n">cell</span><span class="o">=</span><span class="default_value">True</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.EwaldPeaks.plot_proj_int "Permalink to this definition")</dt>

<dd>

plot peak presence instogramm as a function of the cell

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.EwaldPeaks.plot_reduce">`plot_reduce`<span class="sig-paren">(</span>_<span class="n">tollerance</span><span class="o">=</span><span class="default_value">0.1</span>_, _<span class="n">condition</span><span class="o">=</span><span class="default_value">None</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.EwaldPeaks.plot_reduce "Permalink to this definition")</dt>

<dd>

plot collapsed reciprocal space plot the position of the peaks in cell coordinatete and all reduced to a single cell. it create a self.reduce attribute containingt he graph

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.EwaldPeaks.refine_angles">`refine_angles`<span class="sig-paren">(</span>_<span class="n">axes</span><span class="o">=</span><span class="default_value">None</span>_, _<span class="n">tollerance</span><span class="o">=</span><span class="default_value">0.1</span>_, _<span class="n">zero_tol</span><span class="o">=</span><span class="default_value">0.1</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.EwaldPeaks.refine_angles "Permalink to this definition")</dt>

<dd>

refine reciprocal cell basis refine the reciprocal cell basis in respect to data that are indexed in the tollerance range.

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.EwaldPeaks.refine_axang">`refine_axang`<span class="sig-paren">(</span>_<span class="n">axes</span><span class="o">=</span><span class="default_value">None</span>_, _<span class="n">tollerance</span><span class="o">=</span><span class="default_value">0.1</span>_, _<span class="n">zero_tol</span><span class="o">=</span><span class="default_value">0.1</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.EwaldPeaks.refine_axang "Permalink to this definition")</dt>

<dd>

refine reciprocal cell basis refine the reciprocal cell basis in respect to data that are indexed in the tollerance range.

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.EwaldPeaks.refine_axes">`refine_axes`<span class="sig-paren">(</span>_<span class="n">axes</span><span class="o">=</span><span class="default_value">None</span>_, _<span class="n">tollerance</span><span class="o">=</span><span class="default_value">0.1</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.EwaldPeaks.refine_axes "Permalink to this definition")</dt>

<dd>

refine reciprocal cell basis refine the reciprocal cell basis in respect to data that are indexed in the tollerance range.

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.EwaldPeaks.save">`save`<span class="sig-paren">(</span>_<span class="n">filename</span>_, _<span class="n">dictionary</span><span class="o">=</span><span class="default_value">False</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.EwaldPeaks.save "Permalink to this definition")</dt>

<dd>

save EwP

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.EwaldPeaks.set_cell">`set_cell`<span class="sig-paren">(</span>_<span class="n">axes</span><span class="o">=</span><span class="default_value">None</span>_, _<span class="n">axes_std</span><span class="o">=</span><span class="default_value">None</span>_, _<span class="n">tollerance</span><span class="o">=</span><span class="default_value">0.1</span>_, _<span class="n">cond</span><span class="o">=</span><span class="default_value">None</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.EwaldPeaks.set_cell "Permalink to this definition")</dt>

<dd>

calculation of the cell effect the calculation to obtain the cell

<dl class="field-list simple">

<dt class="field-odd">Parameters</dt>

<dd class="field-odd">

**axis** (_np.array 3__,__3_) – the new reciprocal basis to be used in the format if axis is not inoput the programm seach if a new basis has been defined graphically

</dd>

</dl>

<div class="line-block">

<div class="line">axes format: np.array([</div>

<div class="line-block">

<div class="line">[a1, b1, c1],</div>

<div class="line">[a2, b2, c2],</div>

<div class="line">[a3, b3, c3]])</div>

</div>

</div>

<dl class="field-list simple">

<dt class="field-odd">Returns</dt>

<dd class="field-odd">

nothing

</dd>

<dt class="field-even">Variables</dt>

<dd class="field-even">

*   **self.rMT** (_np.array_) – reciprocal metric tensor

*   **self.cell** (_dict_) – a dictionary witht the value of real space cell

*   **self.rMT** – reciprocal metric tensor

*   **self.cell** – a dictionary witht the value of real space cell

</dd>

</dl>

</dd>

</dl>

</dd>

</dl>

<dl class="py function">

<dt id="TEMpcPlot.pt_p">`TEMpcPlot.``pt_p`<span class="sig-paren">(</span>_<span class="n">atom</span>_, _<span class="n">property</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.pt_p "Permalink to this definition")</dt>

<dd>

Atomic properties Tables with atomic properties

<dl class="field-list simple">

<dt class="field-odd">Parameters</dt>

<dd class="field-odd">

**property** (_str_) – property type

</dd>

<dt class="field-even">Returns</dt>

<dd class="field-even">

property of the atoms

</dd>

<dt class="field-odd">Return type</dt>

<dd class="field-odd">

floats, string

</dd>

</dl>

<div class="admonition note">

Note

<div class="line-block">

<div class="line">Examples:</div>

<div class="line-block">

<div class="line">>>>pt_p(34, ‘sym’)</div>

<div class="line">>>>pt_p(‘Cu’, ‘At_w’)</div>

</div>

<div class="line">‘At_w’ : atomic weight</div>

<div class="line">‘Z’ : atomic number</div>

<div class="line">‘cov_r’ : covalent radii</div>

<div class="line">‘sym’ : atomic symbol</div>

<div class="line">‘e_conf’ : electronic conf.</div>

<div class="line">‘ox_st’ : oxydation state</div>

<div class="line">‘bon_dis’ : typical bond distances</div>

<div class="line">‘edges’ : x-ray edges</div>

</div>

</div>

</dd>

</dl>

<span class="target" id="module-TEMpcPlot.TEM.d3plot"></span>

<dl class="py class">

<dt id="TEMpcPlot.TEM.d3plot.D3plot">_class_ `TEMpcPlot.TEM.d3plot.``D3plot`<span class="sig-paren">(</span>_<span class="n">EwPePos</span>_, _<span class="n">size</span><span class="o">=</span><span class="default_value">'o'</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.TEM.d3plot.D3plot "Permalink to this definition")</dt>

<dd>

Class used to plot a set of 3D peaks

<dl class="py method">

<dt id="TEMpcPlot.TEM.d3plot.D3plot.allign_a">`allign_a`<span class="sig-paren">(</span><span class="sig-paren">)</span>[¶](#TEMpcPlot.TEM.d3plot.D3plot.allign_a "Permalink to this definition")</dt>

<dd>

rotate the peaks in order to allign to a* axis to z same command for b* and c*

Example

<div class="doctest highlight-default notranslate">

<div class="highlight">

<pre><span></span><span class="gp">>>></span> <span class="n">Exp1</span><span class="o">.</span><span class="n">EwP</span><span class="o">.</span><span class="n">graph</span><span class="o">.</span><span class="n">allign_a</span><span class="p">()</span>
</pre>

</div>

</div>

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.TEM.d3plot.D3plot.define_axis">`define_axis`<span class="sig-paren">(</span>_<span class="n">abc</span>_, _<span class="n">m</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.TEM.d3plot.D3plot.define_axis "Permalink to this definition")</dt>

<dd>

define axis define axis graphically tracing a line

<dl class="field-list simple">

<dt class="field-odd">Parameters</dt>

<dd class="field-odd">

*   **abc** (_str_) – name of the axis

*   **m** (_int_) – multiple that will be traCED

</dd>

</dl>

Example

>>>Exp1.EwP.graph.define_axis(‘a’, 4)

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.TEM.d3plot.D3plot.filter_int">`filter_int`<span class="sig-paren">(</span>_<span class="n">operator</span><span class="o">=</span><span class="default_value">None</span>_, _<span class="n">lim</span><span class="o">=</span><span class="default_value">None</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.TEM.d3plot.D3plot.filter_int "Permalink to this definition")</dt>

<dd>

conserve only peaks respecting an intensity condition conserve only peaks respecting an intensity condition, to determine the most usefull values use Exp1.EwP.plot_int() .. rubric:: Example

<div class="doctest highlight-default notranslate">

<div class="highlight">

<pre><span></span><span class="gp">>>></span> <span class="n">Exp1</span><span class="o">.</span><span class="n">EwP</span><span class="o">.</span><span class="n">graph</span><span class="o">.</span><span class="n">filter_int</span><span class="p">(</span><span class="s1">'>'</span><span class="p">,</span> <span class="mi">1000</span><span class="p">)</span>
<span class="gp">>>></span> <span class="n">Exp1</span><span class="o">.</span><span class="n">EwP</span><span class="o">.</span><span class="n">graph</span><span class="o">.</span><span class="n">filter_int</span><span class="p">(</span><span class="s1">'<'</span><span class="p">,</span> <span class="mi">1000</span><span class="p">)</span>
</pre>

</div>

</div>

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.TEM.d3plot.D3plot.filter_layer">`filter_layer`<span class="sig-paren">(</span>_<span class="n">listn</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.TEM.d3plot.D3plot.filter_layer "Permalink to this definition")</dt>

<dd>

conserve only the layers in list

Examples

<div class="doctest highlight-default notranslate">

<div class="highlight">

<pre><span></span><span class="gp">>>></span> <span class="n">Exp1</span><span class="o">.</span><span class="n">EwP</span><span class="o">.</span><span class="n">graph</span><span class="o">.</span><span class="n">filter_layer</span><span class="p">([</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">])</span>
</pre>

</div>

</div>

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.TEM.d3plot.D3plot.rotate_0">`rotate_0`<span class="sig-paren">(</span><span class="sig-paren">)</span>[¶](#TEMpcPlot.TEM.d3plot.D3plot.rotate_0 "Permalink to this definition")</dt>

<dd>

rotate to first orientation

Example

<div class="doctest highlight-default notranslate">

<div class="highlight">

<pre><span></span><span class="gp">>>></span> <span class="n">Exp1</span><span class="o">.</span><span class="n">EwP</span><span class="o">.</span><span class="n">graph</span><span class="o">.</span><span class="n">rotate_0</span><span class="p">()</span>
</pre>

</div>

</div>

</dd>

</dl>

<dl class="py method">

<dt id="TEMpcPlot.TEM.d3plot.D3plot.rotatex">`rotatex`<span class="sig-paren">(</span>_<span class="n">deg</span><span class="o">=</span><span class="default_value">90</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.TEM.d3plot.D3plot.rotatex "Permalink to this definition")</dt>

<dd>

rotate along the x axis default value 90 same command for y and z

<dl class="field-list simple">

<dt class="field-odd">Parameters</dt>

<dd class="field-odd">

**deg** (_float_) – angle in degree to rotate

</dd>

</dl>

Examples

<div class="doctest highlight-default notranslate">

<div class="highlight">

<pre><span></span><span class="gp">>>></span> <span class="n">Exp1</span><span class="o">.</span><span class="n">EwP</span><span class="o">.</span><span class="n">graph</span><span class="o">.</span><span class="n">rotatex</span><span class="p">(</span><span class="mi">30</span><span class="p">)</span>
<span class="gp">>>></span> <span class="n">Exp1</span><span class="o">.</span><span class="n">EwP</span><span class="o">.</span><span class="n">graph</span><span class="o">.</span><span class="n">rotatex</span><span class="p">(</span><span class="o">-</span><span class="mi">30</span><span class="p">)</span>
</pre>

</div>

</div>

</dd>

</dl>

</dd>

</dl>

<dl class="py class">

<dt id="TEMpcPlot.TEM.d3plot.D3plotr">_class_ `TEMpcPlot.TEM.d3plot.``D3plotr`<span class="sig-paren">(</span>_<span class="n">EwPePos</span>_, _<span class="n">origin</span>_, _<span class="n">size</span><span class="o">=</span><span class="default_value">'o'</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.TEM.d3plot.D3plotr "Permalink to this definition")</dt>

<dd>

<dl class="py method">

<dt id="TEMpcPlot.TEM.d3plot.D3plotr.define_axis">`define_axis`<span class="sig-paren">(</span>_<span class="n">abc</span>_, _<span class="n">m</span>_, _<span class="n">origin</span><span class="o">=</span><span class="default_value">[0, 0, 0]</span>_<span class="sig-paren">)</span>[¶](#TEMpcPlot.TEM.d3plot.D3plotr.define_axis "Permalink to this definition")</dt>

<dd>

define axis define axis graphically tracing a line

<dl class="field-list simple">

<dt class="field-odd">Parameters</dt>

<dd class="field-odd">

*   **abc** (_str_) – name of the axis

*   **m** (_int_) – multiple that will be traCED

</dd>

</dl>

Example

>>>Exp1.EwP.graph.define_axis(‘a’, 4)

</dd>

</dl>

</dd>

</dl>

</div>

<div class="section" id="indices-and-tables">

# Indices and tables[¶](#indices-and-tables "Permalink to this headline")

*   [<span class="std std-ref">Index</span>](genindex.html)

*   [<span class="std std-ref">Module Index</span>](py-modindex.html)

*   [<span class="std std-ref">Search Page</span>](search.html)

</div>

</div>

</div>

</div>

<div class="sphinxsidebar" role="navigation" aria-label="main navigation">

<div class="sphinxsidebarwrapper">

# [TEMpcPlot](#)

### Navigation

<div class="relations">

### Related Topics

*   [Documentation overview](#)

</div>

<div id="searchbox" style="display: none" role="search">

### Quick search

<div class="searchformwrapper">

<form class="search" action="search.html" method="get"><input type="text" name="q" aria-labelledby="searchlabel"> <input type="submit" value="Go"></form>

</div>

</div>

<script>$('#searchbox').show(0);</script></div>

</div>

</div>

<div class="footer">©2020, C. Prestipino. | Powered by [Sphinx 3.0.4](http://sphinx-doc.org/) & [Alabaster 0.7.12](https://github.com/bitprophet/alabaster) | [Page source](_sources/index.rst.txt)</div>
    
   

## License

See the [License File](./LICENSE.md).

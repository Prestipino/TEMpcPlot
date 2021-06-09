# TEMpcPlot

## Introduction
The library TEMpcPlot has as object the treatments of a Sequence of electropn diffraction cliches to obtain a three dimensional redcipriocal lattice. The idea behind is to find a way of work for TEM, available on pc, with graphycal approach (TEMpcPlot)

The library is divided in two main class
** TEMpcPlot.SeqIm **
    a sequence of image where each element of the class is an image

** EwaldPeaks **
    Set of peaks position and intensity
    this class manages peaks position and intensity and the methods related to
    lattice indexing and refinement
    could be created as an attribute EwP of a SeqIm class by using methods D3_peaks
    or by sum with an another EwaldPeaks class with the same first image

## Typical sequence of commands
The mode of use is relativelly simple :
create a SeqIm object

Ex1 = TEMpcPlot.SeqIm('cr2.sqi')
construct a unindixed reciprocal lattice

Ex1.plot()

Ex1.D3_peaks(tollerance=15)
Index manually the reciprocal space

Ex1.EwP.plot()
reconstruct reciprocal space

Ex1.EwP.set_cell()
take in account the recipriocal cell

Ex1.Ewp.create_layer(hkl)


## Step by step instructions for installation
- install anaconda or miniconda(https://docs.conda.io/en/latest/miniconda.html)
- open an anaconda prompt on the start menu
- type:
```bash
conda install -c cprestip tempcplot
```

##### facultative, but significant better if ipython is installed

```bash
conda install ipython
```
* * *
## In order to work
- create a text file  \*.sqi containing the name of files with the tilts angles in the working directory

      Example \*.sqi:
      # comment 
      UOs025Ge2_02juin21_0006.dm3    3.3  3.3
      UOs025Ge2_02juin21_0007.dm3    6.7  1.6
      UOs025Ge2_02juin21_0008.dm3    9.4  0.4

- open an anaconda prompth in the start menu
- type:
```bash
cd working_dir
```

- chande disk if necessary i.e. C: or D:

- type:
```bash
ipython
```

- type:
```bash
import TEMpcPlot as TP 
```

- create the SeqIm object
> Exp1 = TP.SeqIm('filename.sqi')

* * *
* * *
## usefull command Documentation
the main class documentaion to look are 
*  [SeqIm](#SeqIm)
*  [EwaldPeaks](#EwP)
*  [D3plot](#D3plot)

* * * ****************************




### _class_ TEMpcPlot.SeqIm(_filenames_, _filesangle\=None_, _\*args_, _\*\*kwords_)[¶](#TEMpcPlot.SeqIm "Permalink to this definition") <a name="SeqIm"></a>

sequence of images

this class is supposed to use a sequence of image. each element of the class is an image

***Parameters***
*   **filenames** (_list_) – list string describing the exception.   
*   **filesangle** (_str_) – Human readable file with angles
    

***Attributes***
*   **EwP** ([_TEMpcPlot.EwaldPeaks_](#EwP)) – Ewald peaks 3D set of peaks   
*   **rot\_vect** (_list_) – list of Rotation vector for each image   
*   **scale** (_list_) – scale(magnification) of the images
*   **ima** (_TEMpcPlot.Mimage_) – current image of the sequence
    
___
#### `SeqIm.D3_peaks`(_tollerance\=15_)[¶](#TEMpcPlot.SeqIm.D3_peaks "Permalink to this definition")

sum and correct the peaks of all images :param tollerance () = pixel tollerance to determine if a peak: in two images is the same peak.
___
#### `SeqIm.find_peaks`(_rad\_c\=1.5_, _tr\_c\=0.02_, _dist\=None_, _symf\=None_)[¶](#TEMpcPlot.SeqIm.find_peaks "Permalink to this definition")

findf the peak allows to search again the peaks in all the image witht the same parameter

***Parameters***
*   **tr\_c** (_float_) – total range coefficent the minimal intensity of the peak should be at list tr\_c\*self.ima.max()  
*   **rad\_c** (_float_) – coefficent in respect of the center radious peaks should be separate from at list self.rad\*rad\_c
*   **dist** – (float): maximum distance in pixel  

***Examples***
> Exp1.find\_peaks()

___
#### `SeqIm.help`()[¶](#TEMpcPlot.SeqIm.help "Permalink to this definition")

print class help
___
####  `SeqIm.load`(_filename_)[¶](#TEMpcPlot.SeqIm.load "Permalink to this definition")

load a saved project it is necessary that images remain in the same relative position :param filename: filename to open :type filename: str

Examples

\>>> exp1 \= SeqIm.load('exp1.sqm')
___
####  `SeqIm.plot`(_log\=False_, _fig\=None_, _ax\=None_, _tool\_b\=None_, _\*args_, _\*\*kwds_)[¶](#TEMpcPlot.SeqIm.plot "Permalink to this definition")

plot the images of the sequences with peaks

Parameters

*   **log** (_Bool_) – plot logaritm of intyensity
    
*   **anf keyworg directly of matplotlib plot** (_aargs_) –
    

***Examples***
> Exp1.plot(log\=True)
> Exp1.plot(True)
> Exp1.plot(1)
> Exp1.plot(0)
> Exp1.plot(vmin \= 10, )
___
#### `SeqIm.plot_cal`(_axes_, _log\=False_, _\*args_, _\*\*kwds_)[¶](#TEMpcPlot.SeqIm.plot_cal "Permalink to this definition")

plot the images of the sequences with peaks

Parameters

*   **base for reciprocal space as** (_axes_) – one defined in EwP
    
*   **log** (_Bool_) – plot logaritm of intyensity
    
*   **anf keyworg directly of matplotlib plot** (_aargs_) –
    

Examples

\>>> Exp1.plot(Exp1.EwP.axes, log\=True)
\>>> Exp1.plot(Exp1.EwP.axes)
___
####  `SeqIm.save`(_filesave_)[¶](#TEMpcPlot.SeqIm.save "Permalink to this definition")

> save the project to open later formats available: None: pickel format good for python

***Parameters***
**filename** (_str_) – filename to save

***Examples***
> Exp1.save('exp1')

* * *

### _class_ `TEMpcPlot.EwaldPeaks`(_positions_, _intensity_, _rot\_vect\=None_, _angles\=None_, _r0\=None_, _z0\=None_, _pos0\=None_, _scale\=None_, _axes\=None_, _set\_cell\=True_)<a name="EwP"></a>[¶]

Set of peaks position and intensity this class manages peaks position and intensity and the methods related to lattice indexing and refinement could be created as an attribute EwP of a SeqIm class by using methods D3\_peaks or by sum with an another EwaldPeaks class with the same first image

*Parameters*
*   **positions** (_list_) – list containing the coordonates of peaks
*   **intensity** (_list_) – list containing the intensity of peaks
    

*Variables*
*   **pos** (_list_) – Ewald peaks 3D set of peaks
*   **int** (_list_) – list of Rotation vector for each image    
*   **pos\_cal** (_np.array_) – array witht he position in the new basis    
*   **rMT** (_np.array_) – reciprocal metric tensor   
*   **axis** (_np.array_) – reciprocal basis set, 3 coloums   
*   **cell** (_dict_) – a dictionary witht the value of real space cell   
*   **graph** (_D3plot.D3plot_) – graph Ewald peaks 3D set of peaks used to index

*Examples*:
> Exp1.D3\_peaks(tollerance=5)\
> Exp1.EwP is defined\
> EWT= Exp1.EwP + Exp2.EwP\



___
#### `EwaldPeaks.create_layer`(_hkl_, _n_, _size\=0.25_, _toll\=0.15_, _mir\=0_, _spg\=None_)[¶](#TEMpcPlot.EwaldPeaks.create_layer "Permalink to this definition")

create a specific layer create a reciprocal space layer

Parameters
*   **hkl** (_str_) – constant index for the hkl plane to plot, format(‘k’)
*   **n** (_float__,_ _int_) – value of hkl
*   **size** (_float_) – intensity scaling \* if positive, scale intensity of each peaks respect the max \* if negative, scale a common value for all peaks
*   **mir** (_bool_) – mirror in respect of n meaning =/-n    
*   **tollerance** (_float_) – exclude from the plot peaks at higher distance    
*   **spg** (_str_) – allows to index the peaks, and check if they are extinted
    
___
#### `EwaldPeaks.load`(_filename_)[¶](#TEMpcPlot.EwaldPeaks.load "Permalink to this definition")

load EwP in python format Example: >>>cr1 = EwaldPeaks.load(‘cr1.ewp’)
___
#### `EwaldPeaks.plot`()[¶](#TEMpcPlot.EwaldPeaks.plot "Permalink to this definition")

open a D3plot graph :ivar ~EwaldPeaks.plot.graph: graph Ewald peaks 3D set of peaks used to index
___
#### `EwaldPeaks.plot_int`()[¶](#TEMpcPlot.EwaldPeaks.plot_int "Permalink to this definition")

Plot instogramm of intensity of the peaks
___
#### `plot_proj_int`(_cell\=True_)[¶](#TEMpcPlot.EwaldPeaks.plot_proj_int "Permalink to this definition")

plot peak presence instogramm as a function of the cell
___
#### `EwaldPeaks.plot_reduce`(_tollerance\=0.1_, _condition\=None_)[¶](#TEMpcPlot.EwaldPeaks.plot_reduce "Permalink to this definition")

plot collapsed reciprocal space plot the position of the peaks in cell coordinatete and all reduced to a single cell. it create a self.reduce attribute containingt he graph
___
#### `EwaldPeaks.refine_angles`(_axes\=None_, _tollerance\=0.1_, _zero\_tol\=0.1_)[¶](#TEMpcPlot.EwaldPeaks.refine_angles "Permalink to this definition")

refine reciprocal cell basis refine the reciprocal cell basis in respect to data that are indexed in the tollerance range.
___
#### `EwaldPeaks.refine_axang`(_axes\=None_, _tollerance\=0.1_, _zero\_tol\=0.1_)[¶](#TEMpcPlot.EwaldPeaks.refine_axang "Permalink to this definition")

refine reciprocal cell basis refine the reciprocal cell basis in respect to data that are indexed in the tollerance range.
___
#### `EwaldPeaks.refine_axes`(_axes\=None_, _tollerance\=0.1_)[¶](#TEMpcPlot.EwaldPeaks.refine_axes "Permalink to this definition")

refine reciprocal cell basis refine the reciprocal cell basis in respect to data that are indexed in the tollerance range.
___
#### `EwaldPeaks.save`(_filename_, _dictionary\=False_)[¶](#TEMpcPlot.EwaldPeaks.save "Permalink to this definition")

save EwP
___
#### `EwaldPeaks.set_cell`(_axes\=None_, _axes\_std\=None_, _tollerance\=0.1_, _cond\=None_)[¶](#TEMpcPlot.EwaldPeaks.set_cell "Permalink to this definition")

calculation of the cell effect the calculation to obtain the cell

Parameters

**axis** (_np.array 3__,__3_) – the new reciprocal basis to be used in the format if axis is not inoput the programm seach if a new basis has been defined graphically
axes format: np.array(\[\
\[a1, b1, c1\],\
\[a2, b2, c2\],\
\[a3, b3, c3\]\])



Variables

*   **self.rMT** (_np.array_) – reciprocal metric tensor  
*   **self.cell** (_dict_) – a dictionary witht the value of real space cell
*   **self.rMT** – reciprocal metric tensor
*   **self.cell** – a dictionary witht the value of real space cell
    
* * *
___
### _class_ `TEMpcPlot.TEM.d3plot.D3plot`(_EwPePos_, _size\='o'_)[¶](#TEMpcPlot.TEM.d3plot.D3plot "Permalink to this definition")<a name="D3plot"></a>

Class used to plot a set of 3D peaks
___
#### `allign_a`()[¶](#TEMpcPlot.TEM.d3plot.D3plot.allign_a "Permalink to this definition")

rotate the peaks in order to allign to a\* axis to z same command for b\* and c\*

Example
> Exp1.EwP.graph.allign\_a()
___
####  `define_axis`(_abc_, _m_)[¶](#TEMpcPlot.TEM.d3plot.D3plot.define_axis "Permalink to this definition")

define axis define axis graphically tracing a line

Parameters
*   **abc** (_str_) – name of the axis
*   **m** (_int_) – multiple that will be traCED
    

Example
> Exp1.EwP.graph.define\_axis(‘a’, 4)
___
#### `D3plot.filter_int`(_operator\=None_, _lim\=None_)[¶](#TEMpcPlot.TEM.d3plot.D3plot.filter_int "Permalink to this definition")

conserve only peaks respecting an intensity condition conserve only peaks respecting an intensity condition, to determine the most usefull values use Exp1.EwP.plot\_int()  

Example
> Exp1.EwP.graph.filter\_int('>', 1000)
> Exp1.EwP.graph.filter\_int('<', 1000)
___
#### `D3plot.filter_layer`(_listn_)[¶](#TEMpcPlot.TEM.d3plot.D3plot.filter_layer "Permalink to this definition")

conserve only the layers in list

Examples
> Exp1.EwP.graph.filter\_layer(\[0,1,2\])
___
#### `D3plot.rotate_0`()[¶](#TEMpcPlot.TEM.d3plot.D3plot.rotate_0 "Permalink to this definition")

rotate to first orientation

Example
> Exp1.EwP.graph.rotate\_0()
___
#### `D3plot.rotatex`(_deg\=90_)[¶](#TEMpcPlot.TEM.d3plot.D3plot.rotatex "Permalink to this definition")

rotate along the x axis default value 90 same command for y and z

Parameters
**deg** (_float_) – angle in degree to rotate

Examples
> Exp1.EwP.graph.rotatex(30)\
> Exp1.EwP.graph.rotatex(\-30)

* * *
### `TEMpcPlot.pt_p`(_atom_, _property_)[¶](#TEMpcPlot.pt_p "Permalink to this definition")

Atomic properties Tables with atomic properties

Parameters:\
**property** (_str_) – property type

Returns: property of the atoms

Return type: floats, string


Examples:
> pt\_p(34, ‘sym’)\
>pt\_p(‘Cu’, ‘At\_w’)

  
*Notes*
‘At\_w’ : atomic weight\
‘Z’ : atomic number\
‘cov\_r’ : covalent radii\
‘sym’ : atomic symbol\
‘e\_conf’ : electronic conf.\
‘ox\_st’ : oxydation state\
‘bon\_dis’ : typical bond distances\
‘edges’ : x-ray edges\

©2020, C. Prestipino. | Powered by [Sphinx 3.0.4](http://sphinx-doc.org/) & [Alabaster 0.7.12](https://github.com/bitprophet/alabaster) | [Page source](_sources/index.rst.txt)
   

## License

See the [License File](./LICENSE.md).

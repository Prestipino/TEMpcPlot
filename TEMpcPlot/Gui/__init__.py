"""
The library TEMpcPlot has as object the treatments of a Sequence of
electropn diffraction cliches to obtain a three dimensional redcipriocal 
lattice. The idea behind is to find a way of work for TEM, available on pc,
with graphycal approach (TEMpcPlot) 

The mode of use is relativelly simple :
    1. create a SeqIm object
      >>> Ex1 = TEMpcPlot(filelist, angles)
    2. construct a unindixed reciprocal lattice
      >>> Ex1.D3_peaks(tollerance=15)
    3. Index manually the reciprocal space
      >>> Ex1.EwP.plot()
    4. reconstruct reciprocal space  
      >>> Ex1.Ewp.create_layer(hkl)


-----
"""
from .SIP import(SeqImaPlot)
from .PEW import(EwaldPlot)
from .low_calc import(Bottom_create)
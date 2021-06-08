# TEMpcPlot

## Introduction
The library TEMpcPlot has as object the treatments of a Sequence of electropn diffraction cliches to obtain a three dimensional redcipriocal lattice. The idea behind is to find a way of work for TEM, available on pc, with graphycal approach (TEMpcPlot)

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
- install anaconda or miniconda
https://docs.conda.io/en/latest/miniconda.html
- open an anaconda prompt
- conda install -c cprestip tempcplot
### facultative but significant better if ipython is installed
- conda install ipython



## License

See the [License File](./LICENSE.md).

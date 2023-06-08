from setuptools import setup

# to create the wheel type:
# python3 setup.py sdist bdist_wheel

setup(name='TEMpcPlot',
      version='0.1',
      description='Tools for ED',
      url='https://prestipino.github.io/TEMpcPlot/',
      author='C. Prestipino',
      author_email='carmelo.prestipino@univ-rennes.fr',
      license='MIT',
      keywords=[
        'crystal', 'diffraction', 'crystallography'],
      classifiers=[
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: Apache Software License',
        'Development Status :: 3 - Alpha'],
      packages=['TEMpcPlot', 'TEMpcPlot.Symmetry', 'TEMpcPlot.tables',
                'TEMpcPlot.dm3_lib', 'TEMpcPlot.TEM', 'TEMpcPlot.Gui'],
      install_requires=['numpy', 'matplotlib', 'scipy', 'pillow', 'uncertainties',
                        'keyboard', 'mplcursors'],
      package_data={'TEMpcPlot.Symmetry': ['spacegroup.dat'],
                    'TEMpcPlot.TEM': ['angle.png',
                                      'down.png',
                                      'lenght.png',
                                      'PlotP.png',
                                      'RemP.png',
                                      'UP.png'],
                    'TEMpcPlot.Gui': ['angle.png',
                                      'down.png',
                                      'lenght.png',
                                      'PlotP.png',
                                      'RemP.png',
                                      'UP.png']},
      zip_safe=False)

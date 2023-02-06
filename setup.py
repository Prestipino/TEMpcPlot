from setuptools import setup

# to create the wheel type:
# python3 setup.py sdist bdist_wheel

setup(name='TEMpcPlot',
      version='0.1',
      description='Tools for ED',
      url='http://github.com/stopboring/funniest',
      author='Flying Circus',
      author_email='flyingcircus@example.com',
      license='MIT',
      packages=['TEMpcPlot', 'TEMpcPlot.Symmetry', 'TEMpcPlot.tables',
                'TEMpcPlot.dm3_lib', 'TEMpcPlot.TEM', 'TEMpcPlot.Gui'],
      install_requires=['numpy', 'matplotlib', 'scipy', 'pillow', 'uncertainties',
                        'keyboard', 'mplcursors'],
      package_data={'TEMpcPlot..Symmetry': ['spacegroup.dat'],
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
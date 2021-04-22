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
      packages=['TEMpcPlot', 'TEMpcPlot.GII', 'TEMpcPlot.tables',
                'TEMpcPlot.dm3_lib', 'TEMpcPlot.TEM'],
      install_requires=['numpy', 'matplotlib', 'scipy', 'pillow', 'uncertainties'],
      package_data={'TEMpcPlot.GII': ['libgcc_s_seh-1.dll',
                                      'libgfortran-3.dll',
                                      'libgmp-10.dll',
                                      'libgmpxx-4.dll',
                                      'libquadmath-0.dll',
                                      'libwinpthread-1.dll',
                                      'pydiffax.pyd',
                                      'pydiffax_36.pyd',
                                      'pypowder.pyd',
                                      'pypowder_36.pyd',
                                      'pyspg.pyd',
                                      'pyspg_36.pyd'],
                    'TEMpcPlot.TEM': ['angle.png',
                                      'down.png',
                                      'lenght.png',
                                      'PlotP.png',
                                      'RemP.png',
                                      'UP.png']},
      zip_safe=False)
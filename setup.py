from setuptools import setup

setup(name='tracer_method',
      version='1.0',
      description='Hydrological data analysis package',
      url='https://github.com/paulina-o/tracer_method',
      packages=['tracer_method', 'tracer_method.core', 'tracer_method.tests', 'tracer_method.core.config',
                'tracer_method.core.curve_fitter', 'tracer_method.core.read_data', 'tracer_method.core.tritium'],
      install_requires=[
          'matplotlib==3.2.1',
          'numpy==1.18.2',
          'pandas==1.0.3',
          'scipy==1.10.0',
          'xlrd==1.2.0',
      ],
      zip_safe=False)

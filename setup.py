from setuptools import setup

setup(name='tracer_method',
      version='1.0',
      description='Hydrological data analysis package',
      url='https://github.com/paulina-o/tracer_method',
      packages=['tracer_method', 'tracer_method.core', 'tracer_method.core.config', 'tracer_method.core.curve_fitter',
                'tracer_method.core.read_data', 'tracer_method.core.tritium'],
      zip_safe=False)

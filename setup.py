from setuptools import setup

setup(name='anomdetect',
      version='0.1',
      description='A library for calculating various anomaly detection and SPC charts.',
      url='https://github.com/amanda-park/anomdetect',
      author='Amanda Park & Phil Sattler',
      author_email='apark24@binghamton.edu',
      license='MIT',
      packages=['anomdetect'],
      install_requires=[
            'adtk',
            'pandas',
            'numpy',
            'statistics',
            'scikit-learn'
        ],
      zip_safe=False)

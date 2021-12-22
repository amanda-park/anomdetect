from setuptools import setup

setup(name='Anomaly_Detection',
      version='0.1',
      description='A library for calculating various anomaly detection and SPC charts.',
      url='https://bitbucket.spectrum-health.org:7991/stash/projects/QSE/repos/adtk/',
      author='Phil Sattler',
      author_email='philip.sattler@spectrumhealth.org',
      license='MIT',
      packages=['Anomaly_Detection'],
      install_requires=[
            'adtk',
            'pandas',
            'numpy',
            'statistics',
            'scikit-learn'
        ],
      zip_safe=False)

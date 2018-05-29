from setuptools import setup

setup(name='udaru_anomaly_detection',
      version='0.1',
      description='Detects anomalies in Udaru using Trail',
      url='https://github.com/nearform/udaru-anomaly-detection',
      author='nearForm',
      author_email='support@nearform.com',
      license='Private',
      packages=['udaru_anomaly_detection'],
      scripts=['bin/udaru-anomaly-detection'],
      install_requires=[
          'scipy',
          'numpy',
          'geoip2',
          'tqdm',
          'faker'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)

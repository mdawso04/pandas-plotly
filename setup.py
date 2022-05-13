from distutils.core import setup
setup(
  name = 'pandas-plotly',
  packages = ['pp'],
  version = '0.1',
  license='MIT',
  description = 'Pandas & plotly wrapper for data wrangling & vizualization',
  author = 'Michael Dawson',
  author_email = 'mdawso04@gmail.com',
  url = 'https://github.com/mdawso04/pandas-plotly',
  download_url = 'https://github.com/mdawso04/pandas-plotly/archive/refs/tags/v0.1.tar.gz',
  keywords = ['pandas', 'plotly', 'vizualization'],
  install_requires=[
          'pandas>=1.3.4',
          'plotly-express>=0.4.1',
      ],
  classifiers=[
    'Programming Language :: Python :: 3 :: Only',
  ],
)
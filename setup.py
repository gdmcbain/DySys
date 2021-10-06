from setuptools import setup, find_packages

setup(
    name='DySys',
    version='0.7.1',
    author='G. D. McBain',
    author_email='gdmcbain@pm.me',
    packages=find_packages(),
    scripts=[],
    url='https://github.com/gdmcbain/DySys',
    license='LICENSE.txt',
    description='''some code for dynamical systems,
particularly those of the descriptor type''',
    long_description=open('README.md').read(),
    python_requires='>=3.6',
    setup_requires=['pytest-runner', 'pytest-xdist'],
    tests_require=['pytest'],
    install_requires=['attrs',
                      'toolz',
                      'numpy',
                      'scipy',
                      'pandas',
    ]
)

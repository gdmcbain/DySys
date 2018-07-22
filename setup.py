from setuptools import setup

setup(
    name='DySys',
    version='0.4.0',
    author='G. D. McBain',
    author_email='gmcbain@memjet.com',
    packages=['dysys',
              'dysys.post'],
    scripts=[],
    url='https://gitlab.memjet.local/msm/DySys',
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
                      'scikit-sparse;platform_system!="Windows"',
    ]
)

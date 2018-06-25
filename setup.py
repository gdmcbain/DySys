from distutils.core import setup

setup(
    name='DySys',
    version='0.3.7',
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
    setup_requires=['numpy'],
    install_requires=['numpy',
                      'scipy',
                      'pandas',
                      'scikit-sparse;platform_system!="Windows"',
                      'toolz']
)

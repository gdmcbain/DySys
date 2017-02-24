from distutils.core import setup, Command

setup(
    name='DySys',
    version='0.2.1',
    author='G. D. McBain',
    author_email='gmcbain@memjet.com',
    packages=['dysys',
              'dysys.post'],
    scripts=[],
    url='http://wiki.memjet.local/display/memjet/DySys',
    license='LICENSE.txt',
    description='''some code for dynamical systems,
particularly those of the descriptor type''',
    long_description=open('README.txt').read(),
    install_requires=['numpy',
                      'scipy',
                      'toolz'],
)

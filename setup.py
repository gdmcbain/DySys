from distutils.core import setup

setup(
    name='DySys',
    version='0.3.3',
    author='G. D. McBain',
    author_email='gmcbain@memjet.com',
    packages=['dysys',
              'dysys.post'],
    scripts=[],
    url='https://gitlab.memjet.local/msm/DySys',
    license='LICENSE.txt',
    description='''some code for dynamical systems,
particularly those of the descriptor type''',
    long_description=open('README.rst').read(),
    install_requires=['scipy',
                      'toolz',
                      'scikit-sparse;platform_system!="Windows"',
                      'toolz'],
>>>>>>> e5c557e5fdf58f9ede79b1afad322230b8635c49
)

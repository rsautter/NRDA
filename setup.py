from setuptools import setup

setup(name="NRDA",
      version="1.0",
      author='Rubens Andreas Sautter',
      author_email='rubens.sautter@gmail.com',
      url='https://github.com/rsautter/NRDA',
      install_requires=['numpy','tqdm','scipy'],
      py_modules=['cNoise','ReactionNBurguers']
)

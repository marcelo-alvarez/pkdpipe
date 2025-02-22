from setuptools import setup
pname='pkdpipe'
setup(name=pname,
      version='0.1',
      description='pkdgrav3 analysis pipeline',
      url='http://github.com/marcelo-alvarez/pkdpipe',
      author='Marcelo Alvarez',
      license='MIT',
      packages=['pkdpipe'],
      package_data={
        pname: ["data/*"]
      },
      zip_safe=False)

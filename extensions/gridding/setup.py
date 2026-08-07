from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='gridding',
      version='2.1.0',
      ext_modules=[
          CUDAExtension('gridding', ['gridding_cuda.cpp', 'gridding.cu', 'gridding_reverse.cu']),
      ],
      cmdclass={'build_ext': BuildExtension})

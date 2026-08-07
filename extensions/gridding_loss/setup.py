from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='gridding_distance',
      version='1.0.0',
      ext_modules=[
          CUDAExtension('gridding_distance', ['gridding_distance_cuda.cpp', 'gridding_distance.cu']),
      ],
      cmdclass={'build_ext': BuildExtension})

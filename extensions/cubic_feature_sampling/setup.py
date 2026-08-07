from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='cubic_feature_sampling',
      version='1.1.0',
      ext_modules=[
          CUDAExtension('cubic_feature_sampling', ['cubic_feature_sampling_cuda.cpp', 'cubic_feature_sampling.cu']),
      ],
      cmdclass={'build_ext': BuildExtension})

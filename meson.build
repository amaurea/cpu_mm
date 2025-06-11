project(
	'gpu_mm',
	'cpp',
	version: '0.0',
	default_options: ['cpp_std=c++11'],
)

py = import('python').find_installation(pure: false)
pybind11 = dependency('pybind11')
omp      = dependency('openmp')

py.extension_module('compiled',
	'src/maps.cpp',
	install: true,
	dependencies : [pybind11, omp],
	cpp_args: ['-Wfatal-errors', '-fmax-errors=1'],
	subdir: 'gpu_mm',
)

py.install_sources(['python/__init__.py'], subdir:'gpu_mm')

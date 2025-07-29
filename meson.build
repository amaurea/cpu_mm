project(
	'cpu_mm',
	'cpp',
	version: '0.0',
	default_options: ['cpp_std=c++11'],
)

py = import('python').find_installation(pure: false)
pybind11 = dependency('pybind11')
omp      = dependency('openmp')
blas     = dependency('blas')

py.extension_module('compiled',
	'src/maps.cpp',
	install: true,
	dependencies : [pybind11, omp, blas],
	cpp_args: ['-Wfatal-errors', '-fmax-errors=1'],
	subdir: 'cpu_mm',
)

py.install_sources(['python/__init__.py'], subdir:'cpu_mm')

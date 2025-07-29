from . import compiled
import numpy as np

ncomp  = 3
tshape = (64,64)

sgemm = compiled.sgemm

class LocalPixelization(compiled.LocalPixelization):
	def __init__(self, nypix_global, nxpix_global, periodic_xcoord=True):
		nycells   = (nypix_global+tshape[0]-1)//tshape[0]
		nxcells   = (nxpix_global+tshape[1]-1)//tshape[1]
		cell_inds = np.full((nycells,nxcells),-1,dtype=np.int32)
		compiled.LocalPixelization.__init__(self, nypix_global, nxpix_global, cell_inds)
		self.update_cell_offsets()
	def update_cell_offsets(self):
		"""After cell_inds has been changed, cell_offsets_cpu must be updated
		to reflect those changes. Currently this must be done manually.
		This all of this is clunky..."""
		self.cell_offsets_cpu = self.cell_inds * (ncomp*tshape[0]*tshape[1])

class LocalMap:
	def __init__(self, pixelization, arr):
		self.pixelization = pixelization
		self.arr          = arr
	@property
	def dtype(self): return self.arr.dtype
	@property
	def nypix_global(self): return self.pixelization.nypix_global
	@property
	def nxpix_global(self): return self.pixelization.nxpix_global
	@property
	def periodic_xcoord(self): return self.pixelization.periodic_xcoord

# Basically just an alternative constructor for LocalMap, but
# kept as a separate type to be compatible with gpu_mm
class DynamicMap(LocalMap):
	def __init__(self, nypix_global, nxpix_global, dtype, periodic_xcoord=True):
		if not periodic_xcoord: raise NotImplementedError
		assert dtype in [np.float32, np.float64]
		# Initial empty pixelization
		self.pixelization = LocalPixelization(nypix_global, nxpix_global, periodic_xcoord=periodic_xcoord)
		# Data
		self.arr = np.zeros((0,ncomp,tshape[0],tshape[1]),dtype=dtype)
	def finalize(self):
		return LocalMap(self.pixelization, self.arr)

# I wish this could take a LocalPixelization as argument, but we're mimicing
# the gpu_mm interface, and it doesn't allow that. So we will construct our own
# LocalPixelization internally. Updating the main LocalPixelization will have to
# be done in map2tod.
class PointingPrePlan:
	def __init__(self, xpointing, nypix_global, nxpix_global, periodic_xcoord=True):
		self.pixelization = LocalPixelization(nypix_global, nxpix_global, periodic_xcoord=periodic_xcoord)
		self.plan = compiled.PointingPlan(xpointing, self.pixelization)

# For now, there's no difference between a plan and preplan,
# but we still do it this way for compatibility
def PointingPlan(preplan, xpointing): return preplan.plan


# TODO: I've decided that tod should always be 2d,
# map always 4d, when sogma uses them in the device
# interface. This will require some changes with how
# it's used with gpu_mm

_dtype_msg = "tod, map, xpointing (and response) must have the same dtype, which must be float32 or float64"

def tod2map(lmap, tod, xpointing, plan, response=None):
	if response is None:
		response = np.full((2,len(tod)),1,tod.dtype)
	assert tod.dtype == lmap.arr.dtype, _dtype_msg
	assert tod.dtype == xpointing.dtype, _dtype_msg
	assert tod.dtype == response.dtype, _dtype_msg
	assert tod.dtype in [np.float32, np.float64], _dtype_msg
	if isinstance(lmap, DynamicMap):
		expand_map_if_necessary(lmap, plan)
	#if tod.ndim == 1: tod = tod.reshape(xpointing.shape[1],-1)
	fun = cget("tod2map", tod.dtype)
	fun(lmap.arr, tod, xpointing, response, lmap.pixelization, plan)

def map2tod(tod, lmap, xpointing, plan=None, response=None):
	if response is None:
		response = np.full((2,len(tod)),1,tod.dtype)
	assert tod.dtype == lmap.arr.dtype, _dtype_msg
	assert tod.dtype == xpointing.dtype, _dtype_msg
	assert tod.dtype == response.dtype, _dtype_msg
	assert tod.dtype in [np.float32, np.float64], _dtype_msg
	fun = cget("map2tod", tod.dtype)
	fun(lmap.arr, tod, xpointing, response, lmap.pixelization)

def clear_ranges(tod, dets, starts, lens):
	fun = cget("clear_ranges", tod.dtype)
	fun(tod, dets, starts, lens)

def extract_ranges(tod, junk, offs, dets, starts, lens):
	fun = cget("extract_ranges", tod.dtype)
	fun(tod, junk.reshape(-1), offs, dets, starts, lens)

def insert_ranges(tod, junk, offs, dets, starts, lens):
	fun = cget("insert_ranges", tod.dtype)
	fun(tod, junk.reshape(-1), offs, dets, starts, lens)

def get_border_means(bvals, tod, index_map):
	fun = cget("get_border_means", tod.dtype)
	assert index_map.shape == (bvals.shape[0],5)
	assert bvals.shape[1] == 2
	fun(bvals, tod, index_map)

def deglitch(tod, bvals, index_map2):
	fun   = cget("deglitch", tod.dtype)
	jumps = bvals[:,1]-bvals[:,0]
	cumj  = np.cumsum(jumps)
	assert index_map2.shape == (bvals.shape[0],4)
	assert bvals.shape[1] == 2
	fun(tod, bvals, cumj, index_map2)

def fix_shape(arr, shape):
	assert arr.flags["C_CONTIGUOUS"]
	if arr.ndim == 1: return arr.reshape(shape)
	else: return arr

def cget(name, dtype):
	if   dtype == np.float32: suffix = "_f32"
	elif dtype == np.float64: suffix = "_f64"
	else: raise ValueError("Invalid dtype '%s'" % str(dtype))
	return getattr(compiled, name + suffix)

def expand_map_if_necessary(lmap, plan):
	"""Expand lmap's tiles to cover any previously unseen
	tiles encountered in plan. Plan uses global tile indexing,
	so """
	# lmap.pixelization has a set of active cells
	# plan has its own set of new active cells
	# Update lmap.pixelization and lmap.arr with any new cells
	pactive = plan.active
	cflat   = lmap.pixelization.cell_inds.reshape(-1)
	# global flat cell index of new cells
	new     = pactive[cflat[pactive]<0]
	if new.size == 0: return
	# Allocate new, bigger data array and copy over the old data
	ncell   = len(lmap.arr)+len(new)
	oarr    = np.zeros((ncell,)+lmap.arr.shape[1:], lmap.arr.dtype)
	oarr[:len(lmap.arr)] = lmap.arr
	# Assign new cell offsets
	cflat[new] = np.arange(len(lmap.arr), ncell)
	# And finally replace the array
	lmap.arr   = oarr
	# Yuck. Manually update cell_offsets_cpu. This is needed because
	# cpu_mm, sogma and gpu_mm don't agree on whether one should work
	# with general offsets or simpler tile indices. Offsets are more general,
	# but this generality isn't realized in practice because I assume
	# [ntile,ncomp,64,64] several other places
	lmap.pixelization.update_cell_offsets()

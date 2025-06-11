from . import compiled
import numpy as np

ncomp = 3
tsize = 64

class LocalPixelization(compiled.LocalPixelization):
	def __init__(self, nypix_global, nxpix_global, periodic_xcoord=True, tsize=tsize):
		nycells = (nypix_global+tsize-1)//tsize
		nxcells = (nxpix_global+tsize-1)//tsize
		cell_offsets = np.full((nycells,nxcells),-1,dtype=np.int32)
		compiled.LocalPixelization.__init__(self, nypix_global, nxpix_global, cell_offsets)

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
		self.arr = np.zeros((0,ncomp,tsize,tsize),dtype=dtype)
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

def tod2map(lmap, tod, xpointing, plan, response=None):
	if response is None:
		response = np.full((2,len(tod)),1,tod.dtype)
	if isinstance(lmap, DynamicMap):
		expand_map_if_necessary(lmap, plan)
	fun = cget("tod2map", tod.dtype)
	fun(lmap.arr, tod, xpointing, response, lmap.pixelization, plan)

def map2tod(tod, lmap, xpointing, plan=None, response=None):
	if response is None:
		response = np.full((2,len(tod)),1,tod.dtype)
	fun = cget("map2tod", tod.dtype)
	fun(lmap.arr, tod, xpointing, response, lmap.pixelization)

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
	cflat   = lmap.pixelization.cell_offsets.reshape(-1)
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

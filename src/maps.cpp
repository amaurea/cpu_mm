#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdint.h>
#include <cmath>
#include <vector>
namespace pb = pybind11;

// DynamicMap is python in gpu_mm, so can be that here too
// LocalPixelization is hybrid in gpu_mm. Why? Maybe just so it can be sent as argument

// Let's start with tod2map, and infer what we need from that
template<typename T> using carray      = pb::array_t<T,pb::array::c_style>;
template<typename T> using carray_cast = pb::array_t<T,pb::array::c_style | pb::array::forcecast>;

template<typename T> T clip(T a, int aend) {
	return a < 0 ? 0 : a < aend ? a : aend-1;
}
template<typename T> int wclip(T a, int aperiod) {
	if(a  < 0)       a += aperiod;
	if(a >= aperiod) a -= aperiod;
	return clip(a, aperiod);
}
template<typename T> struct Pixloc {
	Pixloc(T val, int n, bool wrap = false) {
		v     = wrap ? wclip(val, n) : clip(val, n);
		i0    = int(v);
		icell = i0>>6;
		i0   &= 0b111111;
		i1    = i0+1;
		d     = val-i0;
		if(wrap && i1 >= n) i1 = 0;
	}
	int i0, i1, icell;
	T v, d;
};

template<typename T> void add_tqu(T * map, int64_t polstride, int icell, int iy, int ix, T t, T q, T u, T weight) {
	bool in_cell = ((iy|ix)&0b111111111111) == 0;
	if(in_cell) {
		map  += icell<<12;  // Offset to our cell
		int s = (iy<<6) | ix; // index in cell
		map[s+0*polstride] += t*weight;
		map[s+1*polstride] += q*weight;
		map[s+2*polstride] += u*weight;
	}
}

template<typename T> T eval_tqu(const T * map, int64_t polstride, int icell, int iy, int ix, T tr, T qr, T ur, T weight) {
	T val = 0;
	bool in_cell = ((iy|ix)&0b111111111111) == 0;
	if(in_cell) {
		map  += icell<<12;  // Offset to our cell
		int s = (iy<<6) | ix; // index in cell
		val += map[s+0*polstride]*tr*weight;
		val += map[s+1*polstride]*qr*weight;
		val += map[s+2*polstride]*ur*weight;
	}
	return val;
}

struct LocalPixelization {
	LocalPixelization(int nypix_global, int nxpix_global, const carray_cast<int> & cell_offsets):
		nypix_global(nypix_global), nxpix_global(nxpix_global), cell_offsets(cell_offsets) {}
	int nypix_global;
	int nxpix_global;
	carray_cast<int> cell_offsets;
};

struct PointingRange {
	int64_t fsamp0;
	int32_t nsamp;
	int32_t det;
};

struct PointingCell {
	std::vector<PointingRange> ranges;
};

// This is mainly a c++ structure. It will be opaque on the python side
struct PointingPlan {
	PointingPlan(int64_t nsamp, int maxcells):nsamp(nsamp),cells(maxcells) {}
	template<typename T>
	PointingPlan(const carray<T> &xpointing, LocalPixelization &lp):cells(lp.cell_offsets.size()) {
		nsamp = xpointing.shape(2);
		build_pointing_plan(xpointing, lp, *this);
		finish();
	}
	void add_range(int global_cell, int det, int64_t samp0, int rangelen) {
		PointingRange r = { det*nsamp+samp0, rangelen, det };
		cells[global_cell].ranges.push_back(r);
	}
	void finish() {
		for(size_t i = 0; i < cells.size(); i++) {
			if(!cells[i].ranges.empty())
				active.push_back(i);
		}
	}
	const carray<int> get_active() {
		return pb::array_t<int>(active.size(), &active[0]);
	}
	int64_t nsamp;
	std::vector<PointingCell> cells;
	std::vector<int> active;
};

// Building a pointing plan.
template<typename T>
void build_pointing_plan(
	const carray<T> &xpointing,
	LocalPixelization &lp,
	PointingPlan & plan) {

	int     ndet  = xpointing.shape(1);
	int64_t nsamp = xpointing.shape(2);
	auto _pt      = xpointing.template unchecked<3>();
	auto _active  = lp.cell_offsets.mutable_unchecked<2>();

	for(int det = 0; det < ndet; det++) {
		int gprev = -1;
		int64_t psamp = 0;
		for(int64_t samp = 0; samp < nsamp; samp++) {
			// Cell lookup and wrapping
			T y = _pt(0,det,samp);
			T x = _pt(1,det,samp);
			Pixloc<T> py(y, lp.nypix_global);
			Pixloc<T> px(x, lp.nxpix_global, true);
			// global offset of this cell
			int gcell = py.icell*_active.shape(1)+px.icell;
			// Is this the end of a range?
			if(gcell != gprev) {
				if(gprev >= 0) plan.add_range(gprev, det, psamp, samp);
				gprev = gcell;
				psamp = samp;
			}
		}
		// Close any remaining range
		if(gprev >= 0) plan.add_range(gprev, det, psamp, nsamp);
	}
}

template<typename T>
void tod2map(
	carray<T> &map,             // (ncell,{T,Q,U},64,64)
	const carray<T> &tod,       // (ndet,nt)
	const carray<T> &xpointing, // ({y,x,alpha},ndet,nt)
	const carray<T> &response,  // ({T,P},ndet)
	const LocalPixelization &lp,
	const PointingPlan &plan) {

	// Like gpu_mm, we're assuming a contiguous block of memory.
	// This should be ensured by the carray type
	T * _map = map.mutable_data();
	auto _tod = tod.template unchecked<2>();
	auto _pt  = xpointing.template unchecked<3>();
	auto _resp= response.template unchecked<2>();
	auto _coffs = lp.cell_offsets.unchecked<2>();
	int64_t polstride = map.shape(2)*map.shape(3);

	// FIXME: We must distingush between two icells here:
	// 1. The index of the active cells for this plan
	// 2. The index into the total set of active cells for the map
	// These are currently confused
	#pragma omp parallel for
	for(size_t ai = 0; ai < plan.active.size(); ai++) {
		int gcell = plan.active[ai];
		const auto & cell = plan.cells[gcell];
		for(const auto & range : cell.ranges) {
			for(int64_t si = 0; si < range.nsamp; si++) {
				int64_t fsamp = range.fsamp0 + si;
				T val    = _tod(0,fsamp);
				T y      = _pt(0,0,fsamp);
				T x      = _pt(1,0,fsamp);
				T alpha  = _pt(2,0,fsamp);
				T t_resp = _resp(0,range.det);
				T p_resp = _resp(1,range.det);
				// Calculate the response
				T t = val*t_resp;
				T q = val*p_resp*std::cos(2*alpha);
				T u = val*p_resp*std::sin(2*alpha);
				// Cell lookup and wrapping
				Pixloc<T> py(y, lp.nypix_global);
				Pixloc<T> px(x, lp.nxpix_global, true);
				// Loop through our four pixels
				int icell = _coffs(py.icell, px.icell);
				if(icell < 0) continue; // raise error here?
				add_tqu(_map, polstride, icell, py.i0, px.i0, t, q, u, (1-py.d)*(1-px.d));
				add_tqu(_map, polstride, icell, py.i0, px.i1, t, q, u, (1-py.d)*px.d);
				add_tqu(_map, polstride, icell, py.i1, px.i0, t, q, u, py.d*(1-px.d));
				add_tqu(_map, polstride, icell, py.i1, px.i1, t, q, u, py.d*px.d);
			}
		}
	}
}

// Can map2tod use the same data structure? Want to loop over the
// thing we're writing to, so ideally want something that maps from
// samples to the relevant pixels. But we already have that, that's
// xpointing. So this will just be:
//  for each sample, find the cell info, read from cell, update tod
// How can I avoid a divide here? Maybe a fused omp loop does something
// smart.

template<typename T>
void map2tod(
	const carray<T> &map,       // (ncell,{T,Q,U},64,64)
	carray<T> &tod,             // (ndet,nt)
	const carray<T> &xpointing, // ({y,x,alpha},ndet,nt)
	const carray<T> &response,  // ({T,P},ndet)
	const LocalPixelization &lp) {

	// Like gpu_mm, we're assuming a contiguous block of memory.
	// This should be ensured by the carray type
	const T * _map = map.data();
	auto _tod = tod.template mutable_unchecked<2>();
	auto _pt  = xpointing.template unchecked<3>();
	auto _resp= response.template unchecked<2>();
	auto _coffs = lp.cell_offsets.unchecked<2>();
	int     ndet = tod.shape(0);
	int64_t nsamp= tod.shape(1);
	int64_t polstride = map.shape(2)*map.shape(3);

	#pragma omp parallel for collapse(2)
	for(int det = 0; det < ndet; det++) {
		for(int64_t samp = 0; samp < nsamp; samp++) {
			int64_t fsamp = det*nsamp+samp;
			// Get the pointing
			T y      = _pt(0,0,fsamp);
			T x      = _pt(1,0,fsamp);
			T alpha  = _pt(2,0,fsamp);
			// Get the response
			T t_resp = _resp(0,det);
			T p_resp = _resp(1,det);
			// Calculate the response
			T tr = t_resp;
			T qr = p_resp*cos(2*alpha);
			T ur = p_resp*sin(2*alpha);
			// Cell lookup and wrapping
			Pixloc<T> py(y, lp.nypix_global);
			Pixloc<T> px(x, lp.nxpix_global, true);
			int icell = _coffs(py.icell, px.icell);
			// Loop through our four pixels
			T val = 0;
			val += eval_tqu(_map, polstride, icell, py.i0, px.i0, tr, qr, ur, (1-py.d)*(1-px.d));
			val += eval_tqu(_map, polstride, icell, py.i0, px.i1, tr, qr, ur, (1-py.d)*px.d);
			val += eval_tqu(_map, polstride, icell, py.i1, px.i0, tr, qr, ur, py.d*(1-px.d));
			val += eval_tqu(_map, polstride, icell, py.i1, px.i1, tr, qr, ur, py.d*px.d);
			_tod(0,fsamp) += val;
		}
	}
}

// What do we need to make a sotodlib device?
// 1. LocalMap
//    Must contain .arr, a device array reshapable to (ntile,ncomp,tyshape,txshape)
//    Must contain .pixelization, a LocalPixelization
//    Must be constructable from (pixelization, arr)
//    Must be passable to map2tod, tod2map
//    (tyshape,txshape) must be (64,64)
// 2. LocalPixelization
//    Must contain .cell_offsets_cpu. These are offsets from the start of .arr
//    of each tile for a (3,64,64) tile.
// 3. DynamicMap
//    Must be passable to tod2map
//    Must be constructable from (shape, dtype)
//    Must contain .finalize(), which returns a LocalMap
// 4. map2tod
//    Must accept (LocalMap, tod, pointing, plan, [response])
//    pointing is [{y,x,psi},ndet,nsamp]
//    plan is PointingPlan
//    If implemented with sotodlib, we have a mismatch. sotodlib wants
//     (ncomp,tyshape,txshape,ntile), but we have (ntile,ncomp,tyshape,txshape).
//     Will moveaxis be enough, or must we make it contiguous too?
//    sotodlib expects to compute the pointing on the fly. Is there an
//     interface for passing in precomputed pointing? Yes, but it doesn't
//     support bilinear interpolation, which we use in sogma. It also doesn't
//     support response. Must either generalize it, or add a new CoordSys
//     that represents precomputed pointing. The latter is probably simplest.
//    PointingFit calculates the pointing using .dot(), which will hopefully
//     use threads, otherwise it will be very slow.
//    How does sotodlib handle sky wrapping?
// 5. tod2map
//    Must accept (tod, LocalMap/DyamicMap, pointing, plan, [response])
//    sotodlib needs thread_intervals, which must be precomputed using
//    _get_proj_threads. Store a whole P in DynamicMap/LocalMap/PointingPrePlan/PointingPlan?
// 6. PointingPlan
//    Must be constructable from (preplan, pointing)
// 7. PointingPrePlan
//    Must be constructable from (pointing, ny, nx, periodic_xcoord=True)
//    For sotodlib this and PointingPlan can be dummies
// Also these, I think
// * insert_ranges
// * extract_ranges
// * clear_ranges


PYBIND11_MODULE(compiled, m) {
	m.doc() = "cpu_mm low-level implementation";

	pb::class_<LocalPixelization>(m, "LocalPixelization")
		.def(pb::init<int,int,const carray_cast<int>&>())
		.def_readwrite("nypix_global", &LocalPixelization::nypix_global)
		.def_readwrite("nxpix_global", &LocalPixelization::nxpix_global)
		.def_readwrite("cell_offsets", &LocalPixelization::cell_offsets);

	// Har to expose the others currently, since they use c++ vectors etc
	pb::class_<PointingPlan>(m, "PointingPlan")
		.def(pb::init<int64_t, int>())
		.def(pb::init<const carray<float>&, LocalPixelization &>())
		.def(pb::init<const carray<double>&, LocalPixelization &>())
		.def_readwrite("nsamp", &PointingPlan::nsamp)
		.def_property_readonly("active", &PointingPlan::get_active);

	// Float and double versions of these
	m.def("tod2map_f32", &tod2map<float>, "Accumulate tod into map",
		pb::arg("map"), pb::arg("tod"), pb::arg("xpointing"),
		pb::arg("response"), pb::arg("lp"), pb::arg("plan"));
	m.def("tod2map_f64", &tod2map<double>, "Accumulate tod into map",
		pb::arg("map"), pb::arg("tod"), pb::arg("xpointing"),
		pb::arg("response"), pb::arg("lp"), pb::arg("plan"));

	m.def("map2tod_f32", &map2tod<float>, "Project map into tod",
		pb::arg("map"), pb::arg("tod"), pb::arg("xpointing"),
		pb::arg("response"), pb::arg("lp"));
	m.def("map2tod_f64", &map2tod<double>, "Project map into tod",
		pb::arg("map"), pb::arg("tod"), pb::arg("xpointing"),
		pb::arg("response"), pb::arg("lp"));
}

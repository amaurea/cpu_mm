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
template<typename T> T wclip(T a, int aperiod) {
	if(a  < 0)       a += aperiod;
	if(a >= aperiod) a -= aperiod;
	return clip(a, aperiod);
}
template<typename T> struct Pixloc {
	Pixloc(T val, int n, bool wrap = false) {
		v     = wrap ? wclip(val, n) : clip(val, n);
		i[0]  = int(v);
		d[1]  = v-i[0];
		d[0]  = 1-d[1];
		i[1]  = i[0]+1;
		if(wrap && i[1] >= n) i[1] = 0;
		// Get the tile
		icell[0] = i[0]>>6;
		icell[1] = i[1]>>6;
		// Make the pixel tile-relative
		i[0] &= 0b111111;
		i[1] &= 0b111111;
	}
	int i[2], icell[2];
	T v, d[2];
};

#include <stdio.h>
void moo(const carray<float> & arr) {
	float a = 0;
	const float* ptr = arr.data();
	for(int i = 0; i < arr.size(); i++)
		a += ptr[i]*ptr[i];
	a /= arr.size();
	a = sqrt(a);
	fprintf(stderr, "moomoo %15.7e\n", a);
}

template<typename T> void add_tqu(T * cell_data, int64_t polstride, int iy, int ix, T t, T q, T u, T weight) {
	bool in_cell = iy >= 0 && iy < 64 && ix >= 0 && ix < 64;
	if(in_cell) {
		int s = (iy<<6) | ix; // index in cell
		cell_data[s+0*polstride] += t*weight;
		cell_data[s+1*polstride] += q*weight;
		cell_data[s+2*polstride] += u*weight;
	}
}

template<typename T> T eval_tqu(const T * cell_data, int64_t polstride, int iy, int ix, T tr, T qr, T ur, T weight) {
	T val = 0;
	bool in_cell = iy >= 0 && iy < 64 && ix >= 0 && ix < 64;
	if(in_cell) {
		int s = (iy<<6) | ix; // index in cell
		val += cell_data[s+0*polstride]*tr*weight;
		val += cell_data[s+1*polstride]*qr*weight;
		val += cell_data[s+2*polstride]*ur*weight;
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
	int64_t samp0;
	int32_t nsamp;
	int32_t det;
};

struct PointingCell {
	int ycell, xcell;
	std::vector<PointingRange> ranges;
};

// This is mainly a c++ structure. It will be opaque on the python side
struct PointingPlan {
	PointingPlan(int64_t nsamp, int nycell, int nxcell):nycell(nycell),nxcell(nxcell),cells(nycell*nxcell) {
		init_cells();
	}
	template<typename T>
	PointingPlan(const carray<T> &xpointing, LocalPixelization &lp):cells(lp.cell_offsets.size()) {
		nycell = lp.cell_offsets.shape(0);
		nxcell = lp.cell_offsets.shape(1);
		init_cells();
		build_pointing_plan(xpointing, lp, *this);
	}
	void init_cells() {
		// Handy to have the yx coordinates of each cell.
		// Could calculate it on-the-fly too though
		for(int ycell = 0; ycell < nycell; ycell++)
		for(int xcell = 0; xcell < nxcell; xcell++) {
			int icell = ycell*nxcell+xcell;
			cells[icell].ycell = ycell;
			cells[icell].xcell = xcell;
			cells[icell].ranges.clear();
		}
	}
	void add_point(int global_cell, int det, int64_t samp) {
		auto & rs = cells[global_cell].ranges;
		if(rs.empty()) {
			rs.push_back(PointingRange{samp,1,det});
			active.push_back(global_cell);
		}
		else {
			PointingRange & r = rs[rs.size()-1];
			// Already handled
			if(r.det == det && samp < r.samp0 + r.nsamp)
				return;
			// Extend current range
			else if(r.det == det && samp == r.samp0 + r.nsamp)
				r.nsamp++;
			// New range
			else
				rs.push_back(PointingRange{samp,1,det});
		}
	}
	const carray<int> get_active() {
		return pb::array_t<int>(active.size(), &active[0]);
	}
	int nycell, nxcell;
	std::vector<PointingCell> cells;
	std::vector<int> active;
};

// Building a pointing plan. For each cell, we need all
// the samples that touch that cell in some way.
template<typename T>
void build_pointing_plan(
	const carray<T> &xpointing,
	LocalPixelization &lp,
	PointingPlan & plan) {

	int     ndet  = xpointing.shape(1);
	int64_t nsamp = xpointing.shape(2);
	auto    _pt   = xpointing.template unchecked<3>();
	int     nxcell= plan.nxcell;

	for(int det = 0; det < ndet; det++) {
		for(int64_t samp = 0; samp < nsamp; samp++) {
			// Cell lookup and wrapping
			T y = _pt(0,det,samp);
			T x = _pt(1,det,samp);
			Pixloc<T> py(y, lp.nypix_global);
			Pixloc<T> px(x, lp.nxpix_global, true);
			for(int jy = 0; jy < 2; jy++)
			for(int jx = 0; jx < 2; jx++)
				plan.add_point(py.icell[jy]*nxcell+px.icell[jx], det, samp);
		}
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
	auto _map = map.template mutable_unchecked();
	auto _tod = tod.template unchecked<2>();
	auto _pt  = xpointing.template unchecked<3>();
	auto _resp= response.template unchecked<2>();
	auto _coffs = lp.cell_offsets.unchecked<2>();
	int64_t polstride = map.shape(2)*map.shape(3);

	T tmp = 0;
	//#pragma omp parallel for
	for(size_t ai = 0; ai < plan.active.size(); ai++) {
		int gcell = plan.active[ai];
		const auto & cell = plan.cells[gcell];
		int icell = _coffs(cell.ycell, cell.xcell);
		T * cell_data = _map.mutable_data(icell,0,0,0);
		for(const auto & range : cell.ranges) {
			for(int si = 0; si < range.nsamp; si++) {
				int64_t samp = range.samp0 + si;
				T val    = _tod(range.det,samp);
				T y      = _pt(0,range.det,samp);
				T x      = _pt(1,range.det,samp);
				T alpha  = _pt(2,range.det,samp);
				T t_resp = _resp(0,range.det);
				T p_resp = _resp(1,range.det);
				// Calculate the response
				T t = val*t_resp;
				T q = val*p_resp*std::cos(2*alpha);
				T u = val*p_resp*std::sin(2*alpha);
				// Cell lookup and wrapping
				Pixloc<T> py(y, lp.nypix_global);
				Pixloc<T> px(x, lp.nxpix_global, true);

				fprintf(stderr, "A %2d %4d %5d %3d %3d %8.1e %8.1e %8.1e %8.5f %8.5f %8.5f %8.5f\n", ai, range.det, samp, py.i[0], px.i[0], t, q, u, py.d[1], px.d[1], y, x);
				bool in_cell = py.i[0] >= 0 && py.i[0] < 64 && px.i[0] >= 0 && px.i[0] < 64;
				if(in_cell) tmp += *map.data(icell,0,py.i[0],px.i[0]);
				fprintf(stderr, "B %2d %4d %5d %3d %3d\n", ai, range.det, samp, py.i[1], px.i[1]);
				in_cell = py.i[1] >= 0 && py.i[1] < 64 && px.i[1] >= 0 && px.i[1] < 64;
				if(in_cell) tmp += *map.data(icell,0,py.i[1],px.i[1]);

				for(int jy = 0; jy < 2; jy++)
				for(int jx = 0; jx < 2; jx++)
					// We should only process pixels that actually belong to this cell
					if(py.icell[jy]==cell.ycell && px.icell[jx]==cell.xcell)
						add_tqu(cell_data, polstride, py.i[jy], px.i[jx], t, q, u, py.d[jy]*px.d[jx]);
			}
		}
	}
	//fprintf(stderr, "printing this to avoid optimization: %15.7e\n", tmp);
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
	auto _map = map.template unchecked();
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
			// Loop through our four pixels
			T val = 0;
			for(int jy = 0; jy < 2; jy++)
			for(int jx = 0; jx < 2; jx++) {
				int icell = _coffs(py.icell[jy], px.icell[jx]);
				const T * cell_data = _map.data(icell,0,0,0);
				val += eval_tqu(cell_data, polstride, py.i[jy], px.i[jx], tr, qr, ur, py.d[jy]*px.d[jx]);
			}
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
		.def(pb::init<int64_t, int, int>())
		.def(pb::init<const carray<float>&, LocalPixelization &>())
		.def(pb::init<const carray<double>&, LocalPixelization &>())
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

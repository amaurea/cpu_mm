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

template<typename T> T clip(T a, int32_t aend) {
	return a < 0 ? 0 : a < aend ? a : aend-1;
}
template<typename T> T wclip(T a, int32_t aperiod) {
	if(a  < 0)       a += aperiod;
	if(a >= aperiod) a -= aperiod;
	return clip(a, aperiod);
}
template<typename T> struct Pixloc {
	Pixloc(T val, int32_t n, bool wrap = false) {
		v     = wrap ? wclip(val, n) : clip(val, n);
		i[0]  = int32_t(v);
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
	int32_t i[2], icell[2];
	T v, d[2];
};

template<typename T> void add_tqu(T * cell_data, int64_t polstride, int32_t iy, int32_t ix, T t, T q, T u, T weight) {
	bool in_cell = iy >= 0 && iy < 64 && ix >= 0 && ix < 64;
	if(in_cell) {
		int32_t s = (iy<<6) | ix; // index in cell
		cell_data[s+0*polstride] += t*weight;
		cell_data[s+1*polstride] += q*weight;
		cell_data[s+2*polstride] += u*weight;
	}
}

template<typename T> T eval_tqu(const T * cell_data, int64_t polstride, int32_t iy, int32_t ix, T tr, T qr, T ur, T weight) {
	T val = 0;
	bool in_cell = iy >= 0 && iy < 64 && ix >= 0 && ix < 64;
	if(in_cell) {
		int32_t s = (iy<<6) | ix; // index in cell
		val += cell_data[s+0*polstride]*tr*weight;
		val += cell_data[s+1*polstride]*qr*weight;
		val += cell_data[s+2*polstride]*ur*weight;
	}
	return val;
}

struct LocalPixelization {
	LocalPixelization(int32_t nypix_global, int32_t nxpix_global, const carray_cast<int32_t> & cell_inds):
		nypix_global(nypix_global), nxpix_global(nxpix_global), cell_inds(cell_inds) {}
	int32_t nypix_global;
	int32_t nxpix_global;
	carray_cast<int32_t> cell_inds;
};

struct PointingRange {
	int32_t samp0;
	int32_t nsamp;
	int32_t det;
};

struct PointingCell {
	int32_t ycell, xcell;
	std::vector<PointingRange> ranges;
};

// This is mainly a c++ structure. It will be opaque on the python side
struct PointingPlan {
	PointingPlan(int32_t nsamp, int32_t nycell, int32_t nxcell):nycell(nycell),nxcell(nxcell),size(0),cells(nycell*nxcell) {
		init_cells();
	}
	template<typename T>
	PointingPlan(const carray<T> &xpointing, LocalPixelization &lp):cells(lp.cell_inds.size()) {
		nycell = lp.cell_inds.shape(0);
		nxcell = lp.cell_inds.shape(1);
		size   = 0;
		init_cells();
		build_pointing_plan(xpointing, lp, *this);
	}
	void init_cells() {
		// Handy to have the yx coordinates of each cell.
		// Could calculate it on-the-fly too though
		for(int32_t ycell = 0; ycell < nycell; ycell++)
		for(int32_t xcell = 0; xcell < nxcell; xcell++) {
			int32_t icell = ycell*nxcell+xcell;
			cells[icell].ycell = ycell;
			cells[icell].xcell = xcell;
			cells[icell].ranges.clear();
		}
	}
	void add_point(int32_t global_cell, int32_t det, int32_t samp) {
		auto & rs = cells[global_cell].ranges;
		if(rs.empty()) {
			rs.push_back(PointingRange{samp,1,det});
			active.push_back(global_cell);
			size++;
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
			else {
				rs.push_back(PointingRange{samp,1,det});
				size++;
			}
		}
	}
	const carray<int32_t> get_active() {
		return pb::array_t<int32_t>(active.size(), &active[0]);
	}
	int32_t nycell, nxcell, size;
	std::vector<PointingCell> cells;
	std::vector<int32_t> active;
};

// Building a pointing plan. For each cell, we need all
// the samples that touch that cell in some way.
template<typename T>
void build_pointing_plan(
	const carray<T> &xpointing,
	LocalPixelization &lp,
	PointingPlan & plan) {

	int32_t     ndet  = xpointing.shape(1);
	int32_t nsamp = xpointing.shape(2);
	auto    _pt   = xpointing.template unchecked<3>();
	int32_t     nxcell= plan.nxcell;

	for(int32_t det = 0; det < ndet; det++) {
		for(int32_t samp = 0; samp < nsamp; samp++) {
			// Cell lookup and wrapping
			T y = _pt(0,det,samp);
			T x = _pt(1,det,samp);
			Pixloc<T> py(y, lp.nypix_global);
			Pixloc<T> px(x, lp.nxpix_global, true);
			for(int32_t jy = 0; jy < 2; jy++)
			for(int32_t jx = 0; jx < 2; jx++)
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
	auto _cinds = lp.cell_inds.unchecked<2>();
	int64_t polstride = map.shape(2)*map.shape(3);

	#pragma omp parallel for
	for(size_t ai = 0; ai < plan.active.size(); ai++) {
		int32_t gcell = plan.active[ai];
		const auto & cell = plan.cells[gcell];
		int32_t icell = _cinds(cell.ycell, cell.xcell);
		T * cell_data = _map.mutable_data(icell,0,0,0);
		for(const auto & range : cell.ranges) {
			for(int32_t si = 0; si < range.nsamp; si++) {
				int32_t samp = range.samp0 + si;
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
				// Accumulate into the 4 pixels touched by this sample
				for(int32_t jy = 0; jy < 2; jy++)
				for(int32_t jx = 0; jx < 2; jx++)
					// We should only process pixels that actually belong to this cell
					if(py.icell[jy]==cell.ycell && px.icell[jx]==cell.xcell)
						add_tqu(cell_data, polstride, py.i[jy], px.i[jx], t, q, u, py.d[jy]*px.d[jx]);
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
	auto _map = map.template unchecked();
	auto _tod = tod.template mutable_unchecked<2>();
	auto _pt  = xpointing.template unchecked<3>();
	auto _resp= response.template unchecked<2>();
	auto _cinds = lp.cell_inds.unchecked<2>();
	int32_t     ndet = tod.shape(0);
	int32_t nsamp= tod.shape(1);
	int64_t polstride = map.shape(2)*map.shape(3);

	#pragma omp parallel for collapse(2)
	for(int32_t det = 0; det < ndet; det++) {
		for(int32_t samp = 0; samp < nsamp; samp++) {
			// Get the pointing
			T y      = _pt(0,det,samp);
			T x      = _pt(1,det,samp);
			T alpha  = _pt(2,det,samp);
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
			for(int32_t jy = 0; jy < 2; jy++)
			for(int32_t jx = 0; jx < 2; jx++) {
				int32_t icell = _cinds(py.icell[jy], px.icell[jx]);
				const T * cell_data = _map.data(icell,0,0,0);
				val += eval_tqu(cell_data, polstride, py.i[jy], px.i[jx], tr, qr, ur, py.d[jy]*px.d[jx]);
			}
			_tod(det,samp) = val;
		}
	}
}

// Cuts
template<typename T>
void clear_ranges(carray<T> & tod, const carray<int32_t> & dets, const carray<int32_t> & starts, const carray<int32_t> & lens) {
	// Why is this so verbose?
	auto _tod    = tod.template mutable_unchecked<2>();
	auto _dets   = dets.unchecked<1>();
	auto _starts = starts.unchecked<1>();
	auto _lens   = lens.unchecked<1>();
	for(int i = 0; i < _dets.size(); i++) {
		int det = _dets[i], start = _starts[i], len = _lens[i];
		for(int si = 0; si < len; si++)
			_tod(det, start+si) = 0;
	}
}

template<typename T>
void extract_ranges(const carray<T> & tod, carray<T> & data, const carray<int32_t> & offs, const carray<int32_t> & dets, const carray<int32_t> & starts, const carray<int32_t> & lens) {
	auto _tod    = tod.template unchecked<2>();
	auto _data   = data.template mutable_unchecked<1>();
	auto _offs   = offs.unchecked<1>();
	auto _dets   = dets.unchecked<1>();
	auto _starts = starts.unchecked<1>();
	auto _lens   = lens.unchecked<1>();
	#pragma omp parallel for
	for(int i = 0; i < _dets.size(); i++) {
		int off = _offs[i], det = _dets[i], start = _starts[i], len = _lens[i];
		for(int si = 0; si < len; si++)
			_data[off+si] = _tod(det, start+si);
	}
}

template<typename T>
void insert_ranges(carray<T> & tod, const carray<T> & data, const carray<int32_t> & offs, const carray<int32_t> & dets, const carray<int32_t> & starts, const carray<int32_t> & lens) {
	auto _tod    = tod.template mutable_unchecked<2>();
	auto _data   = data.template unchecked<1>();
	auto _offs   = offs.unchecked<1>();
	auto _dets   = dets.unchecked<1>();
	auto _starts = starts.unchecked<1>();
	auto _lens   = lens.unchecked<1>();
	#pragma omp parallel for
	for(int i = 0; i < _dets.size(); i++) {
		int off = _offs[i], det = _dets[i], start = _starts[i], len = _lens[i];
		for(int si = 0; si < len; si++)
			_tod(det, start+si) = _data[off+si];
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
//    Must contain .cell_inds_cpu. These are offsets from the start of .arr
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
		.def(pb::init<int32_t,int32_t,const carray_cast<int32_t>&>())
		.def_readwrite("nypix_global", &LocalPixelization::nypix_global)
		.def_readwrite("nxpix_global", &LocalPixelization::nxpix_global)
		.def_readwrite("cell_inds",    &LocalPixelization::cell_inds);

	// Har to expose the others currently, since they use c++ vectors etc
	pb::class_<PointingPlan>(m, "PointingPlan")
		.def(pb::init<int32_t, int32_t, int32_t>())
		.def(pb::init<const carray<float>&, LocalPixelization &>())
		.def(pb::init<const carray<double>&, LocalPixelization &>())
		.def_property_readonly("active", &PointingPlan::get_active)
		.def_readonly("size", &PointingPlan::size);

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

	// Cuts
	m.def("clear_ranges_f32", &clear_ranges<float>, "Clear cuts",
		pb::arg("tod"), pb::arg("dets"), pb::arg("starts"), pb::arg("lens"));
	m.def("clear_ranges_f64", &clear_ranges<double>, "Clear cuts",
		pb::arg("tod"), pb::arg("dets"), pb::arg("starts"), pb::arg("lens"));
	m.def("extract_ranges_f32", &extract_ranges<float>, "Extract cuts",
		pb::arg("tod"), pb::arg("data"), pb::arg("offs"), pb::arg("dets"), pb::arg("starts"), pb::arg("lens"));
	m.def("extract_ranges_f64", &extract_ranges<double>, "Extract cuts",
		pb::arg("tod"), pb::arg("data"), pb::arg("offs"), pb::arg("dets"), pb::arg("starts"), pb::arg("lens"));
	m.def("insert_ranges_f32", &insert_ranges<float>, "Insert cuts",
		pb::arg("tod"), pb::arg("data"), pb::arg("offs"), pb::arg("dets"), pb::arg("starts"), pb::arg("lens"));
	m.def("insert_ranges_f64", &insert_ranges<double>, "Insert cuts",
		pb::arg("tod"), pb::arg("data"), pb::arg("offs"), pb::arg("dets"), pb::arg("starts"), pb::arg("lens"));

}

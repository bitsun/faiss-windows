/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved

#include "IndexFlat.h"

#include <cstring>
#include "utils.h"
#include "Heap.h"

#include "FaissAssert.h"

#include "AuxIndexStructures.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
namespace faiss {

IndexFlat::IndexFlat (int d, MetricType metric):
            Index(d, metric)
{
}



void IndexFlat::add (idx_t n, const float *x) {
    xb.insert(xb.end(), x, x + n * d);
    ntotal += n;
}


void IndexFlat::reset() {
    xb.clear();
    ntotal = 0;
}


void IndexFlat::search (idx_t n, const float *x, idx_t k,
                               float *distances, idx_t *labels) const
{
    // we see the distances and labels as heaps

    if (metric_type == METRIC_INNER_PRODUCT) {
        float_minheap_array_t res = {
            size_t(n), size_t(k), labels, distances};
        knn_inner_product (x, xb.data(), d, n, ntotal, &res);
    } else if (metric_type == METRIC_L2) {
        float_maxheap_array_t res = {
            size_t(n), size_t(k), labels, distances};
        knn_L2sqr (x, xb.data(), d, n, ntotal, &res);
    }
}

void IndexFlat::range_search (idx_t n, const float *x, float radius,
                              RangeSearchResult *result) const
{
    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            range_search_inner_product (x, xb.data(), d, n, ntotal,
                                        radius, result);
            break;
        case METRIC_L2:
            range_search_L2sqr (x, xb.data(), d, n, ntotal, radius, result);
            break;
    }
}


void IndexFlat::compute_distance_subset (
            idx_t n,
            const float *x,
            idx_t k,
            float *distances,
            const idx_t *labels) const
{
    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            fvec_inner_products_by_idx (
                 distances,
                 x, xb.data(), labels, d, n, k);
            break;
        case METRIC_L2:
            fvec_L2sqr_by_idx (
                 distances,
                 x, xb.data(), labels, d, n, k);
            break;
    }

}

int64_t IndexFlat::remove_ids (const IDSelector & sel)
{
    idx_t j = 0;
    for (idx_t i = 0; i < ntotal; i++) {
        if (sel.is_member (i)) {
            // should be removed
        } else {
            if (i > j) {
                memmove (&xb[d * j], &xb[d * i], sizeof(xb[0]) * d);
            }
            j++;
        }
    }
	int64_t nremove = ntotal - j;
    if (nremove > 0) {
        ntotal = j;
        xb.resize (ntotal * d);
    }
    return nremove;
}



void IndexFlat::reconstruct (idx_t key, float * recons) const
{
    memcpy (recons, &(xb[key * d]), sizeof(*recons) * d);
}

GPU_IndexFlatL2::GPU_IndexFlatL2(int d):IndexFlat(d,METRIC_L2) {
	cublasStatus_t stat;
	stat=cublasCreate(&handle);
}

GPU_IndexFlatL2::GPU_IndexFlatL2() {
	cublasStatus_t stat;
	stat=cublasCreate(&handle);
}

GPU_IndexFlatL2::~GPU_IndexFlatL2() {
	cublasDestroy(handle);
}

void GPU_IndexFlatL2::search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels)const {
	if (metric_type == METRIC_INNER_PRODUCT) {
        float_minheap_array_t res = {
            size_t(n), size_t(k), labels, distances};
        knn_inner_product (x, xb.data(), d, n, ntotal, &res);
    } else if (metric_type == METRIC_L2) {
        float_maxheap_array_t res = {
            size_t(n), size_t(k), labels, distances};
        knn_L2sqr_cublas (handle,x, xb.data(), d, n, ntotal, &res);
    }
}

/***************************************************
 * GPU_IndexFlatFP16L2
 ***************************************************/

GPU_IndexFlatFP16L2::GPU_IndexFlatFP16L2(int d):Index(d,METRIC_L2) {
	cublasStatus_t stat;
	stat=cublasCreate(&handle);
}

GPU_IndexFlatFP16L2::~GPU_IndexFlatFP16L2() {
	cublasDestroy(handle);
}

void GPU_IndexFlatFP16L2::add(idx_t n, const float* x) {
	//convert the vectors into half precision format
	std::vector<uint16_t,aligned_allocator<uint16_t>> pNewdbVectors(n*d);
	//uint16_t* pNewdbVectors = (uint16_t*)(_aligned_malloc(n*d * sizeof(uint16_t), 32));
	float2half(x,pNewdbVectors.data(),n*d);
	xb.insert(xb.end(), pNewdbVectors.begin(), pNewdbVectors.begin()+n*d);
	//compute xnorm
	std::vector<float> ynorms_new(n);
	fvec_norms_L2sqr (ynorms_new.data(), x, d, n);
	yL2norms.insert(yL2norms.end(),ynorms_new.begin(),ynorms_new.end());
    ntotal += n;
	//_aligned_free(pNewdbVectors);
}

void GPU_IndexFlatFP16L2::search(idx_t nx, const float* x, idx_t k, float* distances, idx_t* labels) const {
	float_maxheap_array_t res = {size_t(nx), size_t(k), labels, distances};
	res.heapify ();
	idx_t ny = xb.size()/d;
    // BLAS does not like empty matrices
    if (nx == 0 || ny == 0) return;

    /* block sizes */
    const size_t bs_x = 1024, bs_y = 1024*64;
    // const size_t bs_x = 16, bs_y = 16;
    std::vector<float> ip_block(bs_x * bs_y);
    std::vector<float> x_norms(nx);
    fvec_norms_L2sqr (x_norms.data(), x, d, nx);

    //std::vector<float> y_norms(ny);
    //fvec_norms_L2sqr (y_norms.data(), y, d, ny);
	//if we assume that gpu memory is not big enough, then we should not copy 
	//all the database vectors into gpu once for all,instead we copy them, blokc by block
	cudaError_t cudastat;
	float* d_x;  //the address to the gpu memory storing the current block of the database vectors
	float* d_y;  //the address to the gpu memory storing the current block of the query vectors
	float* d_ip_block;  //the address to the gpu memory storing the inner product result
	//allocate gpu memory for result
	cudastat = cudaMalloc((void**)&d_ip_block,bs_x*bs_y*sizeof(float));
	std::vector<float,aligned_allocator<float>> y(bs_y*d);
    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = i0 + bs_x;
        if(i1 > nx) i1 = nx;
		//copy the current block database vectors into gpu
		cudastat= cudaMalloc((void**)&d_x,(i1-i0)*d*sizeof(float));
		cudastat= cudaMemcpy(d_x,x + i0 * d,sizeof(float)*(i1-i0)*d,cudaMemcpyHostToDevice); 
        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = j0 + bs_y;
            if (j1 > ny) j1 = ny;
			half2float(xb.data()+j0*d,y.data(),(j1-j0)*d);
			cudastat= cudaMalloc((void**)&d_y,(j1-j0)*d*sizeof(float));
			cudastat= cudaMemcpy(d_y,y.data(),sizeof(float)*(j1-j0)*d,cudaMemcpyHostToDevice); 
			
            /* compute the actual dot products */
            {
                float one=1.0f,zero=0.0f;
                int nyi = j1 - j0, nxi = i1 - i0, di = d;
                cublasStatus_t status = cublasSgemm (handle,CUBLAS_OP_T, CUBLAS_OP_N, nyi, nxi, di, &one,
                        d_y, di,d_x, di, &zero,d_ip_block, nyi);
				if (status != CUBLAS_STATUS_SUCCESS) {
					throw std::exception("cublas operation failed");
				}
				//copy result back from gpu
				cudaMemcpy(ip_block.data(),d_ip_block,sizeof(float)*(j1-j0)*(i1-i0),cudaMemcpyDeviceToHost);
            }
			cudaFree(d_y);
            /* collect minima */
#pragma omp parallel for
            for (size_t i = i0; i < i1; i++) {
                float * __restrict simi = res.get_val(i);
				int64_t * __restrict idxi = res.get_ids (i);
                const float *ip_line = ip_block.data() + (i - i0) * (j1 - j0);

                for (size_t j = j0; j < j1; j++) {
                    float ip = *ip_line++;
                    float dis = x_norms[i] + yL2norms[j] - 2 * ip;

                    //dis = corr (dis, i, j);

                    if (dis < simi[0]) {
                        maxheap_pop (k, simi, idxi);
                        maxheap_push (k, simi, idxi, dis, j);
                    }
                }
            }
        }
		cudaFree(d_x);
    }
	cudaFree(d_ip_block);
	//cudaFree(one);
	//cudaFree(zero);
    res.reorder ();
}

void GPU_IndexFlatFP16L2::reset() {
	xb.clear();
	yL2norms.clear();
	ntotal=0;
}
/***************************************************
 * IndexFlatL2BaseShift
 ***************************************************/

IndexFlatL2BaseShift::IndexFlatL2BaseShift (int d, size_t nshift, const float *shift):
    IndexFlatL2 (d), shift (nshift)
{
    memcpy (this->shift.data(), shift, sizeof(float) * nshift);
}

void IndexFlatL2BaseShift::search (
            idx_t n,
            const float *x,
            idx_t k,
            float *distances,
            idx_t *labels) const
{
    FAISS_THROW_IF_NOT (shift.size() == ntotal);

    float_maxheap_array_t res = {
        size_t(n), size_t(k), labels, distances};
    knn_L2sqr_base_shift (x, xb.data(), d, n, ntotal, &res, shift.data());
}



/***************************************************
 * IndexRefineFlat
 ***************************************************/

IndexRefineFlat::IndexRefineFlat (Index *base_index):
    Index (base_index->d, base_index->metric_type),
    refine_index (base_index->d, base_index->metric_type),
    base_index (base_index), own_fields (false),
    k_factor (1)
{
    is_trained = base_index->is_trained;
    FAISS_THROW_IF_NOT_MSG (base_index->ntotal == 0,
                      "base_index should be empty in the beginning");
}

IndexRefineFlat::IndexRefineFlat () {
    base_index = nullptr;
    own_fields = false;
    k_factor = 1;
}


void IndexRefineFlat::train (idx_t n, const float *x)
{
    base_index->train (n, x);
    is_trained = true;
}

void IndexRefineFlat::add (idx_t n, const float *x) {
    FAISS_THROW_IF_NOT (is_trained);
    base_index->add (n, x);
    refine_index.add (n, x);
    ntotal = refine_index.ntotal;
}

void IndexRefineFlat::reset ()
{
    base_index->reset ();
    refine_index.reset ();
    ntotal = 0;
}

namespace {
typedef faiss::Index::idx_t idx_t;

template<class C>
static void reorder_2_heaps (
      idx_t n,
      idx_t k, idx_t *labels, float *distances,
      idx_t k_base, const idx_t *base_labels, const float *base_distances)
{
#pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {
        idx_t *idxo = labels + i * k;
        float *diso = distances + i * k;
        const idx_t *idxi = base_labels + i * k_base;
        const float *disi = base_distances + i * k_base;

        heap_heapify<C> (k, diso, idxo, disi, idxi, k);
        if (k_base != k) { // add remaining elements
            heap_addn<C> (k, diso, idxo, disi + k, idxi + k, k_base - k);
        }
        heap_reorder<C> (k, diso, idxo);
    }
}


}


void IndexRefineFlat::search (
              idx_t n, const float *x, idx_t k,
              float *distances, idx_t *labels) const
{
    FAISS_THROW_IF_NOT (is_trained);
    idx_t k_base = idx_t (k * k_factor);
    idx_t * base_labels = labels;
    float * base_distances = distances;
    ScopeDeleter<idx_t> del1;
    ScopeDeleter<float> del2;


    if (k != k_base) {
        base_labels = new idx_t [n * k_base];
        del1.set (base_labels);
        base_distances = new float [n * k_base];
        del2.set (base_distances);
    }

    base_index->search (n, x, k_base, base_distances, base_labels);

    for (int i = 0; i < n * k_base; i++)
        assert (base_labels[i] >= -1 &&
                base_labels[i] < ntotal);

    // compute refined distances
    refine_index.compute_distance_subset (
        n, x, k_base, base_distances, base_labels);

    // sort and store result
    if (metric_type == METRIC_L2) {
        typedef CMax <float, idx_t> C;
        reorder_2_heaps<C> (
            n, k, labels, distances,
            k_base, base_labels, base_distances);

    } else if (metric_type == METRIC_INNER_PRODUCT) {
        typedef CMin <float, idx_t> C;
        reorder_2_heaps<C> (
            n, k, labels, distances,
            k_base, base_labels, base_distances);
    }

}



IndexRefineFlat::~IndexRefineFlat ()
{
    if (own_fields) delete base_index;
}

/***************************************************
 * IndexFlat1D
 ***************************************************/


IndexFlat1D::IndexFlat1D (bool continuous_update):
    IndexFlatL2 (1),
    continuous_update (continuous_update)
{
}

/// if not continuous_update, call this between the last add and
/// the first search
void IndexFlat1D::update_permutation ()
{
    perm.resize (ntotal);
    if (ntotal < 1000000) {
        fvec_argsort (ntotal, xb.data(), (size_t*)perm.data());
    } else {
        fvec_argsort_parallel (ntotal, xb.data(), (size_t*)perm.data());
    }
}

void IndexFlat1D::add (idx_t n, const float *x)
{
    IndexFlatL2::add (n, x);
    if (continuous_update)
        update_permutation();
}

void IndexFlat1D::reset()
{
    IndexFlatL2::reset();
    perm.clear();
}

void IndexFlat1D::search (
            idx_t n,
            const float *x,
            idx_t k,
            float *distances,
            idx_t *labels) const
{
    FAISS_THROW_IF_NOT_MSG (perm.size() == ntotal,
                    "Call update_permutation before search");

#pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {

        float q = x[i]; // query
        float *D = distances + i * k;
        idx_t *I = labels + i * k;

        // binary search
        idx_t i0 = 0, i1 = ntotal;
        idx_t wp = 0;

        if (xb[perm[i0]] > q) {
            i1 = 0;
            goto finish_right;
        }

        if (xb[perm[i1 - 1]] <= q) {
            i0 = i1 - 1;
            goto finish_left;
        }

        while (i0 + 1 < i1) {
            idx_t imed = (i0 + i1) / 2;
            if (xb[perm[imed]] <= q) i0 = imed;
            else                    i1 = imed;
        }

        // query is between xb[perm[i0]] and xb[perm[i1]]
        // expand to nearest neighs

        while (wp < k) {
            float xleft = xb[perm[i0]];
            float xright = xb[perm[i1]];

            if (q - xleft < xright - q) {
                D[wp] = q - xleft;
                I[wp] = perm[i0];
                i0--; wp++;
                if (i0 < 0) { goto finish_right; }
            } else {
                D[wp] = xright - q;
                I[wp] = perm[i1];
                i1++; wp++;
                if (i1 >= ntotal) { goto finish_left; }
            }
        }
        goto done;

    finish_right:
        // grow to the right from i1
        while (wp < k) {
            if (i1 < ntotal) {
                D[wp] = xb[perm[i1]] - q;
                I[wp] = perm[i1];
                i1++;
            } else {
                D[wp] = std::numeric_limits<float>::infinity();//1.0 / 0.0;
                I[wp] = -1;
            }
            wp++;
        }
        goto done;

    finish_left:
        // grow to the left from i0
        while (wp < k) {
            if (i0 >= 0) {
                D[wp] = q - xb[perm[i0]];
                I[wp] = perm[i0];
                i0--;
            } else {
                D[wp] = std::numeric_limits<float>::infinity();//1.0 / 0.0;
                I[wp] = -1;
            }
            wp++;
        }
    done:  ;
    }

}



} // namespace faiss

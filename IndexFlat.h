/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved
// -*- c++ -*-

#ifndef INDEX_FLAT_H
#define INDEX_FLAT_H

#include <vector>

#include "Index.h"

struct cublasContext;
typedef cublasContext* cublasHandle_t;

namespace faiss {
template <typename T>
struct aligned_allocator {
  using value_type = T;

  aligned_allocator() = default;
  template <class U>
  aligned_allocator(const aligned_allocator<U>&) {}

  T* allocate(std::size_t n) {
    //std::cout << "allocate(" << n << ") = ";
    if (n <= std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      if (auto ptr = _aligned_malloc(n * sizeof(T),32)/*std::malloc(n * sizeof(T))*/) {
        return static_cast<T*>(ptr);
      }
    }
    throw std::bad_alloc();
  }
  void deallocate(T* ptr, std::size_t n) {
    //std::free(ptr);
	_aligned_free(ptr);
  }
};

template <typename T, typename U>
inline bool operator == (const aligned_allocator<T>&, const aligned_allocator<U>&) {
  return true;
}

template <typename T, typename U>
inline bool operator != (const aligned_allocator<T>& a, const aligned_allocator<U>& b) {
  return !(a == b);
}
/** Index that stores the full vectors and performs exhaustive search */
struct IndexFlat: Index {
    /// database vectors, size ntotal * d
    std::vector<float> xb;

    explicit IndexFlat (int d, MetricType metric = METRIC_INNER_PRODUCT);

    void add(idx_t n, const float* x) override;

    void reset() override;

    void search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const override;

    void range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result) const override;

    void reconstruct(idx_t key, float* recons) const override;

    /** compute distance with a subset of vectors
     *
     * @param x       query vectors, size n * d
     * @param labels  indices of the vectors that should be compared
     *                for each query vector, size n * k
     * @param distances
     *                corresponding output distances, size n * k
     */
    void compute_distance_subset (
            idx_t n,
            const float *x,
            idx_t k,
            float *distances,
            const idx_t *labels) const;

    /** remove some ids. NB that Because of the structure of the
     * indexing structre, the semantics of this operation are
     * different from the usual ones: the new ids are shifted */
	int64_t remove_ids(const IDSelector& sel) override;

    IndexFlat () {}
};



struct IndexFlatIP:IndexFlat {
    explicit IndexFlatIP (int d): IndexFlat (d, METRIC_INNER_PRODUCT) {}
    IndexFlatIP () {}
};

struct GPU_IndexFlatL2 :IndexFlat {
	cublasHandle_t handle;
	explicit GPU_IndexFlatL2 (int d);
    GPU_IndexFlatL2 ();
	~GPU_IndexFlatL2();
	void search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const override;
};

/**
flat index which stores the database vectors with fp16 half precision float point numbers,
and internally uses gpu fp16 capabilities to do the knn search
*/
struct GPU_IndexFlatFP16L2 : Index{
	cublasHandle_t handle;
	/**
	database vectors in fp half precision,in column major order
	*/
	std::vector<uint16_t,aligned_allocator<uint16_t>>  xb;
	/**
	L2 norms of database vectors
	*/
	std::vector<float> yL2norms;

	explicit GPU_IndexFlatFP16L2 (int d);
	virtual ~GPU_IndexFlatFP16L2();
	/**
	add data into the index. note that that the data to be added are still stored in a single precision array
	Therefore the caller of this function does not need to worry about the conversion efforts from
	single precision to half precision
	@param n the number of vectors to be added
	@param x the data point to the array of vectors to be added
	*/
    void add(idx_t n, const float* x) override;

	/**
	search  the given vectors against this index. note that the data be searched are still stored in a single precision array
	Therefore the caller of this function does not need to worry about the conversion efforts from
	single precision to half precision
	@param n the number of vectors to be searched
	@param x the data point to the array of vectors to be searched
	@param k number of nearest neighbor to be searched
	@param distances the 
	*/
    void search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const override;

	virtual void reset();
};

struct IndexFlatL2:IndexFlat {
    explicit IndexFlatL2 (int d): IndexFlat (d, METRIC_L2) {}
    IndexFlatL2 () {}
};


// same as an IndexFlatL2 but a value is subtracted from each distance
struct IndexFlatL2BaseShift: IndexFlatL2 {
    std::vector<float> shift;

    IndexFlatL2BaseShift (int d, size_t nshift, const float *shift);

    void search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const override;
};


/** Index that queries in a base_index (a fast one) and refines the
 *  results with an exact search, hopefully improving the results.
 */
struct IndexRefineFlat: Index {

    /// storage for full vectors
    IndexFlat refine_index;

    /// faster index to pre-select the vectors that should be filtered
    Index *base_index;
    bool own_fields;  ///< should the base index be deallocated?

    /// factor between k requested in search and the k requested from
    /// the base_index (should be >= 1)
    float k_factor;

    explicit IndexRefineFlat (Index *base_index);

    IndexRefineFlat ();

    void train(idx_t n, const float* x) override;

    void add(idx_t n, const float* x) override;

    void reset() override;

    void search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const override;

    ~IndexRefineFlat() override;
};


/// optimized version for 1D "vectors"
struct IndexFlat1D:IndexFlatL2 {
    bool continuous_update; ///< is the permutation updated continuously?

    std::vector<idx_t> perm; ///< sorted database indices

    explicit IndexFlat1D (bool continuous_update=true);

    /// if not continuous_update, call this between the last add and
    /// the first search
    void update_permutation ();

    void add(idx_t n, const float* x) override;

    void reset() override;

    /// Warn: the distances returned are L1 not L2
    void search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const override;
};


}

#endif

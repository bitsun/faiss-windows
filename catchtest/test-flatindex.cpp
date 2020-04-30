#include "catch.hpp"
#include "test-util.h"

#include <IndexFlat.h>
#include <utils.h>
/**
test the correctness of flat index using faiss flat index
*/
//BOOST_AUTO_TEST_CASE(FP32FlatIndexCPU) {
TEST_CASE("FP32FlatIndexCpu") {

	int64_t nb = 1024;
	int d = 64;
	std::vector <float> database(nb * d);
	generate_float_vector(d, database.data(), nb);
	decorate_float_vector(d, database.data(), nb);
	L2norm(d, database.data(), nb);

	std::unique_ptr<faiss::Index> index;
	index.reset(new faiss::IndexFlatL2(d));
	index->add(nb, database.data());

	int64_t nq = 128;
	int k = 4;
	std::vector<float> query(nq*d);
	generate_float_vector(d, query.data(), nq);
	decorate_float_vector(d, query.data(), nq);
	L2norm(d, query.data(), nq);

	std::vector<int64_t> I(k * nq);
	std::vector<float> D(k * nq);
	index->search(nq, query.data(), k, D.data(), I.data());

	std::vector<int64_t> I_GT(k * nq);
	std::vector<float> D_GT(k * nq);
	for (int n1 = 0; n1 < nq; ++n1) {
		std::vector<std::pair<float,int64_t>> pResult;
		for (int64_t n2 = 0; n2 < nb; ++n2) {
			float* qv = query.data() + n1*d;
			float* rv = database.data() + n2*d;
			double sum=0;
			for (int d1 = 0; d1 < d; ++d1,++qv,++rv) {
				float r=(*qv-*rv);
				sum += r*r;
			}
			pResult.push_back(std::make_pair((float)sum,n2));
		}
		std::partial_sort(pResult.begin(),pResult.begin()+k,pResult.end(),[](const std::pair<float,long>& p1,
			const std::pair<float,long>& p2)->bool{return p1.first<p2.first;});
		for (int k1 = 0; k1 < k; ++k1) {
			I_GT[n1*k+k1] = pResult[k1].second;
			D_GT[n1*k+k1] = pResult[k1].first;
		}
	}
	int nNumOfInterSec=0;
	for (int64_t n = 0; n < nq; ++n) {
		for (int k1 = 0; k1 < k; k1++) {
			//BOOST_CHECK(I_GT[n*k + k1] == I[n*k + k1]);
			REQUIRE(I_GT[n*k + k1] == I[n*k + k1]);
		}
	}
}

/*

BOOST_AUTO_TEST_CASE(FP32FlatIndexGPU) {
	int64_t nb = 1024;
	int d = 64;
	std::vector <float> database(nb * d);
	generate_float_vector(d, database.data(), nb);
	decorate_float_vector(d, database.data(), nb);
	L2norm(d, database.data(), nb);

	std::unique_ptr<faiss::Index> index;
	index.reset(new faiss::GPU_IndexFlatL2(d));
	index->add(nb, database.data());

	int64_t nq = 128;
	int k = 4;
	std::vector<float> query(nq*d);
	generate_float_vector(d, query.data(), nq);
	decorate_float_vector(d, query.data(), nq);
	L2norm(d, query.data(), nq);

	std::vector<int64_t> I(k * nq);
	std::vector<float> D(k * nq);
	index->search(nq, query.data(), k, D.data(), I.data());

	std::vector<int64_t> I_GT(k * nq);
	std::vector<float> D_GT(k * nq);
	for (int n1 = 0; n1 < nq; ++n1) {
		std::vector<std::pair<float, int64_t>> pResult;
		for (int64_t n2 = 0; n2 < nb; ++n2) {
			float* qv = query.data() + n1 * d;
			float* rv = database.data() + n2 * d;
			double sum = 0;
			for (int d1 = 0; d1 < d; ++d1, ++qv, ++rv) {
				float r = (*qv - *rv);
				sum += r * r;
			}
			pResult.push_back(std::make_pair((float)sum, n2));
		}
		std::partial_sort(pResult.begin(), pResult.begin() + k, pResult.end(), [](const std::pair<float, long>& p1,
			const std::pair<float, long>& p2)->bool {return p1.first<p2.first; });
		for (int k1 = 0; k1 < k; ++k1) {
			I_GT[n1*k + k1] = pResult[k1].second;
			D_GT[n1*k + k1] = pResult[k1].first;
		}
	}
	int nNumOfInterSec = 0;
	for (int64_t n = 0; n < nq; ++n) {
		for (int k1 = 0; k1 < k; k1++) {
			BOOST_CHECK(I_GT[n*k + k1] == I[n*k + k1]);
		}
	}
}

BOOST_AUTO_TEST_CASE(FP16FlatIndexGPU) {
	int64_t nb = 1024;
	int d = 64;
	std::vector <float> database(nb * d);
	generate_float_vector(d, database.data(), nb);
	decorate_float_vector(d, database.data(), nb);
	L2norm(d, database.data(), nb);

	std::unique_ptr<faiss::Index> index;
	index.reset(new faiss::GPU_IndexFlatFP16L2(d));
	index->add(nb, database.data());

	int64_t nq = 128;
	int k = 4;
	std::vector<float> query(nq*d);
	generate_float_vector(d, query.data(), nq);
	decorate_float_vector(d, query.data(), nq);
	L2norm(d, query.data(), nq);

	std::vector<int64_t> I(k * nq);
	std::vector<float> D(k * nq);
	index->search(nq, query.data(), k, D.data(), I.data());

	std::vector<int64_t> I_GT(k * nq);
	std::vector<float> D_GT(k * nq);
	for (int n1 = 0; n1 < nq; ++n1) {
		std::vector<std::pair<float, int64_t>> pResult;
		for (int64_t n2 = 0; n2 < nb; ++n2) {
			float* qv = query.data() + n1 * d;
			float* rv = database.data() + n2 * d;
			double sum = 0;
			for (int d1 = 0; d1 < d; ++d1, ++qv, ++rv) {
				float r = (*qv - *rv);
				sum += r * r;
			}
			pResult.push_back(std::make_pair((float)sum, n2));
		}
		std::partial_sort(pResult.begin(), pResult.begin() + k, pResult.end(), [](const std::pair<float, long>& p1,
			const std::pair<float, long>& p2)->bool {return p1.first<p2.first; });
		for (int k1 = 0; k1 < k; ++k1) {
			I_GT[n1*k + k1] = pResult[k1].second;
			D_GT[n1*k + k1] = pResult[k1].first;
		}
	}
	int nNumOfInterSec=0;
	for (int n = 0; n < nq; ++n) {
		//check intersection
		std::vector<int64_t> GT_Labels;
		std::vector<int64_t> Labels;
		for (int m = 0; m < k; ++m) {
			GT_Labels.push_back(I_GT[n * 4 + m]);
			Labels.push_back(I[n * 4 + m]);
		}
		std::sort(GT_Labels.begin(), GT_Labels.end());
		std::sort(Labels.begin(), Labels.end());
		std::vector<int> v(2 * k);
		auto itr = std::set_intersection(GT_Labels.begin(), GT_Labels.end(), Labels.begin(), Labels.end(), v.begin());
		v.resize(itr - v.begin());
		nNumOfInterSec += v.size();
	}
	BOOST_TEST_MESSAGE("GPU fp16 flat index search results is " << nNumOfInterSec*100.0f / (nq * k) << "% close to that of CPU fp32 flat index ");
}
BOOST_AUTO_TEST_SUITE_END()
*/
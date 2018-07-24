/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include "IndexFlat.h"
#include "utils.h"
#include <chrono>
#include <thread>
#include "utils.h"
#include <unordered_set>
#include <omp.h>
int main(int argc,char** argv) {
	if (argc != 4&&argc!=5) {
		std::cout<<"usage:windows-faiss nr[no. refs] nq[no. queries] index_type[0:cpu_index 1:gpu_fp32_idx 2:gpu_fp16_idx] [num_thread]"<<std::endl;
		std::cout<<"example: wndows-faiss 1000000 100 0"<<std::endl;
	}
	std::mt19937_64 rng;
	int nSeed = std::chrono::system_clock::now().time_since_epoch().count()/4;
	std::seed_seq ss{nSeed,nSeed+1,nSeed+2,nSeed+3};
    rng.seed(ss);
    // initialize a uniform distribution between 0 and 1
    std::uniform_real_distribution<float> unif(0, 1);
    int d = 512;                            // dimension
    int nb = atoi(argv[1]);//1024*1024;                       // database size
    int nq = atoi(argv[2]);//1024;                        // nb of queries
	std::cout<<"generating "<<nb<< " random vectors"<<nq<<" query vectors"<<std::endl;
    float *xb = new float[d * nb];
    float *xq = new float[d * nq];
	
	
    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < d; j++)
            xb[d * i + j] = unif(rng);
        xb[d * i] += (i%1000) / 1000.f;
		//L2 normalisation
		float sum=0.0;
		for (int j = 0; j < d; j++) {
			sum += xb[d * i + j]*xb[d * i + j];
		}
		sum = sqrt(sum);
		for (int j = 0; j < d; j++) {
			xb[d * i + j] = xb[d * i + j]/sum;
		}
    }

    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++)
            xq[d * i + j] = unif(rng);
        xq[d * i] += (i%1000) / 1000.f;
		//L2 normalisation
		float sum=0.0;
		for (int j = 0; j < d; j++) {
			sum += xq[d * i + j]*xq[d * i + j];
		}
		sum = sqrt(sum);
		for (int j = 0; j < d; j++) {
			xq[d * i + j] = xq[d * i + j]/sum;
		}
    }
	std::cout<<"vectors generated..start to test"<<std::endl;
	std::unique_ptr<faiss::Index> index;
	int nType=atoi(argv[3]);
	switch(nType){
		case 0:
			index.reset(new faiss::IndexFlatL2(d));
			break;
		case 1:
			index.reset(new faiss::GPU_IndexFlatL2(d));
			break;
		case 2:
			index.reset(new faiss::GPU_IndexFlatFP16L2(d));
			break;
		default:
			throw std::exception("please give a right index type");
	}
	index->add(nb,xb);
	std::this_thread::sleep_for(std::chrono::milliseconds(5000));
	
	faiss::Timer timer;
	timer.start();
	int k = 4;
	std::vector<long> I(k * nq);
    std::vector<float> D(k * nq);
	if(argc==5){
		int nNumOfThreads = atoi(argv[4]);
		std::cout<<"openmp setting "<<nNumOfThreads<<"threads"<<std::endl;
		omp_set_dynamic(0);     // Explicitly disable dynamic teams
		omp_set_num_threads(nNumOfThreads); // Use 4 threads for all consecutive parallel regions
	}
	index->search(nq,xq,k,D.data(),I.data());
	timer.stop();
	std::cout<<"end,elapsed millisec "<<timer.elapsedMilliseconds()<<std::endl;
	//std::vector<long> I_GT(k * nq);
 //   std::vector<float> D_GT(k * nq);
	//for (int n1 = 0; n1 < nq; ++n1) {
	//	std::vector<std::pair<float,long>> pResult;
	//	for (int n2 = 0; n2 < nb; ++n2) {
	//		float* qv = xq + n1*d;
	//		float* rv = xb + n2*d;
	//		double sum=0;
	//		for (int d1 = 0; d1 < d; ++d1,++qv,++rv) {
	//			float r=(*qv-*rv);
	//			sum += r*r;
	//		}
	//		pResult.push_back(std::make_pair((float)sum,n2));
	//	}
	//	std::partial_sort(pResult.begin(),pResult.begin()+k,pResult.end(),[](const std::pair<float,long>& p1,
	//		const std::pair<float,long>& p2)->bool{return p1.first<p2.first;});
	//	for (int k1 = 0; k1 < k; ++k1) {
	//		I_GT[n1*k+k1] = pResult[k1].second;
	//		D_GT[n1*k+k1] = pResult[k1].first;
	//	}
	//}
	//int nNumOfInterSec=0;
	//for (int n = 0; n < nq; ++n) {
	//	//check intersection
	//	std::vector<int> GT_Labels;
	//	std::vector<int> Labels;
	//	for (int m = 0; m < k; ++m) {
	//		GT_Labels.push_back(I_GT[n*4+m]);
	//		Labels.push_back(I[n*4+m]);
	//	}
	//	std::sort(GT_Labels.begin(),GT_Labels.end());
	//	std::sort(Labels.begin(),Labels.end());
	//	std::vector<int> v(2*k);
	//	auto itr=std::set_intersection(GT_Labels.begin(),GT_Labels.end(),Labels.begin(),Labels.end(),v.begin());
	//	v.resize(itr-v.begin());
	//	nNumOfInterSec += v.size();
	//	if (I_GT[n*4] != I[n*4]) {
	//		std::cout<<"error"<<std::endl;
	//	}
	//}
	//std::cout<<"number of NN "<<nq*k<<" intersection "<<nNumOfInterSec<<std::endl;
}


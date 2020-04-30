#pragma once
#include <stdint.h>
#ifdef  __cplusplus
extern "C" {
#endif 
	/*
	generate n random  d dimensional float vectors in the range of 0,1
	*/
	void generate_float_vector(int d,float* data,int64_t n,int nSeed=-1);

	/**
	decorate the first dimension of the given n d dimensional vectors
	*/
	void decorate_float_vector(int d, float* data, int64_t n);

	/**
	in place L2 normalisation of the given n d dimensional vectors
	*/
	void L2norm(int d, float* data, int64_t n);

#ifdef  __cplusplus
}
#endif 
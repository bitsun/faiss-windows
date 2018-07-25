#include "test-util.h"
#include <random>
#include <chrono>
void generate_float_vector(int d, float* data, int64_t nb, int nSeed) {
	std::mt19937_64 rng;
	if (nSeed == -1) {
		nSeed = std::chrono::system_clock::now().time_since_epoch().count() / 4;
	}
	std::seed_seq ss{ nSeed,nSeed + 1,nSeed + 2,nSeed + 3 };
	rng.seed(ss);
	// initialize a uniform distribution between 0 and 1
	std::uniform_real_distribution<float> unif(0, 1);
	
	for (int64_t i = 0; i < nb; i++) {
		for (int j = 0; j < d; j++)
			data[d * i + j] = unif(rng);
	}
}

void decorate_float_vector(int d, float* data, int64_t nb) {
	for (int64_t i = 0; i < nb; i++) {
		data[d * i] += (i % d) / (float)(d);
	}
}

void L2norm(int d, float* data, int64_t nb) {
	for (int64_t i = 0; i < nb; i++) {
		double sum = 0;
		for (int j = 0; j < d; j++) {
			float v = data[d * i + j];
			sum += v * v;
		}
		sum = sqrt(sum);
		if (sum > 0) {
			for (int j = 0; j < d; j++) {
				data[d * i + j] = data[d * i + j] / (float)sum;
			}
		}
	}
}
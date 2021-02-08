#pragma once
#include "RS.h"
class UniformRS : public RS
{
public:
	UniformRS(double **data, size_t dataset_size, size_t data_dim) :RS(data, dataset_size, data_dim) {};
	void init() {};
	std::vector<std::pair<size_t, double>> sample(size_t sample_size=1);
};


#pragma once
#include <vector>
#include <cstddef>
class RS
{
public:
	RS(double** data, size_t dataset_size, size_t data_dim);
	virtual void init() = 0;
	virtual std::vector<std::pair<size_t, double>> sample(size_t sample_size=1) = 0;
protected:
	double** data;
	size_t dataset_size;
	size_t data_dim;
};


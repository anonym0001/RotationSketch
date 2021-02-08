#pragma once
#include "RS.h"

#include "Eigen/Core"
#include "Eigen/Dense"

class SKARS : public RS
{
public:
	SKARS(double **data, size_t dataset_size, size_t data_dim) :RS(data, dataset_size, data_dim) {};
	void init() { init(1); };
	void init(size_t kernel_p, size_t initial_center = 0);
	std::vector<std::pair<size_t, double>> sample(size_t sample_size=1);
private:
	size_t initial_center;
	size_t kernel_p;
};


#include "SKARS.h"
#include <cmath>
#define PI 3.1415926535897932384626433832795

inline double pAngularDistance(double* x, double* y, size_t dim, size_t p=1)
{
	double ip = 0;
	double lx = 0;
	double ly = 0;
	for (size_t i = 0; i < dim; i++)
	{
		lx += x[i] * x[i];
		ly += y[i] * y[i];
		ip += x[i] * y[i];
	}
	double cossim = ip / sqrt(lx) / sqrt(ly);
	if (cossim > 1)
	{
		return 0;
	}
	else if (cossim < -1)
	{
		return 1;
	}
	else
	{
		return pow(acos(cossim) / PI, p);
	}
}

inline double pAngularKernel(double* x, double* y, size_t dim, size_t p = 1)
{
	double ip = 0;
	double lx = 0;
	double ly = 0;
	for (size_t i = 0; i < dim; i++)
	{
		lx += x[i] * x[i];
		ly += y[i] * y[i];
		ip += x[i] * y[i];
	}
	double cossim = ip / sqrt(lx) / sqrt(ly);
	if (cossim > 1)
	{
		return 1;
	}
	else if (cossim < -1)
	{
		return 0;
	}
	else
	{
		return pow(1 - acos(cossim) / PI, p);
	}
}

void SKARS::init(size_t kernel_p, size_t initial_center)
{
	this->initial_center = initial_center;
	this->kernel_p = kernel_p;
}

std::vector<std::pair<size_t, double>> SKARS::sample(size_t sample_size)
{

	std::vector<size_t> centers;

	centers.reserve(sample_size);
	centers.push_back(initial_center);

	for (size_t i = 1; i < sample_size; i++) {

		size_t max_index = 0;
		double max_distance = 0;

		for (size_t j = 0; j < dataset_size; j++) {
			double min_distance = std::numeric_limits<double>::max();
			for (const size_t &center : centers) {
				double dist = pAngularDistance(data[center], data[j], data_dim, kernel_p);
				if (dist < min_distance)
					min_distance = dist;
			}

			if (min_distance > max_distance) {
				max_distance = min_distance;
				max_index = j;
			}
		}
		centers.push_back(max_index);
	}

	double* kernel_matrix = new double[sample_size*sample_size];
	double* samples_kde = new double[sample_size]();
	for (size_t i = 0; i < sample_size; i++)
	{
		kernel_matrix[i*sample_size + i] = 1.0;
		samples_kde[i] += 1.0;
		for (size_t j = i + 1; j < sample_size; j++)
		{
			double kernel = pAngularKernel(data[centers[i]], data[centers[j]], data_dim, kernel_p);
			kernel_matrix[i*sample_size + j] = kernel;
			kernel_matrix[j*sample_size + i] = kernel;
			samples_kde[i] += kernel;
			samples_kde[j] += kernel;
		}
	}

	Eigen::MatrixXd K = Eigen::Map<Eigen::MatrixXd>(kernel_matrix, sample_size, sample_size);
	Eigen::VectorXd kappa = Eigen::Map<Eigen::VectorXd>(samples_kde, sample_size);

	Eigen::VectorXd w = K.colPivHouseholderQr().solve(kappa);

	std::vector<double> weights;
	weights.reserve(sample_size);

	std::vector<std::pair<size_t, double>> sample_IDs(sample_size);
	for (size_t i = 0; i < sample_size; i++)
	{
		sample_IDs[i].first = centers[i];
		sample_IDs[i].second = w(i) / sample_size;
	}
	delete[] kernel_matrix;
	delete[] samples_kde;
	return sample_IDs;
}

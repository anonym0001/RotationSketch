#include "UniformRS.h"
#include <random>

std::vector<std::pair<size_t, double>> UniformRS::sample(size_t sample_size)
{
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_int_distribution<size_t> uid(0, dataset_size - 1);

	std::vector<std::pair<size_t, double>> sample_IDs(sample_size);

	for (size_t i = 0; i < sample_size; i++)
	{
		size_t ID = uid(generator);
		sample_IDs[i].first = ID;
		sample_IDs[i].second = 1.0 / sample_size;
	}

	return sample_IDs;
}

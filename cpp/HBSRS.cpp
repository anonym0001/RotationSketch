#include "HBSRS.h"
#include <random>
#include "SimHash.h"
#include "SuperBitHash.h"
#include "ORHash.h"

void HBSRS::init(size_t table_num, HashType hash_type, size_t hash_dim_per_table)
{
	const size_t hash_dim = table_num * hash_dim_per_table;
	this->table_num = table_num;
	tables.clear();
	tables.resize(table_num);
	Hash *hf;
	switch (hash_type)
	{
	case htSimHash:
		hf = new SimHash(data_dim, hash_dim);
		break;
	case htSuperbitHash:
		hf = new SuperBitHash(data_dim, hash_dim);
		break;
	case htORHash:
		hf = new ORHash(data_dim, hash_dim);
		break;
	default:
		hf = new ORHash(data_dim, hash_dim);
		break;
	}
	for (size_t i = 0; i < dataset_size; i++)
	{
		int *hashes = hf->getHash(data[i]);
		int *inthash = bin2int(hashes, hash_dim, hash_dim_per_table);
		for (int j = 0; j < table_num; j++)
		{
			tables[j][inthash[j]].push_back(i);
		}
		delete[] hashes;
		delete[] inthash;
	}
	delete hf;
}

std::vector<std::pair<size_t, double>> HBSRS::sample(size_t sample_size)
{
	std::vector<bucket_info> table_info;
	table_info.reserve(1000);
	std::random_device rd;
	std::mt19937 generator(rd());

	std::vector<std::pair<size_t, double> > sample_IDs;
	sample_IDs.reserve(sample_size);

	for (size_t i = 0; i < table_num; ++i)
	{
		size_t n_samples = sample_size / table_num;

		if (i == table_num - 1)
			n_samples = sample_size - (sample_size / table_num)*i;

		table_info.clear();
		double gamma = compute_gamma(i);

		double sum_p = 0;
		for (auto &iterator : tables[i]) {
			size_t bucket_size = iterator.second.size();
			double sampling_probability = pow(bucket_size, gamma);
			sum_p += sampling_probability;

			double weight = pow(bucket_size, 1.0 - gamma);

			bucket_info info = { iterator.first, (int)(bucket_size), sampling_probability, weight };
			table_info.push_back(info);
		}

		std::vector<double> sampling_probabilities;

		for (auto &b : table_info)
		{
			b.bucket_w = b.bucket_w * sum_p / (dataset_size * sample_size);
			sampling_probabilities.push_back(b.bucket_p / sum_p);
		}

		std::discrete_distribution<> bucket_distribution = std::discrete_distribution<>
			(sampling_probabilities.begin(), sampling_probabilities.end());

		for (size_t j = 0; j < n_samples; ++j) {
			int idx = bucket_distribution(generator);
			double weight = table_info[idx].bucket_w;
			int key = table_info[idx].bucket_key;

			std::uniform_int_distribution<int> uniform(0, table_info[idx].bucket_size - 1);

			size_t ID = tables[i][key][uniform(generator)];
			sample_IDs.push_back(std::make_pair(ID, weight));
		}
	}
	return sample_IDs;
}

double HBSRS::compute_gamma(size_t table_number)
{
	size_t max_bucket = 0;
	size_t num_buckets = 0;

	for (auto &iterator : tables[table_number]) {
		if (iterator.second.size() > max_bucket)
			max_bucket = iterator.second.size();
		num_buckets++;
	}

	double eps = 0.5;
	double tau = 0.001;
	double cst = 0.8908987;  // (1/2)^(1/6)
	double eps_cst = eps * eps / 6;
	double nn = max_bucket / tau / dataset_size;
	double log_delta = 1 / eps_cst * 1.1;
	if (num_buckets <= cst / tau) {
		return 1.0 - log(eps_cst * log_delta) / log(nn);
	}
	else
		return 1.0;
}


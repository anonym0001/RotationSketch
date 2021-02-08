#pragma once
#include "RS.h"
#include <unordered_map>
#include "Hash.h"
class HBSRS : public RS
{
public:
	struct bucket_info {
		int bucket_key;
		int bucket_size;
		double bucket_p;
		double bucket_w;
	};
	HBSRS(double ** data, size_t dataset_size, size_t data_dim) :RS(data, dataset_size, data_dim) {};
	void init() { init(1, htORHash, 1); };
	void init(size_t table_num, HashType hash_type, size_t hash_dim_per_table);
	std::vector<std::pair<size_t, double>> sample(size_t sample_size=1);

private:
	double compute_gamma(size_t table_number);
	size_t table_num;
	std::vector< std::unordered_map<int, std::vector<size_t>>> tables;
};


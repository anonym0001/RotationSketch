#include "RS.h"

RS::RS(double ** data, size_t dataset_size, size_t data_dim)
{
	this->data = data;
	this->dataset_size = dataset_size;
	this->data_dim = data_dim;
}

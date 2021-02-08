#pragma once
#include "Hash.h"
class SimHash : public Hash
{
protected:
	double **_projMat;
public:
	SimHash(size_t dimention, size_t numOfHashes);
	int * getHash(double * vector, size_t bckbit=1);
	//int * getHashForTables(double * vector, int K, int tableid, int p_or_n);
	~SimHash();
};


#pragma once
#include "Hash.h"
class ORHash : public Hash
{
private:
	size_t _hashSize;
	bool **_flip;
	int **_rotMat;
public:
	ORHash(size_t dimention, size_t numOfHashes);
	int * getHash(double * vector, size_t bckbit=1);
	//int * getHashForTables(double * vector, int K, int tableid, int p_or_n);
	~ORHash();
};


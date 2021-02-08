#pragma once
#include "Hash.h"
class PWORHash : public Hash
{
private:
	size_t _hashSize;
	bool **_flip;
	size_t piece_num;
	size_t piece_len;
public:
	PWORHash(size_t dimention, size_t numOfHashes);
	int * getHash(double * vector, size_t bckbit = 1);
	~PWORHash();
};


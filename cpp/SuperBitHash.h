#pragma once
#include "SimHash.h"
class SuperBitHash : public SimHash
{
protected:
	size_t _N;
	size_t _L;
public:
	SuperBitHash(size_t dimention, size_t numOfHashes);
	~SuperBitHash();
};


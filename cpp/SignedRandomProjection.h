#include <vector>
#include "Hash.h"
#pragma once
using namespace std;

class SignedRandomProjection : public Hash
{
private:
	int _samSize;
	short ** _randBits;
	int ** _indices;
public:
	SignedRandomProjection(int dimention, int numOfHashes);
	int * getHash(double * vector, int bckbit=1);
	//int * getHashForTables(double * vector, int K, int tableid, int p_or_n);
	~SignedRandomProjection();
};

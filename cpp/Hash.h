#pragma once
#include <cstddef>

enum HashType
{
	htSimHash,
	htSuperbitHash,
	htORHash,
	htPWORHash
};

inline int* bin2int(int* binhash, size_t totaldim, size_t bucketbit)
{
	int* inthash = new int[totaldim / bucketbit]();
	for (size_t i = 0; i < totaldim; i++)
	{
		inthash[i / bucketbit] |= binhash[i] << (i%bucketbit);
	}
	return inthash;
}

class Hash
{
protected:
	size_t _dim;
	size_t _numhashes;
public:
	//virtual int * getHashForTables(double *vector, int K, int tableid, int p_or_n) = 0;
	virtual int * getHash(double * vector, size_t bckbit=1) = 0;
};


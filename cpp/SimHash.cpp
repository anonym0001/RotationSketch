#include "SimHash.h"
#include <ctime>
#include <algorithm>
#include <cmath>
#include <random>

using namespace std;

SimHash::SimHash(size_t dimention, size_t numOfHashes) {
	_dim = dimention;
	_numhashes = numOfHashes;

	random_device rd;
	default_random_engine e(rd());
	normal_distribution<double> n(0, 1);

	_projMat = new double*[_numhashes];
	for (int i = 0; i < _numhashes; i++)
	{
		_projMat[i] = new double[_dim];
		for (int j = 0; j < _dim; j++)
		{
			_projMat[i][j] = n(e);
		}
	}
}

int *SimHash::getHash(double *vector, size_t bckbit)
{
	int *hashes = new int[_numhashes/bckbit];

	int hashcnt = 0;
#pragma omp parallel for
	for (int i = 0; i < _numhashes; i++)
	{
		double ip = 0;
		for (int j = 0; j < _dim; j++)
		{
			ip += _projMat[i][j] * vector[j];
		}
		
		hashes[i] = (ip >= 0 ? 1 : 0);
	}
	return hashes;
}


//int *SimHash::getHashForTables(double *vector, int K, int tableid, int p_or_n) {
//	// length should be = to _dim
//	int *hashes = new int[K];
//	for (int i = tableid * K; i < (tableid + 1)*K; i++)
//	{
//		double ip = 0;
//		for (int j = 0; j < _dim; j++)
//		{
//			ip += _projMat[i][j] * vector[j];
//		}
//		hashes[i - tableid * K] = (ip >= 0 ? 0 : 1);
//	}
//	return hashes;
//}

SimHash::~SimHash() {
	for (int i = 0; i < _numhashes; i++)
	{
		delete[]   _projMat[i];
	}
	delete[]   _projMat;
}

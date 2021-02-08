#include "ORHash.h"
#include <ctime>
#include <algorithm>
#include <cmath>
#include <cstring>
#define DEFAULT_ROTATION_TIMES 3

#define STREAM_MODE true

using namespace std;

//ORHash::ORHash(int dimention, int numOfHashes) {
//	_dim = dimention;
//	_numhashes = numOfHashes;
//
//	_hashSize = 1 << (int)ceil(log2(max(_dim, _numhashes)));
//
//	// initialize sign flipping
//	srand(time(0));
//	_flip = new bool*[DEFAULT_ROTATION_TIMES];
//	for (int i = 0; i < DEFAULT_ROTATION_TIMES; i++)
//	{
//		_flip[i] = new bool[_hashSize];
//		for (int j = 0; j < _hashSize; j++)
//		{
//			_flip[i][j] = rand() & 1;
//		}
//	}
//
//	// initialize rotation matrix
//	_rotMat = new int*[_hashSize];
//	for (int i = 0; i < _hashSize; i++)
//	{
//		_rotMat[i] = new int[_hashSize]();
//		_rotMat[i][i] = 1;
//	}
//	for (int t = 0; t < DEFAULT_ROTATION_TIMES; t++)
//	{
//		// sign-flipping
//		for (int d = 0; d < _hashSize; d++)
//		{
//			if (_flip[t][d])
//			{
//				for (int n = 0; n < _hashSize; n++)
//				{
//					_rotMat[n][d] = -_rotMat[n][d];
//				}
//			}
//		}
//
//		// fast Walsh-Hadamard transform
//		int s = 1;
//		while (s < _hashSize)
//		{
//			for (int i = 0; i < _hashSize; i += (s << 1))
//			{
//				for (int j = i; j < i + s; j++)
//				{
//					for (int n = 0; n < _hashSize; n++)
//					{
//						int p = _rotMat[n][j];
//						int q = _rotMat[n][j + s];
//						_rotMat[n][j] = p + q;
//						_rotMat[n][j + s] = p - q;
//					}
//				}
//			}
//			s <<= 1;
//		}
//	}
//}

ORHash::ORHash(size_t dimention, size_t numOfHashes) {
	_dim = dimention;
	_numhashes = numOfHashes;

	_hashSize = (size_t)1 << (size_t)ceil(log2(max(_dim, _numhashes)));

	// initialize sign flipping
	srand((unsigned int)time(0));
	_flip = new bool*[DEFAULT_ROTATION_TIMES];
	for (size_t i = 0; i < DEFAULT_ROTATION_TIMES; i++)
	{
		_flip[i] = new bool[_hashSize];
		for (size_t j = 0; j < _hashSize; j++)
		{
			_flip[i][j] = rand() & 1;
		}
	}
}

int *ORHash::getHash(double *vector, size_t bckbit)
{
	int *hashes = new int[_numhashes/bckbit]();

#if STREAM_MODE
	double *vectmp = vector;
#else
	double *vectmp = new double[_hashSize]();
	memcpy(vectmp, vector, _dim * sizeof(double));
#endif
	
	// rotation
	for (size_t t = 0; t < DEFAULT_ROTATION_TIMES; t++)
	{
		// sign-flipping
		for (size_t d = 0; d < _hashSize; d++)
		{
			if (_flip[t][d])
			{
				vectmp[d] = -vectmp[d];
			}
		}

		// fast Walsh-Hadamard transform
		size_t s = 1;
		while (s < _hashSize)
		{
			for (size_t i = 0; i < _hashSize; i += (s << 1))
			{
//#pragma omp parallel for
				for (size_t j = i; j < i + s; j++)
				{
					double p = vectmp[j];
					double q = vectmp[j + s];
					vectmp[j] = p + q;
					vectmp[j + s] = p - q;
				}
			}
			s <<= 1;
		}
	}

	for (size_t i = 0; i < _numhashes; i++)
	{
		hashes[i] = (vectmp[i] > 0 ? 1 : 0);
	}
#if STREAM_MODE
#else
	delete[] vectmp;
#endif
	return hashes;
}


ORHash::~ORHash() {
	/*for (int i = 0; i < _hashSize; i++)
	{
		delete[]   _flip[i];
		delete[]   _rotMat[i];
	}
	delete[]   _flip;
	delete[]   _rotMat;*/
	for (int i = 0; i < DEFAULT_ROTATION_TIMES; i++)
	{
		delete[]   _flip[i];
	}
	delete[]   _flip;
}

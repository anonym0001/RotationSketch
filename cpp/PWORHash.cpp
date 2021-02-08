#include "PWORHash.h"
#include <ctime>
#include <algorithm>
#include <cmath>
#include <cstring>
#define DEFAULT_ROTATION_TIMES 3

#define STREAM_MODE true

using namespace std;

PWORHash::PWORHash(size_t dimention, size_t numOfHashes)
{
	_dim = dimention;
	_numhashes = numOfHashes;

	piece_len = (size_t)1 << (size_t)ceil(log2(_dim));
	if (_numhashes % piece_len == 0)
	{
		piece_num = _numhashes / piece_len;
	}
	else
	{
		piece_num = _numhashes / piece_len + 1;
	}
	_hashSize = piece_len * piece_num;

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

int *PWORHash::getHash(double *vector, size_t bckbit)
{
	int *hashes = new int[_numhashes / bckbit]();
#if STREAM_MODE
	double* vectmp = vector;
#else
	double* vectmp = new double[_hashSize]();
	for (size_t i = 0; i < piece_num; i++)
	{
		memcpy(vectmp + i * piece_len, vector, _dim * sizeof(double));
	}
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
		for (size_t piece_cnt = 0; piece_cnt < piece_num; piece_cnt++)
		{
			size_t s = 1;
			while (s < piece_len)
			{
				for (size_t i = piece_cnt * piece_len; i < (piece_cnt + 1)*piece_len; i += (s << 1))
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


PWORHash::~PWORHash() {
	for (int i = 0; i < DEFAULT_ROTATION_TIMES; i++)
	{
		delete[]   _flip[i];
	}
	delete[]   _flip;
}


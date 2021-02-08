#include "SuperBitHash.h"
#include <algorithm>
#include <cmath>

using namespace std;

SuperBitHash::SuperBitHash(size_t dimention, size_t numOfHashes) :SimHash::SimHash(dimention, numOfHashes)
{
	_N = numOfHashes;
	_L = (_numhashes - 1) / _N + 1;
	// Gram-Schmidt
	for (size_t i = 0; i < _L; i++)
	{
		size_t begin = i * _N;
		size_t end = min(_dim, begin + _N);
		for (size_t p = begin; p < end; p++)
		{
			for (size_t q = begin; q < p; q++)
			{
				// inner product of vector p and q
				double ip = 0;
				for (size_t d = 0; d < _dim; d++)
				{
					ip += _projMat[p][d] * _projMat[q][d];
				}
				// norm of vector q is 1, so unnecessary to divide norm of vector q here
				// orthogonalize
				for (size_t d = 0; d < _dim; d++)
				{
					_projMat[p][d] -= ip * _projMat[q][d];
				}
			}
			// normalize
			double norm = 0;
			for (size_t d = 0; d < _dim; d++)
			{
				norm += _projMat[p][d] * _projMat[p][d];
			}
			norm = sqrt(norm);
			for (size_t d = 0; d < _dim; d++)
			{
				_projMat[p][d] /= norm;
			}
		}
	}
}

SuperBitHash::~SuperBitHash()
{
	//for (size_t i = 0; i < _numhashes; i++)
	//{
	//	delete[]   _projMat[i];
	//}
	//delete[]   _projMat;
}

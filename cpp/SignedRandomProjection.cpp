#include "SignedRandomProjection.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <ctime>

SignedRandomProjection::SignedRandomProjection(int dimention, int numOfHashes) {
	_dim = dimention;
	_numhashes = numOfHashes;
	_samSize = (int)ceil(_dim / 3);

	int *a = new int[_dim];
	for (size_t i = 0; i < _dim; i++) {
		a[i] = (int)i;
	}

	srand((unsigned int)time(0));
	_randBits = new short *[_numhashes];
	_indices = new int *[_numhashes];

	for (size_t i = 0; i < _numhashes; i++) {
		random_shuffle(&a[0], &a[_dim - 1]);
		_randBits[i] = new short[_samSize];
		_indices[i] = new int[_samSize];
		for (size_t j = 0; j < _samSize; j++) {
			_indices[i][j] = a[j];
			int curr = rand();
			if (curr % 2 == 0) {
				_randBits[i][j] = 1;
			}
			else {
				_randBits[i][j] = -1;
			}
		}
	}
}

int *SignedRandomProjection::getHash(double *vector, int bckbit) {
    int *hashes = new int[_numhashes];

    for (int i = 0; i < _numhashes; i++) {
        double s = 0;
        for (size_t j = 0; j < _samSize; j++) {
            double v = vector[_indices[i][j]];
            if (_randBits[i][j] >= 0) {
                s += v;
            } else {
                s -= v;
            }
        }
        hashes[i] = (s >= 0 ? 1 : 0);
    }
    return hashes;
}


SignedRandomProjection::~SignedRandomProjection() {
    for (size_t i = 0; i < _numhashes; i++) {
        delete[]   _randBits[i];
        delete[]   _indices[i];
    }
    delete[]   _randBits;
    delete[]   _indices;
}

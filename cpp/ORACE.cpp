#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <cctype>
#include <random>
#include "RACE.h"
#include "Hash.h"
#include "SignedRandomProjection.h"
#include "SimHash.h"
#include "SuperBitHash.h"
#include "ORHash.h"
#include "PWORHash.h"
#include <chrono>
#include "RS.h"
#include "UniformRS.h"
#include "SKARS.h"
#include "HBSRS.h"

#define PI 3.1415926535897932384626433832795
#define ACC_RESULT_FILE "./accres_3.txt"
#define TIME_RESULT_FILE "./timeres_4.txt"
#define SKETCH_RATIO 0.8
// max memory
#define MAX_MEMORY 1l<<30

struct DataSetInfo
{
	string name;
	size_t num;
	size_t dim;
};


inline double AngularKernel(double* x, double* y, size_t dim)
{
	double ip = 0;
	double lx = 0;
	double ly = 0;
	for (size_t i = 0; i < dim; i++)
	{
		lx += x[i] * x[i];
		ly += y[i] * y[i];
		ip += x[i] * y[i];
	}
	double cossim = ip / sqrt(lx) / sqrt(ly);
	if (cossim > 1)
	{
		return 1;
	}
	else if (cossim < -1)
	{
		return 0;
	}
	else
	{
		return 1 - acos(cossim) / PI;
	}
}

inline double* LoadData(std::string filename, size_t dim)
{
	fstream fin(filename, std::ios::in);

	double *dat = new double[dim];

	for (size_t i = 0; i < dim; i++)
	{
		double tmp;
		fin >> tmp;
		dat[i] = tmp;
	}
	fin.close();
	return dat;
}

inline double** LoadData(std::string filename, size_t num, size_t dim)
{
	fstream fin(filename, std::ios::in);

	double **dat = new double*[num];

	for (size_t i = 0; i < num; i++)
	{
		dat[i] = new double[dim];
		for (size_t j = 0; j < dim; j++)
		{
			double tmp;
			fin >> tmp;
			dat[i][j] = tmp;
		}
	}
	fin.close();
	return dat;
}

double AccTest(std::string inputfile, double **data, double *tkd, size_t num, size_t dim, size_t hashtablenum, size_t hashlen, HashType hashtype)
{
	size_t sketchnum = (size_t)floor(SKETCH_RATIO*num);
	size_t testnum = num - sketchnum;
	size_t total_hash_len = hashtablenum * hashlen;

	Hash *hf;
	switch (hashtype)
	{
	case htSimHash:
		hf = new SimHash(dim, total_hash_len);
		break;
	case htSuperbitHash:
		hf = new SuperBitHash(dim, total_hash_len);
		break;
	case htORHash:
		hf = new ORHash(dim, total_hash_len);
		break;
	case htPWORHash:
		hf = new PWORHash(dim, total_hash_len);
		break;
	default:
		hf = new ORHash(dim, total_hash_len);
		break;
	}

	RACE *race = new RACE(hashtablenum, 1 << hashlen);

	for (size_t i = testnum; i < num; i++)
	{
		int *hashes = hf->getHash(data[i]);
		int *inthash = bin2int(hashes, total_hash_len, hashlen);
		race->add(inthash);
		delete[] hashes;
		delete[] inthash;
	}

	double rmse = 0;
	double mre = 0;
	double ekd;
	double err;
	for (size_t i = 0; i < testnum; i++)
	{
		int *hashes = hf->getHash(data[i]);
		int *inthash = bin2int(hashes, total_hash_len, hashlen);
		ekd = (double)race->query(inthash) / sketchnum;
		err = ekd - tkd[i];
		rmse += err * err;
		mre += abs(err) / tkd[i];
		delete[] hashes;
		delete[] inthash;
	}
	rmse /= testnum;
	rmse = sqrt(rmse);
	mre /= testnum;
	delete hf;
	delete race;

	fstream fout(ACC_RESULT_FILE, std::ios::out | std::ios::app);
	// Dataset, HashType, TableNum, HashLength, Memory(KByte), RMSE
	fout << inputfile << "\t" << hashtype << "\t" << hashtablenum << "\t" << hashlen << "\t" << (double)hashtablenum*(1 << hashlen)*4.0 / 1024 << "\t" << rmse << "\t" << mre << "\n";
	fout.close();
	std::cout << inputfile << "\t" << hashtype << "\t" << hashtablenum << "\t" << hashlen << "\t" << (double)hashtablenum*(1 << hashlen)*4.0 / 1024 << "\t" << rmse << "\t" << mre << "\n";
	return rmse;
}

double TimeTest(std::string inputfile, double **data, size_t num, size_t dim, size_t hashtablenum, size_t hashlen, HashType hashtype)
{
	size_t sketchnum = (size_t)floor(SKETCH_RATIO*num);
	size_t testnum = num - sketchnum;
	size_t total_hash_len = hashtablenum * hashlen;

	Hash *hf;
	auto hftime0 = std::chrono::steady_clock::now();
	switch (hashtype)
	{
	case htORHash:
		hf = new ORHash(dim, total_hash_len);
		break;
	case htPWORHash:
		hf = new PWORHash(dim, total_hash_len);
		break;
	case htSimHash:
		hf = new SimHash(dim, total_hash_len);
		break;
	case htSuperbitHash:
		hf = new SuperBitHash(dim, total_hash_len);
		break;
	default:
		hf = new ORHash(dim, total_hash_len);
		break;
	}
	auto hftime1 = std::chrono::steady_clock::now();

	RACE *race = new RACE(hashtablenum, 1 << hashlen);

	auto stime0 = std::chrono::steady_clock::now();
	for (size_t i = testnum; i < num; i++)
	{
		int *hashes = hf->getHash(data[i]);
		int *inthash = bin2int(hashes, total_hash_len, hashlen);
		race->add(inthash);
		delete[] hashes;
		delete[] inthash;
	}
	auto stime1 = std::chrono::steady_clock::now();

	int **hashes = new int*[testnum];
	auto qtime0 = std::chrono::steady_clock::now();
	for (size_t i = 0; i < testnum; i++)
	{
		hashes[i] = hf->getHash(data[i]);
	}
	auto qtime1 = std::chrono::steady_clock::now();
	double ekd;
	for (size_t i = 0; i < testnum; i++)
	{
		int *inthash = bin2int(hashes[i], total_hash_len, hashlen);
		ekd = (double)race->query(inthash) / sketchnum;
		delete[] hashes[i];
		delete[] inthash;
	}
	auto qtime2 = std::chrono::steady_clock::now();

	delete hf;
	delete race;

	auto hft = std::chrono::duration_cast<std::chrono::nanoseconds>(hftime1 - hftime0);
	auto st = std::chrono::duration_cast<std::chrono::nanoseconds>(stime1 - stime0);
	auto qt1 = std::chrono::duration_cast<std::chrono::nanoseconds>(qtime1 - qtime0);
	auto qt2 = std::chrono::duration_cast<std::chrono::nanoseconds>(qtime2 - qtime1);
	auto qt = std::chrono::duration_cast<std::chrono::nanoseconds>(qtime2 - qtime0);

	fstream fout(TIME_RESULT_FILE, std::ios::out | std::ios::app);
	// Dataset, HashType, TableNum, HashLength, Memory(KByte), SketchNum, QueryNum, HashFuctionSetupTime(us), SketchTime(ms), QueryTime(ms), QueryHashTime(us/sample), QueryLookupTime(us/sample)
	fout << inputfile << "\t" << hashtype << "\t" << hashtablenum << "\t" << hashlen << "\t" << (double)hashtablenum*(1 << hashlen)*4.0 / 1024 << "\t" << sketchnum << "\t" << testnum << "\t" << hft.count()*1e-3 << "\t" << st.count()*1e-6 << "\t"<< st.count()*1e-3 / sketchnum << "\t" << qt.count()*1e-6 << "\t" << qt1.count()*1e-3/testnum << "\t" << qt2.count()*1e-3/testnum << "\n";
	fout.close();
	std::cout << inputfile << "\t" << hashtype << "\t" << hashtablenum << "\t" << hashlen << "\t" << (double)hashtablenum*(1 << hashlen)*4.0 / 1024 << "\t" << sketchnum << "\t" << testnum << "\t" << hft.count()*1e-3 << "\t" << st.count()*1e-6 << "\t" << st.count()*1e-3 / sketchnum << "\t" << qt.count()*1e-6 << "\t" << qt1.count()*1e-3/testnum << "\t" << qt2.count()*1e-3/testnum << "\n";
	return ekd;
}

inline pair<double, double> KDCalculate(double** data, double** queries, double* tkd, size_t num, size_t dim, size_t p, const std::vector<std::pair<size_t, double>> &samples)
{
	double *ekd = new double[num];
#pragma omp parallel for
	for (int i = 0; i < num; i++)
	{
		double kd = 0;
		for (auto &s : samples)
		{
			double ak = AngularKernel(queries[i], data[s.first], dim);
			double akp = pow(ak, p);
			kd += s.second*akp;
		}
		ekd[i] = kd;
	}
	double rmse = 0;
	double mre = 0;
	for (size_t i = 0; i < num; i++)
	{
		double err = abs(ekd[i] - tkd[i]);
		rmse += err * err;
		mre += err / tkd[i];
	}
	rmse /= num;
	rmse = sqrt(rmse);
	mre /= num;
	return std::make_pair(rmse, mre);
}

inline void RSOutput(string dataset, string method, size_t p, size_t sample_num, size_t memory, size_t test_num, pair<double, double> accres, std::chrono::time_point<std::chrono::steady_clock> time0, std::chrono::time_point<std::chrono::steady_clock> time1, std::chrono::time_point<std::chrono::steady_clock> time2)
{
	auto sample_time = std::chrono::duration_cast<std::chrono::nanoseconds>(time1 - time0);
	auto query_time = std::chrono::duration_cast<std::chrono::nanoseconds>(time2 - time1);
	const string outfile = "./rstest.txt";
	fstream outf(outfile, std::ios::out | std::ios::app);
	outf << dataset << "\t" << method << "\t" << p << "\t" << sample_num << "\t" << (double)memory / 1024 << "\t" << accres.first << "\t" << accres.second << "\t" << sample_time.count()*1e-6 << "\t" << query_time.count()*1e-3 / test_num << "\n";
	std::cout << dataset << "\t" << method << "\t" << p << "\t" << sample_num << "\t" << (double)memory / 1024 << "\t" << accres.first << "\t" << accres.second << "\t" << sample_time.count()*1e-6 << "\t" << query_time.count()*1e-3 / test_num << "\n";
	outf.close();
}

void RSTest()
{
	const string outfile = "./rstest.txt";
	fstream outf(outfile, std::ios::out | std::ios::app);
	outf << "Dataset\tMethod\tP\tSamples\tMemory(KByte)\tRMSE\tMRE\tSampleTime(ms)\tQueryTime(us/sample)\n";
	std::cout << "Dataset\tMethod\tP\tSamples\tMemory(KByte)\tRMSE\tMRE\tSampleTime(ms)\tQueryTime(us/sample)\n";
	outf.close();

	std::vector<DataSetInfo> dataset_list;
	std::fstream dataconf("./dataset/datasetinfo.txt", std::ios::in);

	while (!dataconf.eof())
	{
		auto ch = dataconf.peek();
		if (ch == EOF)
		{
			break;
		}
		else if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n')
		{
			dataconf.ignore(1);
			continue;
		}
		else if (ch == '#' || ch == '/' || ch == '%')
		{
			dataconf.ignore(256, '\n');
		}
		else if (isalpha(ch) || isdigit(ch))
		{
			string name;
			size_t num;
			size_t dim;
			dataconf >> name >> num >> dim;
			DataSetInfo dsi{ name,num,dim };
			dataset_list.push_back(dsi);
		}
		else
		{
			break;
		}
	}
	dataconf.close();

	//size_t dataset_num;
	//dataconf >> dataset_num;
	//dataset_list.resize(dataset_num);
	//for (size_t i = 0; i < dataset_num; i++)
	//{
	//	string name;
	//	size_t num;
	//	size_t dim;
	//	dataconf >> name >> num >> dim;
	//	dataset_list[i].name = name;
	//	dataset_list[i].num = num;
	//	dataset_list[i].dim = dim;
	//}
	//dataconf.close();

	const std::vector<HashType> hash_list{ htPWORHash, htORHash, htSimHash, htSuperbitHash };
	const std::vector<size_t> sample_num_list{ 1,2,4,8,16,32,64,128,256,512,1024,2048 };
	const std::vector<size_t> kernel_p_list{ 1,2,3,4,5,6 };
	//const std::vector<size_t> kernel_p_list{ 2,3,4,5,6 };
	//const std::vector<size_t> HBS_table_num_list{ 8,64,512,4096 };
	const std::vector<size_t> HBS_table_num_list{ 64,512 };

	UniformRS *urs;
	SKARS *ska;
	HBSRS *hbs;

	for (auto &dataset : dataset_list)
	{
		const size_t sketchnum = (size_t)floor(SKETCH_RATIO*dataset.num);
		const size_t testnum = dataset.num - sketchnum;
		double **data = LoadData("./dataset/" + dataset.name, dataset.num, dataset.dim);
		double **sketchdata = data + testnum;
		urs = new UniformRS(sketchdata, sketchnum, dataset.dim);
		ska = new SKARS(sketchdata, sketchnum, dataset.dim);
		hbs = new HBSRS(sketchdata, sketchnum, dataset.dim);
		for (auto &p : kernel_p_list)
		{
			double *tkd = LoadData("./dataset/gt" + dataset.name + to_string(p), testnum);
			for (auto &sample_num : sample_num_list)
			{
				//if (dataset.name == "SAC" && p < 4)
				//{
				//	continue;
				//}
				//if (dataset.name == "SAC" && p == 4 &&sample_num < 64)
				//{
				//	continue;
				//}
				//if (dataset.name == "KSC" && p == 5 &&sample_num == 512)
				//{
				//	continue;
				//}
				//Uniform
				const size_t mem_urs = sample_num * dataset.dim * sizeof(double);
				if (mem_urs <= MAX_MEMORY)
				{
					auto time0 = std::chrono::steady_clock::now();
					urs->init();
					auto samples = urs->sample(sample_num);
					auto time1 = std::chrono::steady_clock::now();
					auto accres = KDCalculate(sketchdata, data, tkd, testnum, dataset.dim, p, samples);
					auto time2 = std::chrono::steady_clock::now();
					RSOutput(dataset.name, "RS", p, sample_num, mem_urs, testnum, accres, time0, time1, time2);
				}
				//SKA
				// const size_t mem_ska = sample_num * (dataset.dim + 1) * sizeof(double);
				// if (mem_ska <= MAX_MEMORY)
				// {
					// auto time0 = std::chrono::steady_clock::now();
					// ska->init(p);
					// auto samples = ska->sample(sample_num);
					// auto time1 = std::chrono::steady_clock::now();
					// auto accres = KDCalculate(sketchdata, data, tkd, testnum, dataset.dim, p, samples);
					// auto time2 = std::chrono::steady_clock::now();
					// RSOutput(dataset.name, "SKA", p, sample_num, mem_ska, testnum, accres, time0, time1, time2);
				// }
				//HBS
				const size_t mem_hbs = sample_num * (dataset.dim + 1) * sizeof(double);
				if (mem_hbs <= MAX_MEMORY)
				{
					for (auto &tablenum : HBS_table_num_list)
					{
						for (auto &ht : hash_list)
						{
							auto time0 = std::chrono::steady_clock::now();
							hbs->init(tablenum, ht, p);
							auto samples = hbs->sample(sample_num);
							auto time1 = std::chrono::steady_clock::now();
							auto accres = KDCalculate(sketchdata, data, tkd, testnum, dataset.dim, p, samples);
							auto time2 = std::chrono::steady_clock::now();
							string methodstring = "HBS";
							switch (ht)
							{
							case htSimHash:
								methodstring += "-SimHash";
								break;
							case htSuperbitHash:
								methodstring += "-SuperBitHash";
								break;
							case htORHash:
								methodstring += "-ORHash";
								break;
							case htPWORHash:
								methodstring += "-PWORHash";
								break;
							default:
								break;
							}
							methodstring += "@";
							RSOutput(dataset.name, methodstring + to_string(tablenum), p, sample_num, mem_hbs, testnum, accres, time0, time1, time2);
						}
					}
				}
			}
			delete[] tkd;
		}
		delete urs;
		delete ska;
		delete hbs;
		for (size_t i = 0; i < dataset.num; i++)
		{
			delete[] data[i];
		}
		delete[] data;
	}
}

void RACETest()
{
	const string outfile = "./racetest.txt";
	fstream outf(outfile, std::ios::out | std::ios::app);
	outf << "Dataset\tMethod\tP\tTables\tMemory(KByte)\tRMSE\tMRE\tSampleTime(ms)\tQueryTime(us/sample)\n";
	std::cout << "Dataset\tMethod\tP\tTables\tMemory(KByte)\tRMSE\tMRE\tSampleTime(ms)\tQueryTime(us/sample)\n";
	outf.close();

	std::vector<DataSetInfo> dataset_list;
	std::fstream dataconf("./dataset/datasetinfo.txt", std::ios::in);

	while (!dataconf.eof())
	{
		auto ch = dataconf.peek();
		if (ch == EOF)
		{
			break;
		}
		else if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n')
		{
			dataconf.ignore(1);
			continue;
		}
		else if (ch == '#' || ch == '/' || ch == '%')
		{
			dataconf.ignore(256, '\n');
		}
		else if (isalpha(ch) || isdigit(ch))
		{
			string name;
			size_t num;
			size_t dim;
			dataconf >> name >> num >> dim;
			DataSetInfo dsi{ name,num,dim };
			dataset_list.push_back(dsi);
		}
		else
		{
			break;
		}
	}
	dataconf.close();

	const std::vector<HashType> hash_list{ htPWORHash, htORHash, htSimHash, htSuperbitHash };
	const std::vector<int> table_num_list{ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 };
	const std::vector<size_t> kernel_p_list{ 1,2,3,4,5,6 };
	//const std::vector<HashType> hash_list{ htPWORHash, htORHash };
	//const std::vector<int> table_num_list{ 1024, 2048, 4096 };
	//const std::vector<size_t> kernel_p_list{ 3,4,5 };

	RACE *race;
	Hash *hf;

	for (auto &dataset : dataset_list)
	{
		const size_t sketchnum = (size_t)floor(SKETCH_RATIO*dataset.num);
		const size_t testnum = dataset.num - sketchnum;
		double **data = LoadData("./dataset/" + dataset.name, dataset.num, dataset.dim);
		double **sketchdata = data + testnum;
		for (auto &p : kernel_p_list)
		{
			double *tkd = LoadData("./dataset/gt" + dataset.name + to_string(p), testnum);
			for (auto &hash_method : hash_list)
			{
				for (auto &table_num : table_num_list)
				{
					const size_t mem = table_num * (1 << p) * sizeof(int);
					if (mem <= MAX_MEMORY)
					{
						const size_t total_hash_len = table_num * p;
						auto time0 = std::chrono::steady_clock::now();
						switch (hash_method)
						{
						case htSimHash:
							hf = new SimHash(dataset.dim, total_hash_len);
							break;
						case htSuperbitHash:
							hf = new SuperBitHash(dataset.dim, total_hash_len);
							break;
						case htORHash:
							hf = new ORHash(dataset.dim, total_hash_len);
							break;
						case htPWORHash:
							hf = new PWORHash(dataset.dim, total_hash_len);
							break;
						default:
							hf = new ORHash(dataset.dim, total_hash_len);
							break;
						}
						auto time1 = std::chrono::steady_clock::now();
						race = new RACE(table_num, 1 << p);
						for (size_t i = testnum; i < dataset.num; i++)
						{
							int *hashes = hf->getHash(data[i]);
							int *inthash = bin2int(hashes, total_hash_len, p);
							race->add(inthash);
							delete[] hashes;
							delete[] inthash;
						}
						auto time2 = std::chrono::steady_clock::now();
						int **hashes = new int*[testnum];
						for (size_t i = 0; i < testnum; i++)
						{
							hashes[i] = hf->getHash(data[i]);
						}
						auto time3 = std::chrono::steady_clock::now();
						double *ekd = new double[testnum];
						//#pragma omp parallel for
						for (size_t i = 0; i < testnum; i++)
						{
							int *inthash = bin2int(hashes[i], total_hash_len, p);
							ekd[i] = (double)race->query(inthash) / sketchnum;
							delete[] hashes[i];
							delete[] inthash;
						}
						auto time4 = std::chrono::steady_clock::now();
						delete hf;
						delete race;
						double rmse = 0;
						double mre = 0;
						for (size_t i = 0; i < testnum; i++)
						{
							double err = abs(ekd[i] - tkd[i]);
							rmse += err * err;
							mre += err / tkd[i];
						}
						rmse /= testnum;
						rmse = sqrt(rmse);
						mre /= testnum;
						auto hash_init_time = std::chrono::duration_cast<std::chrono::nanoseconds>(time1 - time0);
						auto sketch_time = std::chrono::duration_cast<std::chrono::nanoseconds>(time2 - time1);
						auto set_up_time = std::chrono::duration_cast<std::chrono::nanoseconds>(time2 - time0);
						auto query_hash_time = std::chrono::duration_cast<std::chrono::nanoseconds>(time3 - time2);
						auto look_up_time = std::chrono::duration_cast<std::chrono::nanoseconds>(time4 - time3);
						auto query_time = std::chrono::duration_cast<std::chrono::nanoseconds>(time4 - time2);
						string methodstring = "RACE";
						switch (hash_method)
						{
						case htSimHash:
							methodstring += "-SimHash";
							break;
						case htSuperbitHash:
							methodstring += "-SuperBitHash";
							break;
						case htORHash:
							methodstring += "-ORHash";
							break;
						case htPWORHash:
							methodstring += "-PWORHash";
							break;
						default:
							break;
						}
						auto accres = std::make_pair(rmse, mre);
						RSOutput(dataset.name, methodstring, p, table_num, mem, testnum, accres, time0, time2, time4);
					}
				}
			}
			delete[] tkd;
		}
		for (size_t i = 0; i < dataset.num; i++)
		{
			delete[] data[i];
		}
		delete[] data;
	}
}

inline double ns2ms(long long ns)
{
	return (double)ns * 1e-6;
}

template<typename NumType>
inline void MatCopy(NumType** &dst, NumType** src, size_t num, size_t dstdim, size_t srcdim)
{
	dst = new NumType * [num];
	size_t copydim = min(dstdim, srcdim);
	for (int i = 0; i < num; i++)
	{
		dst[i] = new NumType[dstdim]();
		memcpy(dst[i], src[i], copydim * sizeof(NumType));
	}
}

template<typename NumType>
inline void MatRepCopy(NumType**& dst, NumType** src, size_t num, size_t repdim, size_t repnum, size_t srcdim)
{
	size_t dstdim = repdim * repnum;
	dst = new NumType * [num];
	size_t copydim = min(repdim, srcdim);
	for (int i = 0; i < num; i++)
	{
		dst[i] = new NumType[dstdim]();
		for (int j = 0; j < repnum; j++)
		{
			memcpy(dst[i] + j * repdim, src[i], copydim * sizeof(NumType));
		}
	}
}

template<typename NumType>
inline void MatDelete(NumType** &mat, size_t num)
{
	for (int i = 0; i < num; i++)
	{
		delete[] mat[i];
	}
	delete[] mat;
}

void FastAcc()
{
	const string outfile = "./fastacc.txt";
	fstream outf(outfile, std::ios::out | std::ios::app);
	outf << "Dataset\tMethod\tP\tTables\tMemory(KByte)\tRMSE\tMRE\tSampleTime(ms)\tQueryTime(us/sample)\tTotalTime(ms)\n";
	std::cout << "Dataset\tMethod\tP\tTables\tMemory(KByte)\tRMSE\tMRE\tSampleTime(ms)\tQueryTime(us/sample)\tTotalTime(ms)\n";
	outf.close();

	std::vector<DataSetInfo> dataset_list;
	std::fstream dataconf("./dataset/datasetinfo.txt", std::ios::in);

	while (!dataconf.eof())
	{
		auto ch = dataconf.peek();
		if (ch == EOF)
		{
			break;
		}
		else if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n')
		{
			dataconf.ignore(1);
			continue;
		}
		else if (ch == '#' || ch == '/' || ch == '%')
		{
			dataconf.ignore(256, '\n');
		}
		else if (isalpha(ch) || isdigit(ch))
		{
			string name;
			size_t num;
			size_t dim;
			dataconf >> name >> num >> dim;
			DataSetInfo dsi{ name,num,dim };
			dataset_list.push_back(dsi);
		}
		else
		{
			break;
		}
	}
	dataconf.close();

	const std::vector<HashType> hash_list{ htPWORHash, htORHash, htSimHash, htSuperbitHash };
	//const std::vector<int> table_num_list{ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 };
	//const std::vector<int> table_num_list{ 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304 };
	//const std::vector<int> table_num_list{ 16384, 32768, 65536 };
	const std::vector<int> table_num_list{ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 };
	const std::vector<size_t> kernel_p_list{ 1,2,3,4,5,6 };
	//const std::vector<size_t> kernel_p_list{ 6 };

	RACE* race;
	Hash* hf;

	for (auto& dataset : dataset_list)
	{
		const size_t sketchnum = 100;
		const size_t testnum = 100;
		size_t realnum = dataset.num;
		if (dataset.name == "cifar10")
		{
			realnum = 60000;
		}
		else if (dataset.name == "rcv1")
		{
			realnum = 534135;
		}
		const size_t realsketchnum = (size_t)floor(SKETCH_RATIO * (double)realnum);
		const size_t realtestnum = dataset.num - realsketchnum;
		double** data = LoadData("./dataset/" + dataset.name, dataset.num, dataset.dim);
		double** testdata = data;
		double** sketchdata = data + testnum;
		for (auto& p : kernel_p_list)
		{
			double* tkd = LoadData("./dataset/gt100" + dataset.name + to_string(p), testnum);
			for (auto& table_num : table_num_list)
			{
				const size_t mem = table_num * (1 << p) * sizeof(int);
				if (mem <= MAX_MEMORY)
				{
					size_t vec_dim = dataset.dim;
					size_t hash_dim = table_num * p;
					size_t or_size = (size_t)1 << (size_t)ceil(log2(max(vec_dim, hash_dim)));
					size_t por_psize = (size_t)1 << (size_t)ceil(log2(vec_dim));
					size_t piece_num = (size_t)ceil((double)hash_dim / por_psize);
					size_t por_size = piece_num * por_psize;
					const size_t total_hash_len = table_num * p;
					double** sketch_data_tmp = NULL;
					double** test_data_tmp = NULL;

					for (auto& hash_method : hash_list)
					{
						switch (hash_method)
						{
						case htSimHash:
							MatCopy(sketch_data_tmp, sketchdata, sketchnum, vec_dim, vec_dim);
							MatCopy(test_data_tmp, testdata, testnum, vec_dim, vec_dim);
							break;
						case htSuperbitHash:
							MatCopy(sketch_data_tmp, sketchdata, sketchnum, vec_dim, vec_dim);
							MatCopy(test_data_tmp, testdata, testnum, vec_dim, vec_dim);
							break;
						case htORHash:
							MatCopy(sketch_data_tmp, sketchdata, sketchnum, or_size, vec_dim);
							MatCopy(test_data_tmp, testdata, testnum, or_size, vec_dim);
							break;
						case htPWORHash:
							MatRepCopy(sketch_data_tmp, sketchdata, sketchnum, por_psize, piece_num, vec_dim);
							MatRepCopy(test_data_tmp, testdata, testnum, por_psize, piece_num, vec_dim);
							break;
						default:
							MatCopy(sketch_data_tmp, sketchdata, sketchnum, or_size, vec_dim);
							MatCopy(test_data_tmp, testdata, testnum, or_size, vec_dim);
							break;
						}
						race = new RACE(table_num, 1 << p);

						auto time0 = std::chrono::steady_clock::now();
						switch (hash_method)
						{
						case htSimHash:
							hf = new SimHash(dataset.dim, total_hash_len);
							break;
						case htSuperbitHash:
							hf = new SuperBitHash(dataset.dim, total_hash_len);
							break;
						case htORHash:
							hf = new ORHash(dataset.dim, total_hash_len);
							break;
						case htPWORHash:
							hf = new PWORHash(dataset.dim, total_hash_len);
							break;
						default:
							hf = new ORHash(dataset.dim, total_hash_len);
							break;
						}
						auto time1 = std::chrono::steady_clock::now();
						for (size_t i = 0; i < sketchnum; i++)
						{
							int* hashes = hf->getHash(sketch_data_tmp[i]);
							int* inthash = bin2int(hashes, total_hash_len, p);
							race->add(inthash);
							delete[] hashes;
							delete[] inthash;
						}
						auto time2 = std::chrono::steady_clock::now();
						int** hashes = new int* [testnum];
						for (size_t i = 0; i < testnum; i++)
						{
							hashes[i] = hf->getHash(test_data_tmp[i]);
						}
						auto time3 = std::chrono::steady_clock::now();
						double* ekd = new double[testnum];
						for (size_t i = 0; i < testnum; i++)
						{
							int* inthash = bin2int(hashes[i], total_hash_len, p);
							ekd[i] = (double)race->query(inthash) / sketchnum;
							delete[] hashes[i];
							delete[] inthash;
						}
						auto time4 = std::chrono::steady_clock::now();
						delete hf;
						delete race;
						double rmse = 0;
						double mre = 0;
						for (size_t i = 0; i < testnum; i++)
						{
							double err = abs(ekd[i] - tkd[i]);
							rmse += err * err;
							mre += err / tkd[i];
						}
						rmse /= testnum;
						rmse = sqrt(rmse);
						mre /= testnum;
						auto hash_init_time = std::chrono::duration_cast<std::chrono::nanoseconds>(time1 - time0);
						auto sketch_time = std::chrono::duration_cast<std::chrono::nanoseconds>(time2 - time1);
						auto set_up_time = std::chrono::duration_cast<std::chrono::nanoseconds>(time2 - time0);
						auto query_hash_time = std::chrono::duration_cast<std::chrono::nanoseconds>(time3 - time2);
						auto look_up_time = std::chrono::duration_cast<std::chrono::nanoseconds>(time4 - time3);
						auto query_time = std::chrono::duration_cast<std::chrono::nanoseconds>(time4 - time2);
						string methodstring = "RACE";
						switch (hash_method)
						{
						case htSimHash:
							methodstring += "-SimHash";
							break;
						case htSuperbitHash:
							methodstring += "-SuperBitHash";
							break;
						case htORHash:
							methodstring += "-ORHash";
							break;
						case htPWORHash:
							methodstring += "-PWORHash";
							break;
						default:
							break;
						}
						auto accres = std::make_pair(rmse, mre);

						double sketchtime_ms = ((double)hash_init_time.count() + (double)sketch_time.count() / sketchnum * realsketchnum) * 1e-6;
						double querytime_us = (double)query_time.count() / sketchnum * 1e-3;
						double totaltime_ms = sketchtime_ms + querytime_us * realsketchnum * 1e-3;

						outf.open(outfile, std::ios::out | std::ios::app);
						outf << dataset.name << "\t" << methodstring << "\t" << p << "\t" << table_num << "\t" << (double)mem / 1024 << "\t" << rmse << "\t" << mre << "\t" << sketchtime_ms << "\t" << querytime_us << "\t" << totaltime_ms << "\n";
						std::cout << dataset.name << "\t" << methodstring << "\t" << p << "\t" << table_num << "\t" << (double)mem / 1024 << "\t" << rmse << "\t" << mre << "\t" << sketchtime_ms << "\t" << querytime_us << "\t" << totaltime_ms << "\n";
						outf.close();

						//RSOutput(dataset.name, methodstring, p, table_num, mem, testnum, accres, time0, time2, time4);

						MatDelete(sketch_data_tmp, sketchnum);
						MatDelete(test_data_tmp, testnum);
					}
				}
			}
			delete[] tkd;
		}
		for (size_t i = 0; i < dataset.num; i++)
		{
			delete[] data[i];
		}
		delete[] data;
	}
}

void GetGT()
{
	std::vector<DataSetInfo> dataset_list;
	std::fstream dataconf("./dataset/datasetinfo.txt", std::ios::in);

	while (!dataconf.eof())
	{
		auto ch = dataconf.peek();
		if (ch == EOF)
		{
			break;
		}
		else if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n')
		{
			dataconf.ignore(1);
			continue;
		}
		else if (ch == '#' || ch == '/' || ch == '%')
		{
			dataconf.ignore(256, '\n');
		}
		else if (isalpha(ch) || isdigit(ch))
		{
			string name;
			size_t num;
			size_t dim;
			dataconf >> name >> num >> dim;
			DataSetInfo dsi{ name,num,dim };
			dataset_list.push_back(dsi);
		}
		else
		{
			break;
		}
	}
	dataconf.close();

	const std::vector<size_t> kernel_p_list{ 1,2,3,4,5,6 };

	RACE* race;
	Hash* hf;

	for (auto& dataset : dataset_list)
	{
		const size_t sketchnum = 100;
		const size_t testnum = 100;
		double** data = LoadData("./dataset/" + dataset.name, dataset.num, dataset.dim);
		double** testdata = data;
		double** sketchdata = data + testnum;
		for (auto& p : kernel_p_list)
		{
			double* tkd = new double[testnum];
			//double* tkd = LoadData("./dataset/gt100" + dataset.name + to_string(p), testnum);
			#pragma omp parallel for
			for (int i = 0; i < testnum; i++)
			{
				double kd = 0;
				for (int j = 0; j < sketchnum; j++)
				{
					kd += pow(AngularKernel(testdata[i], sketchdata[j], dataset.dim), p);
				}
				tkd[i] = kd / sketchnum;
			}
			ofstream tkdf;
			tkdf.open("./dataset/gt100" + dataset.name + to_string(p), std::ios::out);
			for (size_t i = 0; i < testnum; i++)
			{
				tkdf << tkd[i] << " ";
			}
			delete[] tkd;
			std::cout << dataset.name << p << "\n";
		}
		for (size_t i = 0; i < dataset.num; i++)
		{
			delete[] data[i];
		}
		delete[] data;
	}
}

void Collision()
{
	std::vector<DataSetInfo> dataset_list;
	std::fstream dataconf("./dataset/datasetinfo.txt", std::ios::in);

	while (!dataconf.eof())
	{
		auto ch = dataconf.peek();
		if (ch == EOF)
		{
			break;
		}
		else if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n')
		{
			dataconf.ignore(1);
			continue;
		}
		else if (ch == '#' || ch == '/' || ch == '%')
		{
			dataconf.ignore(256, '\n');
		}
		else if (isalpha(ch) || isdigit(ch))
		{
			string name;
			size_t num;
			size_t dim;
			dataconf >> name >> num >> dim;
			DataSetInfo dsi{ name,num,dim };
			dataset_list.push_back(dsi);
		}
		else
		{
			break;
		}
	}
	dataconf.close();

	const std::vector<size_t> kernel_p_list{ 1,2,3,4,5,6 };

	Hash* hf;

	for (auto& dataset : dataset_list)
	{
		const size_t sketchnum = 100;
		const size_t testnum = 100;
		const size_t tablenum = 1000;
		double** data = LoadData("./dataset/" + dataset.name, dataset.num, dataset.dim);
		double** testdata = data;
		double** sketchdata = data + testnum;
		for (auto& p : kernel_p_list)
		{
			//double* tkd = new double[testnum];
			//double* tkd = LoadData("./dataset/gt100" + dataset.name + to_string(p), testnum);
			size_t hash_dim = p * tablenum;
			size_t or_size = (size_t)1 << (size_t)ceil(log2(max(dataset.dim, hash_dim)));
			size_t por_psize = (size_t)1 << (size_t)ceil(log2(dataset.dim));
			size_t por_size = (size_t)ceil((double)hash_dim / por_psize) * por_psize;
			hf = new PWORHash(dataset.dim, hash_dim);

			double** test_tmp = NULL;
			MatCopy(test_tmp, testdata, testnum, por_size, dataset.dim);
			double* ekd = new double[testnum];
			for (int i = 0; i < testnum; i++)
			{
				auto tbinhash = hf->getHash(test_tmp[i]);
				auto tinthash = bin2int(tbinhash, por_size, p);
				double kd = 0;
				double** sketch_tmp = NULL;
				MatCopy(sketch_tmp, sketchdata, sketchnum, por_size, dataset.dim);
				for (int j = 0; j < sketchnum; j++)
				{
					double hamsim = 0;
					auto binhash = hf->getHash(sketch_tmp[j]);
					auto inthash = bin2int(tbinhash, por_size, p);
					for (int k = 0; k < tablenum; k++)
					{
						if (inthash[k] == tinthash[k])
						{
							hamsim += 1;
						}
					}
					kd += hamsim / tablenum;
				}
				ekd[i] = kd / sketchnum;
				MatDelete(sketch_tmp, sketchnum);
				delete[] tbinhash;
				delete[] tinthash;
			}
			
			MatDelete(test_tmp, testnum);
			ofstream ccf;
			ccf.open("./dataset/cc100" + dataset.name + to_string(p), std::ios::out);
			for (size_t i = 0; i < testnum; i++)
			{
				ccf << ekd[i] << " ";
			}
			std::cout << dataset.name << p << "\n";
			delete[] ekd;
			delete hf;
		}
		for (size_t i = 0; i < dataset.num; i++)
		{
			delete[] data[i];
		}
		delete[] data;
	}
}

int main()
{
	//RACETest();
	//RSTest();
}
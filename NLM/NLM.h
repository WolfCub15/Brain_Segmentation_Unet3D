#pragma once
#include <iostream>
#include <vector>
#include <QImage.h>
//#include "mpi.h"

#define double long double
using namespace std;

class NLM {
private:
	int n, m,radius;
	QImage photo;
	vector<int> data;

	vector<int> dataR;
	vector<int> dataG;
	vector<int> dataB;

	//vector<vector<int>> st;
	//double calcSt(QImage photo, int t1, int t2, int z1, int z2);
	//double distance(double x, double y);
	//double weightFunction(double u, double h);
	double calcSSIM(int x, int y);
	double averageBrightness(int x);
	double standardDeviation(int x);
	double correlation(int x, int y);
public:

	NLM(int r, QImage& ph);
	~NLM() {
		dataR.clear();
		dataG.clear();
		dataB.clear();
		data.clear();

	};
	QImage calc( );

};





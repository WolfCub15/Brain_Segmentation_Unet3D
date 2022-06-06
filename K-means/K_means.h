#pragma once
#include <iostream>
#include <vector>
#include <QImage.h>

using namespace std;
const int KK = 10;  //количество кластеров
const int max_iterations = 100;  //максимальное количество итераций

typedef struct {
	double r;
	double g;
	double b;
} rgb;

class K_means{
private:
	vector<pair<rgb,pair<int,int>>> pixcel;
	int q_klaster;
	int k_pixcel;
	vector<rgb> centr;

	void identify_centers();
	inline double compute(rgb k1, rgb k2){
		return sqrt(pow((k1.r - k2.r), 2) + pow((k1.g - k2.g), 2) + pow((k1.b - k2.b), 2));
	}
	inline double compute_s(double a, double b) {
		return (a + b) / 2;
	};
public:
	K_means() : q_klaster(0), k_pixcel(0) {};
	K_means(int n, rgb* mas, int n_klaster);
	QImage calc(int n_klaster, const QImage& photo);
	//void clustering(std::ostream& os);
	//void print()const;
	~K_means() {};
	//friend std::ostream& operator<<(std::ostream& os, const K_means& k);
};

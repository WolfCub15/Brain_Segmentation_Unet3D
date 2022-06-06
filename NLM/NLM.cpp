#include "NLM.h"

template <class T>

T clamp(T v, int max, int min) {
	if (v > max) return max;
	else if (v < min) return min;
	else return v;
}

NLM::NLM(int r, QImage& ph) {
	radius = r;
	photo = ph;
	m = photo.width();
	n = photo.height();
	int nm = n * m;
	dataR.resize(nm, 0);
	dataG.resize(nm, 0);
	dataB.resize(nm, 0);
	data.resize(nm, 0);
	int pos = 0;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			QColor c1 = photo.pixelColor(j, i);
			//cout << c1.red() << ' ' << c1.green() << ' ' << c1.blue() << '\n';
			data[pos] = (c1.value());
			dataR[pos] = (c1.red());
			dataG[pos] = (c1.green());
			dataB[pos] = (c1.blue());
			pos++;
		}
	}
	/*cout << dataR.size() << endl;
	for (int i = 0; i < dataR.size(); ++i) {
		cout << i<< "     " <<dataR[i]<<' '<<dataG[i]<<' '<< dataB[i] <<endl;
	}
	exit(0);*/
}
double  NLM::calcSSIM(int x, int y) {
	double u1 = averageBrightness(x);
	double u2 = averageBrightness(y);
	double l = (2. * u1 * u2 + 1.) / (u1 * u1 + u2 * u2 + 1.);
	double o1 = standardDeviation(x);
	double o2 = standardDeviation(y);
	double c = (2. * o1 * o2 + 1.) / (o1 * o1 + o2 * o2 + 1.);
	double g = correlation(x, y);

	double s = (g + 1.) / (o1 * o2 + 1.);
	//cout << l << ' ' << c << ' ' << s << endl;
	//cout << l * c * s << endl;

	return l * c * s;
}

double NLM::averageBrightness(int x) {
	double sum = 0;
	int t = x - m * radius - radius;
	int rSize = 2 * radius + 1;
	for (int q = 1; q <= rSize; ++q) {
		for (int i = 0; i < rSize; i++) {
			if ((t + i) >= 0 && (t + i) < data.size()) sum += data[t + i];
			//else return 0;
		}
		t += m;
	}
	return sum/(rSize* rSize);
}

double NLM::standardDeviation(int x) {
	double u = averageBrightness(x);
	double sum = 0;
	int t = x - m * radius - radius;
	int rSize = 2 * radius + 1;
	for (int q = 1; q <= rSize; ++q) {
		for (int i = 0; i < rSize; i++) {
			if ((t + i) >= 0 && (t + i) < data.size()) sum += abs(data[t + i]-u) * abs(data[t + i]-u);
			//else return 0;
		}
		t += m;
	}
	return sqrt(sum/(rSize * rSize));
}

double NLM::correlation(int x, int y) {
	double sum = 0;
	int t1 = x - m * radius - radius;
	int t2 = y - m * radius - radius;
	double u1 = averageBrightness(x);
	double u2 = averageBrightness(y);

	int rSize = 2 * radius + 1;
	for (int q = 1; q <= rSize; ++q) {
		for (int i = 0; i < rSize; i++) {
			if ((t1 + i) >= 0 && (t1 + i) < data.size() && (t2 + i) >= 0 && (t2 + i) < data.size()) {
				sum += (data[t1 + i]-u1) * (data[t2 + i]-u2);
			}
			//else return 0;

		}
		t1 += m;
		t2 += m;
	}
	return sum / (rSize * rSize);
}



QImage NLM::calc() {
	QImage ph(photo);
	cout << n << ' ' << m << endl;
	int size = dataR.size();
	vector<double> processed_dataR(size, 0);
	vector<double> processed_dataG(size, 0);
	vector<double> processed_dataB(size, 0);
	int rSize = 2 * radius + 1;

	vector<double> norm(size, 0);
	for (int i = 0; i < size; ++i) {
		cout << "Tuuuut  " << i << endl;
		for (int j = 0; j < size; ++j) {
			double weight = abs(calcSSIM(i, j));
			weight /= n * m;
			//cout<<weight<<'\n';
			norm[i] += weight;
			processed_dataR[i] += dataR[j] * weight;
			processed_dataG[i] += dataG[j] * weight;
			processed_dataB[i] += dataB[j] * weight;
		}
	}
	int pos = 0;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			//cout << "toooot!!  " << i << ' ' << j << endl;
			if (norm[pos + j]) {
				QColor c(clamp(processed_dataR[pos + j] / norm[pos + j], 255, 0), clamp(processed_dataG[pos + j] / norm[pos + j], 255, 0), clamp(processed_dataB[pos + j] / norm[pos + j], 255, 0));
				ph.setPixelColor(j, i, c);
			}
		}
		pos += m;
	}
	return ph;
}

#include "K_means.h"
#include <time.h>

template <class T>

T clamp(T v, int max, int min) {//текущее значение, максимальное и минимальное значение
	if (v > max) return max;
	else if (v < min) return min;
	else return v;
}

void K_means::identify_centers() {
	srand((unsigned)time(NULL));
	rgb temp;
	rgb* mas = new rgb[q_klaster];
	for (int i = 0; i < q_klaster; i++) {
		temp = pixcel[0 + rand() % k_pixcel].first;
		for (int j = i; j < q_klaster; j++) {
			if (temp.r != mas[j].r && temp.g != mas[j].g && temp.b != mas[j].b) {
				mas[j] = temp;
			}
			else {
				i--;
				break;
			}
		}
	}
	for (int i = 0; i < q_klaster; i++) {
		centr.push_back(mas[i]);
	}
	delete[]mas;
}



K_means::K_means(int n, rgb* mas, int n_klaster){
	for (int i = 0; i < n; i++) {
		pixcel.push_back(make_pair(*(mas + i),make_pair(0,0)));
	}
	q_klaster = n_klaster;
	k_pixcel = n;
	identify_centers();
}

QImage K_means::calc(int n_klaster, const QImage& photo) {
	rgb temp;
	q_klaster = n_klaster;
	QImage result_image(photo);
	int R, G, B;
	for (int x = 0; x < photo.width(); x++) {
		for (int y = 0; y < photo.height(); y++) {
			QColor color = photo.pixelColor(x, y);
			temp.r = color.red();
			temp.g = color.green();
			temp.b = color.blue();
			pixcel.push_back(make_pair(temp,make_pair(x,y)));
		}
	}
	k_pixcel = pixcel.size();
	identify_centers();

	vector<int> check_1(k_pixcel, -1);
	vector<int> check_2(k_pixcel, -2);
	int iter = 0;
	/*Количество итераций.*/
	while (true) {
		{
			for (int j = 0; j < k_pixcel; j++) {
				double* mas = new double[q_klaster];
				for (int i = 0; i < q_klaster; i++) {
					*(mas + i) = compute(pixcel[j].first, centr[i]);
				}
				/*Определяем минимальное расстояние и в m_k фиксируем номер центра для дальнейшего пересчета.*/
				double min_dist = *mas;
				int m_k = 0;
				for (int i = 0; i < q_klaster; i++) {
					if (min_dist > * (mas + i)) {
						min_dist = *(mas + i);
						m_k = i;//Минимальное расстояние к центру
					}
				}

				//Пересчитываем центр
				centr[m_k].r = compute_s(pixcel[j].first.r, centr[m_k].r);
				centr[m_k].g = compute_s(pixcel[j].first.g, centr[m_k].g);
				centr[m_k].b = compute_s(pixcel[j].first.b, centr[m_k].b);
				
				delete[] mas;
			}
			/*Классифицируем пиксели по кластерам.*/
			int* mass = new int[k_pixcel];
			for (int k = 0; k < k_pixcel; k++) {
				double* mas = new double[q_klaster];
				/*Находим расстояние до каждого центра.*/
				for (int i = 0; i < q_klaster; i++) {
					*(mas + i) = compute(pixcel[k].first, centr[i]);
				}
				/*Определяем минимальное расстояние.*/
				double min_dist = *mas;
				int m_k = 0;
				for (int i = 0; i < q_klaster; i++) {
					if (min_dist > * (mas + i)) {
						min_dist = *(mas + i);
						m_k = i;
					}
				}
				mass[k] = m_k;
			}
			/*Выводим информацию о принадлежности пикселей к кластерам и заполняем вектор для сравнения итераций.*/
			//os << "\nМассив соответствия пикселей и центров: \n";
			for (int i = 0; i < k_pixcel; i++) {
				check_1[i] = *(mass + i);
			}
			//os << std::endl << std::endl;

			//os << "Результат кластеризации: " << std::endl;
			int itr = KK + 1;
			for (int i = 0; i < q_klaster; i++) {
				int r= rand() % 255;
				int g = rand() % 255;
				int b = rand() % 255;

				for (int j = 0; j < k_pixcel; j++) {
					if (mass[j] == i) {
						QColor res = QColor(clamp(r, 255, 0), clamp(g, 255, 0), clamp(b, 255, 0));

						result_image.setPixelColor(pixcel[j].second.first, pixcel[j].second.second, res);
						//os << pixcel[j].r << " " << pixcel[j].g << " " << pixcel[j].b << std::endl;
						mass[j] = ++itr;
					}
				}
			}

			delete[] mass;
			/*Выводим информацию о новых центрах.*/
			//os << "Новые центры: \n";
			for (int i = 0; i < q_klaster; i++) {
				//os << centr[i].r << " " << centr[i].g << " " << centr[i].b << " - #" << i << std::endl;
			}
		}
		/*Если наши векторы равны или количество итераций больше допустимого – прекращаем процесс.*/
		iter++;
		if (check_1 == check_2 || iter >= max_iterations) {
			break;
		}
		check_2 = check_1;
	}
	return result_image;
}



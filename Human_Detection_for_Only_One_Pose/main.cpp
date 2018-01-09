#include <stdio.h>
#include <iostream>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include "svm.h"

#define YUDO_CD 0.5

using namespace std;

struct svm_node *x;
int max_nr_attr = 64;

struct svm_model* CD;
struct svm_model* Stand;
struct svm_model* Squat;
struct svm_model* Lie;

//static char *line = NULL;
static int max_line_len;

class Detect_Place {
public:
	int C_x;
	int C_y;
	int C_width;
	int C_height;
	float C_yudo;

	int territory_num;

	int ratio_num;

	Detect_Place() {
		C_x = C_y = -1;
		C_width = C_height = -1;
		C_yudo = 0.0;

		territory_num = -1;

		ratio_num = -1;
	}

};

void exit_input_error(int line_num)
{
	fprintf(stderr, "Wrong input format at line %d\n", line_num);
	exit(1);
}

double predict(float *hog_vector, int hog_dim, svm_model* Detector)
{
	int svm_type = svm_get_svm_type(Detector);
	int nr_class = svm_get_nr_class(Detector);
	double *prob_estimates = NULL;
	int j;

	int *labels = (int *)malloc(nr_class * sizeof(int));
	svm_get_labels(Detector, labels);
	prob_estimates = (double *)malloc(nr_class * sizeof(double));
	free(labels);


	max_line_len = 1024;
	//	line = (char *)malloc(max_line_len*sizeof(char));
	x = (struct svm_node *) malloc(max_nr_attr * sizeof(struct svm_node));
	int i;
	double target_label, predict_label;
	char *idx, *val, *label, *endptr;
	int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0


							 //	for (i = 0; hog_vector[i] != NULL; i++)
	for (i = 0; i < hog_dim; i++)
	{
		//		clog << i << endl;
		if (i >= max_nr_attr - 1)	// need one more for index = -1
		{
			max_nr_attr *= 2;
			x = (struct svm_node *) realloc(x, max_nr_attr * sizeof(struct svm_node));
		}

		//			clog << i << endl;

		errno = 0;
		x[i].index = i;
		inst_max_index = x[i].index;

		errno = 0;
		x[i].value = hog_vector[i];
		//		cout << i << ":" << hog_vector[i] << endl;
	}
	x[i].index = -1;


	predict_label = svm_predict_probability(Detector, x, prob_estimates);
	//	if (prob_estimates[0] >= YUDO_CD)
	//		printf(" %f\n", prob_estimates[0]);

	free(x);
	//	free(line);

	return prob_estimates[0];
}

int minimum(int a, int b) {
	if (a<b) {
		return a;
	}
	return b;
}

void get_HOG(cv::Mat im, float* hog_vector) {
	int cell_size = 6;
	int rot_res = 9;
	int block_size = 3;
	int x, y, i, j, k, m, n, count;
	float dx, dy;
	float ***hist, *vec_tmp;
	float norm;
	CvMat *mag = NULL, *theta = NULL;
	//	FILE *hog_hist;

	//	fopen_s(&hog_hist,"d_im_hog.txt", "w");
	//	fopen_s(&hog_hist, "d_im_hog.bin", "w");
	//	fprintf(hog_hist, "%c ", '1');
	int counter = 1;

	mag = cvCreateMat(im.rows, im.cols, CV_32F);
	theta = cvCreateMat(im.rows, im.cols, CV_32F);
	for (y = 0; y<im.rows; y++) {
		for (x = 0; x<im.cols; x++) {
			if (x == 0 || x == im.cols - 1 || y == 0 || y == im.rows - 1) {
				cvmSet(mag, y, x, 0.0);
				cvmSet(theta, y, x, 0.0);
			}
			else {
				dx = double((uchar)im.data[y*im.step + x + 1]) - double((uchar)im.data[y*im.step + x - 1]);
				dy = double((uchar)im.data[(y + 1)*im.step + x]) - double((uchar)im.data[(y - 1)*im.step + x]);
				cvmSet(mag, y, x, sqrt(dx*dx + dy * dy));
				cvmSet(theta, y, x, atan(dy / (dx + 0.01)));
			}
		}
	}

	// histogram generation for each cell
	hist = (float***)malloc(sizeof(float**) * (int)ceil((float)im.rows / (float)cell_size));
	if (hist == NULL) exit(1);
	for (i = 0; i<(int)ceil((float)im.rows / (float)cell_size); i++) {
		hist[i] = (float**)malloc(sizeof(float*)*(int)ceil((float)im.cols / (float)cell_size));
		if (hist[i] == NULL) exit(1);
		for (j = 0; j<(int)ceil((float)im.cols / (float)cell_size); j++) {
			hist[i][j] = (float *)malloc(sizeof(float)*rot_res);
			if (hist[i][j] == NULL) exit(1);
		}
	}
	for (i = 0; i<(int)ceil((float)im.rows / (float)cell_size); i++) {
		for (j = 0; j<(int)ceil((float)im.cols / (float)cell_size); j++) {
			for (k = 0; k<rot_res; k++) {
				hist[i][j][k] = 0.0;
			}
		}
	}
	for (i = 0; i<(int)ceil((float)im.rows / (float)cell_size); i++) {
		for (j = 0; j<(int)ceil((float)im.cols / (float)cell_size); j++) {
			for (m = i * cell_size; m<minimum((i + 1)*cell_size, im.rows); m++) {
				for (n = j * cell_size; n<minimum((j + 1)*cell_size, im.cols); n++) {
					hist[i][j][(int)floor((cvmGet(theta, m, n) + CV_PI / 2)*rot_res / CV_PI)] += cvmGet(mag, m, n);
				}
			}
		}
	}

	// normalization for each block & generate vector
	vec_tmp = (float *)malloc(sizeof(float)*block_size*block_size*rot_res);
	for (i = 0; i<(int)ceil((float)im.rows / (float)cell_size) - (block_size - 1); i++) {
		for (j = 0; j<(int)ceil((float)im.cols / (float)cell_size) - (block_size - 1); j++) {
			count = 0;
			norm = 0.0;
			for (m = i; m<i + block_size; m++) {
				for (n = j; n<j + block_size; n++) {
					for (k = 0; k<rot_res; k++) {
						vec_tmp[count++] = hist[m][n][k];
						norm += hist[m][n][k] * hist[m][n][k];
					}
				}
			}
			for (count = 0; count<block_size*block_size*rot_res; count++) {
				vec_tmp[count] = vec_tmp[count] / (sqrt(norm + 1));
				if (vec_tmp[count]>0.2) vec_tmp[count] = 0.2;
				//		fprintf(hog_hist, "%d:%.4f ", counter, vec_tmp[count]);
				hog_vector[counter] = vec_tmp[count];
				//		cout << counter << ":" << hog_vector[counter] << endl;
				//		printf("%d:%.4f ",counter, vec_tmp[count]);
				counter++;
			}
		}
	}
	//	printf("\n");
	//	fprintf(hog_hist, "\n");
	for (i = 0; i<(int)ceil((float)im.rows / (float)cell_size); i++) {
		for (j = 0; j <(int)ceil((float)im.cols / (float)cell_size); j++) {
			free(hist[i][j]);
		}
		free(hist[i]);
	}
	free(hist);
	cvReleaseMat(&mag);
	cvReleaseMat(&theta);
	//	fclose(hog_hist);
}

cv::Mat draw_rectangle(cv::Mat ans_im, int x, int y, int width, int height, int r, int g, int b) {
	rectangle(ans_im, cvPoint(x, y), cvPoint(x + width, y + height), CV_RGB(r, g, b), 1);
	return ans_im;
}

int dimension(int x, int y) {

	return (int)(81 * ((int)ceil((float)x / 6) - 2) * ((int)ceil((float)y / 6) - 2));

	if (x % 6 == 0) {
		return 81 * (x / 6 - 2) * (y / 6 - 1);
	}
	if (y % 6 == 0) {
		return 81 * (x / 6 - 1) * (y / 6 - 2);
	}
	if (x % 6 == 0 && y % 6 == 0) {
		return 81 * (x / 6 - 2) * (y / 6 - 2);
	}
	else
		return 81 * (x / 6 - 1) * (y / 6 - 1);
}


int main(int argc, char** argv) {
	//変数宣言
	//	int x, y;

	int file_num = 1;

	int count = 0;
	int hog_dim;

	//検出器の取り込み
	if ((CD = svm_load_model("C:/model_file/pre_model/CD_123.model")) == 0)exit(1);


	//テスト画像ファイル一覧メモ帳読み込み
	char test_name[1024], result_name[1024];
	FILE *test_data, *result_data;
	if (fopen_s(&test_data, "c:/photo/test_list.txt", "r") != 0) {
		cout << "missing" << endl;
		return 0;
	}

	while (fgets(test_name, 256, test_data) != NULL) {
		string name_tes = test_name;
		char new_test_name[1024];
		for (int i = 0; i < name_tes.length() - 1; i++) {
			new_test_name[i] = test_name[i];
			new_test_name[i + 1] = '\0';
		}
		count = 0;

		char test_path[1024] = "C:/photo/test_data_from_demo/test_data/";
		strcat_s(test_path, new_test_name);

		for (int i = 0; i < 1024; i++) {
			if (new_test_name[i] == 'b') {
				new_test_name[i] = 't';
				new_test_name[i + 1] = 'x';
				new_test_name[i + 2] = 't';
				new_test_name[i + 3] = '\0';
				break;
			}
			else new_test_name[i] = new_test_name[i];
		}
		char result_path[1024] = "result_data/";
		strcat_s(result_path, new_test_name);

		if (fopen_s(&result_data, result_path, "a") != 0) {
			cout << "missing 2" << endl;
			return 0;
		}


		//画像の取り込み
		cv::Mat ans_img_CF = cv::imread(test_path, 1);	//検出する画像
														//		cv::Mat ans_img_CF = cv::imread("Sun_Nov_26_14_02_00_95.bmp", 1);	//検出する画像
		cv::Mat check_img = ans_img_CF.clone();

		cout << file_num << ":" << new_test_name << endl;
		file_num++;

		//	cv::imshow("", ans_img_CF);
		//	cvWaitKey(0);

		//Detect_Placeオブジェクトの作成
		Detect_Place detect[500];

		//Coarse Detectorによる人物検出
		//	cv::Mat CD_img[300];

		float normalize_num[10] = { 128,192,256,320,-1 };
		//		float normalize_num[10] = { 96, 144, 192, 240, 288, 336,-1 };


		for (int img_size = 0; normalize_num[img_size] != -1; img_size++) {
			cv::Mat img;			//検出矩形処理を施す画像
			cvtColor(ans_img_CF, img, CV_RGB2GRAY);
			cv::resize(img, img, cv::Size(), normalize_num[img_size] / img.rows, normalize_num[img_size] / img.rows, CV_INTER_LINEAR);
			for (int y = 0; (y + 128) <= img.rows; y += 8) {
				for (int x = 0; (x + 128) <= img.cols; x += 8) {
					cv::Mat d_im(img, cv::Rect(x, y, 128, 128));
					cv::resize(d_im, d_im, cv::Size(), 96.0 / d_im.cols, 96.0 / d_im.rows);

					hog_dim = dimension(d_im.cols, d_im.rows);
					float hog_vector[20000];						//各次元のHOGを格納
					get_HOG(d_im, hog_vector);	//HOGの取得
					double ans = predict(hog_vector, hog_dim, CD);	//尤度の算出
					if (ans >= YUDO_CD) {//尤度から人物か非人物かの判断
										 //		cout << ans << endl;

						detect[count].C_yudo = ans;
						detect[count].C_x = x * 480 / normalize_num[img_size];
						detect[count].C_y = y * 480 / normalize_num[img_size];
						detect[count].C_width = 128 * 480 / normalize_num[img_size];
						detect[count].C_height = 128 * 480 / normalize_num[img_size];
						detect[count].ratio_num = img_size;
						//		CD_img[count] = img.clone();
						//		CD_img[count] = CD_img[count](cv::Rect(x, y, 128, 128));
						//		check_img = draw_rectangle(check_img, detect[count].C_x, detect[count].C_y, detect[count].C_width, detect[count].C_height, 255, 0, 0);
						count++;
					}
				}
			}
		}

		//	cv::imshow("", check_img);
		//	cvWaitKey(0);

		//領域の統一
		int t_num = 0;
		for (int n = 0; detect[n].C_yudo != 0; n++) {
			if (detect[n].territory_num == -1) {
				t_num++;
				detect[n].territory_num = t_num;
			}
			for (int m = n + 1; detect[m].C_yudo != 0; m++) {
				if ((detect[n].C_x + detect[n].C_width / 2) - 50 <= (detect[m].C_x + detect[m].C_width / 2)
					&& (detect[m].C_x + detect[m].C_width / 2) <= (detect[n].C_x + detect[n].C_width / 2) + 50
					&&
					(detect[n].C_y + detect[n].C_height / 2) - 50 <= (detect[m].C_y + detect[m].C_height / 2)
					&& (detect[m].C_y + detect[m].C_height / 2) <= (detect[n].C_y + detect[n].C_height / 2) + 50) {

					detect[m].territory_num = detect[n].territory_num;
				}
			}

		}

		Detect_Place Det_Fin[20];
		int Fin_count = 0;
		//統一領域ごとに最大尤度を算出
		for (int i = 1; i <= t_num; i++) {
			int final_num = 0;
			float cyudo = 0;
			int area = 0;
			for (int k = 0; detect[k].C_yudo != 0; k++) {
				if (detect[k].territory_num == i && detect[k].C_yudo > cyudo) {
					final_num = k;
					cyudo = detect[k].C_yudo;
				}
			}
			Det_Fin[Fin_count] = detect[final_num];
			Det_Fin[Fin_count].territory_num = -1;
			Fin_count++;
		}
		//最大領域をさらに統一
		t_num = 0;
		for (int n = 0; Det_Fin[n].C_yudo != 0; n++) {
			if (Det_Fin[n].territory_num == -1) {
				t_num++;
				Det_Fin[n].territory_num = t_num;
			}
			for (int m = n + 1; Det_Fin[m].C_yudo != 0; m++) {
				if ((Det_Fin[n].C_x + Det_Fin[n].C_width / 2) - 50 <= (Det_Fin[m].C_x + Det_Fin[m].C_width / 2)
					&& (Det_Fin[m].C_x + Det_Fin[m].C_width / 2) <= (Det_Fin[n].C_x + Det_Fin[n].C_width / 2) + 50
					&&
					(Det_Fin[n].C_y + Det_Fin[n].C_height / 2) - 50 <= (Det_Fin[m].C_y + Det_Fin[m].C_height / 2)
					&& (Det_Fin[m].C_y + Det_Fin[m].C_height / 2) <= (Det_Fin[n].C_y + Det_Fin[n].C_height / 2) + 50) {

					Det_Fin[m].territory_num = Det_Fin[n].territory_num;
				}
			}

		}

		for (int i = 1; i <= t_num; i++) {
			int final_num = 0;
			float cyudo = 0;
			int area = 0;
			for (int k = 0; Det_Fin[k].C_yudo != 0; k++) {
				if (Det_Fin[k].territory_num == i && Det_Fin[k].C_yudo > cyudo) {
					final_num = k;
					cyudo = Det_Fin[k].C_yudo;
				}
			}
			fprintf_s(result_data, "%f", Det_Fin[final_num].C_yudo);
			fprintf_s(result_data, "\n");
			fprintf_s(result_data, "%d", Det_Fin[final_num].C_x);
			fprintf_s(result_data, "\n");
			fprintf_s(result_data, "%d", Det_Fin[final_num].C_y);
			fprintf_s(result_data, "\n");
			fprintf_s(result_data, "%d", Det_Fin[final_num].C_width);
			fprintf_s(result_data, "\n");
			fprintf_s(result_data, "%d", Det_Fin[final_num].C_height);
			fprintf_s(result_data, "\n");

			printf("%f, %d, %d, %d, %d\n", Det_Fin[final_num].C_yudo, Det_Fin[final_num].C_x, Det_Fin[final_num].C_y, Det_Fin[final_num].C_width, Det_Fin[final_num].C_height);
			//	ans_img_CF = draw_rectangle(ans_img_CF, Det_Fin[final_num].C_x, Det_Fin[final_num].C_y, Det_Fin[final_num].C_width, Det_Fin[final_num].C_height, 255, 0, 0);
		}
		//	cv::imshow("", ans_img_CF);
		//	cvWaitKey(0);
		fclose(result_data);
	}

	fclose(test_data);

	return 0;
}

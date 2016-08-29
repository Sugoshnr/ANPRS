//Profiling-top/bottom/left/right
//Generate Training dataset - top_tlbr.txt
#include<iostream>
#include "opencv2/opencv.hpp"
#include <math.h>
#include <fstream>
using namespace std;
using namespace cv;
#define pi 3.14
int limit=100;
cv::Size size(800, 800);
int c1[100][100], c2[100][100], z = 0, v = 1, q = 1, top = -1, top2, top3, top4, sum1 = 0;
int top_all[4];
Mat plot_img = Mat::zeros(100, 100, CV_8UC3);
Vec3b c;
float a1 = 1, a2 = 0, a3 = 0;
//void plot1(int a, int b, Mat x,int flag)
//{
//	Vec3b &color = x.at<Vec3b>(a, b);
//	if (flag == 1)
//	{
//		color.val[0] = 0;
//		color.val[1] = 0;
//		color.val[2] = 255;
//	}
//	if (flag == 2)
//	{
//		color.val[0] = 0;
//		color.val[1] = 255;
//		color.val[2] = 0;
//	}
//	if (flag == 3)
//	{
//		color.val[0] = 255;
//		color.val[1] = 0;
//		color.val[2] = 0;
//	}
//	if (flag == 4)
//	{
//		color.val[0] = 255;
//		color.val[1] = 0;
//		color.val[2] = 255;
//	}
//}
int check(float ang, string& val_bin)
{
	int val;
	if (ang <= 90 && ang >= 67.5){ val = 0, val_bin = "000"; }
	else if (ang <= 67.5 && ang >= 45){ val = 1, val_bin = "001"; }
	else if (ang <= 45 && ang >= 22.5){ val = 2, val_bin = "010"; }
	else if (ang <= 22.5 && ang >= 0){ val = 3, val_bin = "011"; }
	else if (ang <= 0 && ang >= -22.5){ val = 4, val_bin = "100"; }
	else if (ang <= -22.5 && ang >= -45){ val = 5, val_bin = "101"; }
	else if (ang <= -45 && ang >= -67.5){ val = 6, val_bin = "110"; }
	else if (ang <= -67.5 && ang >= -90){ val = 7, val_bin = "111"; }
	return val;
}
int max(int a[])
{
	int x = 0, ind = 0;
	for (int i = 0; i <= 9; i++)
	{
		if (a[i] > x){
			x = a[i]; ind = i;
		}
	}
	return ind;
}
void push(int val, string val_bin, int values[],  int &topx)
{
	topx++;
	values[topx] = val;
}
void profile(int row[], int col[], int grad_row[][100], int grad_col[][100], int count, int len[][100], int values[][100][10], int ind, int c1, int &l)
{
	int k = 1, m = 1; l = 0;
	bool flag;
	if (c1 == 0 || c1 == 2)flag = true;
	else flag = false;
	//cout << grad_row[0][0] << endl;
	int slop1 = 99, slop2 = 99;
	if (flag)
	{
		grad_row[ind][l] = row[0];
		grad_col[ind][l] = col[0];
		//plot1(grad_up_row[q][l], grad_up_col[q][l], x, 1);
		l++;
		while (k < count - 2){
			int slope0 = row[k] - row[k - 1];
			int slope1 = row[k + 1] - row[k];
			int slope2 = row[k + 2] - row[k + 1];
			if ((slope0 > 0 && slope1 > 0) || (slope0 < 0 && slope1 < 0) || (slope0 == 0 && slope1 == 0))k++;
			else{
				if ((col[k] - grad_col[q][l - 1] <= 5 && row[k] - grad_row[q][l - 1] <= 5) || (slop1 == slope0 && slop2 == slope1))goto x1;
				grad_row[q][l] = row[k];
				grad_col[q][l] = col[k];
				slop1 = slope0; slop2 = slope1;
				//plot1(grad_up_row[q][l], grad_up_col[q][l], x, 1);
				m = k;
				l++;
			x1:	k++;
			}
		}
		grad_row[q][l] = row[count - 1];
		grad_col[q][l] = col[count - 1];
		//plot1(grad_up_row[q][l], grad_up_col[q][l], x, 1);
		l++;
		len[c1][ind] = l;
		for (int i = 0; i < l; i++){
			if (i < l - 1){
				float ang = 0;
				int row = grad_row[q][i + 1] - grad_row[q][i];
				int val;
				string val_bin = "";
				int col = grad_col[q][i] - grad_col[q][i + 1];
				if (row == 0){ ang = 0; val = 3; }
				else if (col == 0){
					ang = 90; val = 0;
				}
				else
				{
					float slopex = (float)row / (float)col;
					ang = atan(slopex);
					ang = (180 * ang) / pi;
					val = check(ang, val_bin);
				}
				push(val, val_bin, values[c1][q], top_all[c1]);
			}
			plot_img.at<Vec3b>(grad_row[q][i], grad_col[q][i]) = c;
		}
	}
	else
	{
		grad_row[ind][l] = row[0];
		grad_col[ind][l] = col[0];
		//plot1(grad_up_row[q][l], grad_up_col[q][l], x, 1);
		l++;
		while (k < count - 2){
			int slope0 = col[k] - col[k - 1];
			int slope1 = col[k + 1] - col[k];
			int slope2 = col[k + 2] - col[k + 1];
			if ((slope0 > 0 && slope1 > 0) || (slope0 < 0 && slope1 < 0) || (slope0 == 0 && slope1 == 0))k++;
			else{
				if ((col[k] - grad_col[q][l - 1] <= 5 && row[k] - grad_row[q][l - 1] <= 5) || (slop1 == slope0 && slop2 == slope1))goto x2;
				grad_row[q][l] = row[k];
				grad_col[q][l] = col[k];
				slop1 = slope0; slop2 = slope1;
				//plot1(grad_up_row[q][l], grad_up_col[q][l], x, 1);
				m = k;
				l++;
			x2:	k++;
			}
		}
		grad_row[q][l] = row[count - 1];
		grad_col[q][l] = col[count - 1];
		//plot1(grad_up_row[q][l], grad_up_col[q][l], x, 1);
		l++;
		len[c1][ind] = l;
		for (int i = 0; i < l; i++){
			if (i < l - 1){
				float ang = 0;
				int col = grad_row[q][i + 1] - grad_row[q][i];
				int val;
				string val_bin = "";
				int row = grad_col[q][i] - grad_col[q][i + 1];
				if (row == 0){ ang = 0; val = 3; }
				else if (col == 0){
					ang = 90; val = 0;
				}
				else
				{
					float slopex = (float)row / (float)col;
					ang = atan(slopex);
					ang = (180 * ang) / pi;
					if (c1 == 3)ang *= -1;
					val = check(ang, val_bin);
				}
				push(val, val_bin, values[c1][q], top_all[c1]);
			}
			plot_img.at<Vec3b>(grad_row[q][i], grad_col[q][i]) = c;
		}
	}
}
int main(int argc, char** argv)
{
	//Read
	c.val[0] = 255;
	c.val[1] = 0;
	c.val[2] = 0;
	z = 0;
	ofstream myfile;
	ofstream myfile2;
	myfile.open("avg_lrtb.txt");
	myfile2.open("train_data.txt");
	float val_avg[40][7], key_avg[5] = {};//key_avg=avg keypoints of top/..
	for (int num = 0; num < 36; num++)
	{
		int key_len[100][100] = {}, key_ind = 0, key_sum[5] = {};//key_len=list of keypoints top/..
		int values[4][100][10] = {};//values[top/..][q_sample][list_of_values]
		cout << num << endl;
		int max_top = 0, min_top = 99;
		int val_a[100][100] = {};
		int lim = 100;
		for (q = 0; q < 40; q++)
		{
			top = -1 , top_all[0] = -1; top_all[1] = -1; top_all[2] = -1; top_all[3] = -1;
			string values_bin[100];
			Mat X[100];
			Mat img1;
			//Mat img1 = imread("0.jpg");
			//if (num < 19 || num == 21)
				//img1 = imread("C:\\Users\\Sugosh\\Documents\\Visual Studio 2013\\Projects\\FirstOpenCV\\FirstOpenCV\\Character_Dataset\\Final2\\1 (" + to_string(num + 1) + ")\\" + to_string(q) + " (1).jpg");
			//else
				//img1 = imread("C:\\Users\\Sugosh\\Documents\\Visual Studio 2013\\Projects\\FirstOpenCV\\FirstOpenCV\\Character_Dataset\\Final2\\1 (" + to_string(num + 1) + ")\\" + to_string(q) + ".jpg");
			img1 = imread("C:\\Users\\Sugosh\\Documents\\Visual Studio 2013\\Projects\\FirstOpenCV\\FirstOpenCV\\Character_Dataset\\Final\\" + to_string(num) + "\\1 (" + to_string(q + 1) + ").jpg");
			c.val[0] = a1 * 255;
			c.val[1] = a2 * 255;
			c.val[2] = a3 * 255;
			Mat out2;
			resize(img1, out2, Size(21, 28));
			threshold(out2, img1, 125, 255, THRESH_BINARY);
			Mat out1 = img1.clone();
			cvtColor(out1, out1, CV_RGB2GRAY);
			int count1[1000], count2[1000];
			int k = 0;
			int a, count_1 = 0, count_2 = 0, count_3 = 0, count_4 = 0;
			int i, j, count = 0, upper_row[1000], upper_col[1000], lower_row[1000], lower_col[1000],
				left_row[1000], left_col[1000], right_row[1000], right_col[1000],
				grad_up_row[100][100] = {}, grad_up_col[100][100] = {}, grad_bot_row[100][100] = {}, grad_bot_col[100][100] = {},
				grad_left_row[100][100], grad_left_col[100][100], grad_right_row[100][100], grad_right_col[100][100];
			int slope = 0;
			float slopex[100][200];
			//imshow("..", out1);
			//cout << "upper" << endl;
			//Mat ele = (Mat_<uchar>(3, 3) << 1, 1, 1, 1, 0, 1, 1, 1, 1);
			copyMakeBorder(out1, out1, 4, 4, 4, 4, BORDER_CONSTANT, Scalar(0));
			Mat ele = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
			Mat ele1 = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
			copyMakeBorder(img1, img1, 4, 4, 4, 4, BORDER_CONSTANT, Scalar(0));
			Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
			//morphologyEx(out1, out1, CV_MOP_ERODE, element);
			morphologyEx(out1, out1, CV_MOP_DILATE, ele1);
			morphologyEx(out1, out1, CV_MOP_ERODE, element);
			morphologyEx(out1, out1, CV_MOP_DILATE, ele);
			//morphologyEx(out1, out1, CV_MOP_ERODE, element);
			morphologyEx(out1, out1, CV_MOP_DILATE, ele1);
			//imwrite(to_string(q + 40) + ".jpg", out1);
			imwrite("x.jpg", out1);
			Mat x = imread("x.jpg");
			for (j = 0; j < out1.cols; j++)
			{
				for (i = 0; i < out1.rows; i++)
				{
					a = out1.at<uchar>(i, j);
					if (a == 255)break;
				}
				if (a == 255){
					count1[k] = j;
					upper_row[k] = i;
					upper_col[k] = j;
					k++;
					count_1++;
				}
			}
			k = 0;
			for (j = 0; j < out1.cols; j++)
			{
				for (i = out1.rows - 1; i >= 0; i--)
				{
					a = out1.at<uchar>(i, j);
					if (a == 255)break;
				}
				if (a == 255){
					count2[k] = 400 - i;
					lower_row[k] = i;
					lower_col[k] = j;
					k++;
					count_2++;
				}
			}
			k = 0;
			for (i = 0; i < out1.rows; i++)
			{
				for (j = 0; j < out1.cols; j++)
				{
					a = out1.at<uchar>(i, j);
					if (a == 255)break;
				}
				if (a == 255){
					count1[k] = j;
					left_row[k] = i;
					left_col[k] = j;
					////plot1(upper_row[k], upper_col[k], x);
					k++;
					count_3++;
				}
			}
			k = 0;
			for (i = 0; i < out1.rows; i++)
			{
				for (j = out1.cols - 1; j >= 0; j--)
				{
					a = out1.at<uchar>(i, j);
					if (a == 255)break;
				}
				if (a == 255){
					count2[k] = 400 - i;
					right_row[k] = i;
					right_col[k] = j;
					////plot1(lower_row[k], lower_col[k], x);
					k++;
					count_4++;
				}
			}
			int l = 0;
			profile(upper_row, upper_col, grad_up_row, grad_up_col, count_1, key_len,values,q, 0, l);
			profile(left_row, left_col, grad_left_row, grad_left_col, count_3, key_len, values,q, 1, l);
			profile(lower_row, lower_col, grad_bot_row, grad_bot_col, count_2, key_len,values,q, 2, l);
			profile(right_row, right_col, grad_right_row, grad_right_col, count_4, key_len,values, q, 3, l);
			for (int i = 0; i <= 3; i++)
			{
				for (int j = 0; j < key_len[i][q] - 1; j++)
				{
					myfile2 << values[i][q][j];
				}
				myfile2 << " ";
				//cout << " ";
			}
			myfile2 << endl;
			resize(x, x, Size(100, 100));
			//imwrite("C:\\Users\\Sugosh\\Documents\\Visual Studio 2013\\Projects\\FirstOpenCV\\FirstOpenCV\\A\\" + to_string(num + 25) + "(" + to_string(q + 1) + ").jpg", x);
			//cout << endl << endl;
			switch (q){
			case 2:a1 = 0; a2 = 1; a3 = 0; break;
			case 3:a1 = 0; a2 = 0; a3 = 1; break;
			case 4:a1 = 1; a2 = 1; a3 = 0; break;
			case 5:a1 = 0; a2 = 1; a3 = 1; break;
			case 6:a1 = 1; a2 = 0; a3 = 1; break;
			case 7:a1 = 1; a2 = 1; a3 = 1; break;
			case 8:a1 = 0; a2 = 0.5; a3 = 0; break;
			case 9:a1 = 0.5; a2 = 0; a3 = 0; break;
			}
		}//q loop
	}//num loop
	myfile.close();
	myfile2.close();
	cv::imshow("", plot_img);
	cvWaitKey();
	return 0;
}
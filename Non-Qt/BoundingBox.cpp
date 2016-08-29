#include<iostream>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
cv::Size size(800, 800);
int count1 = 0;
Mat rotate(Mat src, double angle)
{
	Mat dst;
	Point2f pt(src.cols / 2., src.rows / 2.);
	Mat r = getRotationMatrix2D(pt, angle, 1.0);
	warpAffine(src, dst, r, Size(src.cols, src.rows));
	return dst;
}
std::vector<cv::Rect> detectLetters(cv::Mat img)
{
	vector<cv::Rect> boundRect;
	Mat img_gray, img_sobel, img_threshold, element;
	GaussianBlur(img, img, cv::Size(7, 7), 0, 0);
	cvtColor(img, img_gray, CV_BGR2GRAY);
	Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
	//Canny(img_gray, img_sobel, 90, 90 * 3, 3);
	threshold(img_sobel, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	//imshow("x", img_threshold);
	element = getStructuringElement(cv::MORPH_RECT, cv::Size(17, 3));
	morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element); //Does the trick
	//imshow("y", img_threshold);
	vector< std::vector< cv::Point> > contours;
	findContours(img_threshold, contours, 0, 1);
	vector<std::vector<cv::Point> > contours_poly(contours.size());
	for (int i = 0; i < contours.size(); i++)
		if (contours[i].size()>100)
		{
		cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
		cv::Rect appRect(boundingRect(cv::Mat(contours_poly[i])));
		if (appRect.width>2*appRect.height)
			boundRect.push_back(appRect);
		}
	return boundRect;
}
int main(int argc, char** argv)
{
	for (int x = 1; x <= 200; x++)
	{
		//Read
		//if (x == 12)continue;
		Mat X[100];
		//cv::Mat img1 = cv::imread(to_string(x) + ".jpg");
		Mat img1 = imread("C:\\Users\\Sugosh\\Documents\\Visual Studio 2013\\Projects\\FirstOpenCV\\FirstOpenCV\\Datasets\\DS\\1 (" + to_string(x) + ").jpg");
		//Mat img1 = imread("a.jpg");
		//if (img1.cols > img1.rows){
			//resize(img1, img1, Size(img1.cols, img1.rows)); imshow("..", img1); 
			//resize(img1, img1, size);
			//img1 = rotate(img1, -90);
		//}
		resize(img1, img1, size);
		//imshow("", img1);
		std::vector<cv::Rect> letterBBoxes1 = detectLetters(img1);
		for (int i = 0; i < letterBBoxes1.size(); i++){
			//cout << letterBBoxes1[i].x << " " << letterBBoxes1[i].y << " " << letterBBoxes1[i].width << " " << letterBBoxes1[i].width << endl;
			X[i] = img1(letterBBoxes1[i]).clone();
			cv::rectangle(img1, letterBBoxes1[i], cv::Scalar(0, 255, 0), 3, 8, 0);
			//imshow("1", img1);
			//imwrite("C:\\Users\\Sugosh\\Desktop\\DS1\\1 " + to_string(x) + ".jpg", img1);
		}
		cout << x << endl;
		//imshow("", img1);
		//HERE
		int index = -1, max = 0;
		Mat temp;
		for (int i = 0; i < letterBBoxes1.size(); i++){
			int count = 0;
			temp = X[i].clone();
			Canny(X[i], temp, 90, 90 * 3, 3);
			//imshow(to_string(i), X[i]);
			for (int j = 0; j < letterBBoxes1[i].height; j++){
				for (int k = 0; k < letterBBoxes1[i].width; k++){
					if (temp.at<uchar>(j, k) == 255)count++;
				}
			}
			//cout << i << " " << count << endl;
			if (count > max){
				max = count; index = i;
			}
			//imshow(to_string(index + 10), X[index]);
		}
		cout << "index=" << index << endl;
		if (index == -1){
			continue; count1++;
		}
		//imshow("0", X[index]);
		Mat a = X[index].clone();
		//imshow(to_string(x), X[index]);
		///imshow("Original", a);
		cvtColor(X[index], X[index], CV_RGB2GRAY);
		//imshow("Final1", X[index]);
		//threshold(X[index], X[index], 80, 255, THRESH_BINARY_INV);
		adaptiveThreshold(X[index], X[index], 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 3, 1);
		//imshow("Final2", X[index]);

		vector< std::vector< cv::Point> > contours;
		Mat temp1 = X[index].clone();
		findContours(temp1, contours, 0, 1);
		vector<cv::Rect> boundRect;
		vector<std::vector<cv::Point> > contours_poly(contours.size());
		///cout << contours.size() << endl;
		for (int i = 0; i < contours.size(); i++)
		{
			///cout << contours[i].size() << endl;
			if (contours[i].size() > 20)
			{
				cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
				cv::Rect appRect(boundingRect(cv::Mat(contours_poly[i])));
				boundRect.push_back(appRect);
			}
		}
		Mat X1[100];
		if (boundRect.size() > 50)continue;
		for (int i = 0; i < boundRect.size(); i++){
			///cout << boundRect[i].x << " " << boundRect[i].y << " " << boundRect[i].height << " " << boundRect[i].width << endl;
			if (boundRect[i].height > boundRect[i].width)
				{
					X1[i] = a(boundRect[i]).clone();
					//rectangle(a, boundRect[i], cv::Scalar(0, 255, 0), 3, 8, 0);
					cvtColor(X1[i], X1[i], CV_RGB2GRAY);
					//threshold(X1[i], X1[i], 100, 255, THRESH_BINARY_INV);
					adaptiveThreshold(X1[i], X1[i], 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 3, 1);
					imwrite("C:\\Users\\Sugosh\\Desktop\\DS4\\a\\" + to_string(x) + "(" + to_string(i) + ").jpg", X1[i]);
					//imshow(to_string(i), X1[i]);
					//imwrite(to_string(i) + ".jpg", X1[i]);
				}
		}
		
		//cout <<"x="<< x << endl;
		///imshow("Original1", a);
	}
	//cout << count1 << endl;
	//cin >> count1;
		//cv::imshow("imgOut1.jpg", img1);
	cvWaitKey();
	return 0;
}
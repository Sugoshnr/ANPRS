#include "anpr.h"
#include "ui_anpr.h"
#include<QFileDialog>
#include<QtCore>
#include"textloc.h"
#include"textseg.h"
#include "preproc.h"
int choice_img=-1;
cv::Size size(800, 800);
#define pi 3.14
int limit = 100;
int c1[100][100], c2[100][100], z = 0, v = 1, top = -1, top2, top3, top4, sum1 = 0;
int top_all[4];
Vec3b c;
float a1 = 1, a2 = 0, a3 = 0;
float grad_val1[2600][4] = {};
float grad_val2[1300][4] = {};
int tot_ind=0;
//Mat temp_full,temp_plate;
void sort(int centroid[], int sval[],int sind[],int ind)
{
    for (int x = 0; x<ind; x++)
    {
        for (int y = 0; y<ind - 1; y++)
        {
            if (centroid[y]>centroid[y + 1])
            {
                int temp = centroid[y + 1];
                centroid[y + 1] = centroid[y];
                centroid[y] = temp;
            }
        }
    }
    for (int i = 0; i < ind; i++)
    {
        for (int j = 0; j < ind; j++)
        {
            if (sval[i] == centroid[j]){ sind[i] = j; break; }
        }
    }
}
void SVM_Test_Grad(int v1)
{
    ofstream a1, b1;
    a1.open("C:\\Users\\Sugosh\\License_Plate.txt");
    float labels[2600] = {}, k = -1;
    string val[100];
    for (int i = 0; i < 1440; i++)
    {
        if ((i % 40) == 0)k++;
        labels[i] = k;
    }
    int ind = 0;
    ind = 0;
    Mat trainingDataMat(1440, 4, CV_32FC1, grad_val2);
    Mat labelsMat(1440, 1, CV_32FC1, labels);
    CvSVMParams params2;
    params2.svm_type = CvSVM::C_SVC;
    //params2.gamma = 1e-6;
    //params2.C = 10;
    params2.kernel_type = CvSVM::RBF;
    params2.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
    // Train the SVM
    CvSVM SVM2;
    SVM2.train(trainingDataMat, labelsMat, Mat(), Mat(), params2);
    ind = 0;
    for (int i = 0; i <= v1; i++)
    {
        float a = grad_val1[i][0], b = grad_val1[i][1], c = grad_val1[i][2], d = grad_val1[i][3];
        Mat sampleMat = (Mat_<float>(1, 4) << a, b, c, d);
        float response2 = SVM2.predict(sampleMat);
        char ch;
        if (response2 >= 10){
            ch = 55 + response2;
            val[i] = ch;
        }
        else
        {
            val[i] = to_string((long long int)response2);
        }
    }
    for (int i = 0; i <=v1; i++)
    {
        cout << val[i];
        a1 << val[i];
    }
    cout << endl;
    a1.close();

}

void Test_Data_Grad()
{
    ifstream myfile;
    myfile.open("C:\\Users\\Sugosh\\Downloads\\train_data.txt");
    std::string line;
    int key_values[36][50][4][7] = {};//key_values[0..36][sample][0..3][values]
    double length[36][100][4] = {};
    int pos2 = 0, n = 0;
    while (getline(myfile, line))
    {

        int qq = 0, pos1 = 0, k1 = 0;
        int x=line.size()-1;
        int n1=0;
        while (n1 < x+1)
        {
            char a = line[k1];
            if (a == ' '){
                pos1++; qq = 0;
            }
            else
            {
                int z = a - '0';
                key_values[n][pos2][pos1][qq] = z;
                length[n][pos2][pos1]++;
                qq++;
            }
            k1++;
            n1++;
        }
        if (pos2 == 39){ pos2 = -1; n++; }
        pos2++;
    }
    int z = 0;
    for (int n = 0; n < 36; n++)
    {
        for (int q = 0; q < 40; q++)
        {
            for (int i = 0; i <= 3; i++)
            {
                double num1 = 1, sum = 0;
                for (int j = 0; j <= length[n][q][i] - 1; j++)
                {
                    num1 = pow(10.0,(length[n][q][i] - 1 * (j + 1))) * key_values[n][q][i][j];
                    sum += num1;
                }
                grad_val2[z][i] = sum;
            }
            z++;
        }
    }
}

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
void push(int val, string val_bin, int values[], int &topx)
{
    topx++;
    values[topx] = val;
}
//PROFILING
void profile(int row[], int col[], int grad_row[][100], int grad_col[][100], int count, double len[][100], int values[][100][10], int ind, int c1, int &l, int q)
{
    int k = 1, m = 1; l = 0;
    bool flag;
    if (c1 == 0 || c1 == 2)flag = true;
    else flag = false;
    int slop1 = 99, slop2 = 99;
    if (flag)
    {
        grad_row[ind][l] = row[0];
        grad_col[ind][l] = col[0];
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
                m = k;
                l++;
            x1:	k++;
            }
        }
        grad_row[q][l] = row[count - 1];
        grad_col[q][l] = col[count - 1];
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
        }
    }
    else
    {
        grad_row[ind][l] = row[0];
        grad_col[ind][l] = col[0];
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
                m = k;
                l++;
            x2:	k++;
            }
        }
        grad_row[q][l] = row[count - 1];
        grad_col[q][l] = col[count - 1];
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
        }
    }
}
//LOCALIZATION
std::vector<cv::Rect> detectLetters(cv::Mat img)
{
    vector<cv::Rect> boundRect;
    Mat img_gray, img_sobel, img_threshold, element;
    GaussianBlur(img, img, cv::Size(7, 7), 0, 0);
    imwrite("blur.jpg",img);
    cvtColor(img, img_gray, CV_BGR2GRAY);
    imwrite("gray.jpg",img_gray);
    Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    imwrite("sobel.jpg",img_sobel);
    //Canny(img_gray, img_sobel, 90, 90 * 3, 3);
    threshold(img_sobel, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
    imwrite("thresh.jpg",img_threshold);
    element = getStructuringElement(cv::MORPH_RECT, cv::Size(17, 3));
    morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element);
    imwrite("top_hat.jpg",img_threshold);
    vector< std::vector< cv::Point> > contours;
    findContours(img_threshold, contours, 0, 1);
    vector<std::vector<cv::Point> > contours_poly(contours.size());
    for (int i = 0; i < contours.size(); i++)
        if (contours[i].size()>100)
        {
        cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
        cv::Rect appRect(boundingRect(cv::Mat(contours_poly[i])));
        if (appRect.width>2 * appRect.height)
            boundRect.push_back(appRect);
        }
    return boundRect;
}
//////////////////////////////////////////////////////////////////////////


QImage ANPR::convertOpenCVMatToQtQImage(cv::Mat mat) {
    if(mat.channels() == 1) {                   // if grayscale image
        return QImage((uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);     // declare and return a QImage
    } else if(mat.channels() == 3) {            // if 3 channel color image
        cv::cvtColor(mat, mat, CV_BGR2RGB);     // invert BGR to RGB
        return QImage((uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);       // declare and return a QImage
    } else {
        qDebug() << "in convertOpenCVMatToQtQImage, image was not 1 channel or 3 channel, should never get here";
    }
    return QImage();        // return a blank QImage if the above did not work
}

//////////////////////////////////////////////////////////////////////////


ANPR::ANPR(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::ANPR)
{
    ui->setupUi(this);
}

ANPR::~ANPR()
{
    delete ui;
}

void ANPR::on_openfile_clicked()
{

    QString strFileName = QFileDialog::getOpenFileName();

       if(strFileName == "") {
           ui->imgpath->setText("file not chosen");
           return;
       }
       Mat imgOriginal;            // input image
       imgOriginal = imread(strFileName.toStdString());
       Mat img_temp=imgOriginal.clone();
       int w = ui->original_pic->width();
       int h = ui->original_pic->height();
       cv::resize(imgOriginal,imgOriginal,Size(w,h));
       if (imgOriginal.empty()) {
           ui->imgpath->setText("error: image not read from file");
           return;
       }
       ui->imgpath->setText(strFileName);
       ANPR::path=strFileName;
      //Mat img_temp=imgOriginal.clone();
       QImage qimgOriginal = convertOpenCVMatToQtQImage(imgOriginal);
       ui->original_pic->setPixmap(QPixmap::fromImage(qimgOriginal));
       {
           c.val[0] = 255;
               c.val[1] = 0;
               c.val[2] = 0;
               z = 0;
               ofstream myfile;
               Test_Data_Grad();
               myfile.open("test_values1.txt");
               double key_len[100][100] = {}, key_ind = 0, key_sum[5] = {};//key_len=list of keypoints top/..
               int values[4][100][10] = {};//values[top/..][q_sample][list_of_values]
               int max_top = 0, min_top = 99;
               int val_a[100][100] = {};
               int lim = 100;
               float val_avg[40][7], key_avg[5] = {};//key_avg=avg keypoints of top/..
               for (int choice = 0; choice <= 0; choice++)
                   //while (1)
               {
                   Mat X[100];
                   Mat img1=img_temp.clone();
                   cv::resize(img1, img1, Size(800,800));
                   //imshow("Original Image ",img1);
                   //imshow("", img1);
                   std::vector<cv::Rect> letterBBoxes1 = detectLetters(img1);
                   for (int i = 0; i < letterBBoxes1.size(); i++){
                       X[i] = img1(letterBBoxes1[i]).clone();
                       cv::rectangle(img1, letterBBoxes1[i], cv::Scalar(0, 255, 0), 3, 8, 0);
                       //imshow("Text Localization", img1);
                       //moveWindow("Text Localization", 800, 0);
                   }
                   imwrite("loc_car.jpg",img1);
                   int index = -1, max = 0;
                   Mat temp;
                   for (int i = 0; i < letterBBoxes1.size(); i++){
                       int count = 0;
                       temp = X[i].clone();
                       Canny(X[i], temp, 90, 90 * 3, 3);
                       for (int j = 0; j < letterBBoxes1[i].height; j++){
                           for (int k = 0; k < letterBBoxes1[i].width; k++){
                               if (temp.at<uchar>(j, k) == 255)count++;
                           }
                       }
                       if (count > max){
                           max = count; index = i;
                       }
                   }
                   Mat a = X[index].clone();
                   imwrite("loc_plate.jpg",a);
                   //imshow("Plate",a);
                   //moveWindow("Plate", 100, 100);
                   Mat seg=a.clone();
                   cvtColor(X[index], X[index], CV_RGB2GRAY);
                   adaptiveThreshold(X[index], X[index], 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 3, 1);
                   vector< std::vector< cv::Point> > contours;
                   Mat temp1 = X[index].clone();
                   findContours(temp1, contours, 0, 1);
                   vector<cv::Rect> boundRect;
                   vector<std::vector<cv::Point> > contours_poly(contours.size());
                   int ind_x = -1;
                   int centroid[50] = {};
                   for (int i = 0; i < contours.size(); i++)
                   {
                       if (contours[i].size() > 20)
                       {
                           cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
                           cv::Rect appRect(boundingRect(cv::Mat(contours_poly[i])));
                           if (appRect.height < 20 || (appRect.height < appRect.width))continue;
                           boundRect.push_back(appRect);
                           ind_x++;
                           centroid[ind_x] = appRect.x;
                       }
                   }
                   ind_x++;
                   int sval[50], sind[50] = {};
                   for (int i = 0; i < ind_x; i++)sval[i] = centroid[i];
                   sort(centroid, sval, sind, ind_x);
                   //for (int i = 0; i < ind_x; i++)cout << sind[i] << endl;
                   for(int i=0;i<ind_x;i++)
                   {
                       cv::rectangle(seg, boundRect[i], cv::Scalar(0, 255, 0), 3, 8, 0);
                   }
                   imwrite("seg_plate.jpg",seg);
                   //imshow("Text_Segmentation 2",seg);
                   //moveWindow("Text_Segmentation 2", 100, 200);
                   Mat X1[100];
                   int v1 = 0, zx = 0;
                   if (boundRect.size() > 50)continue;
                   for (int v = 0; v < boundRect.size(); v++){
                       if (boundRect[v].height > boundRect[v].width)
                       {
                           X1[zx] = a(boundRect[v]).clone();
                           cvtColor(X1[zx], X1[zx], CV_RGB2GRAY);
                           adaptiveThreshold(X1[zx], X1[zx], 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 3, 1);
                           //imshow(to_string(v+10), X1[zx]);
                           //moveWindow(to_string(v+10), 100 + (v * 75), 600);
                           //imwrite("C:\\Users\\Sugosh\\Documents\\Visual Studio 2013\\Projects\\FirstOpenCV\\FirstOpenCV\\Test\\" + to_string(choice) + " (" + to_string(zx) + ")" + ".jpg", X1[zx]);
                           v1 = zx;
                           zx++;
                       }
                   }
                   Mat temp_mat[15];
                   for (int i = 0; i < zx; i++)	temp_mat[i] = X1[i].clone();
                   for (int i = 0; i < zx; i++)	X1[sind[i]] = temp_mat[i];
                   for (int i = 0; i < zx; i++)
                   {
                       imwrite(to_string((long long int)i)+".jpg",X1[i]);
                       //imshow(to_string((long long int)i), X1[i]);
                       //moveWindow(to_string((long long int)i), 100 + (i * 75), 400);
                   }
                   for (int q = 0; q <= v1; q++)
                   {
                       top = -1, top_all[0] = -1; top_all[1] = -1; top_all[2] = -1; top_all[3] = -1;
                       string values_bin[100];
                       Mat img1 = X1[q].clone();
                       //Mat img1=imread(to_string(q) + ".jpg");
                       c.val[0] = a1 * 255;
                       c.val[1] = a2 * 255;
                       c.val[2] = a3 * 255;
                       Mat out2;
                       cv::resize(img1, out2, Size(21, 28));
                       //cvtColor(out2, out2, CV_RGB2GRAY);
                       threshold(out2, img1, 125, 255, THRESH_BINARY);
                       Mat out1 = img1.clone();
                       //imshow("Segmentation "+to_string((long long int)q+1),out1);
                       //cvtColor(out1, out1, CV_RGB2GRAY);
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
                       profile(upper_row, upper_col, grad_up_row, grad_up_col, count_1, key_len, values, q, 0, l, q);
                       profile(left_row, left_col, grad_left_row, grad_left_col, count_3, key_len, values, q, 1, l, q);
                       profile(lower_row, lower_col, grad_bot_row, grad_bot_col, count_2, key_len, values, q, 2, l, q);
                       profile(right_row, right_col, grad_right_row, grad_right_col, count_4, key_len, values, q, 3, l, q);
                       for (int i = 0; i <= 3; i++)
                       {
                           double num1 = 1; sum1 = 0;
                           for (int j = 0; j < key_len[i][q] - 1; j++)
                           {
                               num1 = pow(10.0,key_len[i][q] - 1 * (j + 1)) * values[i][q][j];
                               sum1 += num1;

                               //myfile << values[i][q][j];
                           }
                           sum1 /= 10;
                           grad_val1[q][i] = sum1;
                           myfile << grad_val1 << " ";
                       }
                       myfile << endl;
                   }
                   tot_ind=v1;
                       SVM_Test_Grad(v1);
               }
               myfile.close();
       }

}




void ANPR::on_textloc_clicked()
{
    TextLoc=new textloc(this);
    TextLoc->show();
}

void ANPR::on_textseg_clicked()
{
    TextSeg=new textseg(this);
    TextSeg->ind=tot_ind;
    TextSeg->show();
}

void ANPR::on_textrec_clicked()
{
    system("C:\\Users\\Sugosh\\License_Plate.txt");
}

void ANPR::on_preproc_lab_clicked()
{
    PreProc=new preproc(this);
    PreProc->show();
}

void ANPR::on_preproc_clicked()
{
    PreProc=new preproc(this);
    PreProc->show();

}

void ANPR::on_textloc_2_clicked()
{
    PreProc=new preproc(this);
    PreProc->show();
}

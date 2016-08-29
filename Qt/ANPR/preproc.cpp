#include "preproc.h"
#include "ui_preproc.h"
#include<QtCore>

preproc::preproc(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::preproc)
{
    ui->setupUi(this);
    Mat blur1=imread("blur.jpg");
    Mat gray1=imread("gray.jpg");
    Mat sobel1=imread("sobel.jpg");
    Mat thresh1=imread("thresh.jpg");
    Mat top_hat1=imread("top_hat.jpg");
    QImage q1,q2,q3,q4,q5;
    Mat temp;
    int w = ui->blur->width();
    int h = ui->blur->height();
    temp=blur1.clone();
    cv::resize(temp,temp,Size(w,h));
    q1 = convertOpenCVMatToQtQImage(temp);
    ui->blur->setPixmap(QPixmap::fromImage(q1));
    temp=gray1.clone();
    cv::resize(temp,temp,Size(w,h));
    q1 = convertOpenCVMatToQtQImage(temp);
    ui->gray->setPixmap(QPixmap::fromImage(q1));
    temp=sobel1.clone();
    cv::resize(temp,temp,Size(w,h));
    q1 = convertOpenCVMatToQtQImage(temp);
    ui->sobel->setPixmap(QPixmap::fromImage(q1));
    temp=thresh1.clone();
    cv::resize(temp,temp,Size(w,h));
    q1 = convertOpenCVMatToQtQImage(temp);
    ui->thresh->setPixmap(QPixmap::fromImage(q1));
    temp=top_hat1.clone();
    cv::resize(temp,temp,Size(w,h));
    q1 = convertOpenCVMatToQtQImage(temp);
    ui->top_hat->setPixmap(QPixmap::fromImage(q1));
}

preproc::~preproc()
{
    delete ui;
}

QImage preproc::convertOpenCVMatToQtQImage(cv::Mat mat) {
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

void preproc::on_pushButton_clicked()
{
    hide();
}

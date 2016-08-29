#include "textseg.h"
#include "ui_textseg.h"
#include<iostream>
#include<QtCore>
using namespace std;
textseg::textseg(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::textseg)
{
    ui->setupUi(this);
    Mat temp;
    Mat seg_plate=imread("seg_plate.jpg");
    int w = ui->segplate->width();
    int h = ui->segplate->height();
    cv::resize(seg_plate,seg_plate,Size(w,h));
    temp=seg_plate.clone();
    QImage Qseg = convertOpenCVMatToQtQImage(seg_plate);
    ui->segplate->setPixmap(QPixmap::fromImage(Qseg));
    Mat X[10];
    for(int i=0;i<=ind;i++)
    {
        X[i]=imread(to_string((long long int)i)+".jpg");
        imshow(to_string((long long int)i),X[i]);
    }
}

textseg::~textseg()
{
    delete ui;
}

void textseg::on_pushButton_clicked()
{
    Mat X[10];
    for(int i=0;i<=ind;i++)
    {
        X[i]=imread(to_string((long long int)i)+".jpg");
        imshow(to_string((long long int)i),X[i]);
        moveWindow(to_string((long long int)i),200+100*(i+1),450);
        //QImage Qseg1=convertOpenCVMatToQtQImage(X[i]);
        //QString x="l"+to_string((long long int)i+1);
        //ui->x->setPixmap(QPixmap::fromImage(Qseg1));
    }
}

QImage textseg::convertOpenCVMatToQtQImage(cv::Mat mat) {
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

void textseg::on_pushButton_2_clicked()
{
    hide();
}

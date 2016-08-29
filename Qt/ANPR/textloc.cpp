#include "textloc.h"
#include "ui_textloc.h"
#include<QtCore>
textloc::textloc(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::textloc)
{
    ui->setupUi(this);
    cv::Mat full_img,plate;
    QImage Qcar,Qplate;
    Mat temp;
    full_img=imread("loc_car.jpg");
    int w = ui->locimg->width();
    int h = ui->locimg->height();
    cv::resize(full_img,full_img,Size(w,h));
    plate=imread("loc_plate.jpg");
    w=ui->locplate->width();
    h=ui->locplate->height();
    cv::resize(plate,plate,Size(w,h));
    temp=full_img.clone();
    Qcar = convertOpenCVMatToQtQImage(full_img);
    temp=full_img.clone();
    Qplate = convertOpenCVMatToQtQImage(plate);
    ui->locimg->setPixmap(QPixmap::fromImage(Qcar));
    ui->locplate->setPixmap(QPixmap::fromImage(Qplate));
    //ui->locplate->setPixmap(QPixmap::fromImage(qplate));

}

textloc::~textloc()
{
    delete ui;
}

QImage textloc::convertOpenCVMatToQtQImage(cv::Mat mat) {
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




void textloc::on_pushButton_clicked()
{
    this->hide();
}

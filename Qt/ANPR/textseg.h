#ifndef TEXTSEG_H
#define TEXTSEG_H

#include <QDialog>
#include "opencv2/opencv.hpp"
using namespace cv;
namespace Ui {
class textseg;
}

class textseg : public QDialog
{
    Q_OBJECT

public:
    int ind;
    explicit textseg(QWidget *parent = 0);
    ~textseg();

private slots:
    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

private:
    Ui::textseg *ui;
    QImage textseg::convertOpenCVMatToQtQImage(cv::Mat mat);
};

#endif // TEXTSEG_H

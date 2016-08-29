#ifndef PREPROC_H
#define PREPROC_H
#include "opencv2/opencv.hpp"
using namespace cv;
#include <QDialog>

namespace Ui {
class preproc;
}

class preproc : public QDialog
{
    Q_OBJECT

public:
    explicit preproc(QWidget *parent = 0);
    ~preproc();

private slots:
    void on_pushButton_clicked();

private:
    Ui::preproc *ui;
    QImage preproc::convertOpenCVMatToQtQImage(cv::Mat mat);

};

#endif // PREPROC_H

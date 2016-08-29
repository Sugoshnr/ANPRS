#ifndef TEXTLOC_H
#define TEXTLOC_H

#include <QDialog>
#include "opencv2/opencv.hpp"
using namespace cv;
namespace Ui {
class textloc;
}

class textloc : public QDialog
{
    Q_OBJECT

public:

    explicit textloc(QWidget *parent = 0);
    ~textloc();

private slots:
    void on_pushButton_clicked();

private:
    Ui::textloc *ui;
    QImage textloc::convertOpenCVMatToQtQImage(cv::Mat mat);
};

#endif // TEXTLOC_H

#ifndef ANPR_H
#define ANPR_H

#include <QMainWindow>
#include "opencv2/opencv.hpp"
#include "textloc.h"
#include "textseg.h"
#include "preproc.h"
#include<iostream>
#include<fstream>
using namespace cv;
using namespace std;
namespace Ui {
class ANPR;
}

class ANPR : public QMainWindow
{
    Q_OBJECT

public:
    QString path;
    explicit ANPR(QWidget *parent = 0);
    ~ANPR();

private slots:
    void on_openfile_clicked();

    void on_textloc_clicked();

    void on_textseg_clicked();

    void on_textrec_clicked();

    void on_textloc_2_clicked();

    void on_preproc_clicked();

    void on_preproc_lab_clicked();

private:
    Ui::ANPR *ui;
    textloc *TextLoc;
    textseg *TextSeg;
    preproc *PreProc;
    QImage ANPR::convertOpenCVMatToQtQImage(cv::Mat mat);

};

#endif // ANPR_H

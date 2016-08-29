#include "anpr.h"
#include <QApplication>
using namespace cv;
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ANPR w;
    w.show();
    if(w.path!=""){
      Mat img=imread(w.path.toStdString());
    imshow("",img);
    }
    return a.exec();
}

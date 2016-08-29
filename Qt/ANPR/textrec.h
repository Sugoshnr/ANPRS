#ifndef TEXTREC_H
#define TEXTREC_H

#include <QDialog>

namespace Ui {
class textrec;
}

class textrec : public QDialog
{
    Q_OBJECT

public:
    explicit textrec(QWidget *parent = 0);
    ~textrec();

private:
    Ui::textrec *ui;
};

#endif // TEXTREC_H

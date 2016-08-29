#include "textrec.h"
#include "ui_textrec.h"

textrec::textrec(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::textrec)
{
    ui->setupUi(this);
}

textrec::~textrec()
{
    delete ui;
}

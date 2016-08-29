#-------------------------------------------------
#
# Project created by QtCreator 2016-05-12T22:01:52
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ANPR
TEMPLATE = app


SOURCES += main.cpp\
        anpr.cpp \
    textloc.cpp \
    textseg.cpp \
    textrec.cpp \
    preproc.cpp

HEADERS  += anpr.h \
    textloc.h \
    textseg.h \
    textrec.h \
    preproc.h

FORMS    += anpr.ui \
    textloc.ui \
    textseg.ui \
    textrec.ui \
    preproc.ui

INCLUDEPATH += C:\\opencv\\build\\include

LIBS += -LC:\\opencv\\build\\x86\\vc10\\lib \
    -lopencv_calib3d2410d \
    -lopencv_contrib2410d \
    -lopencv_core2410d \
    -lopencv_features2d2410d \
    -lopencv_flann2410d \
    -lopencv_gpu2410d \
    -lopencv_highgui2410d \
    -lopencv_imgproc2410d \
    -lopencv_legacy2410d \
    -lopencv_ml2410d \
    -lopencv_nonfree2410d \
    -lopencv_objdetect2410d \
    -lopencv_ocl2410d \
    -lopencv_photo2410d \
    -lopencv_stitching2410d \
    -lopencv_superres2410d \
    -lopencv_ts2410d \
    -lopencv_video2410d \
    -lopencv_videostab2410d

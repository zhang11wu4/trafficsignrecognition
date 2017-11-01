#pragma once
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <tesseract/baseapi.h>
#include <iostream>
#include <stack>
#include <vector>

using namespace std;
using namespace cv;
using namespace tesseract;
#define CIRCLE 1
extern "C" {
	void TrafficSignRecognition(Mat frame, vector<Rect> &vct_rt, vector<char*> &vct_type);
}
Mat ConvertColorSpace(Mat img);
Mat SegDigitalNum(Mat img);
bool FillImage(Mat &img, CvPoint feed_pt, int fillColor, int backColor, int edgeColor);
int GetNumImg(Mat &img);
bool isNum(string str);

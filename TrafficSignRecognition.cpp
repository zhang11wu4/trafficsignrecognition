#include "TrafficSignRecognition.h"

void ShapeDetection(Mat img, vector<Rect> &shape_objs);
extern "C"
{
	void TrafficSignRecognition(Mat frame, vector<Rect> &vct_rt, vector<char*> &vct_type)
	{
		Mat reg_img = ConvertColorSpace(frame);
		vector<Rect> shape_objs;
		ShapeDetection(reg_img, shape_objs);
		Rect rt;
		int img_w = reg_img.cols;
		int img_h = reg_img.rows;
		int shape_num = shape_objs.size();
		TessBaseAPI tess;
		tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
		tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
		tess.SetVariable("tessedit_char_blacklist", ".,!?@#$%&*()<>_-+=/:;'\"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
		tess.SetVariable("tessedit_char_whitelist", "0123456789");
		tess.SetVariable("classify_bln_numeric_mode", "1");
		for (int i = 0; i < shape_num; i++)
		{
			rt = shape_objs[i];
			Mat sub_img = frame(rt);

		        int labimg_w = sub_img.cols;
		        int labimg_h = sub_img.rows;
			Mat res_img = SegDigitalNum(sub_img);

			tess.SetImage((uchar*)res_img.data, res_img.cols, res_img.rows, 1, res_img.cols);
			char* out = tess.GetUTF8Text();
			string str = out;
			printf("out=%d\n", atoi(out));
			bool isnum = isNum(str);
			if (isNum(str) && atoi(out) > 10 && atoi(out) < 130)
			{
				vct_type.push_back(out);
				if (rt.y < (labimg_h / 2))
					vct_rt.push_back(rt);
			}
		}
	}

}
void ShapeDetection(Mat img, vector<Rect> &shape_objs)
	{
		if (img.channels() != 1)
		{
			cout << "input image should be binary" << endl;
		}
		else
		{
			Mat erode_mat;
			int morph_size = 0;
			Mat element = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
			morphologyEx(img, erode_mat, MORPH_ERODE, element);

			Mat edge;
			Canny(erode_mat, edge, 0, 50, 5);

			vector<vector<Point> > contours;
			findContours(edge.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

			for (int i = 0; i < contours.size(); i++)
			{
				Rect rect = boundingRect(contours[i]);
				int rc_x = rect.x;
				int rc_y = rect.y;
				int rc_w = rect.width;
				int rc_h = rect.height;
				Mat src_rect = img(rect);
				Rect src_rect_cen(rc_x + rc_w / 4, rc_y + rc_h / 4, rc_w / 2, rc_h / 2);
				if (src_rect_cen.width < 15 || src_rect_cen.height < 15) continue;
				if (src_rect_cen.width > 28 || src_rect_cen.height >28) continue;
				if (std::abs(1 - ((double)rect.width / rect.height)) <= 0.2)
				{
					shape_objs.push_back(rect);
				}
			}
		}
	}
Mat ConvertColorSpace(Mat img)
{
	Mat res;
	cvtColor(img, res, CV_RGB2HSV);

	int w = res.cols;
	int h = res.rows;
	Mat reg = Mat(h,w, CV_8UC1);
	reg = Scalar::all(0);
	uchar* ptr_reg = reg.ptr(0, 0);
	uchar* ptr_res = res.ptr(0, 0);
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			uchar H_val = *ptr_res++;
			uchar S_val = *ptr_res++;
			uchar V_val = *ptr_res++;
			if ((H_val>90 && H_val < 150) && (S_val>110 && S_val < 200) && (V_val>60 && V_val < 150))
			{
				*ptr_reg++ = 255;
			}
			else
			{
				*ptr_reg++ = 0;
			}
		}
	}
	return reg;
}

Mat SegDigitalNum(Mat img)
{

	Mat gray_img, bina_img, res_img;
	cvtColor(img, gray_img, CV_BGR2GRAY);
	threshold(gray_img, bina_img, 0, 255, CV_THRESH_OTSU);
	CvPoint pt;
	pt.x = 0; pt.y = 0;
	FillImage(bina_img, pt, 255, 0, 255);
	res_img = 255 - bina_img;
	return res_img;
}

bool FillImage(Mat &img, CvPoint feed_pt, int fillColor, int backColor, int edgeColor)
{
	uchar* ptr_img = img.ptr(0, 0);
	int width = img.cols;
	int height = img.rows;
	if (feed_pt.x < 0 || feed_pt.x >= width || feed_pt.y < 0 || feed_pt.y >= height) return false;

	if (backColor == fillColor) return false;
	if (ptr_img[feed_pt.y*width + feed_pt.x] != backColor) return false;

	stack<CvPoint> points;
	points.push(feed_pt);

	int ww = width - 1;
	int hh = height - 1;

	while (points.size() > 0)
	{
		CvPoint p = points.top();
		points.pop();
		ptr_img[p.y*width + p.x] = fillColor;
		if (p.x > 0 && ptr_img[p.y*width + (p.x - 1)] == backColor&&ptr_img[p.y*width + (p.x - 1)] != edgeColor)
		{
			ptr_img[p.y*width + (p.x - 1)] = fillColor;
			CvPoint pt;
			pt.x = p.x - 1;
			pt.y = p.y;
			points.push(pt);
		}

		if (p.x < ww && ptr_img[p.y*width + (p.x + 1)] == backColor&& ptr_img[p.y*width + (p.x + 1)] != edgeColor)
		{
			ptr_img[p.y*width + p.x + 1] = fillColor;
			CvPoint pt;
			pt.x = p.x + 1;
			pt.y = p.y;
			points.push(pt);
		}

		if (p.y > 0 && ptr_img[(p.y - 1)*width + p.x] == backColor&& ptr_img[(p.y - 1)*width + p.x] != edgeColor)
		{
			ptr_img[(p.y - 1)*width + p.x] = fillColor;
			CvPoint pt;
			pt.x = p.x;
			pt.y = p.y - 1;
			points.push(pt);
		}

		if (p.y < hh && ptr_img[(p.y + 1)*width + p.x] == backColor&& ptr_img[(p.y + 1)*width + p.x] != edgeColor)
		{
			ptr_img[(p.y + 1)*width + p.x] = fillColor;
			CvPoint pt;
			pt.x = p.x;
			pt.y = p.y + 1;
			points.push(pt);
		}
	}
	return true;
}
bool isNum(string str)
{
	stringstream sin(str);
	double d;
	char c;
	if (!(sin >> d))
	{
		return false;
	}
	if (sin >> c)
	{
		return false;
	}
	return true;
}

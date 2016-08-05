#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
//#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>
#include <dlib/opencv.h>
#include "svm.h"
#include <bits/stdc++.h>

#define PI 3.14159265;


using namespace cv;
using namespace dlib;
using namespace std;

//ofstream fout_angles;

struct svm_parameter param;     // set by parse_command_line
struct svm_problem prob;        // set by read_problem
struct svm_model *model;
struct svm_node *x_space;

struct svm_node testnode[51];


string svm_predict(std::vector<double> &angles)
{
	for(int i = 0; i < 50; i++)
	{
		testnode[i].index = i;
		testnode[i].value = angles[i];
	}
	testnode[50].index = -1;
	double retval = svm_predict(model,testnode);
	int val = retval;

	switch(val)
	{
		case 0:
				return "Centre Stack";
				break;
		case 1:
				return "Instrument stack";
				break;
		case 2:
				return "Left";
				break;
		case 3:
				return "Right";
				break;
		case 4:
				return "Road";
				break;
		case 5:
				return "Rear view mirror";
				break;
		case 6:
				return "Up";
				break;
		case 7:
				return "Down";
				break;
		default:
				return "Not defined";
				break;
	}
	return "";
}

//Calculate angle of triangle given three coordinates c++ code
void TriangleAngleCalculation(std::vector<double> &angles, int x1, int y1, int x2, int y2, int x3, int y3)
{
	double dist1, dist2, dist3;
	double angle1, angle2, angle3;
	//double total;

	int largestLength = 0;
	dist1 = pow(x1 - x2, 2) + pow(y1 - y2, 2);
	dist2 = pow(x2 - x3, 2) + pow(y2 - y3, 2);
	dist3 = pow(x1 - x3, 2) + pow(y1 - y3, 2);

	angle1 = (180 * acos((dist2 + dist3 - dist1) / (2 * sqrt(dist2 * dist3)))) / PI;
	angle2 = (180 * acos((dist1 + dist3 - dist2) / (2 * sqrt(dist1 * dist3)))) / PI;
	angles.push_back(angle1);
	angles.push_back(angle2);
}


image_window win_triangles;

int triangulation(Mat img, full_object_detection shape, int arr[20])
{
		std::vector<Point2f > vec;
		for (int i = 0; i < 19; i++)
		{
			Point2f pt(shape.part(arr[i]).x(), shape.part(arr[i]).y());
			vec.push_back(pt);

		}
	std::vector<double> angles;
	//1	
	line(img, vec[9],vec[11], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[9], vec[10], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[10], vec[11], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[9].x, vec[9].y, vec[10].x, vec[10].y, vec[11].x, vec[11].y);
	//2
	line(img, vec[13], vec[11], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[9], vec[11], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[9], vec[13], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[9].x, vec[9].y, vec[13].x, vec[13].y, vec[11].x, vec[11].y);
	//3
	line(img, vec[10], vec[11], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[15], vec[11], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[10], vec[15], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[10].x, vec[10].y, vec[15].x, vec[15].y, vec[11].x, vec[11].y);
	//4
	line(img, vec[12], vec[11], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[13], vec[11], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[12], vec[13], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[12].x, vec[12].y, vec[13].x, vec[13].y, vec[11].x, vec[11].y);
	//5
	line(img, vec[12], vec[11], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[15], vec[11], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[12], vec[15], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[15].x, vec[15].y, vec[12].x, vec[12].y, vec[11].x, vec[11].y);
	//6
	line(img, vec[12], vec[13], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[13], vec[14], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[12], vec[14], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[13].x, vec[13].y, vec[12].x, vec[12].y, vec[14].x, vec[14].y);
	//7
	line(img, vec[12], vec[15], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[12], vec[14], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[15], vec[14], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[13].x, vec[13].y, vec[12].x, vec[12].y, vec[14].x, vec[14].y);
	//8
	line(img, vec[10], vec[15], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[10], vec[8], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[15], vec[8], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[10].x, vec[10].y, vec[15].x, vec[15].y, vec[8].x, vec[8].y);
	//9
	line(img, vec[7], vec[15], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[7], vec[8], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[15], vec[8], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[7].x, vec[7].y, vec[15].x, vec[15].y, vec[8].x, vec[8].y);
	//10
	line(img, vec[7], vec[6], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[7], vec[18], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[6], vec[18], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[7].x, vec[7].y, vec[6].x, vec[6].y, vec[18].x, vec[18].y);
	//11
	line(img, vec[7], vec[15], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[7], vec[18], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[15], vec[18], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[7].x, vec[7].y, vec[15].x, vec[15].y, vec[18].x, vec[18].y);
	//12
	line(img, vec[7], vec[15], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[7], vec[18], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[15], vec[18], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[7].x, vec[7].y, vec[15].x, vec[15].y, vec[18].x, vec[18].y);
	//13
	line(img, vec[14], vec[15], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[14], vec[17], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[15], vec[17], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[17].x, vec[17].y, vec[14].x, vec[14].y, vec[15].x, vec[15].y);
	//14
	line(img, vec[14], vec[13], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[14], vec[17], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[13], vec[17], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[17].x, vec[17].y, vec[14].x, vec[14].y, vec[13].x, vec[13].y);
	//15
	line(img, vec[5], vec[6], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[6], vec[18], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[18], vec[5], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[6].x, vec[6].y, vec[5].x, vec[5].y, vec[18].x, vec[18].y);
	//16
	line(img, vec[5], vec[4], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[4], vec[18], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[18], vec[5], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[4].x, vec[4].y, vec[5].x, vec[5].y, vec[18].x, vec[18].y);
	//17
	line(img, vec[18], vec[4], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[4], vec[17], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[17], vec[18], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[4].x, vec[4].y, vec[18].x, vec[18].y, vec[17].x, vec[17].y);
	//18
	line(img, vec[16], vec[4], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[4], vec[17], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[17], vec[16], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[4].x, vec[4].y, vec[16].x, vec[16].y, vec[17].x, vec[17].y);
	//19
	line(img, vec[16], vec[13], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[13], vec[17], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[17], vec[16], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[13].x, vec[13].y, vec[16].x, vec[6].y, vec[17].x, vec[17].y);
	//20
	line(img, vec[16], vec[4], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[4], vec[3], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[3], vec[16], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[4].x, vec[4].y, vec[16].x, vec[16].y, vec[3].x, vec[3].y);
	//21
	line(img, vec[16], vec[2], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[2], vec[3], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[3], vec[16], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[2].x, vec[2].y, vec[16].x, vec[16].y, vec[3].x, vec[3].y);
	//22
	line(img, vec[16], vec[2], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[2], vec[1], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[1], vec[16], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[2].x, vec[2].y, vec[16].x, vec[16].y, vec[1].x, vec[1].y);
	//23
	line(img, vec[16], vec[13], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[13], vec[1], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[1], vec[16], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[13].x, vec[13].y, vec[16].x, vec[16].y, vec[1].x, vec[1].y);
	//24
	line(img, vec[0], vec[13], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[13], vec[1], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[1], vec[0], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[13].x, vec[13].y, vec[0].x, vec[0].y, vec[1].x, vec[1].y);
	//25
	line(img, vec[0], vec[13], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[13], vec[9], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[9], vec[0], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(angles, vec[13].x, vec[13].y, vec[0].x, vec[0].y, vec[9].x, vec[9].y);
	
	string result = svm_predict(angles);

	putText(img, result, Point(  img.cols / 2, img.rows / 4), 1, 2, Scalar(0, 0, 255), 3, 8);

	cv_image<bgr_pixel> x(img);
	array2d<rgb_pixel> temp;
	assign_image(temp, x);
	win_triangles.clear_overlay();
	win_triangles.set_image(temp);
	return 0;
}

int main()
{
	int check;
	try
	{
		model = svm_load_model("model.bin");
		cv::VideoCapture vid(0);
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor sp;
		deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
		int arr[] = { 0,2,4,6,8,10,12,14,16,19,24,27,30,31,33,35,48,51,54};
		image_window win, win_faces;

		while(!win.is_closed())
		{
			cv::Mat temp2;
        	vid >> temp2;
        	cv::Mat temp;
        	flip(temp2, temp, 1);
        	cv_image<bgr_pixel> image(temp);
			array2d<rgb_pixel> img;
			assign_image(img, image);
			
			std::vector<dlib::rectangle> dets = detector(img);
			check = 0;
			// Now we will go ask the shape_predictor to tell us the pose of
			// each face we detected.
			std::vector<full_object_detection> shapes;
			
			for (unsigned long j = 0; j < dets.size(); ++j)
			{
				check = 1;
				full_object_detection shape = sp(img, dets[0]);
				shapes.push_back(shape);
				triangulation(temp, shape, arr);
			}
		}
		getchar();
	}

	catch (exception& e)
	{
		std::cout << "\nexception thrown!" << endl;
		std::cout << e.what() << endl;
		getchar();
	
	return 0;
	}
}
     
// ----------------------------------------------------------------------------------------


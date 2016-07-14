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


void svm_predict(std::vector<double> &angles)
{

	//for(int  i = 0; i < angles.size(); i++)
	//	cout<<angles[i]<<"\n";
	//ifstream remp("Testing_data_matReduced.csv");
	
	//count = 0;
	// while(remp)
	// {
	// 	remp>>str;
	// 	count++;
	// }
	// count--;
	//prob.l = 1;
	//remp.close();
	//cout<<count<<"\n";
//	test.y = Malloc(double,prob.l);
	
	//fin.open("Testing_data_matReduced.csv");
	
	//double correct = 0;
	//cout<<"Here\n";
	for(int i = 0; i < 50; i++)
	{
		testnode[i].index = i;
		testnode[i].value = angles[i];
	}
	testnode[50].index = -1;
	// 	int len = 0;
	// 	fin >> str;
	// 	vals = strtok(str, ",");
	// 	while(vals)
	// 	{
	// 		value = convert(vals);
	// 		//cout<<value<<" ";
	// 		vals = strtok(NULL , ",");
	// 		if(len == 50)
	// 			break;
	// 		 testnode[len].index = len;
 //   			 testnode[len++].value = value;
	// 	}
	// //	test.y[i] = value;
	// 	testnode[50].index = -1; 
		
		//for(int i = 0; i < 50; i++)
		//	cout<<testnode[i].index<<" "<<testnode[i].value<<"\n";
		//cout<<"Example # "<<i<<"\n";
	//cout<<"Here\n";
	double retval = svm_predict(model,testnode);
	//cout<<retval<<"\n";
	int val = retval;

	switch(val)
	{
		case 0:
				cout<<"Centre Stack\n";
				break;
		case 1:
				cout<<"Instrument stack\n";
				break;
		case 2:
				cout<<"Left\n";
				break;
		case 3:
				cout<<"Right\n";
				break;
		case 4:
				cout<<"Road\n";
				break;
		case 5:
				cout<<"Rear view mirror\n";
				break;
		case 6:
				cout<<"Up\n";
				break;
		case 7:
				cout<<"Down\n";
				break;
		default:
				cout<<"Not defined\n";
				break;
	}
		//printf("retval: %f actualval: %f\n",retval,value);
		//if(retval == value)
		//	correct++
  	
  	//cout<<(correct) / (double)(count)<<"\n";


	//This works correctly:
	//double retval = svm_predict(model,testnode);
//	printf("retval: %f\n",retval);
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
	//angle3 = (180 * acos((dist1 + dist2 - dist3) / (2 * sqrt(dist1 * dist2)))) / PI;
	//cout << "Angles are:\n";
	angles.push_back(angle1);
	angles.push_back(angle2);
	//cout << angle1 << " " << angle2 << " " << angle3 << "\n";

	//fout_angles << angle1 << "," << angle2 << ",";// << angle3 << " ";

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

	//cout<<angles.size()<<"\n";
	//string win_delaunay = "Delaunay Triangulation";
	
	svm_predict(angles);


	cv_image<bgr_pixel> x(img);
	array2d<rgb_pixel> temp;
	assign_image(temp, x);
	win_triangles.clear_overlay();
	win_triangles.set_image(temp);
	//imshow("win_delaunay", img);
	//waitKey(0);
	return 0;
}

int main()
{
	//Mat x;
	//testnode = new svm_node[51];
	int check;
	try
	{
		model = svm_load_model("model.bin");
		//fout_angles.open("angles.txt", ios::trunc);
		// This example takes in a shape model file and then a list of images to
		// process.  We will take these filenames in as command line arguments.
		// Dlib comes with example images in the examples/faces folder so give
		// those as arguments to this program.

		// We need a face detector.  We will use this to get bounding boxes for
		// each face in an image.
		cv::VideoCapture vid(0);
		frontal_face_detector detector = get_frontal_face_detector();
		// And we also need a shape_predictor.  This is the tool that will predict face
		// landmark positions given an image and face bounding box.  Here we are just
		// loading the model from the shape_predictor_68_face_landmarks.dat file you gave
		// as a command line argument.
		shape_predictor sp;
		deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
		int arr[] = { 0,2,4,6,8,10,12,14,16,19,24,27,30,31,33,35,48,51,54};
		image_window win, win_faces;
		// Loop over all the images provided on the command line.
		while(!win.is_closed())
		{
			cv::Mat temp;
        	vid >> temp;
        	cv_image<bgr_pixel> image(temp);
			array2d<rgb_pixel> img;
			assign_image(img, image);
			
			//load_image(img, argv[i]);
			
			// Make the image larger so we can detect small faces.
			//pyramid_up(i
			// Now tell the face detector to give us a list of bounding boxes
			// around all the faces in the image.
			std::vector<dlib::rectangle> dets = detector(img);
			//std::cout << "Number of faces detected: " << dets.size() << endl;
			check = 0;
			// Now we will go ask the shape_predictor to tell us the pose of
			// each face we detected.
			std::vector<full_object_detection> shapes;
			
			for (unsigned long j = 0; j < dets.size(); ++j)
			{
				check = 1;
				full_object_detection shape = sp(img, dets[0]);
				//std::cout << "number of parts: " << shape.num_parts() << endl;
				//std::cout << "pixel position of first part:  " << shape.part(0) << endl;
				//std::cout << "pixel position of second part: " << shape.part(1) << endl;
				// You get the idea, you can get all the face part locations if
				// you want them.  Here we just store them in shapes so we can
				// put them on the screen.
				shapes.push_back(shape);
				triangulation(temp, shape, arr);
			}
			

			// Now let's view our face poses on the screen.
			//win.clear_overlay();
			//win.set_image(img);
			//win.add_overlay(render_face_detections(shapes));
			

			// We can also extract copies of each face that are cropped, rotated upright,
			// and scaled to a standard size as shown here:
			//dlib::array<array2d<rgb_pixel> > face_chips;
			//extract_image_chips(img, get_face_chip_details(shapes), face_chips);
			//win_faces.set_image(tile_images(face_chips));
			//string temp_filename = strcat(argv[i], "_o.jpg");

			//save_bmp(img, temp_filename);
                
			//std::cout << "Hit enter to process the next image..." << endl;
			//cin.get();
			//if (check == 1)
			//fout_angles << "\n";
			//cnt++;
		}
		//fout_angles << "Here\n";
		//fout_angles.clear();
		
		//fout_angles.close();
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


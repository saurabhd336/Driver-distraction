// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    


    This face detector is made using the classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image pyramid,
    and sliding window detection scheme.  The pose estimator was created by
    using dlib's implementation of the paper:
        One Millisecond Face Alignment with an Ensemble of Regression Trees by
        Vahid Kazemi and Josephine Sullivan, CVPR 2014
    and was trained on the iBUG 300-W face landmark dataset.  

    Also, note that you can train your own models using dlib's machine learning
    tools.  See train_shape_predictor_ex.cpp to see an example.

    


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
//#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
//#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
//#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
#include<conio.h>
#include<math.h>
#define PI 3.14159265;


using namespace cv;
using namespace dlib;
using namespace std;


// ----------------------------------------------------------------------------------------

ofstream fout_angles;

//std::vector<std::vector<Point2f> > facets;
//std::vector<Point2f> centers;
//subdiv.getVoronoiFacetList(std::vector<int>(), facets, centers);
// Draw a single point
/*
static void draw_point(Mat& img, Point2f fp, Scalar color)
{
	circle(img, fp, 2, color, CV_FILLED, CV_AA, 0);
}
*/

//Calculate angle of triangle given three coordinates c++ code
void TriangleAngleCalculation(int x1, int y1, int x2, int y2, int x3, int y3)
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
	//cout << angle1 << " " << angle2 << " " << angle3 << "\n";
	fout_angles << angle1 << "," << angle2 << ",";// << angle3 << " ";

}

//main function to Calculate angle of triangle given three coordinates c++ code

/*
// Draw delaunay triangles
static void draw_delaunay(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color)
{

	std::vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	std::vector<Point> pt(3);
	Size size = img.size();
	Rect rect(0, 0, size.width, size.height);
	cout << "Size of triangle: "<<triangleList.size()<<endl;
	for (size_t i = 0; i < triangleList.size(); i++)
	{
		Vec6f t = triangleList[i];
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
		cout << "the three points are: " << endl;
		cout << pt[0].x << " " << pt[0].y << endl;
		cout << pt[1].x << " " << pt[1].y << endl;
		cout << pt[2].x << " " << pt[2].y << endl;

		// Draw rectangles completely inside the image.
		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
		{
			line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
			line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
			line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
			TriangleAngleCalculation(pt[0].x, pt[0].y, pt[1].x, pt[1].y, pt[2].x, pt[2].y);
		}
	}
}
*/


int triangulation(string img_path, full_object_detection shape, int arr[20])
{
		cv::vector<Point2f> vec;
		for (int i = 0; i < 19; i++)
		{
			Point2f pt(shape.part(arr[i]).x(), shape.part(arr[i]).y());
			vec.push_back(pt);

		}
	Mat img = imread(img_path);

	//1
	line(img, vec[9],vec[11], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[9], vec[10], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[10], vec[11], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[9].x, vec[9].y, vec[10].x, vec[10].y, vec[11].x, vec[11].y);
	//2
	line(img, vec[13], vec[11], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[9], vec[11], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[9], vec[13], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[9].x, vec[9].y, vec[13].x, vec[13].y, vec[11].x, vec[11].y);
	//3
	line(img, vec[10], vec[11], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[15], vec[11], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[10], vec[15], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[10].x, vec[10].y, vec[15].x, vec[15].y, vec[11].x, vec[11].y);
	//4
	line(img, vec[12], vec[11], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[13], vec[11], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[12], vec[13], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[12].x, vec[12].y, vec[13].x, vec[13].y, vec[11].x, vec[11].y);
	//5
	line(img, vec[12], vec[11], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[15], vec[11], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[12], vec[15], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[15].x, vec[15].y, vec[12].x, vec[12].y, vec[11].x, vec[11].y);
	//6
	line(img, vec[12], vec[13], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[13], vec[14], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[12], vec[14], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[13].x, vec[13].y, vec[12].x, vec[12].y, vec[14].x, vec[14].y);
	//7
	line(img, vec[12], vec[15], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[12], vec[14], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[15], vec[14], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[13].x, vec[13].y, vec[12].x, vec[12].y, vec[14].x, vec[14].y);
	//8
	line(img, vec[10], vec[15], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[10], vec[8], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[15], vec[8], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[10].x, vec[10].y, vec[15].x, vec[15].y, vec[8].x, vec[8].y);
	//9
	line(img, vec[7], vec[15], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[7], vec[8], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[15], vec[8], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[7].x, vec[7].y, vec[15].x, vec[15].y, vec[8].x, vec[8].y);
	//10
	line(img, vec[7], vec[6], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[7], vec[18], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[6], vec[18], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[7].x, vec[7].y, vec[6].x, vec[6].y, vec[18].x, vec[18].y);
	//11
	line(img, vec[7], vec[15], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[7], vec[18], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[15], vec[18], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[7].x, vec[7].y, vec[15].x, vec[15].y, vec[18].x, vec[18].y);
	//12
	line(img, vec[7], vec[15], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[7], vec[18], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[15], vec[18], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[7].x, vec[7].y, vec[15].x, vec[15].y, vec[18].x, vec[18].y);
	//13
	line(img, vec[14], vec[15], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[14], vec[17], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[15], vec[17], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[17].x, vec[17].y, vec[14].x, vec[14].y, vec[15].x, vec[15].y);
	//14
	line(img, vec[14], vec[13], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[14], vec[17], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[13], vec[17], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[17].x, vec[17].y, vec[14].x, vec[14].y, vec[13].x, vec[13].y);
	//15
	line(img, vec[5], vec[6], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[6], vec[18], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[18], vec[5], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[6].x, vec[6].y, vec[5].x, vec[5].y, vec[18].x, vec[18].y);
	//16
	line(img, vec[5], vec[4], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[4], vec[18], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[18], vec[5], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[4].x, vec[4].y, vec[5].x, vec[5].y, vec[18].x, vec[18].y);
	//17
	line(img, vec[18], vec[4], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[4], vec[17], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[17], vec[18], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[4].x, vec[4].y, vec[18].x, vec[18].y, vec[17].x, vec[17].y);
	//18
	line(img, vec[16], vec[4], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[4], vec[17], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[17], vec[16], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[4].x, vec[4].y, vec[16].x, vec[16].y, vec[17].x, vec[17].y);
	//19
	line(img, vec[16], vec[13], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[13], vec[17], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[17], vec[16], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[13].x, vec[13].y, vec[16].x, vec[6].y, vec[17].x, vec[17].y);
	//20
	line(img, vec[16], vec[4], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[4], vec[3], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[3], vec[16], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[4].x, vec[4].y, vec[16].x, vec[16].y, vec[3].x, vec[3].y);
	//21
	line(img, vec[16], vec[2], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[2], vec[3], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[3], vec[16], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[2].x, vec[2].y, vec[16].x, vec[16].y, vec[3].x, vec[3].y);
	//22
	line(img, vec[16], vec[2], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[2], vec[1], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[1], vec[16], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[2].x, vec[2].y, vec[16].x, vec[16].y, vec[1].x, vec[1].y);
	//23
	line(img, vec[16], vec[13], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[13], vec[1], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[1], vec[16], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[13].x, vec[13].y, vec[16].x, vec[16].y, vec[1].x, vec[1].y);
	//24
	line(img, vec[0], vec[13], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[13], vec[1], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[1], vec[0], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[13].x, vec[13].y, vec[0].x, vec[0].y, vec[1].x, vec[1].y);
	//25
	line(img, vec[0], vec[13], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[13], vec[9], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	line(img, vec[9], vec[0], CV_RGB(0, 0, 255), 1, CV_AA, 0);
	TriangleAngleCalculation(vec[13].x, vec[13].y, vec[0].x, vec[0].y, vec[9].x, vec[9].y);

	string win_delaunay = "Delaunay Triangulation";
	imshow(win_delaunay, img);
	//waitKey(0);
	return 0;

	/*
	// Define window names
	string win_delaunay = "Delaunay Triangulation";

	// Turn on animation while drawing triangles
	bool animate = true;

	// Define colors for drawing.
	Scalar delaunay_color(255, 255, 255), points_color(0, 0, 255);

	// Read in the image.
	Mat img = imread(img_path);

	// Keep a copy around
	Mat img_orig = img.clone();

	// Rectangle to be used with Subdiv2D
	Size size = img.size();
	Rect rect(0, 0, size.width, size.height);

	// Create an instance of Subdiv2D
	Subdiv2D subdiv(rect);

	// Create a vector of points.
	std::vector<Point2f> points;

	// Read in the points from a text file
	ifstream ifs("points_list.txt");
	int x, y;
	while (ifs >> x >> y)
	{
		//cout<<"abc";
		cout << x << " " << y << endl;
		points.push_back(Point2f(x, y));
	}
// Insert points into subdiv
	for (std::vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
	{
		subdiv.insert(*it);
		// Show animation
		if (animate)
		{
			Mat img_copy = img_orig.clone();
			// Draw delaunay triangles
			//draw_delaunay(img_copy, subdiv, delaunay_color);
			imshow(win_delaunay, img_copy);
			//waitKey(100);
		}

	}

	// Draw delaunay triangles
	draw_delaunay(img, subdiv, delaunay_color);

	// Draw points
	for (std::vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
	{
		draw_point(img, *it, points_color);
	}


	// Show results.
	imshow(win_delaunay, img);
	waitKey(0);
	

	//return 0;*/

}

int main(int argc, char** argv)
{
	//Mat x;
	int check;
	try
	{
		fout_angles.open("angles.txt", ios::trunc);
		// This example takes in a shape model file and then a list of images to
		// process.  We will take these filenames in as command line arguments.
		// Dlib comes with example images in the examples/faces folder so give
		// those as arguments to this program.
		if (argc == 1)
		{
			std::cout << "Call this program like this:" << endl;
			std::cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
			std::cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
			std::cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
			return 0;
		}

		// We need a face detector.  We will use this to get bounding boxes for
		// each face in an image.
		frontal_face_detector detector = get_frontal_face_detector();
		// And we also need a shape_predictor.  This is the tool that will predict face
		// landmark positions given an image and face bounding box.  Here we are just
		// loading the model from the shape_predictor_68_face_landmarks.dat file you gave
		// as a command line argument.
		shape_predictor sp;
		deserialize(argv[1]) >> sp;
		int arr[] = { 0,2,4,6,8,10,12,14,16,19,24,27,30,31,33,35,48,51,54};
		image_window win, win_faces;
		// Loop over all the images provided on the command line.
		for (int i = 2; i < argc ; ++i)
		{
			std::cout << "processing image " << argv[i] << endl;
			array2d<rgb_pixel> img;
			//array2d<int> img;
			
			load_bmp(img, argv[i]);
			
			// Make the image larger so we can detect small faces.
			//pyramid_up(img);
			// Now tell the face detector to give us a list of bounding boxes
			// around all the faces in the image.
			std::vector<dlib::rectangle> dets = detector(img);
			std::cout << "Number of faces detected: " << dets.size() << endl;
			check = 0;
			// Now we will go ask the shape_predictor to tell us the pose of
			// each face we detected.
			std::vector<full_object_detection> shapes;
			
			for (unsigned long j = 0; j < dets.size(); ++j)
			{
				check = 1;
				full_object_detection shape = sp(img, dets[j]);
				std::cout << "number of parts: " << shape.num_parts() << endl;
				//std::cout << "pixel position of first part:  " << shape.part(0) << endl;
				//std::cout << "pixel position of second part: " << shape.part(1) << endl;
				// You get the idea, you can get all the face part locations if
				// you want them.  Here we just store them in shapes so we can
				// put them on the screen.
				shapes.push_back(shape);
				triangulation(argv[i], shape, arr);
			/*
				ofstream a_file("points_list.txt");
				for (int i = 0; i <19; i++)
				{
					//cout << shapes.part(arr[i]);
					a_file << shape.part(arr[i]).x() << " " << shape.part(arr[i]).y() << "\n";
					//draw_solid_circle(img, shape.part(arr[i]), 2, rgb_pixel(0,0,255));
				}
				//imshow("opencv_window", new_window);
				a_file.close();
				*/
			}
			

			// Now let's view our face poses on the screen.
			win.clear_overlay();
			win.set_image(img);
			win.add_overlay(render_face_detections(shapes));
			

			// We can also extract copies of each face that are cropped, rotated upright,
			// and scaled to a standard size as shown here:
			dlib::array<array2d<rgb_pixel> > face_chips;
			extract_image_chips(img, get_face_chip_details(shapes), face_chips);
			win_faces.set_image(tile_images(face_chips));
			//string temp_filename = strcat(argv[i], "_o.jpg");

			//save_bmp(img, temp_filename);
                
			std::cout << "Hit enter to process the next image..." << endl;
			//cin.get();
			if (check == 1)
			fout_angles << "\n";
			//cnt++;
		}
		//fout_angles << "Here\n";
		//fout_angles.clear();
		
		fout_angles.close();
		//getchar();
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


// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <optional>
using namespace cv;
using namespace std;

int main(int argc, char * argv[]) 
{

	// Declare depth colorizer for pretty visualization of depth data

	rs2::colorizer color_map;

	// Declare RealSense pipeline, encapsulating the actual device and sensors

	rs2::pipeline pipe;

	// Start streaming with default recommended configuration


	rs2::config cfg;

	cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

	rs2::pipeline_profile selection = pipe.start(cfg);

	rs2::device selected_device = selection.get_device();

	auto depth_sensor = selected_device.first<rs2::depth_sensor>();


	if(depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED)) depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1.f);
	if(depth_sensor.supports(RS2_OPTION_LASER_POWER)) depth_sensor.set_option(RS2_OPTION_LASER_POWER, 16.0);
	if(depth_sensor.supports(RS2_OPTION_ACCURACY)) depth_sensor.set_option(RS2_OPTION_ACCURACY, 3.0);
	if(depth_sensor.supports(RS2_OPTION_MOTION_RANGE)) depth_sensor.set_option(RS2_OPTION_MOTION_RANGE, 37.0);
	if(depth_sensor.supports(RS2_OPTION_FILTER_OPTION)) depth_sensor.set_option(RS2_OPTION_FILTER_OPTION, 3.0);
	if(depth_sensor.supports(RS2_OPTION_CONFIDENCE_THRESHOLD)) depth_sensor.set_option(RS2_OPTION_CONFIDENCE_THRESHOLD, 2.0);

	double depth_units_per_meter = depth_sensor.supports(RS2_OPTION_DEPTH_UNITS) ? 1 / depth_sensor.get_option(RS2_OPTION_DEPTH_UNITS) : 1;

	cout << "Depth units per meter: " << depth_units_per_meter << std::endl;

	double robot_height_meters = 0.07;
	double table_distance_meters = 1.00;

	/*
	RS2_OPTION_LASER_POWER 16
	RS2_OPTION_ACCURACY 3
	RS2_OPTION_MOTION_RANGE 37
	RS2_OPTION_FILTER_OPTION 3
	RS2_OPTION_CONFIDENCE_THRESHOLD 2
	RS2_OPTION_FRAMES_QUEUE_SIZE --
	RS2_OPTION_DEPTH_UNITS --
	*/

	using namespace cv;

	const auto window_name = "Display Image";

	namedWindow(window_name, WINDOW_AUTOSIZE);

	double vScaling = 5.0;//5.0;

	auto clahe = cv::createCLAHE(5, cv::Size(8,8));

	SimpleBlobDetector::Params params;
	params.blobColor = 0xFFFF;
	//params.minThreshold = 1;
	//params.maxThreshold =0xFFFF;

	auto blobDetector = SimpleBlobDetector::create(params);

	std::optional<cv::Rect> cropRect;

	int framecnt = 0;

	while (1)
	{
		Mat DD;

		rs2::frameset frames = pipe.wait_for_frames(); //Wait for new set of (visible, IR, depth) to arrive

		rs2::frame depth_frame = frames.get_depth_frame(); //Extract depth frame
		
		Mat depth(Size(640, 480), CV_16UC1, (void*)depth_frame.get_data(), Mat::AUTO_STEP); //Extract frame to Matrix

		Mat& result = depth;

		depth -= (table_distance_meters - robot_height_meters) * depth_units_per_meter; //Black out everything too close (valid range is everything larger than camera height - robot height)
		vector<Rect> blobBounds;
		if (cropRect) {
			Mat cropped = depth(cropRect.value()); //Crop region of interest from image
			clahe->apply(cropped, cropped); //Apply CLAHE transform to clear up contrast
			auto mean_val = mean(cropped);
			Mat thresholded;
			threshold(cropped, thresholded, mean_val[0] * 0.50, 255, THRESH_TOZERO_INV);
			Mat eightBits;
			thresholded.convertTo(eightBits, CV_8UC1);
			Mat cannyOut;
			Canny(eightBits, cannyOut, 0, 255);
			vector<vector<Point>> contours;
			findContours(cannyOut, contours, RETR_LIST, CHAIN_APPROX_NONE);
			vector<vector<Point>> contours_poly(contours.size());
			vector<Rect> rawBlobs(contours.size());
			blobBounds = vector<Rect>(contours.size());

			for (size_t i = 0; i < contours.size(); i++) {
				rawBlobs[i] = boundingRect(contours[i]);
			}

			copy_if(rawBlobs.begin(), rawBlobs.end(), back_inserter(blobBounds), [](Rect bound) {double metric = abs(((double)bound.width / ((double)bound.width + (double)bound.height))-0.5); return bound.area() < 350 && bound.area() > 200 && metric < 0.2;  });

			result = cropped;
		}

		//Convert to visible image
		result.convertTo(DD, CV_8UC1, 1.0/vScaling, 0);

		Mat color;

		cv::applyColorMap(DD, color, COLORMAP_OCEAN);

		for (size_t i = 0; i < blobBounds.size(); i++) {
			rectangle(color, blobBounds[i], Scalar(0, 255, 0));
			putText(color, std::to_string(i), Point(blobBounds[i].x -2, blobBounds[i].y-2), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 255));

			//putText(color, std::to_string(blobBounds[i].width) + "x" + std::to_string(blobBounds[i].height)+ "=" + std::to_string(blobBounds[i].area()), Point(blobBounds[i].x, blobBounds[i].y), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 255));
			//putText(color, std::to_string(abs(((double)blobBounds[i].width / ((double)blobBounds[i].width + (double)blobBounds[i].height)) - 0.5) ), Point(blobBounds[i].x, blobBounds[i].y + 8), FONT_HERSHEY_SIMPLEX,0.4, Scalar(0, 255, 0));
		}

		if (framecnt == 100) {
			imwrite("../sample.png", color);
		}
			framecnt++;

		putText(color, std::to_string(framecnt),Point(0,20), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0));

		if (!cropRect) {
			cropRect = selectROI("Select tracking region", color, false);
			destroyWindow("Select tracking region");
		}
		

		double min, max;

		//minMaxLoc(depth, &min, &max);
		//std::cout << vScaling << ": " << min << "-" << max << "<-";
		//minMaxLoc(DD, &min, &max);
		//std::cout << min << "-" << max << endl;

		/*
		for (int vI = 0; vI < 640 * 480; ++vI) {
			std::cout << vI << ":" << ((uint16_t*)(depth_frame.get_data()))[vI] << "->" << +(DD.at<unsigned char>(vI)) << std::endl;
		}
		*/
		imshow(window_name, color);
		int vKey = waitKey(1);
		switch(vKey) {
		case '+':
			++vScaling;
			break;
		case '-':
			--vScaling;
			break;
		}

	}

	return EXIT_SUCCESS;
}
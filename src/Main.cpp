// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace cv;
using namespace std;

static void get_sensor_option(const rs2::sensor& sensor)
{
	// Sensors usually have several options to control their properties
	//  such as Exposure, Brightness etc.

	std::cout << "Sensor supports the following options:\n" << std::endl;

	// The following loop shows how to iterate over all available options
	// Starting from 0 until RS2_OPTION_COUNT (exclusive)
	for (int i = 0; i < static_cast<int>(RS2_OPTION_COUNT); i++)
	{
		rs2_option option_type = static_cast<rs2_option>(i);
		//SDK enum types can be streamed to get a string that represents them
		std::cout << "  " << i << ": " << option_type;

		// To control an option, use the following api:

		// First, verify that the sensor actually supports this option
		if (sensor.supports(option_type))
		{
			std::cout << std::endl;

			// Get a human readable description of the option
			const char* description = sensor.get_option_description(option_type);
			std::cout << "       Description   : " << description << std::endl;

			// Get the current value of the option
			float current_value = sensor.get_option(option_type);
			std::cout << "       Current Value : " << current_value << std::endl;

			//To change the value of an option, please follow the change_sensor_option() function
		}
		else
		{
			std::cout << " is not supported" << std::endl;
		}
	}
}

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
	get_sensor_option(depth_sensor);

	if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED))
	{
		depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1.f); // Enable emitter
	}
	if (depth_sensor.supports(RS2_OPTION_LASER_POWER))
	{
		// Query min and max values:
		auto range = depth_sensor.get_option_range(RS2_OPTION_LASER_POWER);
		depth_sensor.set_option(RS2_OPTION_LASER_POWER, range.max); // Set max power
		depth_sensor.set_option(RS2_OPTION_LASER_POWER, 0.f); // Disable laser
	}

	/*
	RS2_OPTION_LASER_POWER
	RS2_OPTION_ACCURACY
	RS2_OPTION_MOTION_RANGE
	RS2_OPTION_FILTER_OPTION
	RS2_OPTION_CONFIDENCE_THRESHOLD
	RS2_OPTION_FRAMES_QUEUE_SIZE
	RS2_OPTION_DEPTH_UNITS
	*/

	using namespace cv;

	const auto window_name = "Display Image";

	namedWindow(window_name, WINDOW_AUTOSIZE);

	double vScaling = 5.0;

	while (1)
	{

		Mat DD;



		rs2::frameset frames = pipe.wait_for_frames();



		rs2::frame depth_frame = frames.get_depth_frame();
		
		Mat depth(Size(640, 480), CV_16SC1, (void*)depth_frame.get_data(), Mat::AUTO_STEP);

		depth -= 7500;
		depth.convertTo(DD, CV_8U, 1.0/vScaling, 0);
		
		double min, max;

		minMaxLoc(depth, &min, &max);
		//std::cout << vScaling << ": " << min << "-" << max << "<-";
		minMaxLoc(DD, &min, &max);
		//std::cout << min << "-" << max << endl;

		/*
		for (int vI = 0; vI < 640 * 480; ++vI) {
			std::cout << vI << ":" << ((uint16_t*)(depth_frame.get_data()))[vI] << "->" << +(DD.at<unsigned char>(vI)) << std::endl;
		}
		*/
		imshow(window_name, DD);
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
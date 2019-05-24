#ifndef CREEL_TRACKER_HPP
#define CREEL_TRACKER_HPP

// OpenCV Headers
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

// C++ Standard Libraries
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>

namespace creelsense{
	// takes image and returns vector of [x, y, radius] for each point.
    std::vector<std::array<float, 3>> FindSpools(cv::Mat& src, bool display=false);
}

#endif

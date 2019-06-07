// lib include
#include <flycapture/FlyCapture2.h>
#include <flycapture/FlyCapture2GUI.h>

#include "creel_tracker/CreelTracker.hpp"

// OpenCV Headers
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

extern "C" {
#include "apriltag.h"
#include "apriltag_pose.h"
#include "tag36h11.h"
#include "tag25h9.h"
#include "tag16h5.h"
#include "tagCircle21h7.h"
#include "tagCircle49h12.h"
#include "tagCustom48h12.h"
#include "tagStandard41h12.h"
#include "tagStandard52h13.h"
#include "common/getopt.h"
}

// C++ Standard Libraries
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <sstream>
#include <zmq.hpp>

using namespace FlyCapture2;


apriltag_detection_info_t InitDetectionInfo() {
  apriltag_detection_info_t ret;
  
  ret.tagsize = .0292;//.0352;  // meter
  ret.fx = 1301.54;  // pixel
  ret.fy = 1302.22;  // pixel
  ret.cx = 633.76;  // pixel
  ret.cy = 474.73;  // pixel

  return ret;
}


std::string GetPoses(Camera& camera_FlyCap, apriltag_detector_t *atdt, apriltag_detection_info_t atdit) {

  // Get image from camera
  Error error_FlyCap;
  Image m_Image, m_ImageColor;
  cv::Mat frame, gray;
  error_FlyCap = camera_FlyCap.RetrieveBuffer(&m_Image);
  error_FlyCap = m_Image.Convert(PIXEL_FORMAT_BGR, &m_ImageColor);    
  frame = cv::Mat(cv::Size(m_ImageColor.GetCols(),m_ImageColor.GetRows()), CV_8UC3, (void*)m_ImageColor.GetData());
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

  // Make an image_u8_t header for the Mat data
  image_u8_t im = {.width = gray.cols, .height = gray.rows, .stride = gray.cols, .buf = gray.data};

  // Detect tags
  zarray_t *detections = apriltag_detector_detect(atdt, &im);
  size_t size_detections = zarray_size(detections);
  std::vector<apriltag_pose_t> poses;
  std::vector<std::string> ids;
  std::cout << "Image size = " << m_ImageColor.GetRows() << " x " << m_ImageColor.GetCols() << ",\t" << zarray_size(detections) << " tags detected" << std::endl;  // Debug

  // Add poses to std::vector and draw detection outlines
  for (int i = 0; i < zarray_size(detections); i++) {
    apriltag_detection_t *det;
    zarray_get(detections, i, &det);

    // Pose detection
    atdit.det = det;
    apriltag_pose_t pose;
    double err = estimate_tag_pose(&atdit, &pose);
    poses.push_back(pose);

    // Tag ID number
    std::stringstream ss;
    ss << det->id;
    cv::String text = ss.str();
    ids.push_back(text);

    // Draw tag axes in window
    line(frame, cv::Point(det->p[0][0], det->p[0][1]), cv::Point(det->p[1][0], det->p[1][1]), cv::Scalar(0, 0xff, 0), 2);
    line(frame, cv::Point(det->p[0][0], det->p[0][1]), cv::Point(det->p[3][0], det->p[3][1]), cv::Scalar(0, 0, 0xff), 2);
    line(frame, cv::Point(det->p[1][0], det->p[1][1]), cv::Point(det->p[2][0], det->p[2][1]), cv::Scalar(0xff, 0, 0), 2);
    line(frame, cv::Point(det->p[2][0], det->p[2][1]), cv::Point(det->p[3][0], det->p[3][1]), cv::Scalar(0xff, 0, 0), 2);
    
    int fontface = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
    double fontscale = 1.0;
    int baseline;
    cv::Size textsize = cv::getTextSize(text, fontface, fontscale, 2,
					&baseline);
    putText(frame, text, cv::Point(det->c[0]-textsize.width/2, det->c[1]+textsize.height/2), fontface, fontscale, cv::Scalar(0xff, 0x99, 0), 2);
  }  // End for loop over detections
  
  zarray_destroy(detections);

    
  //  Format return message
  std::ostringstream oss;
  int pose_index = 0;
  for (apriltag_pose_t pose : poses) {
    oss << "[" + ids[pose_index++] + ".0(";  // Including the ".0" is just for parsing the string on the client side, meh
    //      std::cout << "R->nrows = " << pose.R->nrows << "\tR->ncols = " << pose.R->ncols << std::endl;
    //      std::cout << "t->nrows = " << pose.t->nrows << "\tt->ncols = " << pose.t->ncols << std::endl;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
	oss << matd_get(pose.R, i, j) << " ";
	//std::cout << matd_get(pose.R, i, j) << " ";
      }
      //std::cout << std::endl;
    }
    oss << ")(";
    //std::cout << std::endl;
    for (int i = 0; i < 3; i++) {
      oss << matd_get(pose.t, i, 0) << " ";
      //std::cout << matd_get(pose.t, i, 0) << std::endl;
    }
    oss << ")]";
    //std::cout << std::endl << std::endl;
    //std::cout << "pose\n";
  }

  cv::imshow("Tag Detections", frame);
  cv::waitKey(15);
  
  return oss.str();
}


std::string toString(float number) {
    std::ostringstream buff;
    buff<<number;
    return buff.str();
}


std::string GetCircles(Camera& camera_FlyCap) {
  Error error_FlyCap;
  Image m_Image, m_ImageColor;
  cv::Mat* CvImg = nullptr;
  error_FlyCap = camera_FlyCap.RetrieveBuffer(&m_Image);
  error_FlyCap = m_Image.Convert(PIXEL_FORMAT_BGR, &m_ImageColor);

  CvImg = new cv::Mat(cv::Size(m_ImageColor.GetCols(),m_ImageColor.GetRows()), CV_8UC3, (void*)m_ImageColor.GetData());

  cv::imshow("ptgrey_img", *CvImg);
  cv::waitKey(15);
  std::vector<std::array<float, 3>> spools_found = creelsense::FindSpools(*CvImg, true);

  //  Format return message  
  std::string ret = "";
  for (int i(0); i<spools_found.size(); ++i) {
    ret += "[";
    for (int j(0); j<3; ++j) {
      ret += toString(spools_found[i][j]);
      if (j <2) {
	ret += ',';
      }
    }
    ret += "]";
    if (i < spools_found.size()-1) {
      ret +=  ",";
    }
  }

  return ret;
}


// Save an image and return the absolute path to it
std::string GetImagePath(Camera& camera_FlyCap) {
  std::string ret = "/home/mqm/Desktop/current_picture.png";
  Error error_FlyCap;
  Image m_Image, m_ImageColor;
  cv::Mat frame, gray;
  error_FlyCap = camera_FlyCap.RetrieveBuffer(&m_Image);
  error_FlyCap = m_Image.Convert(PIXEL_FORMAT_BGR, &m_ImageColor);
  frame = cv::Mat(cv::Size(m_ImageColor.GetCols(),m_ImageColor.GetRows()), CV_8UC3, (void*)m_ImageColor.GetData());
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

  std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(0);

  try {
    cv::imwrite(ret, frame, compression_params);
  }
  catch (std::runtime_error& ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
    return "Error in GetImagePath";
  }

  return ret;
}


int main(int argc, char *argv[]) {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(NULL);

  getopt_t *getopt = getopt_create();

  getopt_add_bool(getopt, 'h', "help", 0, "Show this help");
  getopt_add_bool(getopt, 'd', "debug", 1, "Enable debugging output (slow)");
  getopt_add_bool(getopt, 'q', "quiet", 0, "Reduce output");
  getopt_add_string(getopt, 'f', "family", "tag36h11", "Tag family to use");
  getopt_add_int(getopt, 't', "threads", "1", "Use this many CPU threads");
  getopt_add_double(getopt, 'x', "decimate", "2.0", "Decimate input image by this factor");
  getopt_add_double(getopt, 'b', "blur", "0.0", "Apply low-pass blur to input");
  getopt_add_bool(getopt, '0', "refine-edges", 1, "Spend more time trying to align edges of tags");

  if (!getopt_parse(getopt, argc, argv, 1) ||
      getopt_get_bool(getopt, "help")) {
    printf("Usage: %s [options]\n", argv[0]);
    getopt_do_usage(getopt);
    exit(0);
  }


  // Prepare our context and socket
  zmq::context_t context(1);
  zmq::socket_t socket(context, ZMQ_REP);
  socket.bind("tcp://127.0.0.1:43001");

  
  // Initialize camera
  Error m_error;
  BusManager m_BusManager;
  Camera m_Cam;
  PGRGuid m_Guid;

  m_error = m_BusManager.GetCameraFromIndex(0,&m_Guid);
  m_error = m_Cam.Connect(&m_Guid);  
  m_error = m_Cam.StartCapture();

  
  // Initialize tag detector with options
  apriltag_family_t *tf = NULL;
  const char *famname = getopt_get_string(getopt, "family");
  if (!strcmp(famname, "tag36h11")) {
    tf = tag36h11_create();
  } else if (!strcmp(famname, "tag25h9")) {
    tf = tag25h9_create();
  } else if (!strcmp(famname, "tag16h5")) {
    tf = tag16h5_create();
  } else if (!strcmp(famname, "tagCircle21h7")) {
    tf = tagCircle21h7_create();
  } else if (!strcmp(famname, "tagCircle49h12")) {
    tf = tagCircle49h12_create();
  } else if (!strcmp(famname, "tagStandard41h12")) {
    tf = tagStandard41h12_create();
  } else if (!strcmp(famname, "tagStandard52h13")) {
    tf = tagStandard52h13_create();
  } else if (!strcmp(famname, "tagCustom48h12")) {
    tf = tagCustom48h12_create();
  } else {
    printf("Unrecognized tag family name. Use e.g. \"tag36h11\".\n");
    exit(-1);
  }

  apriltag_detector_t *td = apriltag_detector_create();
  apriltag_detector_add_family(td, tf);
  td->quad_decimate = getopt_get_double(getopt, "decimate");
  td->quad_decimate = 1.0;
  td->quad_sigma = getopt_get_double(getopt, "blur");
  td->nthreads = getopt_get_int(getopt, "threads");
  td->debug = getopt_get_bool(getopt, "debug");
  td->refine_edges = getopt_get_bool(getopt, "refine-edges");

  // For pose estimation, create an apriltag_detection_info_t struct using known parameters
  apriltag_detection_info_t info = InitDetectionInfo();
  

  while (true) {
    // Wait for next request from client
    std::cout << "Server awaiting message " << std::endl;
    zmq::message_t request;
    std::string request_str;
    socket.recv(&request);
    request_str = std::string(static_cast<char*>(request.data()), request.size());
    std::cout << "request_str = " << request_str << std::endl;
    // Deal with request and create response string
    std::string response_str = "{";

    if (0 == request_str.compare("Request peg poses"))
      response_str += GetPoses(m_Cam, td, info) + "}";
      
    else if (0 == request_str.compare("Request spool circles"))
      response_str += GetCircles(m_Cam) + "}";
      
    else if (0 == request_str.compare("Request image"))
      response_str = GetImagePath(m_Cam);

    else // Did not recieve the right message      
      response_str = "{\"Error\"}";

    std::cout << response_str << std::endl;

    //  Send reply back to client
    zmq::message_t reply(response_str.size());
    memcpy(reply.data(), response_str.data(), response_str.size());
    socket.send(reply);

  }  // End while loop

  
  // Cleanup (but execution currently has no way to reach this point MQM 190415)
  apriltag_detector_destroy(td);

  if (!strcmp(famname, "tag36h11")) {
    tag36h11_destroy(tf);
  } else if (!strcmp(famname, "tag25h9")) {
    tag25h9_destroy(tf);
  } else if (!strcmp(famname, "tag16h5")) {
    tag16h5_destroy(tf);
  } else if (!strcmp(famname, "tagCircle21h7")) {
    tagCircle21h7_destroy(tf);
  } else if (!strcmp(famname, "tagCircle49h12")) {
    tagCircle49h12_destroy(tf);
  } else if (!strcmp(famname, "tagStandard41h12")) {
    tagStandard41h12_destroy(tf);
  } else if (!strcmp(famname, "tagStandard52h13")) {
    tagStandard52h13_destroy(tf);
  } else if (!strcmp(famname, "tagCustom48h12")) {
    tagCustom48h12_destroy(tf);
  }

  getopt_destroy(getopt);

  return 0;
}

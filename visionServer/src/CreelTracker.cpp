#include "creel_tracker/CreelTracker.hpp"



std::vector<std::array<float, 3>> creelsense::FindSpools(cv::Mat& src, bool display){
  using namespace std;
  std::vector<std::vector<cv::Point> > contours;
  cv::Mat contourOutput, plain, gray, edges, blob;
  std::vector<cv::Point> hull;
  cv::Point2f center;
  float aspect_ratio, radius, convex_to_orig_area, area_perim_ratio, score;
  cv::Scalar meanvar, stdvar;
  const int thresh_low(0), thresh_high(65); // light/dark threshold
  std::vector<std::array<float, 3>> ret;
  auto end = chrono::high_resolution_clock::now();
  auto start = chrono::high_resolution_clock::now();

  if(src.empty()){
    cout << "Error: image passed to FindSpools is empty.\n";
    return ret;
  }
  if (src.rows > src.cols){
    cv::rotate(src, src, cv::ROTATE_90_CLOCKWISE);
  }
  src.copyTo(plain);
  aspect_ratio = (float)plain.rows / plain.cols;
  cv::resize(plain, plain, cv::Size(640, (int)(aspect_ratio*640))); // make small for display

  cv::cvtColor(plain, gray, cv::COLOR_BGR2GRAY);
	
  cv::meanStdDev(gray, meanvar, stdvar);
  cout << -1  << '\t' << meanvar[0] << '\t' << stdvar[0] << '\n';
  gray -= meanvar[0]-121;
	
  double minVal; 
  double maxVal; 
  cv::Point minLoc; 
  cv::Point maxLoc;
  cv::minMaxLoc( gray, &minVal, &maxVal, &minLoc, &maxLoc );
  cv::normalize(gray, gray, minVal, maxVal, cv::NORM_MINMAX);
	
  //cv::blur(gray, edges, cv::Size(3,3));
  //~ gray.copyTo(edges);

  //~ cv::Canny(edges, edges, 45, 115, 3);
  //~ for (int i(0); i<3; ++i){
  //~ cv::dilate(edges, edges, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
  //~ }
  //~ for (int i(0); i<4; ++i){
  //~ cv::erode(edges, edges, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
  //~ }
  //~ if (display){
  //~ end = chrono::high_resolution_clock::now();
  //~ cout << "edges morphops: Elapsed time in seconds : " 
  //~ << chrono::duration<double>(end - start).count()
  //~ << " sec" << endl << endl;
  //~ }
  
  //~ cv::imshow("edges", edges);
  
  cv::blur(gray, blob, cv::Size(3,3));
  
  cv::inRange(blob, thresh_low, thresh_high, blob);
  
  for (int i(0); i <5; ++i){
    cv::dilate(blob, blob, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
  }
  for (int i(0); i <3; ++i){
    cv::erode(blob, blob, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
  }
  if (display){
    cv::imshow("blob", blob);
    end = chrono::high_resolution_clock::now();
    cout << "color_threshold morphops: Elapsed time in seconds : " 
	 << chrono::duration<double>(end - start).count()
	 << " sec" << endl;
  }
  
  contourOutput = blob.clone();
  cv::findContours(contourOutput, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE );
  
  for (size_t idx = 0; idx < contours.size(); idx++) {
    if (cv::contourArea(contours[idx]) < blob.rows/20*blob.cols/20)
      {
	continue; // skip small ones
      }

    cv::convexHull(contours[idx], hull);
    convex_to_orig_area = cv::contourArea(contours[idx]) / cv::contourArea(hull);
    if (display){
      cout << "convex to reg area ratio is " << convex_to_orig_area << '\n';
    }
    if (convex_to_orig_area < 0.7){
      continue;
    }
    
    //~ if (display){
    //~ cout << "area/perimeter ratio" << cv::contourArea(contours[idx]) / cv::arcLength(contours[idx], true) << '\n';
    //~ }
    
    //~ area_perim_ratio = cv::contourArea(contours[idx]) / cv::arcLength(contours[idx], true);
    /*if (area_perim_ratio > 16 || area_perim_ratio < .65){
      continue;
      }*/
    
    // mean and std of blobs
    cv::minEnclosingCircle(contours[idx], center, radius);
    cv::Mat1b mask(blob.rows, blob.cols, uchar(0));
    cv::circle(mask, center, radius*0.7, 255, -1);
    cv::meanStdDev(blob, meanvar, stdvar, mask);
    if (display){
      cout << 1 << '\t' << meanvar[0] << '\t' << stdvar[0] << '\n';
    }
    if (meanvar[0] < 210 || stdvar[0] > 110){
      continue;
    }
    
    
    // mean and std of original image
    cv::meanStdDev(plain, meanvar, stdvar, mask);
    if (display){
      cout << 1 << '\t' << meanvar[0] << '\t' << stdvar[0] << '\n';
    }
    if (meanvar[0] >70 || stdvar[0] > 115){
      continue;
    }
    
    // mean and std of area surrounding spool (is there white yarn around the spool)
    mask = mask*0;
    cv::circle(mask, center, radius*2.4, 255, -1); // extermal
    cv::circle(mask, center, radius, 0, -1); // subtract radius
    cv::meanStdDev(blob, meanvar, stdvar, mask);
    if (display){
      cout << 2  << '\t' << meanvar[0] << '\t' << stdvar[0] << '\n';
    }
    if (meanvar[0] > 30){
      continue;
    }
    
    // mean and std of edge images (finds the lines of the yarn)
    //~ cv::meanStdDev(edges, meanvar, stdvar, mask);
    //~ if (display){
    //~ cout << 3  << '\t' << meanvar[0] << '\t' << stdvar[0] << '\n';
    //~ }
    //~ if (meanvar[0] < 10){
    //~ continue;
    //~ }
    
    //    ret.push_back(std::array<float, 3>{center.x / contourOutput.cols, center.y/contourOutput.rows, radius/contourOutput.cols});  // This is AG original.  Normalized values?
    cout << "Image width = " << contourOutput.cols << "\tImage height = " << contourOutput.rows << endl;
    ret.push_back(std::array<float, 3>{center.x, center.y, radius});  // This is MQM 190327
    
    if (display){
      cv::circle(gray, center, radius, cv::Scalar(255,0, 0), 2, 8, 0 );
    }
    
    // Pause after every image and draw a circle
    /*if (display){
      cv::imshow("contourImage",gray);
      cv::waitKey(0);
      }//*/
  }
  if (display){
    end = chrono::high_resolution_clock::now();
    cout << "contour filtering: Elapsed time in seconds : " 
	 << chrono::duration<double>(end - start).count()
	 << " sec" << endl;
    cv::imshow("contourImage",gray);
    cv::waitKey(2);
    //~ cv::waitKey(0);
  }
    
  return ret;
}


#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <iostream>
#include <vector>
 
static double computeReprojectionErrors(
        const std::vector<std::vector<cv::Point3f> >& objectPoints,
        const std::vector<std::vector<cv::Point2f> >& imagePoints,
        const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
        const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
        std::vector<float>& perViewErrors )
{
    std::vector<cv::Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for( i = 0; i < (int)objectPoints.size(); i++ )
    {
        cv::projectPoints(cv::Mat(objectPoints[i]), rvecs[i], tvecs[i],
                      cameraMatrix, distCoeffs, imagePoints2);

        err = cv::norm(cv::Mat(imagePoints[i]), cv::Mat(imagePoints2), cv::NORM_L2);

        int n = (int)objectPoints[i].size();

        perViewErrors[i] = (float)std::sqrt(err*err/n);
        
        totalErr += err*err;
        totalPoints += n;
    }

    // return rms value of errors
    return std::sqrt(totalErr/totalPoints);
}

// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{7,10}; 
 

int main()
{
  // Creating vector to store set of 3D points for each checkerboard image
  std::vector<std::vector<cv::Point3f> > objpoints;
 
  // Creating vector to store set of 2D points for each checkerboard image
  std::vector<std::vector<cv::Point2f> > imgpoints;
 
  // Defining the world coordinates for 3D points
  std::vector<cv::Point3f> objp;
  for(int i{0}; i<CHECKERBOARD[1]; i++) {
    for(int j{0}; j<CHECKERBOARD[0]; j++)
      objp.push_back(cv::Point3f(j,i,0));
  } 
 
  cv::VideoCapture vid(1);
  if(!vid.isOpened()) {
    std::cout << "Error opening device" << std::endl;
    return -1;
  }
 
  cv::Mat frame, gray;
  // vector to store the pixel coordinates of detected checker board corners 
  std::vector<cv::Point2f> corner_pts;
  bool success, save;
  int count = 0;
 
  // Looping over video feed until atleast 20 good frames are captured
  while(count < 20) {
    vid >> frame;
    cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);
 
    // Finding checker board corners
    // If desired number of corners are found in the image then success = true  
    success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), 
      corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
     
    /* 
     * If desired number of corner are detected,
     * we refine the pixel coordinates and display 
     * them on the images of checkered board
    */
    if(success) {
      cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.001);
       
      // refining pixel coordinates for given 2d points.
      cv::cornerSubPix(gray,corner_pts,cv::Size(11,11), cv::Size(-1,-1),criteria);
       
      // Displaying the detected corner points on the checker board
      cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
    
      if(save) {
        objpoints.push_back(objp);
        imgpoints.push_back(corner_pts);
        count++;
        save = false;
      }
    }
 
    cv::putText(frame, "Count: " + std::to_string(count), cv::Point(500, 30), 
      cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0,255,255));

    cv::imshow("Image", frame);
    char key = cv::waitKey(1);

    if(key == 'c')
        save = true;

    if(key == 'q')
        break;
  }
 
  cv::destroyAllWindows();

  /*
   * Performing camera calibration by 
   * passing the value of known 3D points (objpoints)
   * and corresponding pixel coordinates of the 
   * detected corners (imgpoints)
  */

  std::vector<cv::Mat> R;
  std::vector<cv::Mat> T;
  cv::Mat cameraMatrix, distCoeffs;
  
  cv::calibrateCamera(objpoints, imgpoints,
    cv::Size(gray.cols, gray.rows), cameraMatrix, distCoeffs, R, T);
  
  std::cout << "camera Matrix : " << cameraMatrix << std::endl;
  std::cout << "dist Coeffs : " << distCoeffs << std::endl;

  std::vector<float> perViewErrors;
  double rePrjErr = computeReprojectionErrors(objpoints, imgpoints, R, T, 
            cameraMatrix, distCoeffs, perViewErrors);

  std::cout << "Reprojection Error : " << rePrjErr << std::endl;
 
  return 0;
}

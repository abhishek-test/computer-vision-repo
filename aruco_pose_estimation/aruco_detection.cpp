
#include "opencv2/aruco.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;
 
int main(int argc, char* argv[])
{
  int camId = 1; //atoi(argv[1]);
  int dictId = 5; //atoi(argv[2]);

  cv::VideoCapture vid(camId);
  Mat inputImage;
  
  vid.set(CAP_PROP_FRAME_HEIGHT, 480);
  vid.set(CAP_PROP_FRAME_WIDTH,  640);  
  
  std::vector<int> markerIds;
  std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;

  aruco::DetectorParameters parameters;
  cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(dictId);
  aruco::ArucoDetector detector(dictionary, parameters); 

  std::vector<int> markerIds_selected;
  std::vector<std::vector<cv::Point2f>> markerCorners_selected, rejectedCandidates_selected;

  while(1) {
  
    vid >> inputImage;
    detector.detectMarkers(inputImage, markerCorners, markerIds, rejectedCandidates);

    int neededId = 0;
    for(int idx=0; idx<markerIds.size(); idx++) {
      if(markerIds[idx] == 24) {
        neededId = idx;
        break;
      }
    }    

    markerIds_selected.push_back(neededId);
    markerCorners_selected.push_back(markerCorners[neededId]);
    
    //if(markerIds.size() > 0)
      //cv::aruco::drawDetectedMarkers(inputImage, markerCorners, markerIds);

    if(markerIds.size() > 0)
      cv::aruco::drawDetectedMarkers(inputImage, markerCorners_selected, markerIds_selected);

    markerIds_selected.clear();
    markerCorners_selected.clear();
      
    imshow("Input Video", inputImage);
    
    if(waitKey(1) == 'q')
      break;
  }
  
}
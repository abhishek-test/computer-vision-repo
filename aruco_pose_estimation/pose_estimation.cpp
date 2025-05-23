
#include "opencv2/opencv.hpp"
#include "opencv2/aruco.hpp"

using namespace cv;
using namespace std;

int main()
{
    cv::VideoCapture cap(1);
    
    cv::Mat frame;
    cv::Mat cameraMatrix, distCoeffs;
    float markerLength = 0.05;    
   
    // Set coordinate system
    cv::Mat objPoints(4, 1, CV_32FC3);
    objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength/2.f,  markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerLength/2.f,   markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerLength/2.f,  -markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength/2.f, -markerLength/2.f, 0);

    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(5);
    aruco::ArucoDetector detector(dictionary, detectorParams);    

    while (cap.read(frame)) {

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        detector.detectMarkers(frame, corners, ids);

        // If at least one marker detected
        if (ids.size() > 0) {

            cv::aruco::drawDetectedMarkers(frame, corners, ids);
            int nMarkers = corners.size();
            std::vector<cv::Vec3d> rvecs(nMarkers), tvecs(nMarkers);

            // Calculate pose for each marker
            for (int i = 0; i < nMarkers; i++) {
                solvePnP(objPoints, corners.at(i), cameraMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));
            }

            // Draw axis for each marker
            for(unsigned int i = 0; i < ids.size(); i++) {
                cv::drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
            }
        }

        // Show resulting frame
        cv::imshow("Estimated Pose", frame);
        char key = (char) cv::waitKey(1);

        if (key == 'q')
            break;
    }
}
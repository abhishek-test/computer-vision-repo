
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/opencv.h>

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

using namespace std;
using namespace cv;
using namespace dlib;

float dist(Point2f& P1, Point2f& P2) {
  float temp = (P1.x - P2.x)*(P1.x - P2.x) + (P1.y - P2.y)*(P1.y - P2.y);
  return (sqrtf32(temp));
}

float calculateEAR(std::vector<cv::Point2f>& landmarks) {

  float earL = (dist(landmarks[37], landmarks[41]) + dist(landmarks[38], landmarks[40])) / 
                  (2.0*(dist(landmarks[36], landmarks[39])));
  float earR = (dist(landmarks[43], landmarks[47]) + dist(landmarks[44], landmarks[46])) / 
                  (2.0*(dist(landmarks[42], landmarks[45])));

  return ((earL + earR)/2.0);
}

static void draw_delaunay( Mat& img, Subdiv2D& subdiv)
{
  std::vector<cv::Vec6f> triangleList;
  subdiv.getTriangleList(triangleList);
  std::vector<cv::Point> pt(3);
  Size size = img.size();
  Rect rect(0,0, size.width, size.height);
 
  for( size_t i = 0; i < triangleList.size(); i++ )
  {
    Vec6f t = triangleList[i];
    pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
    pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
    pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
 
    // Draw rectangles completely inside the image.
    if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
    {
      cv::line(img, pt[0], pt[1], cv::Scalar(0,255,255), 1, 8, 0);
      cv::line(img, pt[1], pt[2], cv::Scalar(0,255,255), 1, 8, 0);
      cv::line(img, pt[2], pt[0], cv::Scalar(0,255,255), 1, 8, 0);
    }
  }

  triangleList.clear();
  pt.clear();

}

int main(int argc,char** argv)
{
  VideoCapture cam(0);
  Mat frame, gray; 

  shape_predictor pose_model;
  std::vector<full_object_detection> shapes;
  std::vector<dlib::rectangle> faces_dlib;
  std::vector<cv::Point2f> landmarks;
  std::vector<cv::Rect> faces;
  int eyeClosedCounter = 0;

  deserialize("./shape_predictor_68_face_landmarks.dat") >> pose_model;
  CascadeClassifier faceDetector("./haarcascade_frontalface_alt2.xml");
  
  // Read a frame
  while(cam.read(frame)) {    
      
    faces.clear();
    shapes.clear();
    landmarks.clear();
    faces_dlib.clear();

    cv_image<bgr_pixel> cimg(frame);
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    faceDetector.detectMultiScale(gray, faces);

    Subdiv2D subdiv(Rect(0, 0, frame.cols, frame.rows));
      
    if(faces.size() > 0) 
    {
      for(int i = 0; i < faces.size(); i++) {
        faces_dlib.push_back(dlib::rectangle(faces[i].x, faces[i].y, 
          faces[i].x + faces[i].width, faces[i].y + faces[i].height));
        cv::rectangle(frame, faces[i], Scalar(255,0,0), 2, 8);
      }

      for(unsigned long i = 0; i < faces.size(); ++i) {
        shapes.push_back(pose_model(cimg, faces_dlib[i]));
      }

      for(int j=0; j<shapes[0].num_parts(); j++) {
        cv::circle(frame, cv::Point(shapes[0].part(j).x(), shapes[0].part(j).y()), 3, Scalar(0,0,255), -1, 8);
        landmarks.push_back(cv::Point2f(shapes[0].part(j).x(), shapes[0].part(j).y()));
      }

      subdiv.insert(landmarks);     

      float ear = calculateEAR(landmarks);
      cv::putText(frame, "EAR : " + to_string(ear), Point(20, 20), 
        cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0,0,0), 2, 8);

      if(ear < 0.28) { 
        eyeClosedCounter++;
      }

      else {
        eyeClosedCounter = 0;
      }

      if(eyeClosedCounter > 50) {
        cv::circle(frame, cv::Point(20, 70), 8, cv::Scalar(0,0,255), -1, 8);
        system("canberra-gtk-play -f ./alarm2.wav"); // blocking system call, need to be replaced with threads ??
      }

      draw_delaunay(frame, subdiv);
    }

    else {
      cv::circle(frame, cv::Point(20, 70), 8, cv::Scalar(0,0,255), -1, 8);
    }

    cv::putText(frame, "Counter : " + to_string(eyeClosedCounter), 
        cv::Point(20, 40), cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0,0,0), 2, 8);

    cv::imshow("Pose Estimation", frame);

    if(cv::waitKey(1)=='q')
      break;
  
  }
}
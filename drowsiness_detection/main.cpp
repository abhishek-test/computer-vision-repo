

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/generic_image.h>

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

using namespace std;
using namespace cv;
using namespace dlib;

int main(int argc,char** argv)
{
  shape_predictor pose_model;
  CascadeClassifier faceDetector(".\\haarcascade_frontalface_alt2.xml");
  deserialize(".\\shape_predictor_68_face_landmarks.dat") >> pose_model;
  std::vector<full_object_detection> shapes;

  VideoCapture cam(1);
  Mat frame, gray; 

  std::vector<cv::Rect> faces;
  std::vector<dlib::rectangle> faces_dlib;
  image_window win;

  // Read a frame
  while(cam.read(frame)) {
      
    faces.clear();
    faces_dlib.clear();
    shapes.clear();

    cv_image<bgr_pixel> cimg(frame);
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    faceDetector.detectMultiScale(gray, faces);
      
    if(faces.size() > 0) {
      for(int i = 0; i < faces.size(); i++)
        faces_dlib.push_back(dlib::rectangle(faces[i].x, faces[i].y, faces[i].x + faces[i].width, faces[i].y + faces[i].height));
      }

    for(unsigned long i = 0; i < faces.size(); ++i)
      shapes.push_back(pose_model(cimg, faces_dlib[i]));

    win.clear_overlay();
    win.set_image(cimg);
    win.add_overlay(render_face_detections(shapes));
      
  }
}
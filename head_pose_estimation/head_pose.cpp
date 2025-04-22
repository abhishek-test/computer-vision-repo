
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

bool isRotationMatrix(Mat &R)
{
    Mat Rt;
    transpose(R, Rt);
    Mat shouldBeIdentity = Rt * R;
    Mat I = Mat::eye(3,3, shouldBeIdentity.type());
    float temp = norm(I, shouldBeIdentity);
    //cout << "Temp: " << temp << endl;  // 9.7
 
    return  temp < 1e-6;
}

Vec3f rotationMatrixToEulerAngles(Mat &R)
{
    assert(isRotationMatrix(R));
 
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) ); 
    bool singular = sy < 1e-6; // If
 
    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return Vec3f(x, y, z);
 
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
  //VideoCapture cam("./input_video.mp4");
  VideoCapture cam(0);
  Mat frame, gray; 

  shape_predictor pose_model;
  std::vector<full_object_detection> shapes;
  std::vector<dlib::rectangle> faces_dlib;
  std::vector<cv::Point2f> landmarks;
  std::vector<cv::Point3f> landmarks_3D;
  std::vector<cv::Point2f> landmarks_2D;
  std::vector<cv::Rect> faces;
  cv::Vec3f eulerAngles;
  int eyeClosedCounter = 0;

  deserialize("./shape_predictor_68_face_landmarks.dat") >> pose_model;
  CascadeClassifier faceDetector("./haarcascade_frontalface_alt2.xml");

  landmarks_3D.push_back(cv::Point3f(-225.0, 170.0, -135.0));
  landmarks_3D.push_back(cv::Point3f( 225.0, 170.0, -135.0));
  landmarks_3D.push_back(cv::Point3f( 0.0, 0.0, 0.0));
  landmarks_3D.push_back(cv::Point3f(-150.0, -150.0, -125.0));
  landmarks_3D.push_back(cv::Point3f(150.0, -150.0, -125.0));
  landmarks_3D.push_back(cv::Point3f(0.0, -330.0, -65.0));

  double focal_length = cam.get(CAP_PROP_FRAME_WIDTH); 
  Point2d center = cv::Point2d(cam.get(CAP_PROP_FRAME_WIDTH)/2, cam.get(CAP_PROP_FRAME_HEIGHT)/2);
  cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
  cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type);

  cv::Mat rotation_vector; // Rotation in axis-angle form
  cv::Mat translation_vector;

  int t1, t2;
  double tickFreq = cv::getTickFrequency();
  
  // Read a frame
  while(cam.read(frame)) {    

    t1 = t2 = 0;    

    t1 = getTickCount();
      
    faces.clear();
    shapes.clear();
    landmarks.clear();
    faces_dlib.clear();
    landmarks_2D.clear();
    
    cv_image<bgr_pixel> cimg(frame);
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    
    faceDetector.detectMultiScale(gray, faces);  // 100 ms

    Subdiv2D subdiv(Rect(0, 0, frame.cols, frame.rows));  //46 ms

    if(faces.size() > 0) 
    {
      for(int i = 0; i < faces.size(); i++) {
        faces_dlib.push_back(dlib::rectangle(faces[i].x, faces[i].y, 
          faces[i].x + faces[i].width, faces[i].y + faces[i].height));

        cv::rectangle(frame, faces[i], Scalar(0,255,0), 2, 8);
      }

      for(unsigned long i = 0; i < faces.size(); ++i) {
        shapes.push_back(pose_model(cimg, faces_dlib[i]));  // 45 ms
      }

      for(int j=0; j<shapes[0].num_parts(); j++) {
        //cv::circle(frame, cv::Point(shapes[0].part(j).x(), shapes[0].part(j).y()), 1, Scalar(0,0,255), -1, 8);
        landmarks.push_back(cv::Point2f(shapes[0].part(j).x(), shapes[0].part(j).y()));
      }      

      //subdiv.insert(landmarks);
      //draw_delaunay(frame, subdiv);

      landmarks_2D.push_back(landmarks[36]);
      landmarks_2D.push_back(landmarks[45]);
      landmarks_2D.push_back(landmarks[30]);
      landmarks_2D.push_back(landmarks[48]);
      landmarks_2D.push_back(landmarks[54]);
      landmarks_2D.push_back(landmarks[8]);      

      cv::solvePnP(landmarks_3D, landmarks_2D, camera_matrix, dist_coeffs, rotation_vector, translation_vector); // 1 ms

      // calculate euler angles from rot_vec
      Mat rot_mat;
      cv::Rodrigues(rotation_vector, rot_mat);
      eulerAngles = rotationMatrixToEulerAngles(rot_mat);
      cv::putText(frame, "X : " + to_string(eulerAngles(0)), cv::Point(30,60), FONT_HERSHEY_COMPLEX, 0.6, Scalar::all(0), 1, 8 );
      cv::putText(frame, "Y : " + to_string(eulerAngles(1)), cv::Point(30,90), FONT_HERSHEY_COMPLEX, 0.6, Scalar::all(0), 1, 8 );
      cv::putText(frame, "Z : " + to_string(eulerAngles(2)), cv::Point(30,120), FONT_HERSHEY_COMPLEX, 0.6, Scalar::all(0), 1, 8 );

      // 0 ms

      std::vector<Point3f> nose_end_point3D;
      std::vector<Point2f> nose_end_point2D;

      nose_end_point3D.push_back(Point3f(0,0,200.0));
      nose_end_point3D.push_back(Point3f(0,200.0,0));
      nose_end_point3D.push_back(Point3f(200.0,0,0));
 
      projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);
 
      for(int i=0; i < landmarks_2D.size(); i++) {
        circle(frame, landmarks_2D[i], 3, Scalar(0,255,255), -1);
      }
 
      cv::line(frame, landmarks_2D[2], nose_end_point2D[0], cv::Scalar(255,0,0), 2);
      cv::line(frame, landmarks_2D[2], nose_end_point2D[1], cv::Scalar(0,255,0), 2);
      cv::line(frame, landmarks_2D[2], nose_end_point2D[2], cv::Scalar(0,0,255), 2);
    }

    t2 = getTickCount();
    int time = (int)(1000.0 * (1.0*(t2-t1)/(1.0*tickFreq)));

    cv::putText(frame, "Time (ms) : " + to_string(time), cv::Point(30,30), FONT_HERSHEY_COMPLEX, 0.6, Scalar::all(0), 1, 8 );

    cv::imshow("Pose Estimation", frame);

    if(cv::waitKey(1)=='q')
      break;
  
  }
}
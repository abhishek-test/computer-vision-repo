#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main()
{
    VideoCapture vid("C:\\Abhishek_Data\\My_Data\\Datasets\\videos\\drone_traffic_2.mp4");
    Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2(); 
    Mat frame, thresh, fgMask, filtered;    
    
    while(1) {

        vid >> frame;
        resize(frame, frame, Size(1280, 720));

        pBackSub->apply(frame, fgMask);
        threshold(fgMask, thresh, 150, 255, THRESH_BINARY);
        medianBlur(thresh, filtered, 7);

        vector<vector<Point> > contours;
        findContours(filtered, contours, RETR_TREE, CHAIN_APPROX_SIMPLE );

        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect(contours.size());
        
        for( size_t i = 0; i < contours.size(); i++ ) {        
            approxPolyDP( contours[i], contours_poly[i], 3, true );
            boundRect[i] = boundingRect( contours_poly[i] );        
        }
        
        for( size_t i = 0; i< contours.size(); i++ )  {      
            int area = boundRect[i].area();
            if((area > 25) && (area < 300))
                rectangle(frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0,0,255), 2 );
        }

        imshow("Detections", frame);

        char c = waitKey(1);

        if(c == 'p')
            waitKey(0);

        if(c=='q') 
            break;       
    }
}


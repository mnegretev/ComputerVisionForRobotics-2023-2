#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <math.h>
using namespace cv;
using namespace std;

int k_size = 5; //Variable kernel size for the gaussian filter
int t1=1; // lower threshold
int t2=4; // upper threshold
double pi = 3.14159265358979323846;

cv::Mat filter_gaussian(int k,float sigma){
    int siz=(int)k/2;
    float suma=0;
    float H[2*siz+1][2*siz+1]={}; //-1
    float arg=0;// 1 2 3
    for(int i=0; i<= 2*siz; i++){
        for(int j=0; j<= 2*siz; j++){
            arg=(pow(i-siz,2)+pow(j-siz,2))/(2*pow(sigma,2));
            H[i][j]=exp(-arg)/(2*pi*pow(sigma,2));
            suma=suma+H[i][j];
        }
    }
    for(int i=0; i<= 2*siz; i++){
        for(int j=0; j<= 2*siz; j++){
            H[i][j]=H[i][j]/suma*255;
            cout<<H[i][j]<<endl;
        }
    }
        //for j in range(2*siz+1):
            //arg=-((i-siz)**2+(j-siz)**2)/(2*sigma**2)
            //H[i,j]=math.exp(arg)/(2*math.pi*sigma**2)
    //H=H/np.sum(H)
    //uint8_t Array=atoi(H);
    //cout<<H<<endl;
    //CV_32F  CV_8UC1
    cv::Mat Kernel = cv::Mat(2*siz+1, 2*siz+1, CV_32F, &H);
    cout<<Kernel<<endl;
    return Kernel;
}

void on_threshold_changed(int, void*){}

int main(int, char**)
{
    Point anchor;
    double delta;
    int ddepth;
    cv::Mat frame;
    cv::Mat gray;
    cv::Mat img_filter;
    float sigma=1;
    cv::Mat kernel=filter_gaussian(k_size, sigma);
    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture cap;
    // open the default camera using default API
    cap.open(0);
    cap.set(CAP_PROP_FRAME_WIDTH, 320);//Setting the width of the video
    cap.set(CAP_PROP_FRAME_HEIGHT, 240);//Setting the height of the video//
    cv::namedWindow("Original");
    cv::createTrackbar("Umbral inferior:", "Original", &t1, 14, on_threshold_changed);
    // check if we succeeded
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    //--- GRAB AND WRITE LOOP
    cout << "Start grabbing" << endl
        << "Press ESC to terminate" << endl;
    while(cap.isOpened())
    {
        // wait for a new frame from camera and store it into 'frame'
        cap.read(frame);
        // check if we succeeded
        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        gray = frame.clone();
        //cv::Mat gray;
        //cvtColor(image, grayImage, CV_BGR2GRAY);
        cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
        // Initialize arguments for the filter
        anchor = Point( -1, -1 );
        delta = 0;
        ddepth = -1;
        //cout << gray.type() << " == " << CV_8U << endl;
        //gray.convertTo(gray,CV_32FC1);
        filter2D(gray, img_filter, -1 , kernel, cv::Point(-1, -1), delta, BORDER_DEFAULT );
        //img_filter.convertTo(img_filter,CV_8UC1);
        //show live and wait for a key with timeout long enough to show images
        imshow("Original", frame);
        imshow("Grey",gray);
        imshow("Kernel", img_filter);
        //float sigma;
        //cv::Mat kernel=filter_gaussian(k_size, sigma);
        //cout<<kernel<<endl;
        if (waitKey(30) == 27)
            break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
    //https://stackoverflow.com/questions/69338519/generated-gaussian-kernel-saturates-the-image-to-white-color-with-opencv-in-c
}

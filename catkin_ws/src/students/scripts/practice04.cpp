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
int t1=5; // lower threshold
int t2=10; // upper threshold
double pi = 3.14159265358979323846;
double delta=0;
int ddepth=-1;
int hough_threshold = 100;

cv::Mat get_sobel_x_gradient(cv::Mat& img){
    float Gx[3][3]={-1/8.0,0,1/8.0,-2/8.0,0,2/8.0,-1/8.0,0,1/8.0};
    cv::Mat SobelX=Mat(3, 3, CV_32F, &Gx);
    cv::Mat img_filter;
    //cout<<SobelX<<endl;
    filter2D(img, img_filter, ddepth , SobelX, cv::Point(-1, -1), delta, BORDER_DEFAULT );
    //CV_32F  CV_8UC1
    return img_filter.clone();
}

cv::Mat get_sobel_y_gradient(cv::Mat& img){
    float Gy[3][3]={-1/8.0,-2/8.0,-1/8.0,0,0,0,1/8.0,2/8.0,1/8.0};
    cv::Mat SobelY=Mat(3, 3, CV_32F, &Gy);
    cv::Mat img_filter;
    //cout<<SobelY<<endl;
    filter2D(img, img_filter, ddepth , SobelY, cv::Point(-1, -1), delta, BORDER_DEFAULT );
    //CV_32F  CV_8UC1
    return img_filter.clone();
}


cv::Mat Covariance_Matrix(cv::Mat &M, int window_size){
    int w=int(window_size/2);
    cv::Mat Gx=get_sobel_x_gradient(M);
    cv::Mat Gy=get_sobel_y_gradient(M);
    cv::Mat C=cv::Mat::zeros(M.rows, M.cols, CV_32FC3);
    for(size_t i=w; i<M.rows-w; i++)
        for(size_t j=w; j<M.cols-w; j++)
            for(int k1=-w; k1<=w; k1++)
                for(int k2=-w; k2<=w; k2++){
                    C.at<cv::Vec3f>(i,j)[0]+=Gx.at<float>(i+k1,j+k2)*Gx.at<float>(i+k1,j+k2); //Covarianza GxGx
                    C.at<cv::Vec3f>(i,j)[1]+=Gx.at<float>(i+k1,j+k2)*Gy.at<float>(i+k1,j+k2); //Covarianza GyGx
                    C.at<cv::Vec3f>(i,j)[2]+=Gy.at<float>(i+k1,j+k2)*Gy.at<float>(i+k1,j+k2); //Covarianza GyGy
                }
    return C.clone();
}

cv::Mat Eigen_values(cv::Mat &C){
    cv::Mat E=cv::Mat::zeros(C.rows, C.cols, CV_32FC2);
    for(size_t i=0; i<C.rows; i++)
        for(size_t j=0; j<C.cols; j++){ //Polinomio de la forma l²-(Gx+Gy)+GxGy+Gxy²
            float b=-C.at<cv::Vec3f>(i,j)[0]-C.at<cv::Vec3f>(i,j)[2];
            float c=C.at<cv::Vec3f>(i,j)[0]*C.at<cv::Vec3f>(i,j)[2]+C.at<cv::Vec3f>(i,j)[1]*C.at<cv::Vec3f>(i,j)[1];
            E.at<cv::Vec2f>(i,j)[0] = (-b + sqrt(b*b - 4*c))/2;
            E.at<cv::Vec2f>(i,j)[1] = (-b - sqrt(b*b - 4*c))/2;
        }
    return E.clone();
}

cv::Mat get_harris_response(cv::Mat& lambda, float k){
    float max=-10000;
    cv::Mat R = cv::Mat(lambda.rows, lambda.cols, CV_32FC1);
    for(size_t i=0; i < lambda.rows; i++)
        for(size_t j=0; j < lambda.cols; j++)
        {
            float l1 = lambda.at<cv::Vec2f>(i,j)[0];
            float l2 = lambda.at<cv::Vec2f>(i,j)[1];
            R.at<float>(i,j) = l1*l2 - k*(l1+l2)*(l1+l2);
            if (R.at<float>(i,j)>max)
                max=R.at<float>(i,j);
        }
    cout<<max<<endl;
    return R.clone();
}


void on_threshold_changed(int, void*){}

int main(int, char**)
{
    cv::Mat frame,Gm,G,Ga,gray,img_filter,T;
    cv::Mat SobelX,SobelY,kernel,F, Hough;
    //cout<<Arr<<endl;
    float sigma=1.1;
    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture cap;
    // open the default camera using default API
    cap.open(0);
    cap.set(CAP_PROP_FRAME_WIDTH, 640);//Setting the width of the video
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);//Setting the height of the video//
    cv::namedWindow("Original");
    cv::createTrackbar("Umbral de Harris:", "Original", &t1, 30, on_threshold_changed);
    cv::createTrackbar("Ventana:", "Original", &k_size, 30, on_threshold_changed);
    cv::createTrackbar("Umbral número de votos:", "Original", &hough_threshold, 255, on_threshold_changed);
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
        //gray=gray/255.0;
        cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
        // Initialize arguments for the filter
        gray.convertTo(gray,CV_32FC1); //Change the class type in order to work with float numbers
        gray/=255.0;
        cv::Mat C=Covariance_Matrix(gray,k_size);
        cv::Mat E=Eigen_values(C);
        //E/=(255.0*255.0);
        cv::Mat R=get_harris_response(E, t1/1000.0);
        //R=R/(255.0*255.0*255.0);
        imshow("Original",frame);
        imshow("Covarianza", C);
        imshow("Eigen", R);
        //imshow("F",F);
        if (waitKey(30) == 27)
            break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

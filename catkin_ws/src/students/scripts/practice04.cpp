#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <math.h>
using namespace cv;
using namespace std;

int k_fino=1;
int w_size = 3; //Variable kernel size for the gaussian filter
int k=4; // lower threshold
double delta=0;
int ddepth=-1;
float harris_avg=0;

cv::Mat get_sobel_x_gradient(cv::Mat& img){
    float Gx[3][3]={-1,0,1,-2,0,2,-1,0,1};
    //float Gx[3][3]={-1/8.0,0,1/8.0,-2/8.0,0,2/8.0,-1/8.0,0,1/8.0};
    cv::Mat SobelX=Mat(3, 3, CV_32F, &Gx);
    cv::Mat img_filter;
    //cout<<SobelX<<endl;
    filter2D(img, img_filter, ddepth , SobelX, cv::Point(-1, -1), delta, BORDER_DEFAULT );
    //CV_32F  CV_8UC1
    return img_filter.clone();
}

cv::Mat get_sobel_y_gradient(cv::Mat& img){
    float Gy[3][3]={-1,-2,-1,0,0,0,1,2,1};
    //float Gy[3][3]={-1/8.0,-2/8.0,-1/8.0,0,0,0,1/8.0,2/8.0,1/8.0};
    cv::Mat SobelY=Mat(3, 3, CV_32F, &Gy);
    cv::Mat img_filter;
    //cout<<SobelY<<endl;
    filter2D(img, img_filter, ddepth , SobelY, cv::Point(-1, -1), delta, BORDER_DEFAULT );
    //CV_32F  CV_8UC1
    return img_filter.clone();
}


cv::Mat Covariance_Matrix(cv::Mat &M, int window_size){
    int w=int(window_size/2);
    float wi=2*w+1;
    wi=wi*wi;
    cv::Mat Gx=get_sobel_x_gradient(M);
    cv::Mat Gy=get_sobel_y_gradient(M);
    cv::Mat C=cv::Mat::zeros(M.rows, M.cols, CV_32FC3);
    for(size_t i=w; i<M.rows-w; i++)
        for(size_t j=w; j<M.cols-w; j++)
            for(int k1=-w; k1<=w; k1++)
                for(int k2=-w; k2<=w; k2++){
                    C.at<cv::Vec3f>(i,j)[0]+=(Gx.at<float>(i+k1,j+k2)*Gx.at<float>(i+k1,j+k2)/wi); //Covarianza GxGx
                    C.at<cv::Vec3f>(i,j)[1]+=(Gx.at<float>(i+k1,j+k2)*Gy.at<float>(i+k1,j+k2)/wi); //Covarianza GyGx
                    C.at<cv::Vec3f>(i,j)[2]+=(Gy.at<float>(i+k1,j+k2)*Gy.at<float>(i+k1,j+k2)/wi); //Covarianza GyGy
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
    float max=-1000;
    cv::Mat R = cv::Mat(lambda.rows, lambda.cols, CV_32FC1);
    for(size_t i=0; i < lambda.rows; i++)
        for(size_t j=0; j < lambda.cols; j++)
        {
            float l1 = lambda.at<cv::Vec2f>(i,j)[0];
            float l2 = lambda.at<cv::Vec2f>(i,j)[1];
            R.at<float>(i,j) = l1*l2 - k*(l1+l2)*(l1+l2);
            if( R.at<float>(i,j) > max) //Check if the value is nan, nan has the property to always be false
                max=R.at<float>(i,j) ;
        }//ventana tamaño 13
    return R.clone();
}

cv::Mat non_max_supress(cv:: Mat& R, int window_size){
    int w=window_size/2;
    cv::Mat W=cv::Mat::zeros(R.rows, R.cols, CV_32FC1);
    for(size_t i=w; i<R.rows-w; i++)
        for(size_t j=w; j<R.cols-w; j++){
            float max=-10000000;
            if (R.at<float>(i,j)< 0.1) //harris_avg
                continue;
            for(int k1=-w; k1<=w; k1++)
                for(int k2=-w; k2<=w; k2++){
                    if(R.at<float>(i+k1, j+k2) > max)
                        max=R.at<float>(i+k1,j+k2);
                }
            if(R.at<float>(i,j)==max)
                W.at<float>(i,j)=255.0;
        }
    W.convertTo(W,CV_8UC1);
    return W.clone();

}

void draw_corners(cv::Mat img, std::vector<cv::Point> corners)
{
    for(size_t i=0; i< corners.size(); i++)
        cv::circle(img, corners[i], 5, cv::Scalar(0,0,255), -1);
}


void on_threshold_changed(int, void*){}

int main(int, char**)
{
    cv::Mat frame,Gm,G,Ga,gray,img_filter,T;
    cv::Mat SobelX,SobelY,kernel,F, Hough;
    //cout<<Arr<<endl;
    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture cap;
    // open the default camera using default API
    cap.open(0);
    cap.set(CAP_PROP_FRAME_WIDTH, 352);//Setting the width of the video
    cap.set(CAP_PROP_FRAME_HEIGHT, 288);//Setting the height of the video//
    cv::namedWindow("Original");
    cv::createTrackbar("k", "Original", &k, 30, on_threshold_changed);
    cv::createTrackbar("k ajuste fino", "Original", &k_fino, 10, on_threshold_changed);
    cv::createTrackbar("Tamaño de Ventana", "Original", &w_size, 9, on_threshold_changed);
    cv::setTrackbarMin("k", "Original", 4);
    cv::setTrackbarMin("Tamaño de Ventana", "Original", 1);
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
        //harris_avg=float(umbral);
        gray = frame.clone();
        cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
        // Initialize arguments for the filter
        gray.convertTo(gray,CV_32FC1); //Change the class type in order to work with float numbers
        gray/=255.0;
        cv::Mat C=Covariance_Matrix(gray,w_size);
        cv::Mat E=Eigen_values(C);
        //E/=(255.0*255.0);
        cv::Mat R=get_harris_response(E, k/100.0+k_fino/1000.0);
        cv:Mat W=non_max_supress(R, w_size);
        std::vector<cv::Point> corners;
        cv::findNonZero(W, corners);
        draw_corners(frame, corners);
        //R=R/(255.0*255.0*255.0);
        imshow("Original",frame);
        imshow("Corner", W);
        //cout<<harris_avg<<endl;
        //imshow("F",F);
        if (waitKey(30) == 27)
            break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

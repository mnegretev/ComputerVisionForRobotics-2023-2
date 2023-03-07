#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <math.h>
using namespace cv;
using namespace std;

int k_size = 3; //Variable kernel size for the gaussian filter
int t1=1; // lower threshold
int t2=4; // upper threshold
double pi = 3.14159265358979323846;
double delta=0;
int ddepth=-1;

cv::Mat filter_gaussian(int k,float sigma){
    int siz=(int)k/2;
    float suma=0;
    float H[2*siz+1][2*siz+1]={};
    float arg=0;
    for(int i=0; i<= 2*siz; i++){
        for(int j=0; j<= 2*siz; j++){
            arg=(pow(i-siz,2)+pow(j-siz,2))/(2*pow(sigma,2));
            H[i][j]=exp(-arg)/(2*pi*pow(sigma,2));
            suma=suma+H[i][j];
        }
    }
    for(int i=0; i<= 2*siz; i++){
        for(int j=0; j<= 2*siz; j++){
            H[i][j]=H[i][j]/suma;
        }
    }
    //CV_32F  CV_8UC1
    return cv::Mat(2*siz+1, 2*siz+1, CV_32F, &H).clone();
}

cv::Mat threshold(cv::Mat& G){
    cv::Mat T=G.clone();
    for(size_t i=1; i < G.rows-1; i++)
        for(size_t j=1; j < G.cols-1; j++){
            if(G.at<unsigned char>(i,j) > t1 && G.at<unsigned char>(i,j) <= t2)
                T.at<unsigned char>(i,j)=60;
            else if(G.at<unsigned char>(i,j) > t2)
                T.at<unsigned char>(i,j)=255;
        }
    return T.clone();
}

cv::Mat get_sobel_x_gradient(cv::Mat& img){
    float Gx[3][3]={-1,0,1,-2,0,2,-1,0,1};
    cv::Mat SobelX=Mat(3, 3, CV_32F, &Gx);
    cv::Mat img_filter;
    filter2D(img, img_filter, ddepth , SobelX, cv::Point(-1, -1), delta, BORDER_DEFAULT );
    //CV_32F  CV_8UC1
    return img_filter.clone();
}

cv::Mat get_sobel_y_gradient(cv::Mat& img){
    float Gy[3][3]={-1,-2,-1,0,0,0,1,2,1};
    cv::Mat SobelY=Mat(3, 3, CV_32F, &Gy);
    cv::Mat img_filter;
    filter2D(img, img_filter, ddepth , SobelY, cv::Point(-1, -1), delta, BORDER_DEFAULT );
    //CV_32F  CV_8UC1
    return img_filter.clone();
}

cv::Mat supress_non_maximum(cv::Mat& Gm, cv::Mat& Ga){
    cv::Mat G=Gm.clone();
    size_t di=0;
    size_t dj=0;
    for(size_t i=1; i < G.rows-1; i++)
        for(size_t j=1; j < G.cols-1; j++){
            if(Ga.at<unsigned char>(i,j)<=22 || Ga.at<unsigned char>(i,j) > 157){
                di=0;
                dj=1;
            }
            else if(Ga.at<unsigned char>(i,j)>22 && Ga.at<unsigned char>(i,j) <= 67){
                di=1;
                dj=1;
            }
            else if(Ga.at<unsigned char>(i,j)>67 && Ga.at<unsigned char>(i,j) <= 112){
                di=1;
                dj=0;
            }
            else{
                di=1;
                dj=-1;
            }
            if(Gm.at<unsigned char>(i,j) >= Gm.at<unsigned char>(i+di,j+dj) && Gm.at<unsigned char>(i,j) > Gm.at<unsigned char>(i-di,j-dj) )
                G.at<unsigned char>(i,j)=Gm.at<unsigned char>(i,j);
            else
               G.at<unsigned char>(i,j)=0;
    }
    return G.clone();
}

void mag_angle(cv::Mat& A, cv::Mat& Gm, cv::Mat& Ga){
    cv::Mat SobelX=get_sobel_x_gradient(A);
    cv::Mat SobelY=get_sobel_y_gradient(A);
    for(size_t i=0; i < SobelX.rows; i++)
        for(size_t j=0; j < SobelX.cols; j++){
            Gm.at<unsigned char>(i,j)=sqrt(pow(SobelX.at<unsigned char>(i,j),2)+pow(SobelY.at<unsigned char>(i,j),2));
            Ga.at<unsigned char>(i,j)=atan2(SobelY.at<unsigned char>(i,j),SobelX.at<unsigned char>(i,j));
            if(Ga.at<unsigned char>(i,j)<0)
                Ga.at<unsigned char>(i,j)+=pi;
            Ga.at<unsigned char>(i,j)=Ga.at<unsigned char>(i,j)*180/pi;
        }
    Ga.convertTo(Ga,CV_8UC1);
}

void on_threshold_changed(int, void*){}

int main(int, char**)
{
    cv::Mat frame,Gm, G,Ga,gray, img_filter,T;
    //cv::Mat G;
    //cv::Mat Gm;
    //cv::Mat Ga;
    //cv::Mat gray;
    //cv::Mat img_filter;
    float sigma=1;
    cv::Mat kernel=filter_gaussian(k_size, sigma);
    //cout<<kernel<<endl;
    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture cap;
    // open the default camera using default API
    cap.open(0);
    cap.set(CAP_PROP_FRAME_WIDTH, 320);//Setting the width of the video
    cap.set(CAP_PROP_FRAME_HEIGHT, 240);//Setting the height of the video//
    cv::namedWindow("Original");
    cv::createTrackbar("Umbral inferior:", "Original", &t1, 14, on_threshold_changed);
    cv::createTrackbar("Umbral superior:", "Original", &t2, 30, on_threshold_changed);
    cv::createTrackbar("Tamaño filtro:", "Original", &k_size, 14, on_threshold_changed);
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
        cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
        // Initialize arguments for the filter
        //cout << gray.type() << " == " << CV_8U << endl;
        //gray.convertTo(gray,CV_32F);
        kernel=filter_gaussian(k_size, sigma);
        filter2D(gray, img_filter, ddepth , kernel, cv::Point(-1, -1), delta, BORDER_DEFAULT );
        Gm=img_filter.clone();
        Ga=img_filter.clone();
        //img_filter.convertTo(img_filter,CV_8UC1);
        mag_angle(img_filter, Gm, Ga);
        G=supress_non_maximum(Gm, Ga);
        T=threshold(G);
        //show live and wait for a key with timeout long enough to show images
        imshow("Original", frame);
        imshow("Grey",gray);
        //imshow("Sobel Y", SobelX);
        imshow("Magnitud", Gm);
        //imshow("Angulo", Ga);
        imshow("G",G);
        imshow("T",T);
        //cout<<kernel<<endl;
        if (waitKey(30) == 27)
            break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

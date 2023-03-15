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

std::vector<cv::Vec2f> hough_lines(cv::Mat img_bin, float d_min, float d_max, int d_res, float theta_min,
                                   float theta_max, float theta_res, int threshold)
{
    int dist_n  = (int)ceil((d_max - d_min)/d_res);
    int theta_n = (int)ceil((theta_max - theta_min)/theta_res);
    int votes[dist_n][theta_n]={};
    float theta_values[theta_n];
    std::cout << "Bins for d=" << dist_n << "  bins for theta=" << theta_n<< std::endl;
    for(size_t i=0; i<theta_n; i++)
        theta_values[i] = theta_min + theta_res*i;
    //for(size_t i=0; i<dist_n; i++)
      //  for(size_t j=0; j<theta_n; j++)
        //    votes[i][j] = 0;
    for(size_t i=0; i < img_bin.rows; i++)
        for(size_t j=0; j < img_bin.cols; j++)
            if(img_bin.at<unsigned char>(i,j) == 255)
                for(size_t k=0; k<theta_n; k++)
                {
                    int d = (int)((j*cos(theta_values[k]) + i*sin(theta_values[k]) - d_min)/d_res);
                    if(d >= 0 && d < dist_n)  votes[d][k]++;
                }

    std::vector<cv::Vec2f> lines;
    for(size_t i=0; i<dist_n; i++)
        for(size_t j=0; j<theta_n; j++)
            if(votes[i][j] > threshold)
            {
                float d = i*d_res + d_min;
                float a = j*theta_res + theta_min;
                lines.push_back(cv::Vec2f(d,a));
            }
    return lines;
}

void draw_lines(cv::Mat& img, std::vector<cv::Vec2f>& lines)
{
    for(size_t i=0; i< lines.size(); i++)
    {
        float d = lines[i][0], theta = lines[i][1];
        cv::Point p1, p2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*d, y0 = b*d;
        p1.x = round(x0 + 1000*(-b));
        p1.y = round(y0 + 1000*(a));
        p2.x = round(x0 - 1000*(-b));
        p2.y = round(y0 - 1000*(a));
        cv::line(img, p1, p2, cv::Scalar(0,255,0), 2, cv::LINE_AA);
    }
}

cv:: Mat final_supress(cv::Mat& T){
    cv::Mat F=T.clone();
    for(size_t i=1; i < T.rows-1; i++)
        for(size_t j=1; j < T.cols-1; j++){
            if(T.at<float>(i,j)==60&&(T.at<float>(i+1,j)==255||T.at<float>(i+1,j+1)==255||T.at<float>(i+1,j-1)==255||T.at<float>(i-1,j)==255||T.at<float>(i-1,j+1)==255||T.at<float>(i-1,j-1)==255||T.at<float>(i,j+1)==255||T.at<float>(i,j-1)==255 ))
                F.at<float>(i,j)=255;
            else if(T.at<float>(i,j)==60)
                F.at<float>(i,j)=0;
        }
    F.convertTo(F,CV_8UC1);
    return F.clone();
}

cv::Mat threshold(cv::Mat& G){
    cv::Mat T=G.clone();
    for(size_t i=1; i < G.rows-1; i++)
        for(size_t j=1; j < G.cols-1; j++){
            if(G.at<float>(i,j) > (float)t1 && G.at<float>(i,j) <= (float)t2)
                T.at<float>(i,j)=60;
            else if(G.at<float>(i,j) > (float)t2)
                T.at<float>(i,j)=255;
        }
    return T.clone();
}

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


void mag_angle(cv::Mat& A, cv::Mat& Gm, cv::Mat& Ga){
    cv::Mat SobelX=get_sobel_x_gradient(A);
    cv::Mat SobelY=get_sobel_y_gradient(A);
    for(size_t i=0; i < SobelX.rows; i++)
        for(size_t j=0; j < SobelX.cols; j++){
            Gm.at<float>(i,j)=sqrt(pow(SobelX.at<float>(i,j),2)+pow(SobelY.at<float>(i,j),2));
            Ga.at<float>(i,j)=atan2( SobelY.at<float>(i,j), SobelX.at<float>(i,j) );
            if(Ga.at<float>(i,j)<0)
                Ga.at<float>(i,j)+=pi;
            Ga.at<float>(i,j)=Ga.at<float>(i,j)*180/pi;
        }
    //Ga.convertTo(Ga,CV_8UC1);
}

cv::Mat supress_non_maximum(cv::Mat& Gm, cv::Mat& Ga){
    cv::Mat G=Gm.clone();
    size_t di=0;
    size_t dj=0;
    for(size_t i=1; i < G.rows-1; i++)
        for(size_t j=1; j < G.cols-1; j++){
            if(Ga.at<float>(i,j)<=22 || Ga.at<float>(i,j) > 157){
                di=0;
                dj=1;
            }
            else if(Ga.at<float>(i,j)>22 && Ga.at<float>(i,j) <= 67){
                di=1;
                dj=1;
            }
            else if(Ga.at<float>(i,j)>67 && Ga.at<float>(i,j) <= 112){
                di=1;
                dj=0;
            }
            else{
                di=1;
                dj=-1;
            }
            if(Gm.at<float>(i,j) >= Gm.at<float>(i+di,j+dj) && Gm.at<float>(i,j) > Gm.at<float>(i-di,j-dj) )
                G.at<float>(i,j)=Gm.at<float>(i,j);
            else
                G.at<float>(i,j)=0;
    }
    return G.clone();
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
    cv::createTrackbar("Umbral inferior Canny:", "Original", &t1, 12, on_threshold_changed);
    cv::createTrackbar("Umbral superior Canny:", "Original", &t2, 22, on_threshold_changed);
    cv::createTrackbar("Tamaño filtro gaussiano:", "Original", &k_size, 9, on_threshold_changed);
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

        //kernel=filter_gaussian(k_size, sigma);
        //filter2D(gray, img_filter, ddepth , kernel, cv::Point(-1, -1), delta, BORDER_DEFAULT );
        cv::Mat C=Covariance_Matrix(gray,k_size);
        //F_clone.convertTo(F_clone,CV_8U);
        //imshow("Grey",gray/255.0);
        imshow("Original",frame);
        imshow("Covarianza", C);
        //imshow("F",F);
        if (waitKey(30) == 27)
            break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

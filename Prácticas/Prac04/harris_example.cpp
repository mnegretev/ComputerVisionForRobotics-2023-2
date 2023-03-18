#include "opencv2/opencv.hpp"
using namespace cv;
double pi = 3.14159265358979323846;
double delta=0;
int ddepth=-1;
int trackbar_K = 10;
int trackbar_W = 3;
int trackbar_Kfino=2;
float t=0;

cv::Mat convolve2d(cv::Mat& A, cv::Mat& kernel)
{
    cv::Mat convolved;
    cv::filter2D(A, convolved, CV_32F, kernel);
    return convolved;
}

cv::Mat get_sobel_x_gradient(cv::Mat& A)
{
    cv::Mat Gx = cv::Mat(3, 3, CV_32FC1);
    Gx.at<float>(0,0) = -1; Gx.at<float>(0,1) = 0; Gx.at<float>(0,2) = 1;
    Gx.at<float>(1,0) = -2; Gx.at<float>(1,1) = 0; Gx.at<float>(1,2) = 2;
    Gx.at<float>(2,0) = -1; Gx.at<float>(2,1) = 0; Gx.at<float>(2,2) = 1;
    return convolve2d(A, Gx);
}

cv::Mat get_sobel_y_gradient(cv::Mat& A)
{
    cv::Mat Gy = cv::Mat(3, 3, CV_32FC1);
    Gy.at<float>(0,0) = -1; Gy.at<float>(0,1) = -2; Gy.at<float>(0,2) = -1;
    Gy.at<float>(1,0) =  0; Gy.at<float>(1,1) =  0; Gy.at<float>(1,2) =  0;
    Gy.at<float>(2,0) =  1; Gy.at<float>(2,1) =  2; Gy.at<float>(2,2) =  1;
    return convolve2d(A, Gy);
}

cv::Mat matrix_second_moment(cv::Mat& A, int window_size)
{
    int w = window_size/2;
    float wi=2*w+1;
    cv::Mat Gx = get_sobel_x_gradient(A);
    cv::Mat Gy = get_sobel_y_gradient(A);
    cv::Mat M      = cv::Mat::zeros(A.rows, A.cols, CV_32FC4);
    for(size_t i=w; i < A.rows-w; i++)
        for(size_t j=w; j < A.cols-w; j++)
            for(int k1=i-w; k1<=i+w; k1++)
                for(int k2=j-w; k2<=j+w; k2++)
                {
                    M.at<cv::Vec4f>(i,j)[0] +=( Gx.at<float>(k1,k2)*Gx.at<float>(k1,k2)/wi);
                    M.at<cv::Vec4f>(i,j)[1] +=( Gx.at<float>(k1,k2)*Gy.at<float>(k1,k2)/wi);
                    M.at<cv::Vec4f>(i,j)[2] +=( Gx.at<float>(k1,k2)*Gy.at<float>(k1,k2)/wi);
                    M.at<cv::Vec4f>(i,j)[3] +=( Gy.at<float>(k1,k2)*Gy.at<float>(k1,k2)/wi);
                }
    return M;
}

cv::Mat get_eigenvalues(cv::Mat& M)
{
    cv::Mat lambda = cv::Mat::zeros(M.rows, M.cols, CV_32FC2);
    for(size_t i=0; i < M.rows; i++)
        for(size_t j=0; j < M.cols; j++)
        {
            cv::Vec4f m = M.at<cv::Vec4f>(i,j);
            float b =-m[0] - m[3];
            float c = m[0]*m[3] + m[1]*m[2];
            lambda.at<cv::Vec2f>(i,j)[0] = (-b + sqrt(b*b - 4*c))/2;
            lambda.at<cv::Vec2f>(i,j)[1] = (-b - sqrt(b*b - 4*c))/2;
        }
    return lambda;
}

cv::Mat get_harris_response(cv::Mat& lambda, float k)
{
    cv::Mat R = cv::Mat(lambda.rows, lambda.cols, CV_32FC1);
    for(size_t i=0; i < lambda.rows; i++)
        for(size_t j=0; j < lambda.cols; j++)
        {
            float l1 = lambda.at<cv::Vec2f>(i,j)[0];
            float l2 = lambda.at<cv::Vec2f>(i,j)[1];
            R.at<float>(i,j) = l1*l2 - k*(l1+l2)*(l1+l2);
        }
    return R;
}

cv::Mat suppress_non_maximum(cv::Mat& R, int window_size)
{
    int w = window_size/2;
    cv::Mat H = cv::Mat::zeros(R.rows, R.cols, CV_8UC1);
    for(size_t i=0; i<R.rows; i++)
        for(size_t j=0; j<R.cols; j++)
        {
            if(R.at<float>(i,j) < 0.5)
                continue;
            float max = -999999;
            for(int k1=i-w; k1<=i+w; k1++)
                for(int k2=j-w; k2<=j+w; k2++)
                    if(R.at<float>(k1,k2) > max)
                        max = R.at<float>(k1,k2);
            H.at<unsigned char>(i,j) = max == R.at<float>(i,j) ? 255: 0;
        }
    return H;
}


std::vector<cv::Point> corners_harris(cv::Mat& A, int window_size, float k)
{
    cv::Mat M = matrix_second_moment(A, window_size);
    cv::Mat lambda = get_eigenvalues(M);
    cv::Mat R = get_harris_response(lambda, k);
    cv::Mat H = suppress_non_maximum(R, window_size);
    cv::imshow("R",H);
    std::vector<cv::Point> corners;
    cv::findNonZero(H, corners);
    return corners;
}

void draw_corners(cv::Mat img, std::vector<cv::Point> corners)
{
    for(size_t i=0; i< corners.size(); i++)
        cv::circle(img, corners[i], 5, cv::Scalar(0,0,255), -1);
}


void on_k_changed(int, void*){}
void on_w_changed(int, void*){}

int main()
{
    cv::Mat img_original = cv::imread("prueba.jpg");
    cv::namedWindow("Corners");
    cv::createTrackbar("K:", "Corners", &trackbar_K, 30, on_k_changed);
    cv::setTrackbarMin("K:", "Corners", 4);
    cv::createTrackbar("K ajuste fino:", "Corners", &trackbar_Kfino, 10, on_k_changed);
    cv::createTrackbar("W:", "Corners", &trackbar_W, 9, on_k_changed);
    cv::setTrackbarMin("W:", "Corners", 1);
    while(cv::waitKey(100) != 27)
    {
        cv::Mat img = img_original.clone();
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(gray, CV_32F);
        gray /= 255.0f;
        std::vector<cv::Point> corners = corners_harris(gray, trackbar_W, trackbar_K/100.0+trackbar_Kfino/1000.0);
        draw_corners(img, corners);
        cv::imshow("Corners", img);
    }
    return 0;
}

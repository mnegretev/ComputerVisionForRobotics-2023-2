#include "opencv2/opencv.hpp"


int hough_threshold = 100;

std::vector<cv::Vec2f> hough_lines(cv::Mat img_bin, float d_min, float d_max, int d_res, float theta_min,
                                   float theta_max, float theta_res, int threshold)
{
    int dist_n  = (int)ceil((d_max - d_min)/d_res);
    int theta_n = (int)ceil((theta_max - theta_min)/theta_res);
    int votes[dist_n][theta_n];
    float theta_values[theta_n];
    std::cout << "Bins for d=" << dist_n << "  bins for theta=" << theta_n<< std::endl;
    for(size_t i=0; i<theta_n; i++)
        theta_values[i] = theta_min + theta_res*i;
    for(size_t i=0; i<dist_n; i++)
        for(size_t j=0; j<theta_n; j++)
            votes[i][j] = 0;
    
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

void on_threshold_changed(int, void*){}

int main()
{
    cv::Mat img_original = cv::imread("TestLines.png");
    cv::namedWindow("Lines");
    cv::createTrackbar("Thr:", "Lines", &hough_threshold, 255, on_threshold_changed);

    while(cv::waitKey(100) != 27)
    {
        cv::Mat img = img_original.clone();
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Vec2f> lines = hough_lines(gray, 0, 356, 2, -M_PI, M_PI, 0.15, hough_threshold);
        draw_lines(img, lines);
        std::cout << "Number of lines: " << lines.size() << std::endl;
        cv::imshow("Lines", img);
    }
    return 0;
}

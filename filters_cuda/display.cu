#include <opencv2/opencv.hpp>
#include <string>

void disp(cv::Mat &frame, const std::string &win_name)
{
    cv::namedWindow(win_name, 1);
    cv::imshow(win_name, frame);
    cv::waitKey(1);
}
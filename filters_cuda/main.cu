#include <iostream>
#include<opencv2/opencv.hpp>
#include "filters.h"
#include <typeinfo>

int main(int argc,char *argv[])
{
    char src_type, key, filter_type;
    std::cout << "Enter source type: i:Image, v:Video stream" << std::endl;
    std::cin >> src_type;

    std::cout << "Enter filter type: g:Grayscale, b:Gaussian blur, x:Sobel x, y: Sobel y, m: Magnitude, l: Blur+Quantization, c: Cartoonize" << std::endl;
    std::cin >> filter_type;

    cv::Mat src_img, mid_img, filtered_img;

    std::vector<float> filter1, filter2;
    std::vector<int> var (2,0);

    bool filter_init = false;

    if(src_type == 'i')
    {
        std::string src;
        std::cout << "Enter image source: ";
        std::cin >> src;
        if(src.empty())
            src = "G:/dog.jpg";

        src_img = cv::imread(src);
        filtered_img = cv::Mat::zeros(src_img.size(), src_img.type());

        filters(src_img, filtered_img, filter1, filter2, filter_type, var, filter_init, false);
        disp(src_img, "raw_img");
        cv::waitKey(0);
    }

    else if(src_type == 'v')
    {   std::cout << "Videostream started!" << std::endl;
        cv::VideoCapture *capdev;

        capdev = new cv::VideoCapture(0);
        if (!capdev->isOpened())
        {
            std::cout << "Unable to open device!" <<std::endl;
            exit(-1);
        }

        while(true)
        {
            *capdev >> src_img;
            if (src_img.empty())
            {
                std::cout << "Empty frame" << std::endl;
                exit(-1);
            }

            filtered_img = cv::Mat::zeros(src_img.size(), src_img.type());
            filters(src_img, filtered_img, filter1, filter2, filter_type, var,filter_init, false);
            disp(src_img, "raw_vid");
            cv::waitKey(10);
        }
        delete capdev;
    }

    else
    {
        std::cout << "Invalid entry" << std::endl;
        exit(-1);
    }

    cv::destroyAllWindows();
}

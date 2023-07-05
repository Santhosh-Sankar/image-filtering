#include<opencv2/opencv.hpp>
#include "utils.h"
#include"filters.h"
#include<cmath>

void GrayScale(cv::Mat &src, cv::Mat &gray_src)
{
    cv::cvtColor(src, gray_src, cv::COLOR_BGR2GRAY);
}

void filter_fn(cv::Mat &src, cv::Mat &blur_src, std::vector<float> &filter1, std::vector<float> &filter2)
{
    int filter_size = filter1.size();
    cv::Mat mid_img, img_patch;
    mid_img = cv::Mat::zeros(src.size(), CV_32FC3);

    for(int img_i=0;img_i<src.rows;img_i++)
    {
        cv::Vec3f *rptr_dstn = mid_img.ptr<cv::Vec3f>(img_i);

        for(int img_j=0;img_j<src.cols;img_j++)
        {
            img_patch = retrieve_imgpatch<cv::Vec3b>(src.rows, src.cols, img_i,  img_j, 'x', filter_size, src);
            rptr_dstn[img_j] = dot<cv::Vec3b>(img_patch, filter1);

//            std:: cout << "Patch: " <<img_patch << std::endl;
//            for(int a=0;a<3;a++)
//            {
//                std::cout << rptr_dstn[img_j][a] << std::endl;
//            }
        }
    }
//    cv::Mat inter_img;
//    cv::convertScaleAbs(mid_img, inter_img);
//    disp(inter_img, "mid");

    for(int img_i=0;img_i<src.rows;img_i++)
    {
        cv::Vec3f *rptr_dstn = blur_src.ptr<cv::Vec3f>(img_i);

        for(int img_j=0;img_j<src.cols;img_j++)
        {
            img_patch = retrieve_imgpatch<cv::Vec3f>(mid_img.rows, mid_img.cols,  img_i, img_j, 'y', filter_size, mid_img);
            rptr_dstn[img_j] = dot<cv::Vec3f>(img_patch, filter2);
        }
    }
}

void filters(cv::Mat &src, cv::Mat &filtered_src, std::vector<float> &filter1, std::vector<float> &filter2, char &filter_type, std::vector<int> &var, bool &filter_init, bool supress_vid)
{
    std::string filter_name;

    switch (filter_type)
    {
        case 'g':
            filter_name = "Grayscale";
            GrayScale(src, filtered_src);
            break;

        case 'b': {
            filter_name = "Blur";

            cv::Mat filtered_src32 = cv::Mat::zeros(src.size(), CV_32FC3);

            if (not(filter_init)) {
                filter_input(filter1);
                filter_init = true;
            }

            filter_fn(src, filtered_src32, filter1, filter1);
            cv::convertScaleAbs(filtered_src32, filtered_src);
            break;
        }

        case 'x': {
            filter_name = "Sobel x";
            cv::Mat filtered_src32 = cv::Mat::zeros(src.size(), CV_32FC3);
            std::vector<float> filter_diff = {-1, 0, 1};
            if (not(filter_init)) {
                filter_input(filter1);
                filter_init = true;
            }
            if (filter1.size() != filter_diff.size()) {
                std::cout << "Filter sizes not matching!";
                exit(-1);
            }

            filter_fn(src, filtered_src32, filter_diff, filter1);
            cv::convertScaleAbs(filtered_src32, filtered_src);
            break;
        }

        case 'y': {
            filter_name = "Sobel y";
            cv::Mat filtered_src32 = cv::Mat::zeros(src.size(), CV_32FC3);
            std::vector<float> filter_diff = {-1, 0, 1};
            if (not(filter_init)) {
                filter_input(filter1);
                filter_init = true;
            }
            if (filter1.size() != filter_diff.size()) {
                std::cout << "Filter sizes not matching!";
                exit(-1);
            }

            filter_fn(src, filtered_src32, filter1, filter_diff);
            cv::convertScaleAbs(filtered_src32, filtered_src);
            break;
        }

        case 'l': {
            filter_name = "Blur + Quantize";

            cv::Mat filtered_src32 = cv::Mat::zeros(src.size(), CV_32FC3);
            cv::Mat blur;
            int level;
            char filter1_type = 'b';

            if (not(filter_init)) {
                std::cout << "Enter number of levels: ";
                std::cin >> level;
                var[0] = level;
            }
            else
                level = var[0];

            float bucket_size = 255/level;

            blur = cv::Mat::zeros(src.size(), src.type());
            filters(src, blur, filter1, filter2, filter1_type, var, filter_init, true);

            for(int i=0;i<src.rows;i++)
            {
                cv::Vec3b *rptr = blur.ptr<cv::Vec3b>(i);
                cv::Vec3f *dstn_ptr = filtered_src32.ptr<cv::Vec3f>(i);

                for(int j=0;j<src.cols;j++)
                {
                    dstn_ptr[j][0] = (int(rptr[j][0] / bucket_size)) * bucket_size;
                    dstn_ptr[j][1] = (int(rptr[j][1] / bucket_size)) * bucket_size;
                    dstn_ptr[j][2] = (int(rptr[j][2] / bucket_size)) * bucket_size;
                }
            }
            cv::convertScaleAbs(filtered_src32, filtered_src);
            break;
        }

        case 'm': {
            filter_name = "Magnitude";
            char filter1_type = 'x', filter2_type = 'y';
            cv::Mat filtered_src32 = cv::Mat::zeros(src.size(), CV_32FC3);
            cv::Mat sobel_x, sobel_y;
            sobel_x = cv::Mat::zeros(src.size(), src.type());
            sobel_y = cv::Mat::zeros(src.size(), src.type());

            filters(src, sobel_x, filter1, filter2, filter1_type, var, filter_init, true);
            filters(src, sobel_y, filter1, filter2, filter2_type, var, filter_init, true);

            for (int i = 0; i < sobel_x.rows; i++) {
                cv::Vec3b *xptr = sobel_x.ptr<cv::Vec3b>(i);
                cv::Vec3b *yptr = sobel_y.ptr<cv::Vec3b>(i);
                cv::Vec3f *dstn_ptr = filtered_src32.ptr<cv::Vec3f>(i);

                for (int j = 0; j < sobel_x.cols; j++) {
                    dstn_ptr[j][0] = sqrt(pow(xptr[j][0], 2) + pow(yptr[j][0], 2));
                    dstn_ptr[j][1] = sqrt(pow(xptr[j][1], 2) + pow(yptr[j][1], 2));
                    dstn_ptr[j][2] = sqrt(pow(xptr[j][2], 2) + pow(yptr[j][2], 2));
                }
            }
            cv::convertScaleAbs(filtered_src32, filtered_src);
            break;

        }
        case 'c': {
            filter_name = "Cartoonized";
            char filter1_type = 'm', filter2_type = 'l';
            cv::Mat magnitude, quant;
            magnitude = cv::Mat::zeros(src.size(), src.type());
            quant = cv::Mat::zeros(src.size(), src.type());
            int threshold, flag=1;

            if(not(filter_init)) {
                std::cout << "Enter threshold: ";
                std::cin >> threshold;
                var[1]  = threshold;

                filters(src, magnitude, filter1, filter2, filter1_type, var, filter_init, true);
                filter_init = false;
                filters(src, quant, filter2, filter1, filter2_type, var, filter_init, true);
            }
            else {
                threshold = var[1];
                filters(src, magnitude, filter1, filter2, filter1_type, var, filter_init, true);
                filters(src, quant, filter2, filter1, filter2_type, var, filter_init, true);
            }

            for (int i = 0; i < src.rows; i++) {
                cv::Vec3b *mptr = magnitude.ptr<cv::Vec3b>(i);
                cv::Vec3b *qptr = quant.ptr<cv::Vec3b>(i);
                cv::Vec3b *dstn_ptr = filtered_src.ptr<cv::Vec3b>(i);

                for (int j = 0; j < src.cols; j++) {

                    for(int k=0;k<3;k++)
                        if(mptr[j][k] > char(threshold))
                        {
                            flag=0;
                            break;
                        }

                    if (flag == 0)
                    {
                        flag = 1;
                        for(int k=0;k<3;k++)
                            dstn_ptr[j][k] = char(0);
                    }
                    else
                    {
                        for(int k=0;k<3;k++)
                            dstn_ptr[j][k] = qptr[j][k];
                    }
                }

            }
            break;
        }

    }

    if(not(supress_vid))
        disp(filtered_src, filter_name);

}
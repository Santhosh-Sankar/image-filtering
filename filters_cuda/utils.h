//
// Created by Santhosh S on 5/14/2023.
//

#ifndef CV_ASS1_UTILS_H
#define CV_ASS1_UTILS_H

void filter_input(std::vector<float> &filter);
void errCheck(cudaError_t error);

//template<typename T>
//cv::Vec3f dot(cv::Mat &img_patch, std::vector<float> &filter)
//{
//    cv::Vec3f prod = cv::Vec3f(0.0);
//    T *rptr = img_patch.ptr<T>(0);
//
//    for(int i=0; i<filter.size(); i++)
//    {
//        prod[0] += filter[i]*rptr[i][0];
//        prod[1] += filter[i]*rptr[i][1];
//        prod[2] += filter[i]*rptr[i][2];
//    }
//    return prod;
//}
//
//template<typename T>
//cv::Mat retrieve_imgpatch(int &row_size, int &col_size, int &img_i, int &img_j,const char &dir, const int filter_size, cv::Mat &src)
//{
//
//    T *patch_ptr = patch.ptr<T>(0);
//
//    int offset = filter_size/2;
//
//    if(dir=='x')
//    {
//        T *rptr = src.ptr<T>(img_i);
//        for (int j = -offset; j<1+offset; j++)
//            if (img_j + j >= 0 && img_j + j < col_size)
//                patch_ptr[j + offset] = rptr[img_j + j];
//    }
//
//    else if (dir=='y')
//    {
//        for(int i=-offset;i<1+offset;i++)
//            if (img_i + i >= 0 && img_i + i < row_size)
//            {
//                T *rptr = src.ptr<T>(img_i+i);
//                patch_ptr[i+offset] = rptr[img_j];
//            }
//    }
//
//    else
//    {
//        std::cout << "Invalid option" <<std::endl;
//        exit(-1);
//    }
//
//    return patch;
//}


#endif //CV_ASS1_UTILS_H

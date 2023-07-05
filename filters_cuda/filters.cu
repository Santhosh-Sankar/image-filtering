#include<opencv2/opencv.hpp>
#include "utils.h"
#include"filters.h"
#include<cmath>

void GrayScale(cv::Mat &src, cv::Mat &gray_src)
{
    cv::cvtColor(src, gray_src, cv::COLOR_BGR2GRAY);
}

__global__
void init(float *ptr, int size) {

    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < size)
        ptr[idx] = 0;
}



__global__
void filter_kernel_x(int src_rows, int src_cols, int filter_size, uchar *src_ptr, float *mid_ptr, float *patch_ptr_x, float *filter_x)
{
    int rows = threadIdx.x + blockDim.x * blockIdx.x;
    int cols = threadIdx.y + blockDim.y * blockIdx.y;

    if(rows < src_rows && cols < src_cols)
    {
        int offset = filter_size / 2;
        int postn = filter_size*3*(src_cols*rows+cols);
        int i, j, k;


        //retrieve patch x
        for (i = -offset; i < 1 + offset; i++)
            if (cols + i >= 0 && cols + i < src_cols)
            {
                patch_ptr_x[postn + 3 * (i + offset)] = src_ptr[3 * (rows * src_cols + (cols + i))];
                patch_ptr_x[postn + 3 * (i + offset) + 1] = src_ptr[3 * (rows * src_cols + cols + i) + 1];
                patch_ptr_x[postn + 3 * (i + offset) + 2] = src_ptr[3 * (rows * src_cols + cols + i) + 2];
            }

        //dot product
        for (j = 0; j < filter_size; j++)
        {

            for (k = 0; k < 3; k++)
                mid_ptr[3 * (rows * src_cols + cols) + k] += patch_ptr_x[postn + 3 * j + k] * filter_x[j];

        }

    }
}

__global__
void filter_kernel_y(int src_rows, int src_cols, int filter_size, float *mid_ptr, float *fimg_ptr, float *patch_ptr_y, float *filter_y)
{
    int rows = threadIdx.x + blockDim.x * blockIdx.x;
    int cols = threadIdx.y + blockDim.y * blockIdx.y;

    if(rows < src_rows && cols < src_cols)
    {
        int offset = filter_size / 2;
        int postn = filter_size*3*(src_cols*rows+cols);
        int i, j, k;


        //retrieve patch y
        for (i = -offset; i < 1 + offset; i++)
            if (rows + i >= 0 && rows + i < src_rows)
            {
                patch_ptr_y[postn + 3 * (i + offset)] = mid_ptr[3 * ((rows + i) * src_cols + cols)];
                patch_ptr_y[postn + 3 * (i + offset) + 1] = mid_ptr[3 * ((rows + i) * src_cols + cols) + 1];
                patch_ptr_y[postn + 3 * (i + offset) + 2] = mid_ptr[3 * ((rows + i) * src_cols + cols) + 2];
            }

        for (j = 0; j < filter_size; j++)
        {
            for (k = 0; k < 3; k++)
                fimg_ptr[3 * (rows * src_cols + cols) + k] += patch_ptr_y[postn + 3 * j + k] * filter_y[j];
        }


    }
}




void filter_fn(cv::Mat &src, uchar *src_ptr, float *mid_ptr, float *fimg_ptr, std::vector<float> &filter1, std::vector<float> &filter2, cudaStream_t stream)
{


    int filter_size = filter1.size();
    cv::Mat img_patch, dot_prod, filter1_mat, filter2_mat;

    int rows = src.rows, cols = src.cols;

    float *patch_ptr_x, *patch_ptr_y, *filter1_ptr_host = &filter1[0], *filter2_ptr_host = &filter2[0], *filter1_ptr_gpu, *filter2_ptr_gpu, *temp1, *temp2;
    size_t patch_size = rows*cols*filter_size*3*sizeof(float), filter_size_b = filter_size*sizeof(float);

    cudaMalloc(&patch_ptr_x, patch_size);
    cudaMalloc(&patch_ptr_y, patch_size);
    cudaMalloc(&filter1_ptr_gpu, filter_size_b);
    cudaMalloc(&filter2_ptr_gpu, filter_size_b);
    cudaMallocHost(&temp1, filter_size_b);
    cudaMallocHost(&temp2, filter_size_b);

    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    int num_SMs = prop.multiProcessorCount;

    init<<<int((rows*cols*filter_size)/(prop.warpSize*3))+1, prop.warpSize*3, 0, stream>>>(patch_ptr_x, rows*cols*filter_size*3);
    init<<<int((rows*cols*filter_size)/(prop.warpSize*3))+1, prop.warpSize*3, 0, stream>>>(patch_ptr_y, rows*cols*filter_size*3);

    errCheck(cudaGetLastError());

    memcpy(temp1, filter1_ptr_host, filter_size_b);
    errCheck(cudaMemcpy(filter1_ptr_gpu, temp1, filter_size_b, cudaMemcpyHostToDevice));

    memcpy(temp2, filter2_ptr_host, filter_size_b);
    errCheck(cudaMemcpy(filter2_ptr_gpu, temp2, filter_size_b, cudaMemcpyHostToDevice));

    cudaFreeHost(temp1);
    cudaFreeHost(temp2);

//    errCheck(cudaDeviceSynchronize());
    dim3 threads(prop.warpSize, prop.warpSize, 1);

    int blocks_x = int(rows/threads.x) + 1, blocks_y = int(cols/threads.y) + 1;

    int rem_x = blocks_x%num_SMs, rem_y = blocks_y%num_SMs;

    if(rem_x != 0)
        blocks_x += num_SMs - rem_x;

    if(rem_y != 0)
        blocks_y += num_SMs - rem_y;

    dim3 blocks(blocks_x, blocks_y, 1);

    filter_kernel_x<<<blocks, threads,0, stream>>>(rows, cols, filter_size, src_ptr, mid_ptr, patch_ptr_x, filter1_ptr_gpu);
//
//    errCheck(cudaGetLastError());

    filter_kernel_y<<<blocks, threads,0, stream>>>(rows, cols, filter_size, mid_ptr, fimg_ptr, patch_ptr_y, filter2_ptr_gpu);

    errCheck(cudaGetLastError());

    errCheck(cudaDeviceSynchronize());

    cudaFree(patch_ptr_x);
    cudaFree(patch_ptr_y);
    cudaFree(filter1_ptr_gpu);
    cudaFree(filter2_ptr_gpu);


}

void filters(cv::Mat &src, cv::Mat &filtered_src, std::vector<float> &filter1, std::vector<float> &filter2, char &filter_type, std::vector<int> &var, bool &filter_init, bool supress_vid)
{
    std::string filter_name;
    cv::Mat mid_img = cv::Mat::zeros(src.size(), CV_32FC3), filtered_src32 = cv::Mat::zeros(src.size(), CV_32FC3);;

    int rows = src.rows, cols = src.cols;
    uchar *src_ptr_cpu, *src_ptr_gpu; float *mimg_ptr_cpu, *mimg_ptr_gpu, *fimg_ptr_cpu, *fimg_ptr_gpu;
    size_t size = rows*cols*3*sizeof(uchar), size_float = rows*cols*3*sizeof(float);

    errCheck(cudaMalloc(&src_ptr_gpu, size));
    errCheck(cudaMalloc(&mimg_ptr_gpu, size_float));
    errCheck(cudaMalloc(&fimg_ptr_gpu, size_float));

    errCheck(cudaMallocHost(&src_ptr_cpu, size));
    errCheck(cudaMallocHost(&mimg_ptr_cpu, size_float));
    errCheck(cudaMallocHost(&fimg_ptr_cpu, size_float));

    memcpy(src_ptr_cpu, src.data, size);
    errCheck(cudaMemcpy(src_ptr_gpu, src_ptr_cpu, size, cudaMemcpyHostToDevice));

    memcpy(mimg_ptr_cpu, mid_img.data, size_float);
    errCheck(cudaMemcpy(mimg_ptr_gpu, mimg_ptr_cpu, size_float, cudaMemcpyHostToDevice));

    memcpy(fimg_ptr_cpu, filtered_src32.data, size_float);
    errCheck(cudaMemcpy(fimg_ptr_gpu, fimg_ptr_cpu, size_float, cudaMemcpyHostToDevice));


    switch (filter_type)
    {
        case 'g':
            filter_name = "Grayscale";
            GrayScale(src, filtered_src);
            break;

        case 'b': {
            filter_name = "Blur";
            cudaStream_t stream_gaussian;
            cudaStreamCreate(&stream_gaussian);

            if (!(filter_init)) {
                filter_input(filter1);
                filter_init = true;
            }

            filter_fn(src, src_ptr_gpu, mimg_ptr_gpu, fimg_ptr_gpu, filter1, filter1, stream_gaussian);

            cudaMemcpy(fimg_ptr_cpu, fimg_ptr_gpu, size_float, cudaMemcpyDeviceToHost);
            cudaStreamDestroy(stream_gaussian);

            memcpy(filtered_src32.data, fimg_ptr_cpu, size_float);


            cv::convertScaleAbs(filtered_src32, filtered_src);

            break;
        }

        case 'x': {
            filter_name = "Sobel x";
                cudaStream_t stream_sobelx;
                cudaStreamCreate(&stream_sobelx);

            std::vector<float> filter_diff = {-1, 0, 1};
            if (!(filter_init)) {
                filter_input(filter1);
                filter_init = true;
            }
            if (filter1.size() != filter_diff.size()) {
                std::cout << "Filter sizes not matching!";
                exit(-1);
            }



            filter_fn(src, src_ptr_gpu, mimg_ptr_gpu, fimg_ptr_gpu, filter_diff, filter1, stream_sobelx);
            cudaMemcpy(fimg_ptr_cpu, fimg_ptr_gpu, size_float, cudaMemcpyDeviceToHost);
            cudaStreamDestroy(stream_sobelx);

            memcpy(filtered_src32.data, fimg_ptr_cpu, size_float);
            cv::convertScaleAbs(filtered_src32, filtered_src);
            break;
        }

        case 'y': {
            filter_name = "Sobel y";
            cudaStream_t stream_sobely;
            cudaStreamCreate(&stream_sobely);

            std::vector<float> filter_diff = {-1, 0, 1};
            if (!(filter_init)) {
                filter_input(filter1);
                filter_init = true;
            }
            if (filter1.size() != filter_diff.size()) {
                std::cout << "Filter sizes not matching!";
                exit(-1);
            }

            filter_fn(src, src_ptr_gpu, mimg_ptr_gpu, fimg_ptr_gpu, filter1, filter_diff,  stream_sobely);
            cudaMemcpy(fimg_ptr_cpu, fimg_ptr_gpu, size_float, cudaMemcpyDeviceToHost);
            cudaStreamDestroy(stream_sobely);

            memcpy(filtered_src32.data, fimg_ptr_cpu, size_float);
            cv::convertScaleAbs(filtered_src32, filtered_src);
            break;
        }

        case 'l': {
            filter_name = "Blur + Quantize";

            //cv::Mat filtered_src32 = cv::Mat::zeros(src.size(), CV_32FC3);
            cv::Mat blur;
            int level;
            char filter1_type = 'b';

            if (!(filter_init)) {
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
            //cv::Mat filtered_src32 = cv::Mat::zeros(src.size(), CV_32FC3);
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

            if(!(filter_init)) {
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

    if(!(supress_vid)) {
        disp(filtered_src, filter_name);
//        cv::imwrite("G:/house1.jpg", filtered_src);
    }


    errCheck(cudaFree(src_ptr_gpu));
    errCheck(cudaFree(fimg_ptr_gpu));
    errCheck(cudaFree(mimg_ptr_gpu));

    errCheck(cudaFreeHost(src_ptr_cpu));
    cudaFreeHost(mimg_ptr_cpu);
    errCheck(cudaFreeHost(fimg_ptr_cpu));


}
#include<opencv2/opencv.hpp>
#include <algorithm>
#include <vector>
#include<iostream>
#include <cmath>
#include<numeric>


void filter_processing(std::vector<float> &filter_in,std::vector<float> &norm_filter)
{
    float sum;
    if (std::any_of(filter_in.begin(), filter_in.end(), [](int elem) { return elem < 0; }))
    {
        std::vector<float> abs_filter (filter_in.size());
        std::transform(filter_in.begin(), filter_in.end(), abs_filter.begin(), static_cast<float (*) (float)> (std::abs));

        std::sort(abs_filter.begin(), abs_filter.end());

        auto new_end = std::unique(abs_filter.begin(), abs_filter.end());

        abs_filter.erase(new_end, abs_filter.end());

        sum = std::accumulate(abs_filter.begin(), abs_filter.end(), 0.0f);
    }

    else
        sum = std::accumulate(filter_in.begin(), filter_in.end(), 0.0f);

    std::cout << sum << std::endl;

    //std::transform(filter_in.begin(), filter_in.end(), norm_filter.begin(), [sum](float x) { return x / sum; });
    for(int i=0;i<filter_in.size(); i++)
    {
        norm_filter[i] = filter_in[i]/sum;
    }
}

void filter_input(std::vector<float> &filter)
{
    std::vector<float> filter_in;
    float temp; // declare a temporary variable to store input
    std::cout << "Input the filter (Input any number > 100 to stop reading): " << std::endl;
    while (std::cin >> temp)
    {   if(temp > 100)
            break;
        filter_in.push_back (temp);
        filter.push_back(0.0f);// add input to the vector
    }
    filter_processing(filter_in, filter);
}
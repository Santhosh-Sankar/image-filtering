#include<string>

#ifndef CV_ASS1_FILTERS_H
#define CV_ASS1_FILTERS_H

void disp(cv::Mat &frame, const std::string &win_name);
void filters(cv::Mat &src, cv::Mat &filtered_src, std::vector<float> &filter1, std::vector<float> &filter2, char &filter_type, std::vector<int> &var, bool &filter_init, bool supress_vid);

#endif //CV_ASS1_FILTERS_H

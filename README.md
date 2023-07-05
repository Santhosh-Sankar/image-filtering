# image-filtering
Filtering an image or a live stream video feed using OpenCV and C++ with CPU and GPU (using CUDA).  

There exist two versions of this code. Both the versions perform Gaussian, Sobel, Magnitude, Blur+Quantization and Cartoonization filteringon a given image or a live feed video stream.  

The first is the sequential CPU code and the second is the parallelized code using CUDA which parallelizes the filtering process.


## Outputs
The outputs after applying the filters to an input image are shown below. 

### Raw Image
<p align='center'>
    <img src="/images/house.jpg" alt="Image src" width="500"/>
</p>


### Blurred image
<p align='center'>
    <img src="/images/blur.jpg" alt="Image 1" width="300" />
</p>


### Sobel X Image
<p align='center'>
    <img src="/images/sobel_x.jpg" alt="Image 1" width="300" />
</p>


### Sobel Y Image
<p align='center'>
    <img src="/images/sobel_y.jpg" alt="Image 2" width="300" />
</p>


### Magnitude Image
<p align='center'>
    <img src="/images/magnitude.jpg" alt="Image 2" width="300" />
</p>

### Blur+Quantized Image
<p align='center'>
    <img src="/images/quantization.jpg" alt="Image 1" width="300" />
</p>

### Cartoonized Image
<p align='center'>
    <img src="/images/cartoonization.jpg" alt="Image 2" width="300" />
</p>


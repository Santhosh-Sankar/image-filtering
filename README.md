# image-filtering
FIltering an limage or a live stream video feed using OpenCV and C++ with CPU and GPU (using CUDA).  

There exist two versions of this code. Both the versions perform Gaussian, Sobel, Magnitude, Blur+Quantization and Cartoonization filteringon a given image or a live feed video stream.  

The first is the sequential CPU code and the second is the parallelized code using CUDA which parallelizes the filtering process.


##Outputs
The outputs after applying the filters to an input image are shown below. 

<p align='center'>
    <img src="/images/dog.jpg" alt="Image src" width="1000"/>
    <figcaption>"Raw Image"</figcaption>
</p>

<div>
    <img src="/images/blur.jpg" alt="Image 1" width="400" />
    <figcaption>"Blurred image"</figcaption>
    <img src="/images/magnitude.jpg" alt="Image 2" width="400" />
    <figcaption>"Magnitude Image"</figcaption>
</div>

<div>
    <img src="/images/sobel_x.jpg" alt="Image 1" width="400" />
    <figcaption>"Sobel X Image"</figcaption>
    <img src="/images/sobel_y.jpg" alt="Image 2" width="400" />
    <figcaption>"Sobel Y Image"</figcaption>
</div>

<div>
    <img src="/images/quantization.jpg" alt="Image 1" width="400" />
    <figcaption>"Quantized Image"</figcaption>
    <img src="/images/cartoonization.jpg" alt="Image 2" width="400" />
    <figcaption>"Cartoonized Image"</figcaption>
</div>
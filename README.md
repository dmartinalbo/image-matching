# Image Classification and Matching Using Local Features and Homography

#### Requierements
- python
- opencv + opencv_contrib with python bindings

More info on how to install opencv3 + python [here](http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/). Follow step 10 and so on. Caution: opencv3.0 and/or opencv3.1 contains a bug with some python bindings. Install the last commited version.

- numpy

#### Usage:
```bash
$ python image-matching.py \
            -t <TEMPLATEs> \
            -q <QUERYs> \ 
```

##### Parameters:
- `-t template_names`: List of template images. If several templates are provided, the program will classify each query as a template, providing the posterior probability. Successive steps will be performed assuming that the query image is similar to the chosen template.
- `-q query_names`: List of query images

##### Optionals:
- `-o output_path`: Output directory
- `-c bounding_boxes`: Bounding boxes to crop. The format is "WxH+X+Y". If no bounding box is provided (or more than one template is provided) the whole image will be saved.
- `-v`: Increase output verbosity
- `-p`: Use only if the image is scanned or photocopied. Do not use with images captured with a camera. This parameter transforms the 3D homography to a 2D affine transformation.
- `--matches`: Shows the matching result and the good matches

## Examples

### Finding a template inside an image

We have the following images:
![Template image](https://github.com/dmartinalbo/image-matching/blob/master/example/lena_eyes.png "Template image")
![Query image](https://github.com/dmartinalbo/image-matching/blob/master/example/lena.png "Query image")

And we want to find the first one called `example/lena_eyes.png` (template) inside the other, called `example/lena.png` (query). The query image can contain distortions (rotation, scale, translation) that can alter the aspect of template.

We execute `image-matching.py` with the proper parameters:
```bash
$ python image-matching.py \
          -t example/lena_eyes.png \
          -q example/lena.png \
          -v \
          -p \
          --matches
```
Here, we include `-p` given that there is no perspective distortion. 

Execution log:
```bash
2016-06-20 16:19:31,883 Loading template image example/lena_eyes.png
2016-06-20 16:19:31,883   Calculating SIFT features ...
2016-06-20 16:19:31,888 Loading query image example/lena.png
2016-06-20 16:19:31,888   Calculating SIFT features ...
2016-06-20 16:19:31,930 Estimating match between example/lena_eyes.png and example/lena.png
2016-06-20 16:19:31,931 p(t_0 | x) = 1.0000
2016-06-20 16:19:31,931 Estimating homography between example/lena_eyes.png and example/lena.png
2016-06-20 16:19:31,931 Simplifying transformation matrix ...
```

In the screen we can observe the matching result:
![Matching and good matches](https://github.com/dmartinalbo/image-matching/blob/master/example/matches.png "Matching and good matches")

And the file `example/lena_fix.png` is:
![Result](https://github.com/dmartinalbo/image-matching/blob/master/example/lena_fix.png "Result")

### Finding a template inside an image and crop a particular region

Again, we have the following images:
![Template image](https://github.com/dmartinalbo/image-matching/blob/master/example/lena_eyes.png "Template image")
![Query image](https://github.com/dmartinalbo/image-matching/blob/master/example/lena.png "Query image")

And we want to find the first one called `example/lena_eyes.png` (template) inside the other, called `example/lena.png` (query). But, in this case, we want to crop a particular region of the query image, which is the eye. We know that the eye is located at (10,5) in the template image and its size is 40x50 ("40x50+10+5").

We execute `image-matching.py` with the proper parameters:
```bash
$ python image-matching.py \
          -t example/lena_eyes.png \
          -q example/lena.png \
          -v \
          -p \
          -o example \
          -c 30x30+10+5
```
Here we include `-c` with the proper region to crop.

Execution log:
```bash
2016-06-30 09:51:15,123 Loading template image example/lena_eyes.png
2016-06-30 09:51:15,124   Calculating SIFT features ...
2016-06-30 09:51:15,128 Loading query image example/lena.png
2016-06-30 09:51:15,129   Calculating SIFT features ...
2016-06-30 09:51:15,169 Estimating match between example/lena_eyes.png and example/lena.png
2016-06-30 09:51:15,171 p(t_0 | x) = 1.0000
2016-06-30 09:51:15,171 Estimating homography between example/lena_eyes.png and example/lena.png
2016-06-30 09:51:15,171 Simplifying transformation matrix ...
2016-06-30 09:51:15,173 Cropping "30x30+10+5" from example/lena.png
2016-06-30 09:51:15,173   Saved in ./example/lena_crop_0.png
```

As a result we obtain the image `lena_crop_0.png` ![Result](https://github.com/dmartinalbo/image-matching/blob/master/example/lena_crop_0.png "Result")
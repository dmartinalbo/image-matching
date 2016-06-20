# Image Classification and Matching Using Local Features and Homography

#### Usage:
```bash
$ python image-matching.py \
            -t <TEMPLATEs> \
            -q <QUERYs> \ 
```

##### Parameters:
- `-t template_names`: List of template images
- `-q query_names`: List of query images
##### Optionals:
- `-o output_path`: Output directory
- `-c bounding_boxes`: Bounding boxes to crop. The format is "WxH+X+Y". If no bounding box is provided the whole image will be saved.
- `-v`: Increase output verbosity
- `-p`: Use only if the image is scanned or photocopied, do not with photos! This parameter transforms the homography to a 2D affine transformation.
- `--matches`: Shows the matching result and the good matches

## Example

We have the following images:
![Template image](https://github.com/dmartinalbo/image-matching/blob/master/example/lena_eyes.png "Template image")
![Query image](https://github.com/dmartinalbo/image-matching/blob/master/example/lena.png "Query image")

And we want to find the first one called `example/lena_eyes.png` inside the other, called `example/lena.png`. We execute `image-matching.py` with the proper parameters:
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

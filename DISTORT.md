# Compression artifact removal

In addition to being a quite good upscaling solution, FSRCNNX can be trained to mitigate compression artifacts. It performs far better than many conventional post-processing methods, which are mainly targeted at older video coding formats and are too aggressive for the compression artifacts you find in most videos nowadays, provided you choose the right approach.

To teach the model how to mitigate artifacts, the --distort flag introduces a chance for each training image to get degraded via selectable lossy compression.

How much the model will eventually remove artifacts (and small details) is influenced by the following factors:

* Chance to trigger lossy compression
* Compression quality range
* Training duration
* Model parameters
* Training data
* Compression types

***

#### Chance to trigger lossy compression

This is by far the most influential factor. Originally the chance was 33%, which was too low to produce considerable results. After several tests by multiple people, 80-90% are recommended. Avoid 100%.

#### Lossy compression quality range

Less important than one might think. Once you set the range low enough for considerable compression to occur, the results will stay the same. Still, I think very low values are probably not beneficial, as the image gets distorted to such degree, that it's sometimes unrecognisable and would require magic to properly restore. It's probably best to compress a few of the training pictures with different settings and see which range produces the quality you typically want to enhance. Use compression debug to see what your settings look like.

#### Training duration

The longer you train (epochs), the more the model will smooth over an image in the end. Just another reason to be careful not to overtrain your model. Ideally, you find the right training duration to reach peak sharpness. Interestingly enough, this duration will be very similar for different models regardless of their layer's width (e.g. 16-0-4-1 and 56-16-4-1). Only when you go as low as 8-0-4-1 will behavior change and the necessary training duration increases dramatically. 

#### Model parameters

Another influential factor, but still important to keep in mind. Increasing a model's d and s parameters will also increase the strength of its artifact mitigation. Decreasing the training duration just a bit should be enough to counteract this behavior. Alternatively, if you want to go for maximum artifact removal, you should work with high d and s values. For example, d=8 is not capable of removing pixelated diagonals, while d=12 does it well.

#### Training data

This one should be obvious, and I only include it for completeness sake. Large, sharp datasets of >1000Mpx total work well for me. Add computer-generated images if you want and use "keywords_of_renders" to keep track of them in your dataset. See "keywords_of_renders_sample_lightly" to help with some video game datasets.

#### Compression types

Two modes are available: "h264" and "avif". H.264 is a very common video codec. AVIF is AV1's video compression applied to a single image. Use the codec that matches your videos. You can adjust settings in "utils.py".
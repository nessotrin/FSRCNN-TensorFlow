# Compression artifact removal

In addition to being a quite good upscaling solution, FSRCNNX can be trained to mitigate compression artifacts. It performs far better than many conventional post-processing methods, which are mainly targeted at older video coding formats and are too aggressive for the compression artifacts you find in most videos nowadays, provided you choose the right approach.

To teach the model how to mitigate artifacts, the --distort flag introduces a chance for each training image to get degraded via lossy JPEG compression.

How much the model will eventually remove artifacts (and small details) is influenced by the following factors:

* Chance to trigger JPEG compression
* JPEG compression quality range
* Training duration
* Model parameters
* Training data

***

#### Chance to trigger JPEG compression

This is by far the most influential factor. Originally the chance was 33%, which was too low to produce considerable results. After several tests I concluded that 75-80% give good results, without introducing too much blur. 100% should be avoided as it leads to extreme detail loss.

#### JPEG compression quality range

Less important that one might think. Once you set the range low enough for considerable JPEG compression to occur, the results will stay the same. Still, I think very low values (1-25) are probably not beneficial, as the image gets distorted to such a degree, that it's sometimes unrecognisable and would require magic to properly restore. It's probably best to compress a few of the training pictures with different settings and see which range produces the quality you typically want to enhance. In my case that led me to choose 50-75.

#### Training duration

The longer you train (epochs), the more the model will smooth over an image in the end. Just another reason to be careful not to overtrain your model. Ideally you find the right training duration to reach peak sharpness. Interestingly enough, this duration will be very similar for different models regardless of their layer's width (e.g. 16-0-4-1 and 56-16-4-1). Only when you go as low as 8-0-4-1 will the behaviour change and the necessary training duration increases dramatically. 

#### Model parameters

Another less influential factor, but still important to keep in mind. Increasing a model's d and s parameters will also increase the strength of its artifact mitigation. Decreasing the training duration just a bit should be enough to counteract this behaviour. Alternatively, if you want to go for maximum artifact removal, you should work with high d and s values.

#### Training data

This one should be obvious and I only include it for completeness sake.

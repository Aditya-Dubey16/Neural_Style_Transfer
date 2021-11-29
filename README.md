# Neural Style Transfer

We will learn how to use deep learning to compose images in the style of another image (ever wish you could paint like Picasso or Van Gogh?). This is known as neural style transfer! This is a technique outlined in Leon A. Gatys’ paper, A Neural Algorithm of Artistic Style, which is a great read, and you should definitely check it out.

Neural style transfer is an optimization technique used to take three images, a content image, a style reference image (such as an artwork by a famous painter), and the input image you want to style — and blend them together such that the input image is transformed to look like the content image, but “painted” in the style of the style image.
### VGG19 model is used. 
## Style Image
![alt text](https://github.com/Aditya-Dubey16/Neural_Style_Transfer/blob/main/2.jpg)
## Content Image 
![alt text](https://github.com/Aditya-Dubey16/Neural_Style_Transfer/blob/main/5.jpg)

## Calculate Loss
We have content loss, style loss and Total variation loss.

The content of an image is represented by the values of the intermediate feature maps.It turns out, the style of an image can be described by the means and correlations across the different feature maps. Calculate a Gram matrix that includes this information by taking the outer product of the feature vector with itself at each location, and averaging that outer product over all locations. This Gram matrix can be calculated for a particular layer as:

          𝐺𝑙𝑐𝑑=∑𝑖𝑗𝐹𝑙𝑖𝑗𝑐(𝑥)𝐹𝑙𝑖𝑗𝑑(𝑥)𝐼𝐽
          

###### Total variation loss:
One downside to this basic implementation is that it produces a lot of high frequency artifacts. Decrease these using an explicit regularization term on the high frequency components of the image. In style transfer, this is often called the total variation loss.
####
We perform backpropagation in the usual way such that we minimize this  loss. We thus change the initial image until it generates a similar response in a certain layer (defined in content_layer) as the original content image. We use the Adam optimizer in order to minimize our loss. We iteratively update our output image such that it minimizes our loss: we don’t update the weights associated with our network, but instead we train our input image to minimize loss.

Here we use tf.GradientTape to compute the gradient. It allows us to take advantage of the automatic differentiation available by tracing operations for computing the gradient later.
## Style Image
###### Result after 100 iteration.
![alt text](https://github.com/Aditya-Dubey16/Neural_Style_Transfer/blob/main/output.jpg)
##### For better results run for more number of iteration.

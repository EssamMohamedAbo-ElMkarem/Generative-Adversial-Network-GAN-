# Generative Adversial Network

In this problem we'll be trying to develop a GAN based on the CIFAR-10 Dataset
Understanding Generative Adversarial Networks (GANs)

Generative adversarial networks (GANs) are algorithmic architectures that use two neural networks, pitting one against the other (thus the “adversarial”) in order to generate new, synthetic instances of data that can pass for real data. They are used widely in image generation, video generation and voice generation.


<p align="center">
  <img src="https://github.com/EssamMohamedAbo-ElMkarem/Generative-Adversial-Network-GAN-/blob/main/docs/GANs.png" style="width:800px;"/>
</p>

## CIFER-10 Dataset

CIFAR is an acronym that stands for the Canadian Institute For Advanced Research and the CIFAR-10 dataset was developed along with the CIFAR-100 dataset (covered in the next section) by researchers at the CIFAR institute. The dataset is comprised of 60,000 32×32 pixel color photographs of objects from 10 classes, such as frogs, birds, cats, ships, airplanes, etc.

<p align="center">
  <img src="https://github.com/EssamMohamedAbo-ElMkarem/Generative-Adversial-Network-GAN-/blob/main/docs/cifer.png" style="width:400px;"/>
</p>


## The Generator

The generator uses tf.keras.layers.Conv2DTranspose (upsampling) layers to produce an image from a seed (random noise). Start with a Dense layer that takes this seed as input, then upsample several times until you reach the desired image size of 32x32x3. Notice the tf.keras.layers.LeakyReLU activation for each layer, except the output layer which uses tanh.

## The discriminator

The discriminator model has a normal convolutional layer followed by three convolutional layers using a stride of 2×2 to downsample the input image. The model has no pooling layers and a single node in the output layer with the sigmoid activation function to predict whether the input sample is real or fake. The model is trained to minimize the binary cross entropy loss function, appropriate for binary classification.
<p align="center">
  <img src="https://github.com/EssamMohamedAbo-ElMkarem/Generative-Adversial-Network-GAN-/blob/main/docs/cnn.jpeg" style="width:800px;"/>
</p>

## Final Results
Once a final generator model is selected, it can be used in a standalone manner for your application. This involves first loading the model from file, then using it to generate mages. The generation of each image requires a point in the latent space as input. generated_plot_e200.png In this case, we used the model saved after 200 training epochs, but the model saved after 100 epochs would work just as well.
<p align="center">
  <img src="https://github.com/EssamMohamedAbo-ElMkarem/Generative-Adversial-Network-GAN-/blob/main/training_logs/file/content/generated_plot_e200.png" style="width:800px;"/>
</p>

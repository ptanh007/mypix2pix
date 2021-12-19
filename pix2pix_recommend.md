### Generator

	* Each block in the encoder is: Convolution -> Batch normalization -> Leaky ReLU
		<downsample: use_bias=False>
		<downsample: 01 block đầu tiên apply_batchnorm=False>
	* Each block in the decoder is: Transposed convolution -> Batch normalization -> Dropout (applied to the first 3 blocks) -> ReLU
		<upsample: use_bias=False>
		<upsample: 03 block đầu tiên apply_dropout=True> 
		<last: activation='tanh'>
	* There are skip connections between the encoder and decoder (as in the U-Net).
	
	* generator loss is a sigmoid cross-entropy loss of the generated images and an array of ones.
		<BinaryCrossentropy(from_logits=True)>
	* The formula to calculate the total generator loss is gan_loss + LAMBDA * l1_loss, where LAMBDA = 100. This value was decided by the authors of the paper.


### Discriminator

	* Each block in the discriminator is: Convolution -> Batch normalization -> Leaky ReLU.
		<downsample: 01 block đầu tiên use_bias=False>
		<conv1: use_bias=False>
	* The shape of the output after the last layer is (batch_size, 30, 30, 1).
	* Each 30 x 30 image patch of the output classifies a 70 x 70 portion of the input image.


### Training

	* Optimizer
		generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
		discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
		
	* Check that neither the generator nor the discriminator model has "won". If either the gen_gan_loss or the disc_loss gets very low, it's an indicator that this model is dominating the other, and you are not successfully training the combined model.
	
	* The value log(2) = 0.69 is a good reference point for these losses, as it indicates a perplexity of 2 - the discriminator is, on average, equally uncertain about the two options.
	
	* For the disc_loss, a value below 0.69 means the discriminator is doing better than random on the combined set of real and generated images.
	
	* For the gen_gan_loss, a value below 0.69 means the generator is doing better than random at fooling the discriminator.
	* As training progresses, the gen_l1_loss should go down
	* each epoch can take around 15 seconds on a single V100 GPU 
[dataset (80k steps), train 200 epoch]

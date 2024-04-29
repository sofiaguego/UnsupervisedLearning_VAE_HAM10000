**PROJECT 1: Variational Autoencoder (VAE) Implemented on VAE 10000**

**Project Overview**

**Objective:** Implement a VAE for feature learning and image generation using 5K skin lesion images from the HAM10000 dataset.

**Architecture:** The VAE comprises an encoder and decoder. The encoder transforms input images into a lower-dimensional latent space representation, while the decoder reconstructs the original images from the latent space.

**Training Process**
**Reparameterization Trick:** VAE utilizes a reparameterization trick for efficient sampling of latent space representations.
**Forward Pass:** Input images are encoded into a latent space representation by the encoder, and then sampled using the reparameterization trick. The decoder reconstructs the input images from the sampled latent space representations.

**Results**
**Hyperparameters:** After experimenting with different hyperparameters, the best reconstruction results by using 5K which is half of the original dataset, we achieved:
- Latent dimension: 120
- Batch size: 32
- Learning rate: 1e-3
- Number of epochs: 100
  
**Performance:** While the results are not perfect, the VAE showed significant improvement with optimized hyperparameters and the reconstructed image shows the main characteristics of the original image and it is visible that the reconstruction is for a skin lesion image.

---

**PROJECT 2: Anomaly detection: Generation Approach Autoencoder**

**Project Overview**

**Objective:** Implement anomaly detection on the full HAM10000 dataset consisting of 10,000 skin lesion images. The dataset contains 7 classes of skin lesions, with cancerous ones categorized as "abnormal" and the remaining 4 classes considered as "normal". 
0: 'nv' - Melanocytic nevus, 1: 'mel' - Melanoma, 2: 'bkl' - Benign keratosis, 3: 'bcc' - Basal cell carcinoma, 4: 'akiec' - Actinic keratosis, 5: 'vasc' - Vascular lesion, 6: 'df' - Dermatofibroma.

Abnormal Classes = [1, 3, 4] 
Normal Classes = [0, 2, 5, 6] 


**Architecture:** The architecture employed for anomaly detection consisted of an Autoencoder (AE) model. The AE model comprised an encoder and a decoder, each consisting of multiple fully connected layers with ReLU activation functions. The encoder reduced the input image dimensionality to a lower-dimensional latent space, while the decoder attempted to reconstruct the original image from this latent representation.

**Training Process**: The training process involves the following steps:

1. **Model Training Setup**: 
   - The model was set to training mode using `model.train()` to enable dropout and batch normalization layers.
   - The training loop iterated over epochs, with each epoch consisting of multiple batches.

2. **Loss Calculation and Optimization**: 
   - For each batch in the training DataLoader, the model reconstructed the input data and computed the mean squared error (MSE) loss between the input and reconstructed samples.
   - Gradients of the loss with respect to model parameters were computed using backpropagation, and the model parameters were updated using the optimizer.

**Evaluation**: The evaluation process includes the following steps:

1. **Model Evaluation Setup**: 
   - The model was set to evaluation mode using `model.eval()` to disable dropout and batch normalization layers.

2. **Loss Calculation**: 
   - For each batch in the test DataLoader, the model reconstructed the input data.
   - MSE loss was calculated between the input data and reconstructed samples, and the loss distribution was flattened.

3. **Loss Distribution and Labels**: 
   - The loss distribution and corresponding labels were collected for further analysis.

4. **Evaluation Metrics**: 
   - Evaluation metrics such as precision, recall, and F1-score were calculated using the loss distribution and labels for anomaly detection.

**Results**: When checking the Loss distribution of the data we can see that the Anomalies have a normal distribution, however, the Normal does not have a good distribution at all and this is due to the class imbalance of the original dataset. 

**Performance:** Overall, the model did not perform well when reconstructing the image, however, after changing the hyper-parameters it did improve, which is the main objective rather than having a perfect model, plus, we learnt a lot!

**Future Improvements:** Future improvements for this model include incorporating convolutional neural networks (CNNs) to capture more intricate patterns in the images. Additionally, proper normalization of the images and addressing class imbalance issues, possibly by utilizing techniques such as Variational Autoencoders (VAEs), could enhance the model's performance.


**Note**: Both projects utilized the HAM10000 dataset. However, during the loading process, the pixel values were extracted in tabular form. This approach may have introduced a discrepancy, as the visualization of the images revealed a decrease in quality. Addressing this issue is essential to ensure accurate representation and analysis of the dataset.


Thanks for checking this repo out :)




  

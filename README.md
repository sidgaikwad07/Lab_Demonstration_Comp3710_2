# Lab_Demonstration_Comp3710_2

Here is a README file based on the provided lab details:

---

# Pattern Recognition Lab - Dimensionality Reduction, Classification, Convolutional Neural Networks(CNNs) & General Adversarial Networks(GANs)

## Overview

This lab is aimed at understanding the basics of TensorFlow/PyTorch through dimensionality reduction and classification. The tasks are split into three main parts:

1. **Part 1 - Eigenfaces & PCA**: Implement PCA of human faces and use Random Forest classification.
2. **Part 2 - CNN Classifier**: Implement a CNN-based classifier to recognize human faces.
3. **Part 3 - Recognition Problem**: Solve a recognition problem using deep learning frameworks such as TensorFlow, PyTorch, or JAX.

Each task involves implementing and demonstrating your solutions to your instructor in practical sessions. Ensure you have demonstrated all tasks before the due date to receive full marks.

---

## Part 1 - Eigenfaces

### Task Description:
We compute **Eigenfaces** (PCA of human faces) using Numpy and the **Labeled Faces in the Wild (LFW)** dataset. The process involves:

- Loading the LFW dataset
- Computing PCA to reduce dimensionality
- Visualizing the resulting eigenfaces
- Using the PCA subspace to build a **Random Forest Classifier** to classify faces

### Instructions:
1. Load the LFW dataset using Scikit-learn.
2. Perform PCA by computing the eigen-decomposition of the training data matrix after centering.
3. Visualize the eigenfaces.
4. Evaluate the performance of dimensionality reduction with a compactness plot.
5. Build a Random Forest classifier and evaluate its performance.

### Your Task:
Re-implement the PCA algorithm using TensorFlow or PyTorch.

---

## Part 2 - CNN Classifier

### Task Description:
You will implement a **Convolutional Neural Network (CNN)** to classify the same LFW dataset as in Part 1. The CNN is expected to outperform the PCA + Random Forest approach.

### Instructions:
1. Implement a CNN with two convolution layers of 3x3, each with 32 filters.
2. Connect the CNN to fully connected (dense) layers for classification.
3. Use the Adam optimizer and sparse categorical cross-entropy loss.
4. Normalize and resize the input images into 4D tensors (batch_size, channels, height, width).
5. Train and evaluate the model.

---

## Part 3 - Recognition Problem

### Task Description:
Solve one of the recognition problems using TF/Keras/PyTorch, achieving reasonable results. The task difficulty and complexity determine the maximum marks available.

### Project Title: Generative Adversarial Network for Realistic Brain Image Synthesis
1. **Generative Adversarial Networks (GANs)** for realistic brain generation from the OASIS dataset.

### Your Task:
- Choose one of the recognition problems.
- Develop your solution and explain all results, network layers, and code.
- Commit and push your code to a GitHub repository.

### Description:
- This project implements a Generative Adversarial Network (GAN) to generate realistic brain images based on the OASIS dataset. The GAN consists of a generator network that learns to produce synthetic brain images, and a discriminator network that differentiates between real and generated images. By training them adversarially, the generator aims to improve its realism, while the discriminator works to accurately identify real and fake images.

### Dataset

1) Source: OASIS dataset located at /home/groups/comp3710/OASIS/keras_png_slices* directories (train, test, and validation)
2) Preprocessing:
   2.1) Images are resized to a fixed size of 128x128 pixels.
   2.2) Converted to grayscale format.
   2.3) Pixel values are normalized to the range [-1, 1] for better training performance.

## Model Architecture
1) Generator :
   - Uses a fully-connected layer as the initial step to map a random noise vector (100 dimensions) to a higher-dimensional feature space.
   - Employs transposed convolutional layers (ConvTranspose2d) to progressively upsample the feature maps and generate a final image with the desired size (128x128) and grayscale channel (1).
   - Leaky ReLU activation functions are used for non-linearity.
   - Batch normalization layers are added for improved convergence and stability.
   - Tanh activation is used in the final layer to constrain the output pixel values between -1 and 1.
2) Discriminator :
   - Utilizes convolutional layers (Conv2d) to extract features from the input images.
   - Employs Leaky ReLU activations for non-linearity.
   - Includes Dropout layers to prevent overfitting.
   - Flattens the output feature maps before feeding them into a final linear layer that outputs a single value representing the probability of the input being a real image.

## Training Process

1) Hyperparameters:
       i) Learning rate: 1e-4 (Adam optimizer) for both generator and discriminator.
      ii) Batch size: 350
     iii) Epochs: 500
      iv)Noise dimension: 100 (controls the complexity of the generated images)
   
2) Loss Function: Binary Cross-Entropy with Logits (BCEWithLogitsLoss) is used to measure the difference between the discriminator's predictions and the true labels (real or fake).

3) Training Loop:
   ### In each training step:
      a) Real images are fetched from a batch of the training data.
      b) Random noise vectors are generated for the generator.
         i) The discriminator is trained first:
            A) It takes real and fake images (generated from noise) as input.
            B) It tries to classify them accurately (real as 1, fake as 0).
            C) The discriminator loss is calculated based on its classification performance.
         ii) The generator is trained next:
            A) It aims to generate images that fool the discriminator into classifying them as real.
            B) The generator loss is calculated based on how well it deceives the discriminator.

## Evaluation
1) Training losses (generator and discriminator) are monitored and visualized as plots to assess convergence and training progress.
2) Generated images at specific epochs are saved for qualitative evaluation of their realism compared to real brain images.
---

## Additional Requirements

### Advanced Git Course:
Complete the "Version Control for Teams using Git" short course on edX.

### DAWNBench Challenge:
Create a fast CIFAR-10 dataset classification network, aiming for more than 93% accuracy within the fastest time (preferably under 30 minutes on the Rangpur cluster). Use the NVIDIA A100 GPU for faster training. Mixed precision and optimized architectures such as ResNet-18 may help achieve this.

---

## Appendix

### Rangpur HPC Cluster
For this lab, you will use UQ’s **Rangpur HPC cluster** to run models and evaluate results. Detailed information can be found in the lab resources provided by your instructor.

---

## References

### 1) https://www.datacamp.com/tutorial/principal-component-analysis-in-python
### 2) https://towardsai.net/p/l/impact-of-optimizers-in-image-classifiers
### 3) https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
### 4) https://www.geeksforgeeks.org/generative-adversarial-network-gan/
### 5) 


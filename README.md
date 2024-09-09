# Lab_Demonstration_Comp3710_2

Here is a README file based on the provided lab details:

---

# Pattern Recognition Lab - Dimensionality Reduction and Classification

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

### Options:
1. **Variational Autoencoder (VAE)** on Magnetic Resonance (MR) images of the brain.
2. **UNet** for MR image segmentation of the brain (Max 8 Marks).
3. **Generative Adversarial Networks (GANs)** for realistic brain generation from the OASIS dataset.

### Your Task:
- Choose one of the recognition problems.
- Develop your solution and explain all results, network layers, and code.
- Commit and push your code to a GitHub repository.

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

### DAWNBench Resources
Check the **Crash Course in Deep Learning** and the available DAWNBench code resources for help in solving the challenge.

---

## Mark Breakdown

1. **Part 1 - Eigenfaces**:
2. **Part 2 - CNN Classifier**: 
   - CNN Classifier:
   - Advanced Git Course: 
   - DAWNBench Challenge: 
3. **Part 3 - Recognition Problem**: 

### GitHub Requirements:
1. Code must be hosted in your GitHub project with relevant commit logs.
2. Code should be well-commented and structured.
3. Commit messages must be meaningful.

---

## Submission

- Push all your code to a GitHub repository.
- Ensure proper commit messages and well-documented code.
- Demonstrate your work to the instructor before the due date to be awarded marks.

---


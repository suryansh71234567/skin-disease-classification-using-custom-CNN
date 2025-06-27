## Abstract

Skin diseases are a prevalent global health concern, and their accurate detection plays a critical role in early diagnosis and treatment. This project focuses on leveraging deep learning techniques, specifically convolutional neural networks (CNNs), to classify skin diseases from dermoscopic images. Two distinct approaches were explored: one involved designing a custom CNN model from scratch, and the other utilized the ResNet-50 architecture as a baseline. I also trained Resnet 50 model on the same dataset for comparing the efficiency of my models.

The custom CNN model was designed to optimize computational efficiency, incorporating significantly fewer parameters than ResNet-50, and other famous architectures while maintaining robust performance. Techniques such as Squeeze-and-Excitation (SE) blocks were integrated into the architecture to enhance the model's ability to focus on fine-grained skin textures and subtle disease-specific features. On the other hand, the ResNet-50-based model was adapted by customizing specific layers to better suit the task of skin disease classification. Both models were trained and evaluated on a curated and augmented skin disease dataset, utilizing transformations such as random flipping, rotation, brightness adjustment, and normalization to enhance generalizability.

Experimental results showed that the custom CNN model achieved competitive performance, surpassing the ResNet-50 baseline in accuracy with approximately one-fourth the number of parameters. The custom model achieved an accuracy of 65% after 25 epochs of training, demonstrating its ability to efficiently extract disease-relevant features. The ResNet-50-based model also performed effectively, providing insights into the potential of transfer learning and architectural customization for this domain.

This project highlights the advantages of lightweight, task-specific CNN architectures in medical imaging applications, alongside the utility of established architectures like ResNet-50 as a foundation for customization. The final deliverable includes both a custom CNN model and a modified ResNet-50-based model, offering scalable and effective solutions for skin disease classification. Future work will focus on extending the dataset, improving model generalization, and introducing more features which will help the model detect skin disease with greater accuracy.

---

## Section 2: Methodology

### Problem Statement

The primary goal of this project was to build a robust convolutional neural network (CNN) for detecting skin diseases. Initially, the objective was to train a CNN for skin disease classification, but as the project progressed, the focus shifted to understanding the inner workings of CNNs and exploring their potential. Transfer learning with pre-trained models like ResNet-50 often provides strong baselines but comes with significant computational costs due to their depth and complexity. Therefore, the project evolved to design a custom CNN that is lightweight, efficient, and optimized specifically for the task of skin disease classification. Additionally, a modified version of ResNet-50 was developed to improve its performance for this specific domain.

### Data Collection and Preprocessing

#### Data Sources
The dataset was sourced from Kaggle, consisting of high-quality images for various skin diseases. However, the dataset exhibited class imbalance, with some categories significantly underrepresented.

#### Preprocessing Steps

**Data Augmentation:**

- Rotation: Randomly rotated images to simulate real-world variability.
- Flipping: Horizontal and vertical flips to increase data diversity.
- Brightness Adjustment: Modified brightness to make the model invariant to lighting conditions.

**Normalization:**

- Pixel values were rescaled to the range [0, 1] for faster convergence during training.

**Class Balancing:**

- Oversampling techniques and weighted loss functions were considered to address the class imbalance.

### Algorithms and Techniques

#### Custom CNN Architecture

- Focused on efficiency, using fewer layers compared to ResNet-50.
- Emphasized smaller kernel sizes (e.g., 3×3) to capture fine-grained details.
- Included SE blocks to improve feature representation.
- Architecture deep enough to understand skin defects and small enough for inexpensive computation.

#### Modified ResNet-50

- Customized by fine-tuning layers and reducing complexity.
- Reduced number of filters and added dropout layers for regularization.

---

## Model Development

### Architecture Details
####Input Layer
- Shape: `(3, H, W)`

#### 🔹 First Convolutional Block
- `Conv2d`: (3 → 32), Kernel: 3×3, Padding: 1  
- `BatchNorm2d`: (32)  
- `ReLU` Activation  
- `MaxPool2d`: Kernel: 3×3, Stride: 2  
- `Dropout`: 0.25

#### 🔹 Second Convolutional Block
- `Conv2d`: (32 → 64), Kernel: 3×3, Padding: 1  
- `ReLU` Activation  
- `Conv2d`: (64 → 64), Kernel: 3×3, Padding: 1  
- `SEBlock`: (64)  
- `BatchNorm2d`: (64)  
- `ReLU` Activation  
- `MaxPool2d`: Kernel: 2×2, Stride: 2  
- `Dropout`: 0.25

#### 🔹 Third Convolutional Block
- `Conv2d`: (64 → 128), Kernel: 3×3, Padding: 1  
- `ReLU` Activation  
- `Conv2d`: (128 → 128), Kernel: 3×3, Padding: 1  
- `SEBlock`: (128)  
- `BatchNorm2d`: (128)  
- `ReLU` Activation  
- `MaxPool2d`: Kernel: 2×2, Stride: 2  
- `Dropout`: 0.25

#### 🔹 Flatten Layer
- `Flatten`

#### 🔹 Fully Connected Layers
- `Linear`: (6272 → 1024)  
- `BatchNorm1d`: (1024)  
- `ReLU` Activation  
- `Dropout`: 0.5  
- `Linear`: (1024 → 10)

#### 🟢 Output
- Shape: `(10,)`  
- Activation: `Softmax` (multi-class classification)

#### Custom CNN

- **Convolution layers** with kernels ≤ 3×3.
- **SE Blocks** enhance feature learning.
- **Batch normalization** after each layer.
- **Dropout** after pooling to reduce overfitting.
- **Final layer**: Fully connected with softmax over 10 classes.

#### Outline of the Model:


#### Modified ResNet-50 (Summary of changes)

- Replaced 7×7 kernel with 3×3
- Reduced stride in early layers
- Used dilated convolutions in place of pooling
- Added SE blocks
- Increased filters in early layers
- Used multi-scale input and stronger augmentation

### Optimization

- **Loss**: Categorical Cross-Entropy
- **Optimizer**: Adam (lr = 0.001)
- **Regularization**: Dropout, BatchNorm, Data Augmentation

---

## Evaluation

- Accuracy as primary metric.
- Trained both models for 25 epochs.
- Custom CNN achieved 65% accuracy, outperforming ResNet-50 baseline.

### Implementation Workflow

- Split dataset: train/val/test
- Applied data augmentation
- Saved model checkpoints
- Evaluated on unseen data

---

## Results

### Custom CNN

- 6 million parameters (vs 25 million in ResNet-50)
- 65% validation accuracy
- Strong feature extraction using 3×3 kernels
- Lightweight and efficient

### ResNet-50

- Similar accuracy (~65%)
- Higher complexity and computational cost

---

## Final Deliverable

- Lightweight CNN with only 6M parameters
- Comparable accuracy to ResNet-50
- Training pipeline and codebase included
- Suitable for edge/mobile deployment

---

## Challenges Faced

- Calculating neurons in fully connected layers was complex
- Colab RAM limitations required image size and batch size reduction
- Overfitting mitigated with dropout and data augmentation
- Slow internet at home delayed training
- Long training cycles per change made iteration slow
- Debugging modified ResNet-50 was especially hard as a beginner

---

## Secondary Goals and Future Scope

- **HSV Color Space** instead of RGB
- Use parallel filters per HSV channel
- Inspired by Google’s Inception v4
- More robust and detail-aware feature extraction

---

## References

- [Python Crash Course](https://www.youtube.com/playlist?list=PL-osiE80TeTsqhIuOqKhwlXsIBIdSeYtc)
- [Machine Learning Intro](https://youtu.be/i_LwzRVP7bg?si=8EWzDA1a8GMhWHsj)
- [PyTorch Basics](https://youtu.be/EMXfZB8FVUA?si=OspKX21trOuTYXmA)
- [DL/NN Overview](https://youtu.be/aircAruvnKk?si=0y2gG56dbOw9GZlK)
- [Medium CNN Guide](https://medium.com/)
- [CS231n Convolutions](http://cs231n.github.io/convolutional-networks/)
- [SE Block Paper](http://arxiv.org/abs/1602.07261)
- [LeCun CNN Paper](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)

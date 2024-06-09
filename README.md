#Intel Image Classification using GradientBoostingClassifier

Overview
This repository provides an implementation of image classification for Intel's dataset using the GradientBoostingClassifier. While Convolutional Neural Networks (CNNs) are a popular choice for image classification, GradientBoostingClassifier can also be highly effective, particularly when combined with robust feature extraction techniques. This approach leverages the power of GradientBoostingClassifier to achieve high accuracy without the complexity and computational demands of CNNs.

Why GradientBoostingClassifier?
GradientBoostingClassifier is a powerful machine learning algorithm that excels in various tasks, including image classification, due to its following advantages:

Feature Engineering: It allows for sophisticated feature engineering, which can significantly improve the model's performance. This is particularly useful for datasets where domain knowledge can be applied to extract relevant features.

Handling Diverse Features: GradientBoostingClassifier can handle diverse types of features, including those extracted through traditional image processing techniques such as histograms, edges, and textures.

Robustness: It is robust to overfitting when properly tuned and can handle noisy data well, which is common in real-world image datasets.

Interpretable Models: Models built using GradientBoostingClassifier are often more interpretable than deep learning models, providing insights into which features are most important for classification.

Less Computationally Intensive: Unlike CNNs, GradientBoostingClassifier does not require GPUs for training, making it more accessible for environments with limited computational resources.

Dataset
The dataset used for this project is Intel's Image Classification dataset, which contains images of various categories such as buildings, forest, glacier, mountain, sea, and street.

Feature Extraction
To leverage the GradientBoostingClassifier, we employ several image processing techniques to extract meaningful features from the images:

Histogram Equalization: Enhances the contrast of images.
Gray-scale Transformation: Converts images to gray-scale to simplify the feature space.
Image Smoothing: Reduces noise using Gaussian blur.
Edge Detection: Identifies edges using Canny and Sobel filters.
Line Detection: Detects lines using Hough transforms.
SIFT Features: Extracts Scale-Invariant Feature Transform (SIFT) descriptors to capture local features.
HOG Features: Computes Histogram of Oriented Gradients (HOG) features for capturing gradient structure.

Dataset Description
This Data contains around 25k images of size 150x150 distributed under 6 categories:

'buildings' -> 0,

'forest' -> 1,

'glacier' -> 2,

'mountain' -> 3,

'sea' -> 4,

'street' -> 5


Conclusion
This repository demonstrates how to effectively use GradientBoostingClassifier for image classification without relying on CNNs. By leveraging robust feature extraction techniques, we achieve high classification accuracy while maintaining model interpretability and computational efficiency.

Feel free to explore the code and modify it to suit your specific needs. Contributions and suggestions are welcome!

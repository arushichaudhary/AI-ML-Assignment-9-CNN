# Assignment 9 – Image Classification using CNN (Cats vs Dogs)

## Objective
Build a Convolutional Neural Network (CNN) that classifies pet images into two categories — **Cat** and **Dog** — for an animal welfare organization looking to automate image sorting.

## Dataset
- Kaggle: [Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

## Libraries Used
- TensorFlow / Keras (model building, training, `ImageDataGenerator`)
- NumPy
- Matplotlib (visualizations)
- Pillow (image loading)
- scikit-learn (precision, recall, F1-score, confusion matrix)

## Methodology
1. **Data Understanding** – walked the folder structure, counted images per class, checked original image dimensions, and displayed 5 sample images with labels.
2. **Data Preprocessing** – resized all images to 128×128, normalized pixel values to 0–1, used an 80/20 train/test split, and built Keras `ImageDataGenerator` data generators.
3. **Model Development** – trained a CNN (architecture below) for 10 epochs using the Adam optimizer and binary cross-entropy loss.
4. **Model Evaluation** – computed test accuracy, precision, recall, F1-score, a confusion matrix, and accuracy/loss curves across epochs.
5. **Conclusion** – summarized findings and CNN vs ANN trade-offs.

## CNN Architecture
```
Conv2D(32, 3x3, ReLU) -> MaxPooling2D(2x2)
Conv2D(64, 3x3, ReLU) -> MaxPooling2D(2x2)
Conv2D(128, 3x3, ReLU) -> MaxPooling2D(2x2)
Flatten
Dense(128, ReLU)
Dense(1, Sigmoid)
```
Optimizer: Adam · Loss: Binary Crossentropy · Metric: Accuracy · Epochs: 10

## Results
> **Note:** The full dataset has ~12,500 images per class (~25,000 total). To keep training time reasonable on a single-CPU machine, a balanced subset of 2,000 training images and 500 test images per class (5,000 total) was used here. The exact same code runs unchanged on the full dataset — just point `TRAIN_DIR`/`TEST_DIR` at the complete folders.

Trained on a balanced 4,000 image training set (2,000 cats / 2,000 dogs) and evaluated on a held-out 1,000 image test set (500 cats / 500 dogs):

| Metric | Value |
|---|---|
| Test Accuracy | 0.771 |
| Test Loss | 0.896 |
| Precision | 0.790 |
| Recall | 0.738 |
| F1-Score | 0.763 |

Confusion Matrix:
```
              Predicted Cat   Predicted Dog
Actual Cat         402             98
Actual Dog         131            369
```

Training accuracy rose from ~59% to ~98% over 10 epochs while validation accuracy plateaued around 70–77%, showing the model overfit the training data somewhat — expected for a plain CNN with no dropout/augmentation trained on a moderate amount of data.

## Conclusion
This project built a CNN to classify cat and dog images, reaching 77% test accuracy after 10 epochs, though training accuracy (98%) outpaced validation accuracy, showing overfitting on the training set used. Convolutional layers are essential because they automatically learn spatial features, such as edges, textures and shapes, directly from raw pixels, removing the need for manual feature engineering. Pooling layers then downsample these feature maps, reducing the parameter count and making the model more tolerant of small shifts or distortions. Compared to a plain ANN, a CNN's key advantage is that it preserves an image's 2D spatial structure through shared convolutional filters, rather than flattening pixels into one long vector and losing that structure. A limitation of CNNs is that they need large amounts of labeled data to generalize well; with limited data, as seen here, they can overfit quickly and lose accuracy on unseen images.



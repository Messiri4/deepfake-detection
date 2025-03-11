# Deepfake Detection System

## Overview

This project aims to detect deepfake images using machine learning techniques. It involves exploratory data analysis (EDA), statistical tests for comparisons, and hypothesis evaluation.

## Requirements

Ensure the following dependencies are installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

If using Google Colab, mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Dataset Preparation

1. Upload the dataset to Google Drive.
2. Extract it into the working directory:

```python
!unzip /content/drive/MyDrive/Dataset.zip -d /content/Dataset
```

## Exploratory Data Analysis (EDA)

1. Count the number of images in each subfolder (`real` and `fake`).
2. Check for class imbalances in training, validation, and test sets.
3. Visualize sample images using Matplotlib:

```python
import matplotlib.pyplot as plt
import cv2
import os

image_path = '/content/Dataset/fake/sample_image.jpg'
image = cv2.imread(image_path)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Fake Image Sample")
plt.show()
```

## Statistical Test of Comparisons

Perform statistical tests to compare real vs. fake images. Example:

```python
from scipy.stats import ttest_ind

# Example data: Replace with actual feature extraction values
real_features = [0.75, 0.80, 0.78, 0.82, 0.79]
fake_features = [0.55, 0.58, 0.60, 0.54, 0.57]

stat, p_value = ttest_ind(real_features, fake_features)
print(f"T-test statistic: {stat}, P-value: {p_value}")
```

## Hypothesis Evaluation

Define and test hypotheses:

**Null Hypothesis (H0):** There is no significant difference between real and fake image features.

**Alternative Hypothesis (H1):** There is a significant difference between real and fake image features.

Interpretation:

- If `p_value < 0.05`, reject the null hypothesis, meaning a significant difference exists.
- Otherwise, fail to reject the null hypothesis.

## Running the Notebook

To execute the steps above, open `Deepfake_detection_system.ipynb` in Jupyter Notebook or Google Colab and run each cell sequentially.

## Conclusion

This project explores deepfake detection through EDA, statistical analysis, and hypothesis evaluation, forming the basis for training a machine learning model for classification.

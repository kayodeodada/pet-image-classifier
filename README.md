# Pet Image Classifier

Welcome to the **Pet Image Classifier** project! This repository implements a Python-based application that uses a pretrained Convolutional Neural Network (CNN) model to classify pet images, compare predictions to ground-truth labels, and summarize the performance of various CNN architectures. The repository is a fork of [AIPND-revision/intropyproject-classify-pet-images](https://github.com/udacity/AIPND-revision/tree/master/intropyproject-classify-pet-images), developed as part of Udacity’s **AI Programming with Python** Nanodegree.

---

## Purpose

1. **Classify Pet Images**: Utilizes pretrained CNN models to classify images into corresponding labels.
2. **Extract Labels from Filenames**: Derives the true label from the pet image filenames.
3. **Compare CNN Architectures**: Evaluates the performance of three CNN architectures: `vgg`, `alexnet`, and `resnet`.
4. **Summarize Performance**: Provides comprehensive performance metrics for each model, including precision, recall, and accuracy.

---

## Features

- **Multiple CNN Architectures**: Supports transfer learning with `vgg`, `alexnet`, and `resnet`.
- **Command-Line Interface**: Flexible input via command-line arguments.
- **Performance Summary**: Detailed metrics for evaluating classification accuracy.
- **File-Based Labeling**: Automatically extracts true labels from filenames for validation.
- **Dog Breed Identification**: Incorporates a file of valid dog breed names for more robust classification.

---

## Project Structure

```plaintext
pet-image-classifier/
├── pet_images/           # Directory containing pet images for classification
├── dognames.txt          # Reference file for dog breed names
├── check_images.py       # Main script for running the classifier
├── README.md             # Documentation for the repository
└── LICENSE               # License for the repository
```

## Usage

### Command-Line Interface

Run the classifier using the following format:

```bash
python check_images.py --dir <directory_with_images> --arch <model> --dogfile <dognames_file>
```

### Example

```bash
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
```

### Arguments

- `--dir`: Path to the directory containing pet images (default: `pet_images/`).
- `--arch`: Pretrained CNN model architecture (`vgg`, `alexnet`, `resnet`).
- `--dogfile`: Path to the file containing valid dog breed names.

---

## Results Summary

The classification results for the three CNN architectures are as follows:

| **Metric**                  | **AlexNet**   | **ResNet**   | **VGG**      |
|-----------------------------|---------------|--------------|--------------|
| **Total Images**            | 40            | 40           | 40           |
| **Dog Images**              | 30            | 30           | 30           |
| **Not-Dog Images**          | 10            | 10           | 10           |
| **% Correct Dogs**          | 100.00%       | 100.00%      | 100.00%      |
| **% Correct Not-Dogs**      | 100.00%       | 90.00%       | 100.00%      |
| **% Correct Breed**         | 80.00%        | 90.00%       | 93.33%       |
| **% Match**                 | 75.00%        | 82.50%       | 87.50%       |

The program evaluates the performance of each model architecture based on the following metrics:

| **Metric**       | **AlexNet**   | **ResNet**   | **VGG**      |
|------------------|---------------|--------------|--------------|
| **Accuracy**     | 75.00%        | 82.50%       | 87.50%       |
| **Precision**    | 100.00%       | 95.00%       | 97.00%       |
| **Recall**       | 80.00%        | 90.00%       | 93.33%       |
| **F1-Score**     | 88.89%        | 92.44%       | 95.08%       |

### Observations

1. **Dog Classification**: All three models achieved 100% accuracy for identifying dogs.
2. **Not-Dog Classification**:
   - AlexNet and VGG achieved 100%.
   - ResNet achieved 90%, showing slightly lower performance in identifying non-dog images.
3. **Breed Accuracy**:
   - VGG had the highest accuracy at 93.33%, followed by ResNet at 90%, and AlexNet at 80%.
4. **Overall Match**:
   - VGG had the best match rate (87.50%), followed by ResNet (82.50%) and AlexNet (75.00%).

---

## Key Highlights

1. **Pretrained Models**: Explores state-of-the-art CNN architectures to classify pet images effectively.
2. **Filename Label Extraction**: Extracts true labels directly from filenames to streamline the workflow.
3. **Comprehensive Comparison**: Evaluates multiple pretrained models for an informed performance analysis.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- Forked from [AIPND-revision/intropyproject-classify-pet-images](https://github.com/udacity/AIPND-revision/tree/master/intropyproject-classify-pet-images).
- Part of Udacity’s **AI Programming with Python** Nanodegree.
- Pretrained models provided by PyTorch and the open-source deep learning community.

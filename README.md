This project implements an end-to-end text classification pipeline:

Data preprocessing and cleaning
Text tokenization and sequence generation
Feature extraction through word embeddings
Multi-level news category prediction
Evaluation and validation of classification performance

The model predicts:

Category Level 1 (main category)
Category Level 2 (subcategory)

using a single neural network architecture with multiple output layers

--------------- Features -------------------

Text preprocessing and normalization
Tokenization and sequence padding
Label encoding for hierarchical categories
Word embedding layer
Convolutional Neural Network (CNN)
Multi-output classification
Model training and validation
Prediction on unseen news articles
Automatic CSV export of predictions

--------------- Architecture -------------------

Input Text

↓

Text Preprocessing

↓

Tokenization

↓

Embedding Layer

↓

1D Convolution Layer

↓

Global Max Pooling

↓

Dense Layer

↓

├── Output Level 1

└── Output Level 2

------------ Technologies -------------

Python
TensorFlow
Keras
Pandas
Scikit-Learn
NumPy

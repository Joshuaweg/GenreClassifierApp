# Genre Classifier - Interpretable Book Genre Classification

A deep learning project that classifies book descriptions into Fantasy, Fiction, or Nonfiction genres with built-in model interpretability.
![image](https://github.com/user-attachments/assets/31eebc2f-41fd-4ebd-889a-139aed6ec591)

## Overview

This project demonstrates an interpretable machine learning application that uses a custom deep learning model to classify book descriptions. The model processes text descriptions from Amazon.com and GoodReads to determine if a book belongs to Fantasy, Fiction, or Nonfiction genres.

### Key Features

- **High Accuracy**: Achieves 90% accuracy on genre classification
- **Large Training Dataset**: Trained on 8,000 different books
- **Simple Architecture**: Uses feedforward neural network with learnable bag-of-word embeddings
- **Web Interface**: Flask web application for easy model interaction
- **Model Interpretability**: Multiple techniques to understand model decisions
- **Nearest Neighbor Analysis**: Comparison with similar books in training data

## Technical Details

### Model Architecture

The model uses a simple feedforward neural network that leverages learnable bag-of-word embeddings. The core assumption is that collections of words in book descriptions create unique vectors for each book, allowing the model to establish clear decision boundaries between genres.

### Interpretability Methods
![image](https://github.com/user-attachments/assets/172bb0aa-b69d-4a57-bcb6-b6eb2ec5ad0c)

The Flask web application provides multiple techniques for understanding model decisions:

1. **LIME (Local Interpretable Model-agnostic Explanations)**
   - Explains individual predictions by approximating the model locally

2. **SampleSHAP (Shapley Additive Explanations)**
   - Attributes prediction importance to individual input features

3. **LRP (Layer-wise Relevance Propagation)**
   - Traces relevant patterns in the input that lead to the prediction

4. **Integrated Gradients**
   - Examines the model's behavior along a path from a baseline to the input

### Visualization
![image](https://github.com/user-attachments/assets/2f2a12ce-e0be-4474-b862-f1de10b898f6)

- **t-SNE Projections**: Visualize the training data distribution based on the trained classifier
- **Nearest Neighbors**: Display Top 10 most similar books from training data
- **Farthest Neighbors**: Show 10 most dissimilar books for contrast

## Getting Started

### Prerequisites

```
Python 3.7+
Flask
PyTorch
scikit-learn
matplotlib
numpy
pandas
```

### Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

For development:
```bash
python runserver.py
```

For production:
```bash
python runserver_live.py
```

The application will be available at `http://localhost:8501` by default.

## Usage

1. Enter a book description in the web interface
2. Select the interpretability method
3. View the classification result along with:
   - Confidence scores for each genre
   - Feature importance visualization
   - Similar and dissimilar books from the training data
   - t-SNE projection of the book's position relative to the training data

## Project Structure

```
GenreClassifier/
├── GenreClassifierNN.py   # Neural network model implementation
├── views.py              # Flask routes and view functions
├── plots.py             # Visualization functions
├── templates/           # HTML templates
├── models/             # Trained model weights
└── data/               # Training data and embeddings
```

## License

[MIT License](LICENSE)

## Acknowledgments

- Data sourced from Amazon.com and GoodReads
- Interpretability implementations based on LIME, SHAP, LRP, and Integrated Gradients papers 

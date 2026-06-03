# ML Facial Expressions Model

As we know, sign language is very important for deaf people to communicate. In sign
languages, people use hand gestures but they also use facial expressions a lot. These facial
expressions are not just for showing emotions; they also help in the grammar of the sentence, like
asking a question or showing doubt. These are called Grammatical Facial Expressions (GFEs). In this
project, I worked on the "Grammatical Facial Expressions" dataset which is taken from the UCI
Machine Learning Repository. It contains spatial face coordinates extracted from videos. The main
aim was to apply machine learning models to classify these expressions and also group them using
clustering. After pre-processing, I applied classification models including Random Forest, Support
Vector Machine (SVM), and Multilayer Perceptron (MLP) to predict the correct grammatical
expression. Also, I applied clustering models including Principal Component Analysis (PCA) and K-
Means to find natural groups in the face data. Finally, I compared the performance of these
models using Accuracy, F1-Score, and Silhouette Score. This project will help computer systems to
understand sign language better.

## Dataset

I used the "Grammatical Facial Expressions Dataset" from the UCI Machine
Learning website. This dataset has data from a Kinect sensor tracking a user's face. It contains 100
spatial coordinates (x, y, z) of face parts like eyes, nose, and mouth for thousands of video frames.
The x and y values in the dataset are in pixels, but the z value is in millimeters.
If we feed this directly to our models, distance-based models will give bad results. So, we applied
Standard Scaler to make all features on the same scale. Additionally, text-based headers were filtered
out, and a frame-by-frame mapping was applied to isolate 9 active expressions alongside a 10th
"neutral" class for resting faces.
To classify the face frames into different grammatical categories, I
applied three supervised machine learning models:
1. Random Forest Classifier
2. Support Vector Machine (SVM)
3. Multilayer Perceptron (MLP) Neural Network
I applied unsupervised clustering models to group similar face
coordinates together:
4. Principal Component Analysis (PCA) for dimensionality reduction
5. K-Means Clustering
For comparing the classification models, we checked their Accuracy, Precision,
Recall, and F1-Score. For comparing the clustering models, I used the Silhouette Score.

## Project Structure

* **`MLModel.ipynb`**: The primary Jupyter Notebook containing data exploration, preprocessing, various classification model training, clustering using models.
* **`main.py`**: The main Python script, for executing the core application.
* **`datasets/`**: Directory dedicated to storing the datasets used for training, validation, and testing the model.
* **`pyproject.toml` & `uv.lock`**: Project metadata and dependency management files, configured for use with Astral's `uv` package manager.
* **`.python-version`**: Defines the specific Python version required to run this project seamlessly.

## Getting Started

### Prerequisites

This project utilizes `uv` for lightning-fast dependency management. Ensure you have Python installed, along with `uv`. 

To install `uv` (if you haven't already):
```bash
pip install uv
```

### Installation
Clone the repository:

```Bash
git clone [https://github.com/Kamp7/ML-facial-expressions-model.git](https://github.com/Kamp7/ML-facial-expressions-model.git)
cd ML-facial-expressions-model
```
Install dependencies:
Since this project uses uv, you can quickly sync your virtual environment and install all required packages defined in the pyproject.toml and uv.lock files:

Bash
```
uv venv
uv sync
```
### Usage
Model Training & Exploration
To explore the dataset, tweak hyperparameters, or retrain the model, open the Jupyter Notebook:

Bash
```
jupyter notebook MLModel.ipynb
```
Running the Script
To execute the main application or test the model's inference capabilities:

Bash
```
python main.py
```

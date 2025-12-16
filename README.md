# US Citizen Income Classifier (Adult Income)
## Project overview
This project tackles a binary classification task: predicting whether a person’s annual income is greater than 50,000 USD (>50K) or less than or equal to 50,000 USD (<=50K) using demographic and social attributes from the Adult (Census Income) dataset.​
The solution uses an MLP (Multi-Layer Perceptron) trained in TensorFlow, with preprocessing done in Pandas (including One-Hot Encoding).​

## Dataset
The model is trained on the Adult Dataset from the UCI Machine Learning Repository (also referred to as Census Income).​
In the notebook, data is loaded directly from the adult.data file hosted by UCI.​

## Columns
The dataset includes attributes such as age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, and the target label income.​​

## Preprocessing
The notebook applies the following preprocessing steps:​

Load the CSV and remove missing rows via dropna.​

Convert the label to a binary target: >50K mapped to 1, otherwise 0.​

Apply One-Hot Encoding with pd.get_dummies(..., drop_first=True).​

Standardize features using mean/std with a small constant epsilon = 1e-7 for numerical stability.​

Split the data into train and test sets using train_test_split(test_size=0.2, random_state=42).​

## Model and training
The MLP parameters are initialized with Glorot/Xavier initialization (tf.keras.initializers.GlorotNormal).​
Forward propagation uses ReLU activations in hidden layers and a Sigmoid activation in the output layer for binary classification.​

## Optimization
Optimizer: Adam (tf.keras.optimizers.Adam).​

Training loop uses tf.GradientTape, and mini-batches are built with tf.data.Dataset.batch(...).prefetch(...).​

Example configuration from the notebook
One run in the notebook uses the layer layout 96, 20, 10, 50, 10, 1 with learning_rate=0.01, num_epochs=100, and minibatch_size=512.​

## Prediction and metrics
Predictions are produced by thresholding output probabilities at 0.5 (after Sigmoid).​
Accuracy is computed as the mean of correct predictions on the test set.​

## Result
The notebook prints a test accuracy of approximately 0.845 (84.5%).​

## Requirements
This project uses:​

Python 3

TensorFlow

NumPy

Pandas

scikit-learn (only for train_test_split)

## How to run
Open adult_income_classifier_model.ipynb.​

Run the cells in order: data loading, preprocessing, training, and evaluation.​

The notebook prints training cost during learning and the final test accuracy.​

## Key implementation components
The notebook includes (among others):​

Parameter initialization (initialize_parameters).​

Forward steps for ReLU and Sigmoid (forward_step_relu, forward_step_sig).​

Full forward propagation (forward_propagation).​

Binary cross-entropy loss computed with from_logits=True.​

Training function (model) built on tf.GradientTape + Adam.​

Prediction (predict) and accuracy (calculate_accuracy).​

## License
The project can be released under the MIT License (as stated in the project description).

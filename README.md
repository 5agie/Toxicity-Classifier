# Toxicity Classifier

The Toxicity Classifier is a machine learning model developed to predict the toxicity level of text comments. It utilizes Natural Language Processing (NLP) techniques and pre-trained machine learning models to classify comments into different toxicity categories. This project aims to contribute to applications such as moderating online discussion platforms and minimizing hate speech or abusive content on social media.

## Project Overview

The main components of the Toxicity Classifier project include:

- Data Loading: Loading the training and testing data from CSV files.
- Data Preprocessing: Splitting the data into features and target labels.
- Feature Extraction: Using TF-IDF vectorization to convert text comments into numerical features.
- Model Training: Training logistic regression models for each toxicity category.
- Prediction Functions: Functions to predict toxicity categories and print information about comments.
- Dataframe Manipulation: Adding predicted categories to a dataframe and saving it to a new CSV file.

## Usage

To use the Toxicity Classifier, follow these steps:

1. Install the required dependencies mentioned in the `requirements.txt` file.
2. Prepare your training data in CSV format, ensuring it has a column named 'comment_text' for text comments.
3. Run the script `train_classifier.py` to train the toxicity classification models using the training data.
4. Once the models are trained, you can utilize the provided functions for making predictions and analyzing comments:
   - Use the `predict_toxicity_categories(sentence)` function to predict toxicity categories for a given sentence.
   - Use the `print_information(sentence)` function to print predicted toxicity categories and additional features for a given sentence.
   - Use the `add_predicted_categories_to_dataframe(dataframe)` function to add predicted categories to a dataframe containing comment text.
5. Customize the code according to your specific use case or integrate it into your application.

## Example

Here's an example usage of the Toxicity Classifier:

```python
from predict_toxicity_categories import predict_toxicity_categories

# Call the function to make toxicity predictions on a sample text
example_comment = 'This is a sample comment to check toxicity level.'
toxicity_scores = predict_toxicity_categories(example_comment)
print(toxicity_scores)
```
This will output the predicted toxicity categories as a list of Boolean values.

Please make sure to adjust and customize the content of the README.md file to accurately represent your project. Include any relevant information, instructions, and guidelines for users to understand and use the Toxicity Classifier effectively.

Let me know if there's anything else I can help you with!

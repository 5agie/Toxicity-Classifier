"""
Created on Mon Mar 20 15:04:01 2023

@author: Sagie

Project name: Toxicity Classifier

The main idea of the project is to develop a machine learning model that can predict the toxicity level of a given
text comment. The model is designed to predict several types of toxicity, such as toxic, severe toxic, obscene, threat,
insult, and identity hate. The project leverages the power of Natural Language Processing (NLP) techniques to analyze
the text data and pre-trained machine learning models to classify the text into different toxicity categories.
The project can be used for various applications, such as moderating online discussion platforms and minimizing hate
speech or abusive content on social media.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the train and test CSV files
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Split the train data into features and target labels
train_features = train_data['comment_text']
train_labels = train_data.iloc[:, 2:]

# Create a TF-IDF vectorizer to convert text into numerical features
vectorizer = TfidfVectorizer(max_features=10000)
train_features_vectorized = vectorizer.fit_transform(train_features)
test_features_vectorized = vectorizer.transform(test_data['comment_text'])

# Train a logistic regression model for each toxicity category on the vectorized train data
toxic_model = LogisticRegression(max_iter=10000)
toxic_model.fit(train_features_vectorized, train_labels['toxic'])

severe_toxic_model = LogisticRegression(max_iter=10000)
severe_toxic_model.fit(train_features_vectorized, train_labels['severe_toxic'])

obscene_model = LogisticRegression(max_iter=10000)
obscene_model.fit(train_features_vectorized, train_labels['obscene'])

threat_model = LogisticRegression(max_iter=10000)
threat_model.fit(train_features_vectorized, train_labels['threat'])

insult_model = LogisticRegression(max_iter=10000)
insult_model.fit(train_features_vectorized, train_labels['insult'])

identity_hate_model = LogisticRegression(max_iter=10000)
identity_hate_model.fit(train_features_vectorized, train_labels['identity_hate'])


def predict_toxicity_categories(sentence):
    """
    Module: predict_toxicity_categories

    This module defines a function that takes a sentence as input (a string), vectorizes it using a pre-trained
    vectorizer, and predicts the toxicity categories of the sentence based on a pre-trained model for different types of
    toxicity, such as toxic, severe toxic, obscene, threat, insult, identity hate.

    The function returns a list of predicted categories consisting of six Boolean values (True if the sentence is
    predicted to belong to the corresponding toxicity category, False otherwise), one for each category.

    Functions
    ---------
    predict_toxicity_categories(sentence):
        Predicts the toxicity category for the input sentence based on a pre-trained model and returns a list of Boolean
        values corresponding to different categories.

    Parameters
    ----------
    sentence : A string representing a sentence or comment.

    Usage
    -----
    from predict_toxicity_categories import predict_toxicity_categories
    # Call the function to make toxicity predictions on a sample text
    example_comment = 'This is a sample comment to check toxicity level.'
    toxicity_scores = predict_toxicity_categories(example_comment)
    """
    # Vectorized the sentence
    sentence_vectorized = vectorizer.transform([sentence])

    # Predict the toxicity categories
    toxic = bool(toxic_model.predict(sentence_vectorized)[0])
    severe_toxic = bool(severe_toxic_model.predict(sentence_vectorized)[0])
    obscene = bool(obscene_model.predict(sentence_vectorized)[0])
    threat = bool(threat_model.predict(sentence_vectorized)[0])
    insult = bool(insult_model.predict(sentence_vectorized)[0])
    identity_hate = bool(identity_hate_model.predict(sentence_vectorized)[0])

    # Return the predicted toxicity categories as a list
    return [toxic, severe_toxic, obscene, threat, insult, identity_hate]


def print_information(sentence):
    """
    Module: print_information

    This module defines a function `print_information` that takes a sentence as input and prints the predicted toxicity
    categories of the input sentence based on pre-trained models for different types of toxicity (such as toxic, severe
    toxic, obscene, threat, insult, identity hate) along with additional features extracted from the input sentence like
    comment length, number of sentences and number of words.

    Functions
    ---------
    print_information(sentence):
        Prints the predicted toxicity categories and other extracted features of the input sentence.

    Parameters
    ----------
    sentence : A string representing a sentence or comment.

    Usage
    -----
    from print_information import print_information

    # Call the function to print toxicity predictions and additional features on a sample text
    example_comment = 'This is a sample comment to check toxicity level.'
    print_information(example_comment)
    """
    # Extract additional features
    comment_length = len(sentence)
    num_sentences = len(sentence.split('.'))
    num_words = len(sentence.split())

    # Print the predicted toxicity categories and additional features
    toxic, severe_toxic, obscene, threat, insult, identity_hate = predict_toxicity_categories(sentence)
    print("Toxic: ", toxic)
    print("Severe Toxic: ", severe_toxic)
    print("Obscene: ", obscene)
    print("Threat: ", threat)
    print("Insult: ", insult)
    print("Identity Hate: ", identity_hate)
    print(f"comment length: {comment_length}, number of sentences: {num_sentences}, number of words: {num_words}")


def add_predicted_categories_to_dataframe(dataframe):
    """
    Module : add_predicted_categories_to_dataframe

    This module defines a function that takes a Pandas dataframe as input and adds six new columns to the input
    dataframe. These columns represent the predicted category for each row of comment text in the input dataframe for
    the corresponding type of toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate) based on a
    pre-trained toxicity classification model. The function calls a separate `predict_toxicity_categories` function,
    defined in another module, to generate these predicted categories.

    The input dataframe should have a 'comment_text' column containing the text of the comments.

    The updated dataframe with added columns is then saved to a new CSV file called 'updated_data.csv', stored in the
    current working directory.

    Functions
    ---------
    add_predicted_categories_to_dataframe(dataframe) :
        Adds new columns to the input dataframe with predicted toxicity categories and saves the updated dataframe
        to a new CSV file.

    Parameters
    ----------
    dataframe : Pandas DataFrame with a 'comment_text' column containing the text of the comments.

    Usage
    -----
    import pandas as pd
    from predict_toxicity import add_predicted_categories_to_dataframe

    # Load the test_data from a CSV file into a Pandas DataFrame
    test_data = pd.read_csv('test_data.csv')

    # Call the function to add predicted categories to the test_data DataFrame and save the updated DataFrame to a new
    CSV file add_predicted_categories_to_dataframe(test_data)
    """
    # Add a new column to the dataframe for each predicted category
    dataframe['toxic'] = dataframe['comment_text'].apply(lambda x: predict_toxicity_categories(x)[0])
    dataframe['severe_toxic'] = dataframe['comment_text'].apply(lambda x: predict_toxicity_categories(x)[1])
    dataframe['obscene'] = dataframe['comment_text'].apply(lambda x: predict_toxicity_categories(x)[2])
    dataframe['threat'] = dataframe['comment_text'].apply(lambda x: predict_toxicity_categories(x)[3])
    dataframe['insult'] = dataframe['comment_text'].apply(lambda x: predict_toxicity_categories(x)[4])
    dataframe['identity_hate'] = dataframe['comment_text'].apply(lambda x: predict_toxicity_categories(x)[5])

    # Save the updated dataframe to a new CSV file
    dataframe.to_csv('updated_data.csv', index=False)


# Example usage
sentence1 = "The meaning of GAY is of, relating to, or characterized by sexual or romantic attraction to people of one's same sex"
print_information(sentence1)

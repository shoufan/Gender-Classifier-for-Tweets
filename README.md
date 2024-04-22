# Naïve Bayes Gender Classifier for Twitter Tweets

## Overview
This project implements a Naïve Bayes classifier to predict the gender of Twitter users based on their tweets. It leverages natural language processing techniques to analyze the text content of tweets and classify them as either male or female. The classifier is trained on a labeled dataset containing tweets from male and female users.

## Dataset
The dataset used in this project consists of Twitter tweets collected from various users. It is split into three subsets:
- **Training Set**: Used to train the Naïve Bayes classifier.
- **Test Set**: Used to evaluate the performance of the trained classifier.
- **Validation Set**: Used for additional validation and tuning of hyperparameters.
- **https://www.kaggle.com/datasets/aharless/tweet-files-for-gender-guessing**

## Implementation Details
- **Data Cleaning**: The tweets are preprocessed to remove URLs, special characters, and emojis. The text is then converted to lowercase for consistency.
- **Tokenization**: The preprocessed text is tokenized by splitting it into individual words. This step facilitates further analysis and feature extraction.
- **Naïve Bayes Classifier**: The classifier is implemented using the Naïve Bayes algorithm, which assumes independence between features. It calculates class probabilities and word counts for each label (male and female) based on the training data.
- **Evaluation**: The classifier's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score on the test set.

## Usage
To use the classifier:
1. Ensure Python and the required libraries (e.g., pandas, numpy) are installed.
2. Clone the repository and navigate to the project directory.
3. Run the main script (`gender_classifier.py`) with the desired configuration.
4. Follow the prompts to input sentences for classification.

## Results
The classifier achieves an accuracy of around 57% to 58% on the test set, with moderate sensitivity and specificity values. Performance may vary depending on the size of the training set and the specific characteristics of the dataset.

## Future Improvements
- Experiment with different classification algorithms to compare performance.
- Explore advanced feature engineering techniques such as word embeddings.
- Fine-tune hyperparameters to optimize classifier performance.
- Handle unbalanced classes more effectively to improve precision and recall.

## Contributors
- Mazin Khider

## License
This project is licensed under the MIT License.

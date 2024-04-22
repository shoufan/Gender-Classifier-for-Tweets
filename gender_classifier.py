import pandas as pd
import re
import sys
from collections import defaultdict
import numpy as np

# Load the dataset
train_df = pd.read_csv("twitgen_train_201906011956.csv")
test_df = pd.read_csv("twitgen_test_201906011956.csv")
valid_df = pd.read_csv("twitgen_valid_201906011956.csv")

# Data Cleaning
def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove special characters and emojis
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Convert text to lowercase
    text = text.lower()
    return text

# Apply cleaning function to tweet text
train_df['clean_text'] = train_df['text'].apply(clean_text)
test_df['clean_text'] = test_df['text'].apply(clean_text)
valid_df['clean_text'] = valid_df['text'].apply(clean_text)

# Tokenize text (split into individual words)
train_df['tokens'] = train_df['clean_text'].str.split()
test_df['tokens'] = test_df['clean_text'].str.split()
valid_df['tokens'] = valid_df['clean_text'].str.split()

# Training Classifier
class NaiveBayesClassifier:
    def __init__(self):
        self.class_prior = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.vocabulary = set()

    def fit(self, X_train, y_train):
        # Calculate class prior probabilities
        for label in y_train:
            self.class_prior[label] += 1
        
        total_documents = len(y_train)
        for label, count in self.class_prior.items():
            self.class_prior[label] = count / total_documents
        
        # Calculate word counts for each class
        for doc, label in zip(X_train, y_train):
            for word in doc:
                self.word_counts[label][word] += 1
                self.vocabulary.add(word)

    def predict(self, X_test):
        predictions = []
        for doc in X_test:
            class_scores = {}
            for label in self.class_prior:
                log_score = np.log(self.class_prior[label])
                for word in doc:
                    log_score += np.log((self.word_counts[label][word] + 1) / 
                                        (sum(self.word_counts[label].values()) + len(self.vocabulary)))
                class_scores[label] = log_score
            predicted_label = max(class_scores, key=class_scores.get)
            predictions.append(predicted_label)
        return predictions

def parse_train_size_argument():
    default_train_size = 80
    if len(sys.argv) != 2:
        print("Invalid number of arguments. Using default TRAIN_SIZE=80.")
        return default_train_size
    try:
        train_size = int(sys.argv[1])
        if train_size < 20 or train_size > 80:
            print("TRAIN_SIZE argument out of range. Using default TRAIN_SIZE=80.")
            return default_train_size
        return train_size
    except ValueError:
        print("Invalid TRAIN_SIZE argument. Using default TRAIN_SIZE=80.")
        return default_train_size

# Parse command line argument for TRAIN_SIZE
train_size = parse_train_size_argument()

# Display the chosen TRAIN_SIZE
print(f"Training set size: {train_size}%\n")

# Split dataset into features (X) and labels (y)
X_train = train_df['tokens'].tolist()
y_train = train_df['male'].tolist()

X_test = test_df['tokens'].tolist()
y_test = test_df['male'].tolist()

# Initialize and train the Naive Bayes classifier
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X_train[:int(len(X_train)*(train_size/100))], y_train[:int(len(X_train)*(train_size/100))])

# Testing Classifier
# Predict labels for test set
predictions = nb_classifier.predict(X_test)

# Calculate and display classification metrics
print("Training classifier...")
print("Testing classifier...")
true_positives = sum((pred == 1) and (true == 1) for pred, true in zip(predictions, y_test))
true_negatives = sum((pred == 0) and (true == 0) for pred, true in zip(predictions, y_test))
false_positives = sum((pred == 1) and (true == 0) for pred, true in zip(predictions, y_test))
false_negatives = sum((pred == 0) and (true == 1) for pred, true in zip(predictions, y_test))
sensitivity = true_positives / (true_positives + false_negatives)
specificity = true_negatives / (true_negatives + false_positives)
precision = true_positives / (true_positives + false_positives)
negative_predictive_value = true_negatives / (true_negatives + false_negatives)
accuracy = (true_positives + true_negatives) / len(predictions)
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

print("\nTest results / metrics:")
print(f"Number of true positives: {true_positives}")
print(f"Number of true negatives: {true_negatives}")
print(f"Number of false positives: {false_positives}")
print(f"Number of false negatives: {false_negatives}")
print(f"Sensitivity (recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Negative predictive value: {negative_predictive_value:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F-score: {f1_score:.4f}")

# Classifying User-inputed sentences
def classify_sentence(classifier, vocabulary, sentence):
    # Tokenize and clean the input sentence
    cleaned_sentence = clean_text(sentence)
    tokens = cleaned_sentence.split()
    
    # Ensure the sentence contains tokens
    if not tokens:
        return "Invalid input. Please enter a valid sentence."

    # Ensure all tokens are in the vocabulary
    tokens = [token for token in tokens if token in vocabulary]
    if not tokens:
        return "None of the words in the sentence are in the vocabulary."
    
    # Predict label and probabilities
    prediction = classifier.predict([tokens])[0]
    class_probabilities = {}
    for label in classifier.class_prior:
        log_score = np.log(classifier.class_prior[label])
        for word in tokens:
            log_score += np.log((classifier.word_counts[label][word] + 1) / 
                                (sum(classifier.word_counts[label].values()) + len(vocabulary)))
        class_probabilities[label] = np.exp(log_score)
    
    # Display classification result
    result = f"\nSentence '{sentence}' was classified as {prediction}.\n"
    result += f"P({prediction} | S) = {class_probabilities[prediction]}\n"
    opposite_label = not prediction  # Assuming the opposite label is binary (0 or 1)
    result += f"P({opposite_label} | S) = {class_probabilities[opposite_label]}\n"
    return result

# Prompt user input and classify sentences
while True:
    user_input = input("\nEnter your sentence (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    else:
        # Classify user-entered sentence
        classification_result = classify_sentence(nb_classifier, nb_classifier.vocabulary, user_input)
        print(classification_result)
        # Ask user if they want to enter another sentence
        continue_classification = input("\nDo you want to enter another sentence [Y/N]? ").strip().lower()
        if continue_classification != 'y':
            break
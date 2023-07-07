

# FakeNewsDetection
Fake news Detection 
This code implements a text classification model using the PassiveAggressiveClassifier algorithm to classify news articles as either fake or real. 

The code starts by importing the necessary libraries: numpy, pandas, train_test_split from sklearn.model_selection, TfidfVectorizer from sklearn.feature_extraction.text, PassiveAggressiveClassifier from sklearn.linear_model, accuracy_score and confusion_matrix from sklearn.metrics.

The dataset is read from a CSV file using Pandas, and its shape and head are printed. The labels column is extracted and stored in the 'labels' variable.

Next, the dataset is split into training and testing sets using the train_test_split function, with 80% for training and 20% for testing. The TfidfVectorizer is initialized, which converts text into numerical feature vectors using the TF-IDF (Term Frequency-Inverse Document Frequency) approach. It removes stopwords and ignores terms that appear in more than 70% of the documents.

The training and testing sets are transformed using the TfidfVectorizer. Then, a PassiveAggressiveClassifier is initialized and trained on the training set.

The model makes predictions on the test set, and the accuracy of the predictions is calculated and printed. Finally, a confusion matrix is built to evaluate the model's performance, showing the number of true positives, true negatives, false positives, and false negatives.

Overall, this code demonstrates how to train a text classification model using the PassiveAggressiveClassifier algorithm and evaluate its accuracy using the TF-IDF vectorization technique.

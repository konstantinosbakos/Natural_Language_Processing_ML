
# Natural Language Processing with Machine Learning [![Project Status: Inactive – The project has reached a stable, usable state but is no longer being actively developed; support/maintenance will be provided as time allows.](https://www.repostatus.org/badges/latest/inactive.svg)](https://www.repostatus.org/#inactive) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Summary

This is a project that puts into use different machine learning methods with the goal to compare their efficiency in classifying reviews as positive or negative. 

The reviews are contained in the `.csv` file found in the project and are either positive (6-10) or negative (0-4), since the classification methods used, are binary. There are no reviews with a score of "5" in the dataset. The end product should be machine learning models that are capable of understanding if a review is positive or negative with high accuracy.

## Data Preprocessing

The data preprocessing is essentially the same - up to a point - for all three projects.

### Loading and converting to boolean

The dataset is loaded into a pandas dataframe. Then the ratings of the reviews are changed to 0 or 1 (0-4 reviews are negative, so they become 0, 6-10 reviews are positive, so they become 1).

### Proof that the dataset is balanced

It is very important to use a balanced dataset to train a machine learning algorithm for various reasons, including but not limited to:
- **Better Generalisation**: If all classes are represented equally, then the model is able to be trained equally on all of them. On the contrary, if a class is underepresented, there is a high chance that there will be poor performance in its recognition.
- **Bias of Majority Class**: If a dataset is unbalanced, there is a bias introduced towards detecting the majority class, since the model will be better versed in recognising it.
- **More Robust Model**: The model produced via training on a balanced dataset is better at detecting both the classes and dealing with the edge cases in a more responsible and desirable way, compared to a model that is trained on an unbalanced dataset.

### Tokenisation

The last common process done in all projects is the vectorisation of the reviews. Essentially the reviews turn from `"strings of words"` to `"[strings", "of", "words"]`(lists).

### Further Preprocessing

#### Logistic Regression

For the logistic regression model, there are used two different vectorizer types, [Count](https://towardsdatascience.com/basics-of-countvectorizer-e26677900f9c) and [TF-IDF](https://towardsdatascience.com/how-tf-idf-works-3dbf35e568f0), while the text is processed by creating [n_grams](https://en.wikipedia.org/wiki/N-gram) of size (1,2) and (2,2). The latter parentheses express the lenght of the n_grams, where in the first case both unigrams and bigrams are used, while in the second case, only bigrams are used.

#### Linear NN and BRNN

For the Linear Neural Network and the Recurrent Neural Network, the words in the reviews are converted to 300-dimentional vectors that are unique for every word, using [Project Glove](https://nlp.stanford.edu/projects/glove/).

## Logistic Regression

A Logistic Regression model that is tuned using grid_search and the `liblinear` solver that supports L2-loss linear SVM and L1-loss linear SVM.

## Linear Neural Network

A Linear NN that uses 6 linear layers and the sigmoid activation function (binary classification). The Network is trained three times using three different optimisers, NAdam, AdamW and SGD.

## Bidirectional Recurrent Neural Network

A BRNN is essentially an RNN that processes data both forwards and backwards. The two architectures utilised are [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory#:~:text=The%20Long%20Short%2DTerm%20Memory,gate%20and%20a%20forget%20gate.) and [GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit).  These architectures enchance the learning capabilities and regulate the flow of information in the network, aiming to tackle the problem of vanishing gradients that is frequently present in RNNs.

The training and tuning is done using [Optuna](https://optuna.org/). 

## Evaluation Methods

For the Logistic Regression model only the first three evaluation methods are used, while for the other two, all of them are used.

### Learning Curve

The [learning curve](https://en.wikipedia.org/wiki/Learning_curve_(machine_learning)) is an evaluation method that can be used to tune the model so that it achieves its best performance. Some of the benefits it provides are that overfitting and underfitting can be detected with it and it can also indicate to the user if they should stop the training in a smaller number of epochs.

It is important to note that, although most of the times the train loss curve should be lower than the test loss curve, depending on the dataset, the nature of the model used and the optimisation techniques used, it is plausible that the test curve is lower than the train curve without implying underlying problems with the model. In the specific case of these models, train loss is calculated between batches and not at the end of the epochs, which has as a result - on average - a higher loss. Optimisation methods like dropout and cliping can also increase the train loss, which are present in the RNN.

### Confusion Matrix

A [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) summarizes the performance of the model in the classification problem. It provides the number of predicted labels that correspond with the actual labels, as well as the predicted labels that do not. If the numbers of falsely predicted labels is high, this indicates that something is wrong with the model (assuming the preprocessing of the data is done correctly).

### F1-score

The [F1-score](https://en.wikipedia.org/wiki/F-score) is a measure of predictive performance and essentially, it uses the values of a confusion matrix to be calculated.

### ROC Curve

The [ROC (Receiver Operating Characteristic) curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic), is a curve that compares the true positive rate (TPR) against the false positive rate (FPR). TPR could also be reffered to as "the probability of detection", while FPR could be reffered to as "the probability of false alarm".

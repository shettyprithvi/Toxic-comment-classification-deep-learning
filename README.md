# Toxic comment classifier app using deep learning and machine learning
## Classification of toxic comments

**Disclaimer : Please keep in mind that the comments might be really racist and toxic. But it's strictly used for the purpose of academic studies and comparison.**


The architecture of the flask application built : 

![alt text](https://github.com/shettyprithvi/Toxic_comment_classification/blob/master/application.PNG)


## Abstract:

NLP at its core deals with classification of text, which entails training a model to
understand the pattern of a corpus of text and compare it or distinguish it with a set of known
categories. The pattern of text is the vector representation of the document/corpus, which is
used by the designated classification techniques. There are many ways to classify textual data
into predefined categories but the accuracy of classification can vary based on the vectorization
technique, classification technique, length of the corpus and the source of text etc. Our goal
through this project was to compare the performance of the models in terms of F1-Score,
which is the harmonic mean of precision and recall. We used comments data from wikipedia
talk page, pre-processed it and converted it to vector representation using TF-IDF and Doc2vec,
which was then passed on to our chosen classification techniques :

1. Naive Bayes
2. Logistic Regression
3. Bidirectional L.S.T.M.


Topic: Toxic comments classifier using deep learning
Overview:
Built four different classifiers for identifying different levels of toxicity among the comments from our dataset (Source : Wikipedia review comments).

Among the four classifiers which I built (Bi-Directional LSTM model, Doc2Vec Logistic Regression, Tf-IDF Logistic Regression and TF-IDF Naive Bayes), the Bi-Directional LSTM model performed the best with the highest average F-1 score across different test datasets.

The reason I have built a separate classifier for each class (toxic, severe_toxic, obscene, threat, insult, identity_hate) is that we can compare the results much more efficiently and with much more clarity for each class for the four different models that we have used.

Code guideline: To run the code. You will need to follow the code and run the program as it is. To load the loaded model, you can use the respective model names for each class ('toxic.hd5','severe_toxic.hd5','obscene.hd5','threat.hd5','insult.hd5','identity_hate.hd5'). To load the model, use model= load_model(model name)

NOTE : It took upto 50 minutes for each deep learning model to train on our PC (16 GB RAM , i-7 Core)

Code Index:
1. Data reading and cleaning

    1.1 Importing essential libraries

    1.2 Null cleaning function

    1.3 Reading data and aplying null cleaning function

    1.4 Cleaning all non-alphanumeric characters

    1.5 Converting to lowercase and removing punctuations

    1.6 Cleaned output


Experiments

2. Deep learning using Bi-Directional LSTM( Long Short Term Memory)

    2.1 Importing all the keras libraries

    2.2 Splitting into training and test data

    2.3 Tokenizer function of keras

    2.4 Vectorization

    2.5 Converting all the output labels into appropriate categorical form for model training

    2.6 Building the neural network

    2.7 Training the model with 30 epochs of the training data

    2.8 Saving and loading the model

        2.8.1 'Toxic' category classification

            2.8.1.1 Test set results

            2.8.1.2 Checking the classification metrics of the model

        2.8.2 'Severely Toxic' category classification

            2.8.2.1 Test set results

            2.8.2.2 Checking the classification metrics of the model

        2.8.3 'Obscene' category classification

            2.8.3.1 Test set results

            2.8.3.2 Checking the classification metrics of the model

        2.8.4 'Threat' category classification

            2.8.4.1 Test set results

            2.8.4.2 Checking the classification metrics of the model

        2.8.5 'Insult' category classification

            2.8.5.1 Test set results

            2.8.5.2 Checking the classification metrics of the model

        2.8.6 'Hate' category classification

            2.8.6.1 Test set results

            2.8.6.2 Checking the classification metrics of the model



3. Machine learning approach

    3.1 Importing Machine learning classifiers from scikit learn

    3.2 Lemmatization class using nltk

    3.3 Importing the Vectorization libraries

    3.4 Fitting the training data on TfIdfVectorizer

    3.5 Vectorizing the test and train data

    3.6 Results on different classifiers using Naive bayes and Logistic Regression

        3.6.1 Logistic regression

            3.6.1.1 'Toxic' category classification and results

            3.6.1.2 'Severe Toxic' category classification and results

            3.6.1.3 'Obscene' category classification and results

            3.6.1.4 'Threat' category classification and results

            3.6.1.5 'Insult' category classification and results

            3.6.1.6 'Hate' category classification and results

        3.6.2 Naive Bayes

            3.6.2.1 'Toxic' category classification and results

            3.6.2.2 'Severe Toxic' category classification and results

            3.6.2.3 'Obscene' category classification and results

            3.6.2.4 'Threat' category classification and results

            3.6.2.5 'Insult' category classification and results

            3.6.2.6 'Hate' category classification and results


4. Custom dataset experiment (Twitter data)

    4.1 Bidirectional LSTM deep learning results

        4.1.1 'Toxic' category classifier

        4.1.2 'Severe toxic' category classifier

        4.1.3 'Obscene' category classifier

        4.1.4 'Threat' category classifier

        4.1.5 'Insult' category classifier

        4.1.6 'Hate' category classifier

    4.2 Logistic regression results

        4.2.1 'Toxic' category classifier

        4.2.2 'Severe toxic' category classifier

        4.2.3 'Obscene' category classifier

        4.2.4 'Threat' category classifier

        4.2.5 'Insult' category classifier

        4.2.6 'Hate' category classifier

    4.3 Naive Bayes results

        4.3.1 'Toxic' category classifier

        4.3.2 'Severe toxic' category classifier

        4.3.3 'Obscene' category classifier

        4.3.4 'Threat' category classifier

        4.3.5 'Insult' category classifier

        4.3.6 'Hate' category classifier


5. Doc2Vec for classification

    5.1 Importing essential libraries

    5.2 Converting into vectors using Doc2Vec

    5.3 Logistic regression test set results and metrics


6. Scattertext for identifying themes in classified comments

    6.1 Importing essential libraries

    6.2 Creating a corpus

    6.3 Rending the scattertext on a html page



Each classified text (classified as toxic, severely toxic,obscene, threat, insult, hate) was
compared with its label to determine the precision and recall based on which the cumulative
F1-Score for each classifier was calculated and compared.
We also compared the performance of the semantic vector space model ( TFIDF Vs Doc2Vec) on
the classifier performance.

At the end of the experiments, we were able to find quantitative results supporting that Deep
Neural Network Model performed better in terms of F1-Score than machine learning models (
Naive Bayes and Logistic Regression) for both known and unknown datasets and that TFIDF
performs better than Word2Vec, at least when the training corpus size is ~140,000.

Test results :

![alt text](https://github.com/shettyprithvi/Toxic_comment_classification/blob/master/results.PNG)
![alt text](https://github.com/shettyprithvi/Toxic_comment_classification/blob/master/results2.PNG)


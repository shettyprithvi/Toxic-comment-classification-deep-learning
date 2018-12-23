# Toxic_comment_classification
## Classification of toxic comments

**Disclaimer : Please keep in mind that the comments mind be really racist and toxic. But it's strictly used for the purpose of academic studies and comparison.**

## Abstract:

NLP at its core deals with classification of text, which entails training a model to
understand the pattern of a corpus of text and compare it or distinguish it with a set of known
categories. The pattern of text is the vector representation of the document/corpus, which is
used by the designated classification technique. There are many ways to classify textual data
into predefined categories but the accuracy of classification can vary based on the vectorization
technique, classification technique, length of the corpus and the source of text etc. Our goal
through this project was to compare the performance of the models in terms of F1-Score,
which is the harmonic mean of precision and recall. We used comments data from wikipedia
talk page, pre-processed it and converted it to vector representation using TF-IDF and Doc2vec,
which was then passed on to our chosen classification techniques :

1. Naive Bayes
2. Logistic Regression
3. Bidirectional L.S.T.M.

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

The architecture of the flask application built : 

![alt text](https://github.com/shettyprithvi/Toxic_comment_classification/blob/master/application.PNG)

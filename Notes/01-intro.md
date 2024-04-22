# (ML)^2
_**Me Learning Machine Learning**_

I have been learning Data Science, Machine Learning, Deep Learning from many sources, especially from,
- "Data Science from Scratch (Ed.2) by Joel Grus"
- "Hands-on Machine Learning with Scikit-Learn, Keras, and Tensor Flow (Ed.3) by Aurélien Geron"
- Machine Learning Specialization by Andrew Ng (I am a big fan), DeepLearning.ai
- Kaggle Competitions

_**Below notes and excerpts are either from the above mentioned sources or influenced by them**_

_**I will mention if something is based on my own thoughts (is it even possible?)**_

# Introduction to Machine Learning

## Machine Learning
ML is the field of study that gives computers the ability to learn without being explicitly programmed. - **Arthur Samuel, 1959**

A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E. - **Tom Mitchell, 1997**

Example, email spam filter. It learns from user marking an email as spam or not. Each such marking forms a training data (E). The learning from this data and predicting (T) an email as spam or not is called a model.

### Examples
- Detecting defects in products on a production line (uses Convolutional Neural Network (CNN))
- Classify news articles (uses Natural Language Processing, specifically text classification based on recurrent neural networks (RNN) and CNN. Transformers work better for this)
- Summarizing long documents (uses NLP)
- Chatbots (uses NLP, including Natural Language Understanding)
- Revenue forecast (uses linear or polynomial regression, regression support vector machine, regression random forest...)
- Detecting credit card fraud (anomaly detection using isolation forests, Gaussian mixture models or autoencoders)
- Customer segmentation (Clustering like k-means, DBSCAN...)

## Types of Machine Learning System
- Supervised learning (supervised, semi-supervised, self-supervised...)
- Learning incremental on the fly (online) or batch learning (offline?)
- Simple prediction by comparing new data to known data points or detecting patterns in the training data to build predictive model

## Supervised Learning
Training set is given to an algorithm with labels (or targets).
- Classification (email spam filter) - Algorithm is given training emails (examples) with their class (spam or not). Model generated based on this learns how to classify a new email as spam or not
- Regression is to predict a target numeric value, like revenue, fuel price, stock value, etc., Training data with features and labels (or targets) are provided to the algorithm to learn. Then the model knows how to predict the label/target for a new set of features
- Logistic Regression - Sometimes regression is used for classification, like Logistic Regression. It can output value that corresponds to probability of result belonging to a class.

## Unsupervised Learning
Training data is unlabeled. Algorithm tries to learn without reference label.
- Clustering, Hierarchical clustering (clustering a blog's visitors)
- Visualization algorithms (provide a lot of complex and unlabeled data, they output 2D or 3D representation of the data, that can be easily plotted)
- Dimensionality reduction (to simplify the data without losing much information. Correlated features will be merged into one feature, also called feature extraction)
- Anomaly detection (detect unusual (anomaly) data pattern - used in credit card transaction, detecting manufacturing defects, spot outlier...)
- Association rule learning (use large amount unlabeled data and discover relation between features - ex - people watching Jackie Chan movies also enjoy movies containing slapstick comedy)

## Semi-supervised learning
It is generally a combination of unsupervised learning and supervised learning. Dataset contains some labeled data, while the rest is all unlabeled.
- Google Photos recognizing same person in different photos, but does not who is that person. Once the user identifies (labels) the person, then Google Photos starts using that label to search for that particular person in many photos

## Self-supervised learning
Labeling a fully unlabeled data. After this step, fully labeled data can be used with Supervised learning algorithm.
- TBD

## Reinforcement learning
An agent observes the environment, then performs action that will either yield rewards or penalties. Based on the reward and penalty the agent learns the best strategy (course of action or rather selection of action), which is called as Policy. This policy defines what action the agent should choose in a given situation.
- DeepMind's AlphaGo is an example of Reinforcement learning

## Batch (offline?) and Online learning
### Batch learning
- System is incapable of learning incrementally. It has to be trained with all the data, including the new data, every time when there is new data.
- Batch system performance decays slowly over time. This happens as new data continues to evolve and the system has not learnt about it.
- This is called model rot or data drift.

### Online learning
- System is incrementally fed with new data for it to learn sequentially.
- Each learning step is fast and cheap, making the system to learn about new data on the fly as it is fed.
- Online learning algorithms generally works on huge datasets, which cannot fit in one machine's main memory (out-of-core). Hence, algorithm loads part of the data, trains on that data, then repeats the process until it has learnt all the data, incrementally.

### Out-of-core learning
- This is usually done offline (i.e. not on the live system). This can be seen as incremental learning.

## Instance-Based Learning Vs Model-Based Learning
This is a categorization based on how the predictions are made.

### Instance-based learning
- Trivial form of learning. For example, spam filter will mark all the new emails (inference) that are exactly same as the ones (training data) user has already marked as spam.
- Such a prediction algorithm will miss spam emails that are similar but not exactly same as training data.
- Not so bad, but not the best!
- Algorithm uses similarity measure, by comparing with the learned data

### Mode-based learning
- Builds model and uses it make predictions
- Linear model, quadratic model... trying to fit to training data as much as possible (not to overfit) and uses that fit to make new predictions

## Main Challenges of ML
- Insufficient quantity of training data
- Non-representative training data (missing variations)
- Missing data
- Not knowing what feature is missing, until it is too late

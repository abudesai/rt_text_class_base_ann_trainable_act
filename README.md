ANN Classifier With Trainable Activation in TensorFlow/Keras for Text Classification - Base problem category as per Ready Tensor specifications.

* tensorflow
* keras
* neural network
* pandas
* numpy
* activation
* python
* adam optimizer
* text classification
* scikit-optimize
* flask
* nginx
* uvicorn
* docker
* tf-idf 
* nltk

This Artificial Neural Network (ANN) is comprised of 3 layers, one of which includes a layer with trainable parameters to control the slope of the activation function, that uses the Adam optimizer to evaluate the performance of the model. 

The data preprocessing step includes tokenizing the input text, applying a tf-idf vectorizer to the tokenized text, and applying Singular Value Decomposition (SVD) to find the optimal factors coming from the original matrix. In regards to processing the labels, a label encoder is used to turn the string representation of a class into a numerical representation. 

The Hyperparameter Tuning (HPT) involves finding the optimal values for the L1 and L2 regularization, learning rate for Adam optimmizer, and the number of changepoints for the trainable activation layer.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as clickbait, drug, and movie reviews as well as spam texts and tweets from Twitter.

The main programming language is Python. The main algorithm is created using TensorFlow and Keras while Scikitlearn is used to calulate the model metrics and preprocess the data. NLTK, pandas, and numpy are used to help preprocess the data while Scikit-Optimize is used for HPT. Flask, Nginx, gunicorn are used to allow for web services. The web service provides two endpoints- /ping for health check and /infer for predictions in real time.

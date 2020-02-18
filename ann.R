install.packages('caTools')
library(caTools)

# Artificial Neural Network

# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]


# Encoding the categorical variables
dataset$Geography = as.numeric(factor(dataset$Geography, levels = c('France', 'Spain', 'Germany'), labels = c(1,2,3)))
dataset$Gender = as.numeric(factor(dataset$Gender, levels = c('Female', 'Male'), labels = c(1,2)))


# Splitting the dataset into train and test sets
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
trainingSet = subset(dataset, split == TRUE)
testSet = subset(dataset, split == FALSE)


# Feature Scaling
trainingSet[-11] = scale(trainingSet[-11])
testSet[-11] = scale(testSet[-11])


# Fitting ANN to the training set
# It is connecting to an h2o instance - for building efficient deep learning models
install.packages('h2o')
library(h2o)
h2o.init(nthreads=-1)
model = h2o.deeplearning(y = 'Exited',
                         training_frame = as.h2o(trainingSet),
                         activation = 'Rectifier',
                         hidden = c(6,6),
                         epochs = 100,
                         train_samples_per_iteration = -2)


# Predicting the Test set results
predicted = h2o.predict(model, newdata = as.h2o(testSet[-11]))
yPred = (predicted > 0.5)
yPred = as.vector(yPred)


# Making the confusion matrix
cm = table(testSet[,11], yPred)


# Shutting down the h2o instance
h2o.shutdown()








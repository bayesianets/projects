library(caret)
library(MASS)


#
# Activity Type recognition
#
# Input dataset contains observations of 561 parameter collected from a group of 30 individuals using data
# from a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, 3-axial
# linear acceleration and 3-axial angular velocity at a constant rate of 50Hz was captured. See data/README for
# more details on data collection and preprocessing.
#
# References:
#
# Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. 
# A Public Domain Dataset for Human Activity Recognition Using Smartphones.
# 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013.
# Bruges, Belgium 24-26 April 2013.
#
# Data set: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

trainX = read.table("~/src/Kate/activity/data/train/X_train.txt",
                    quote="\"",
                    stringsAsFactors=FALSE)

testX = read.table("~/src/Kate/activity/data/test/X_test.txt",
                   quote="\"",
                   stringsAsFactors=FALSE)

#
# Each person performed six activities:
#   - walking
#   - walking upstairs
#   - walking downstairs
#   - sitting
#   - standing
#   - laying
#
activities = c("walking", "walkingUp", "walkingDown", "sitting", 
               "standing", "laying")

# The experiments have been labeled manually after collection.
trainY = read.table("~/src/Kate/activity/data/train/y_train.txt",
                    col.names=c("Activity"),
                    quote="\"")

testY = read.table("~/src/Kate/activity/data/test/y_test.txt",
                    col.names=c("Activity"),
                    quote="\"")

trainY$Factor = factor(trainY$Activity, labels = activities)
testY$Factor = factor(testY$Activity, labels = activities)

trainData = cbind(trainX, trainY$Factor)
testData = cbind(testX, testY$Factor)

names(trainData) = c(names(trainX), "ActivityFactor")
names(testData) = c(names(testX), "ActivityFactor")

# Preprocessing the data, removing highly correlated variables
corM = cor(trainX)
highCorrelated = findCorrelation(corM, cutoff = 0.8)
reducedData = data.matrix(trainX[,  -c(highCorrelated)])
preProcessFit = preProcess(reducedData)
preProcessedData = predict(preProcessFit, newdata = reducedData)

reducedTest = data.matrix(testX[, -c(highCorrelated)])
preProcessedTest = predict(preProcessFit, newdata = reducedTest)

# Applying LDA for activity recognition
ldaFit = lda(x = preProcessedData, grouping = trainY$Factor)
predictedActivities = predict(ldaFit, newdata = preProcessedTest)
score = mean(predictedActivities$class == testY$Factor)

# "LDA performance: 0.944350186630472"
print(paste("LDA performance: ", score, sep=""))

obsCount = length(testY$Activity)
classCount = length(activities)

# Now, applying HMM on top of LDA
updateActivitiesTransitionsMatrix = function(activities, obsCount, classCount) {
  m = matrix(0, classCount, classCount)
  
  for (i in 1:classCount) {
    for (j in 1:classCount) {
      # count of transitions from class i into class j
      nTransitions = length(which(activities[1:obsCount - 1] == i & activities[2:obsCount] == j))
      m[i, j] = nTransitions
    }
  }
  
  transitionMatrix = m/colSums(m)
  return(transitionMatrix) 
}

initialState = as.vector(c(0, 0, 0, 0, 1, 0))
transitionMatrix = updateActivitiesTransitionsMatrix(testY$Activity, obsCount, classCount)

# Estimation maximization, estimates probability of every activity.
simpleEm = function(transitionMatrix, initialState, outputProb, nObs, mClass) {
  priorState = initialState%*%transitionMatrix
  
  posteriorProb = matrix(0, nObs, mClass)
  posteriorProbBack = matrix(0, nObs, mClass)
  
  posteriorProb[1, ] = outputProb[1,] * t(priorState)/sum(outputProb[1,] * t(priorState))
  
  for (time in 2:nObs) {
    prob = outputProb[time,] * (posteriorProb[time - 1, ]%*%transitionMatrix)
    posteriorProb[time, ] = prob/sum(prob)
  }
  return(posteriorProb)
}

testYProb = predictedActivities$posterior
probabilities = simpleEm(transitionMatrix, initialState, testYProb, obsCount, classCount)

getMostProbableActivity = function(probabilities, obsCount, y) {
  activities = matrix(0, obsCount, 1)
  for (i in 1:obsCount) {
    index = which(probabilities[i, ] == max(probabilities[i, ]))
    activities[i, 1] = index[1] 
  }
  return (activities)
}
activities = getMostProbableActivity(probabilities, obsCount)

table(activities, testY$Activity)
newScore = mean(activities == testY$Activity)

# "LDA+HMM performance: 0.979300984051578"
print(paste("LDA+HMM performance: ", newScore, sep=""))

# SVM for activity recognition
fitControl = trainControl(method = "cv", number=5, classProbs=TRUE)
svmFit = train(x = preProcessedData,
               y = trainY$Factor,
               method = "svmRadial",
               trControl = fitControl,
               tuneLength = 8)

svmClass = predict(svmFit$finalModel, preProcessedTest)
table(svmClass, testY$Factor)
svmScore = mean(svmClass == testY$Factor)

# "SVM (radial) performance:0.943332202239566"
print(paste("SVM (radial) performance:", svmScore, sep=""))

fitControl = trainControl(method = "cv", number=5, classProbs=TRUE)
gbmGrid = expand.grid(interaction.depth = c(1, 5, 9),
                      n.trees = (1:30)*50,
                      shrinkage = 0.1,
                      n.minobsinnode = 20)

gbmFit = train(x = preProcessedData,
               y = trainY$Factor,
               method = "gbm",
               trControl = fitControl,
               tuneGrid = gbmGrid,
               verbose = FALSE)

# > gbmFit$bestTune
# n.trees interaction.depth shrinkage n.minobsinnode
# 85    1250                 9       0.1             20

gbmClass = predict(gbmFit, newdata = preProcessedTest, type = "raw")
table(gbmClass, testY$Factor)
gbmScore = mean(gbmClass == testY$Factor)

# "Boosted tree performance:0.955208686800136"
print(paste("Boosted tree performance:", gbmScore, sep=""))

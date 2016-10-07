library(class)
library(parallel)

# References:
# Michael Lindenbaum and Shaul Markovitch and Dmitry Rusakov. 
# Selective Sampling Using Random Field Modelling. 1999
#
# Data Set
# https://archive.ics.uci.edu/ml/datasets/Ionosphere
# 
# This radar data was collected by a system in Goose Bay, Labrador. This system consists of a phased array 
# of 16 high-frequency antennas with a total transmitted power on the order of 6.4 kilowatts. The targets
# were free electrons in the ionosphere. "Good" radar returns are those showing evidence of some type of
# structure in the ionosphere. "Bad" returns are those that do not; their signals pass through the ionosphere.

# Received signals were processed using an autocorrelation function whose arguments are the time of a pulse
# and the pulse number. There were 17 pulse numbers for the Goose Bay system. Instances in this databse are
# described by 2 attributes per pulse number, corresponding to the complex values returned by the function
# resulting from the complex electromagnetic signal.
ionosphere.data <- read.csv("~/src/Kate/LSS/ionosphere.data.txt", header=FALSE, stringsAsFactors=FALSE)

# Attribute Information:
#
# - All 34 are continuous
# - The 35th attribute is either "good" or "bad" according to the definition summarized above.
lastAttr = ncol(ionosphere.data)

dataX = data.matrix(ionosphere.data[, -lastAttr])

dataY = matrix(1, nrow(dataX), 1)
dataY[which(ionosphere.data[, lastAttr] == 'g'), 1] = -1  # extracting "good" radar returns

distance = function(x1, x2) {
  distance = sqrt(sum((x1 - x2)^2))
  return (distance)
}

# The labels of every pair of points are correlated with covariance decreasing
# with the distance between points.
#
# scale - scaling parameter
covariance = function(dist, scale) {
  return (0.25 * exp(-dist/scale))
}

nearestNeighbors = function(X, point) {
  dists = apply(X, 1, function(x) distance(x, point))
  sortedDists = sort(dists, decreasing = TRUE)

  return (match(sortedDists, dists))
}

# Conditional probability that unknown point belongs to class=1 basing on two closest labeled neighbors
calcProb = function(trainX, trainY, x, neighbors, scale) {
  point1 = trainX[neighbors[1],]
  point2 = trainX[neighbors[2],]

  dist1  = distance(point1, x)
  dist2  = distance(point2, x)
  dist12 = distance(point1, point2)

  cov1  = covariance(dist1, scale)
  cov2  = covariance(dist2, scale)
  cov12 = covariance(dist12, scale)

  label1 = trainY[neighbors[1]]
  label2 = trainY[neighbors[2]]

  return (0.5 + (label1*cov1 + label2*cov2)/(0.5 + 2*label1*label2*cov12))
}

# Utility function that estimates the merit of adding an instance x to the training
# set as a training example.
utility = function(unlabeledX, trainX, trainY, pointX, pointY, neighbors, scale) {
  candidateX = rbind(trainX, pointX)
  candidateY = c(trainY, pointY)

  prob = apply(unlabeledX, 1, function(u) {
    p = calcProb(candidateX, candidateY, u, neighbors, scale)
    return(max(p, 1 - p))
  })
  accuracy = mean(prob)
  return (accuracy)
}

# One-step lookahead selective sampling, chooses the next example in order
# to maximize the expected utility of the resulting classifier.
selectNextExample = function(unlabeledX, trainX, trainY) {
  # calculating 2D matrix of pairwise distances for all observations
  pairwiseDist = apply(trainX, 1,
    function(x1) apply(trainX, 1, function(x2) distance(x1, x2)))

  distScale = mean(pairwiseDist)/4  # 4 is scaling factor

  utilityValues = apply(unlabeledX, 1, function(u) {
    neighbors = nearestNeighbors(trainX, u)
    
    u1 = max(utility(unlabeledX, trainX, trainY, u, -1, neighbors, distScale))
    u2 = max(utility(unlabeledX, trainX, trainY, u,  1, neighbors, distScale))

    p = calcProb(trainX, trainY, u, neighbors, distScale)
    u = (1 - p)*u1 + p*u2

    return (u)
  })
  index = which.max(utilityValues)
  return (index)
}

randomSubsetKNN = function(n) {
  trainIndex = sample(1:200, n) #sample(1:nrow(dataX), n)
  trainX = dataX[trainIndex,]
  trainY = dataY[trainIndex]

  testX = dataX[201:351,]
  testY = dataY[201:351]
  #testX = dataX[-trainIndex,]
  #testY = dataY[-trainIndex]

  knnPred = knn(trainX, testX, trainY, k=1)
  return (mean(testY != knnPred))
}

selectiveSamplingKNN = function(n) {
  unlabeledX = dataX[1:200,]
  unlabeledY = dataY[1:200]
  
  trainSet = sample(1:nrow(unlabeledX), 2)
  
  trainX = unlabeledX[trainSet,]
  trainY = unlabeledY[trainSet]
  
  for (i in 1:n) {
    print(i)
    nextPoint = selectNextExample(unlabeledX, trainX, trainY)
  
    trainX = rbind(trainX, unlabeledX[nextPoint,])
    trainY = c(trainY, unlabeledY[nextPoint])
  
    unlabeledX = unlabeledX[-nextPoint,]
    unlabeledY = unlabeledY[-nextPoint]
  }
  
  testX = dataX[201:351,]
  testY = dataY[201:351]
  knnPred = knn(trainX, testX, trainY, k=1)

  return (mean(testY != knnPred))
}

# Selective Sampling
exp1 = mclapply(1:30, function(n) {
  print(n)
  ret = sapply(1:100, function(x) selectiveSamplingKNN(n))
  return (list("mean" = mean(ret), "sd" = sd(ret)))
}, mc.cores=4)

exp1 = simplify2array(mclapply(1:100, function(x) selectiveSamplingKNN(20), mc.cores=4))
print(paste("mean=", mean(exp1), "; stdev=", sd(exp1), sep=""))

# Random Subset Sampling
exp2 = sapply(1:30, function(n) {
  ret = sapply(1:100, function(x) randomSubsetKNN(n))
  return (list("mean" = mean(ret), "sd" = sd(ret)))
})
print(paste("mean=", mean(exp2), "; stdev=", sd(exp2), sep=""))
print("Done")

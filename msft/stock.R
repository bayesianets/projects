# Fits a model for prediction of next day MSFT stock price depending on values
# of todays stock prices for other tech companies.
#
#References:
#Paper: Akbilgic, O., Bozdogan, H., Balaban, M.E., (2013) A novel Hybrid RBF Neural Networks model as a forecaster, Statistics and Computing. DOI 10.1007/s11222-013-9375-7 
#PhD Thesis: Oguz Akbilgic, (2011) Hibrit Radyal TabanlÄ± Fonksiyon AÄŸlarÄ± ile DeÄŸiÅŸken SeÃ§imi ve Tahminleme: Menkul KÄ±ymet YatÄ±rÄ±m KararlarÄ±na Ä°liÅŸkin Bir Uygulama, Istanbul University
#Data source: www.finance.yahoo.com

library(caret)
library(astsa)
library(rpart)
library(parallel)


data = data.frame()
for(symbol in c("MSFT", "AAPL", "ADBE", "AMD", "EMC", "HP", "IBM", "INTC", "NVDA", "SNE", "YHOO")) {
    p = paste("~/src/Kate/stock/data/", symbol, ".csv", sep = "")
    d = read.csv(p)

    # Removing trends from the source data. If a trend exist, we need to 
    # remove it before fitting a model to uncover underlying dependencies.
    m = lm(d$close ~ time(date), data = d)
    d$close = m$residuals
    
    d = d[c("date", "close")]
    names(d) = c("date", symbol)

    # Creating additional variables for time series with lag of 1-2 days.
    ts = paste(symbol, ".TS", sep = "")
    d[[ts]] = ts(d[[symbol]])
    
    ts1 = paste(symbol, ".TS.1", sep = "")
    d[[ts1]] = lag(d[[ts]], -1)
    
    ts2 = paste(symbol, ".TS.2", sep = "")
    d[[ts2]] = lag(d[[ts]], -2)
    
    if (length(data) == 0) {
        data = d
    } else {
        data = merge(data, d)
    }
}

# Creating feature plots
dataNames = names(data)
dataTs = dataNames[grep(".TS$", dataNames)]
dataTs = dataTs[-grep("MSFT", dataTs)]

# plot.ts can only display 10 plots at once, splitting into two batches
plot.ts(data[c("MSFT.TS", dataTs[1:5])])
plot.ts(data[c("MSFT.TS", dataTs[6:length(dataTs)])])

featurePlot(x = data[dataTs], y = data[, "MSFT"], type = c("g", "p", "smooth"))

# Creating intersected dataset with original plus lagged data
data = ts.intersect(data[dataNames[grep("TS", dataNames)]])

dataNames = names(data)
msftIndex = grep("MSFT.TS$", dataNames)

# Declaring RBF functions for transformation of the data.
gaussRbf = function(x, center, radius, alpha) {
  obsCount = nrow(x)
  dimCount = nrow(radius)
  phi = matrix(0, obsCount, dimCount)
  
  for (i in 1:obsCount) {
    for (j in 1:dimCount) {
      phi[i, j] = exp(sum(-alpha^2/2*((x[i, ] - center[j, ])/radius[j, ])^2))
    }
  }
  return (phi)
}

# Estimating initial coefficients for NN using ridge regression for given candidate 
# transformation function.
fitRbfModel = function(data, response) {
  ridgeGrid = data.frame(.lambda = seq(0, 1, length = 5))
  response = as.numeric(response)
  
  ridgeRegFit = train(data, response, method = "ridge", 
                      tuneGrid = ridgeGrid, 
                      trCtrl = trainControl(method="cv", number=5), 
                      preProc = c("center", "scale"))
  return (ridgeRegFit)
}

# Calculating ICOMP information criteria.
calcICOMP = function(ridgeRegFit, phi, y) {
  n = nrow(phi)
  P = ncol(phi)
  yPred = predict(ridgeRegFit)
  sigma2 = t(y - yPred)%*%(y - yPred)/n
  invFisher = matrix(0, P + 1, P + 1)
  invFisher[1:P, 1:P] = as.numeric(sigma2) * solve(t(phi)%*%phi)
  invFisher[P + 1, P + 1] = 2*sigma2^2/4
  rankF = min(dim(invFisher))
  traceF = sum(diag(invFisher)/rankF)
  ICOMP = n * log(2 * pi) + n * log(sigma2) + n + rankF * log(traceF) - log(det(invFisher))
  
  return (ICOMP)
}

calculateICOMPRatio = function(icompValues) {
  icompMax = max(icompValues)
  icompDelta = sapply(icompValues, function(x) icompMax - x)
  
  averageDelta = mean(icompDelta)
  icompRatio = sapply(icompDelta, function(d) d/averageDelta)
  return (icompRatio)
}

#
# Next, we need to select appropriate features for construction of the model. 
# With lag variables input data has 16 distinct features, so exhaustive search
# of best subset is very expensive. 
#
generateInitialPopulation = function(populationSize, featureCount) {
  initialPop = matrix(0, populationSize, featureCount)
  for (i in 1:populationSize) {
    repeat {
      initialPop[i,] = sample(c(0, 1), featureCount, replace = TRUE)
      
      if (sum(initialPop[i, ]) > 1) {
        break
      }
    }
  }
  return (initialPop)
}

generateChildren  = function(population, icompRatios, populationSize, featureCount, probCross, probMut) {
  children = matrix(0, 0, featureCount)
  while (nrow(children) < populationSize) {
    parent1Index = sample(1:populationSize, 1, prob=icompRatios)
    parent2Index = sample(1:populationSize, 1, prob=icompRatios)
    
    parent1 = population[parent1Index,]
    parent2 = population[parent2Index,]
    
    crossMask = rbinom(featureCount, 1, probCross)
    newChild = ifelse(crossMask, parent1, parent2)
    
    mut_mask = rbinom(featureCount, 1, probMut)
    newChild[which(mut_mask == 1)] = abs(newChild[which(mut_mask == 1)] - 1)
    if (sum(newChild) > 1)  # at least one variable is enabled
      children = rbind(children, newChild)
  }
  return (children)
}

selectBestSubset = function(trainX,
                              trainY,
                              population,
                              populationSize, generationCount,
                              featureCount, 
                              probCross, probMut) {
  children = population
  
  for (i in 1:generationCount) {
    models = apply(children, 1, function(mask) {
      dataSubset = trainX[, which(mask == 1)]
      
      model = fitRbfModel(dataSubset, trainY)
      icomp = calcICOMP(model, dataSubset, trainY)
      
      return(list("mask" = mask, "model" = model, "ICOMP" = icomp))
    })
    
    icompValues = sapply(models, function(x) return(x$ICOMP))
    icompRatio = calculateICOMPRatio(icompValues)
    
    children = generateChildren(population, icompRatio, populationSize, featureCount, probCross, probMut)
    
    if (i == generationCount) {
      index = min(which(icompRatio == max(icompRatio)))
      bestModel = models[[index]]
      
      return(list("alpha" = alpha,
                  "probCross" = probCross,
                  "probMut" = probMut,
                  "family" = "gauss",
                  "model" = bestModel$model,
                  "ICOMP" = bestModel$ICOMP,
                  "mask" = bestModel$mask))
    }
  }
}

runModel = function(trainData, testData) {
  trainX = as.matrix(trainData[, -1])
  trainY = as.matrix(trainData[, 1])
  colnames(trainY) = c("Y")
  
  testX = as.matrix(testData[, -1])
  testY = as.matrix(testData[, 1])
  colnames(testY) = c("Y")
  
  # Next we are going to apply NN to model ISE for T+1 depending on current
  # and past state of the other indices. There are multiple possible kernel
  # transformations that we evaluate using regression trees for parameters
  # initialization.
  tree = rpart(MSFT.TS ~ ., data = trainData, control = rpart.control(minsplit=nrow(trainData)/20))
  leaf = unique(tree$where)  # list of terminal nodes
  
  featureCount = ncol(trainX)  # count of predictors
  regionCount = length(leaf)  # count of hyperrectangles
  
  center = matrix(0, regionCount, featureCount)
  radius = matrix(0, regionCount, featureCount)
  for (i in 1:regionCount) {
    elements = trainX[tree$where == leaf[i],]
    
    for (j in 1:featureCount) {
      center[i, j] = (max(elements[, j]) + min(elements[, j]))/2
      radius[i, j] = max(elements[, j]) - min(elements[, j])
    }
  }
  
  # Generate Initial population for all models
  populationSize = 25
  generationCount = 30
  #generationCount = 3
  featureCount = regionCount
  
  initialPop = generateInitialPopulation(populationSize, featureCount)
  
  # The parameter aplpha controls the spread of the gaussian function. Detailed experimentation
  # with different values of alpha (ranging from 0.5 through 2.5) showed that TB-RBF was relatively
  # insensitive to the setting of this parameter.
  #
  # "Decision Trees Can Initialize Radial-Basis Function Networks" by Miroslav Kubat.
  #
  alpha = 2
  
  # applying set of RBFs to transform input values
  rbf = gaussRbf(trainX, center, radius, alpha)
  colnames(rbf) = paste("RBF", 1:ncol(rbf), sep="")
  
  # tuning grid for genetic algorithm
  probCross = 0.5 + (0:8)/20  # 0.5 0.55 0.6 ...
  #probCross = 0.5 + (0:2)/20  # 0.5 0.55 0.6 ...
  probMut = 0.005
  
  bestSubsets = lapply(probCross, function(p) {
    print(paste("probCross: ", p, sep = ""))
    return(selectBestSubset(rbf,
                              trainY,
                              initialPop,
                              populationSize,
                              generationCount,
                              featureCount,
                              p,
                              probMut))
  })
  
  # Select best model
  icomp = sapply(bestSubsets, function(s) s$ICOMP)
  ind = min(which(icomp == max(icomp)))
  bestSubset = bestSubsets[[ind]]
  
  rbf = gaussRbf(testX, center, radius, alpha)
  colnames(rbf) = paste("RBF", 1:ncol(rbf), sep="")
  rbfSubset = rbf[, which(bestSubset$mask == 1)]
  
  predictY = predict(bestSubset$model, newdata = rbfSubset)
  
  predictY1 = predictY[1:(length(predictY) - 1)]
  predictY2 = predictY[2:length(predictY)]
  
  predictDirection = sign((predictY2 - predictY1)/predictY1) #cbind(predictY1, predictY2, (predictY2 - predictY1)/predictY1)
  
  testY1 = testY[1:(length(testY) - 1)]
  testY2 = testY[2:length(testY)]
  
  testDirection = sign((testY2 - testY1)/testY1) #cbind(testY1, testY2, (testY2 - testY1)/testY1)
  
  print(predictDirection == testDirection)  
  return(predictDirection == testDirection)
}

rowCount = nrow(data)
trainSize = 250
testSize = 20

print(paste("Running ", ceiling((rowCount - trainSize)/testSize), "iterations"))
result = mclapply(1:ceiling((rowCount - trainSize)/testSize), function(x) {
  print(x)
  trainData = as.data.frame(data[((x - 1)*testSize + 1):((x - 1)*testSize + trainSize),])
  testData = as.data.frame(data[(trainSize + (x - 1)*testSize + 1):min(rowCount, (trainSize + x*testSize)),])
  
  return (runModel(trainData, testData))
}, mc.cores = 4)


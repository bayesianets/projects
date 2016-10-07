using DataFrames


data = readtable("ionosphere.data.txt")

dataY = zeros(Int, nrow(data))
dataY[data[:g] .== "g"] = 1

dataX = Array{Float64, 2}(data[names(data)[1:(length(data) - 1)],])

function distance(x1, x2)
    sqrt(sum(x1 .- x2)^2)
end

function covariance(dist, scale)
    0.25*exp(-dist/scale)
end

function nearestNeighbors(X, point)
    (row, col) = size(X)
    dists = [distance(X[row, :], point) for row in 1:row]
    sortedDists = sort(dists)

    indexin(sortedDists, dists)
end

function calcProb(trainX, trainY, x, neighbors, scale)
    point1 = trainX[neighbors[1],:]
    point2 = trainX[neighbors[2],:]

    dist1  = distance(point1, x)
    dist2  = distance(point2, x)
    dist12 = distance(point1, point2)

    cov1  = covariance(dist1, scale)
    cov2  = covariance(dist2, scale)
    cov12 = covariance(dist12, scale)

    label1 = trainY[neighbors[1]]
    label2 = trainY[neighbors[2]]

    0.5 + (label1*cov1 + label2*cov2)/(0.5 + 2*label1*label2*cov12)
end

function utility(unlabeledX, trainX, trainY, pointX, pointY, neighbors, scale)
    candidateX = [trainX; pointX]
    candidateY = [trainY; pointY]

    (row, col) = size(unlabeledX)
    prob = zeros(row)

    for i in 1:row
        u = unlabeledX[i,:]
        p = calcProb(candidateX, candidateY, u, neighbors, scale)
        prob[i] = max(p, 1 - p)
    end
    mean(prob)
end

function selectNextExample(unlabeledX, trainX, trainY)
    count = size(trainX)[1]
    pairwiseDist = zeros(count, count)

    for i in 1:count
        for j in 1:count
            pairwiseDist[i, j] = distance(trainX[i,:], trainX[j,:])
        end
    end
    distScale = mean(pairwiseDist)/4

    (row, col) = size(unlabeledX)
    utilityValues = zeros(row)

    for i in 1:row
        u = unlabeledX[i,:]
        neighbors = nearestNeighbors(trainX, u)
        u1 = utility(unlabeledX, trainX, trainY, u, -1, neighbors, distScale)
        u2 = utility(unlabeledX, trainX, trainY, u,  1, neighbors, distScale)

        p = calcProb(trainX, trainY, u, neighbors, distScale)
        utilityValues[i] = (1 - p)*u1 + p*u2
    end
    (value, index) = findmax(utilityValues)

    index
end

function selectiveSamplingKNN(n)
    println("selectiveSamplingKNN(", n, ")")
    pointCount = 200
    unlabeledX = dataX[1:pointCount,:]
    unlabeledY = dataY[1:pointCount]

    trainSet = sample(1:pointCount, 2)

    trainX = unlabeledX[trainSet,:]
    trainY = unlabeledY[trainSet]

    for i in 1:n
        nextPoint = selectNextExample(unlabeledX, trainX, trainY)

        trainX = [trainX; unlabeledX[nextPoint,:]]
        trainY = [trainY; unlabeledY[nextPoint]]

        unlabeledX = unlabeledX[1:end .!= nextPoint,:]
        unlabeledY = unlabeledY[1:end .!= nextPoint]
    end

    testX = dataX[(pointCount + 1):end,:]
    testY = dataY[(pointCount + 1):end]

    (row, col) = size(testX)
    result = zeros(row)

    for i in 1:row
        p = testX[i,:]
        d = [distance(p, trainX[row,:]) for row in 1:size(trainX)[1]]

        (value, index) = findmin(d)

        if trainY[index] != testY[index]
            result[i] = 1
        end
    end
    mean(result)
end

data = Array{Float64}(50)
for i in 1:50
    data[i] = selectiveSamplingKNN(20)
end
println(data)
println("mean=", mean(data), ";sd=", std(data))

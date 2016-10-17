# References:
#
# The dataset is taken from https://archive.ics.uci.edu/ml/datasets/CalIt2+Building+People+Counts
#
# "Adaptive event detection with time-varying Poisson processes" 
# A. Ihler, J. Hutchins, and P. Smyth 
# Proceedings of the 12th ACM SIGKDD Conference (KDD-06), August 2006. 
# http://www.ics.uci.edu/~ihler/code/event.html

library(gtools)

FLOW_IN  = 9
FLOW_OUT = 7

#
# CalIt2 building people counts, format:
#
# 1. Flow ID: 7 is out flow, 9 is in flow
# 2. Date: MM/DD/YY
# 3. Time: HH:MM:SS
# 4. Count: Number of counts reported for the previous hour.
#
# Rows: Each half hour time slice is represented by 2 rows: one row for the out flow during that time
# period (ID=7) and one row for the in flow during that time period (ID=9)
#
data = read.csv("~/src/Kate/poisson/CalIt2.data.txt",
                header=FALSE, stringsAsFactors=FALSE,
                col.names=c('FlowId', 'Date', 'Time', 'Entries'))

data = data[which(data$FlowId == FLOW_OUT), ]
date = strptime(data$Date, '%m/%d/%y')

timeIntervalCount = list(
    # count of unique days
    "day"       = length(unique(data$Date)),
    # count of observed days of week
    "dayOfWeek" = length(unique(weekdays(date))),
    # count of time intervals/day
    "timeOfDay" = length(unique(data$Time))
    )

countData = matrix(data$Entries, timeIntervalCount$timeOfDay, timeIntervalCount$day)

Niter = 50
Nburn = 10

# By adjusting these priors one can also increase or decrease the model’s sensitivity to deviations
# and thus the number of events detected;
priors = list(
    # Normal event rate is decomposed as following:
    #
    #   rate = baseRate*dayEffect*timeOfDayEffect
    #
    # where following conjugate prior distributions are used: 
    #
    #                                     baseRate ~ Gamma(shape, rate)
    #                        (1/7)*(dayEffect1..7) ~ Dir(a1, .., a7)
    #   (1/D)*(timeOfDayEffect1..timeOfDayEffectD) ~ Dir(a1, .., aD)
    "baseRate"        = list("type" = "gamma", "shape" = 1, "rate" = 1),
    "dayEffect"       = matrix(5, 1,                           timeIntervalCount$dayOfWeek),
    "timeOfDayEffect" = matrix(1, timeIntervalCount$timeOfDay, timeIntervalCount$dayOfWeek),

    # Increase in observation counts due to the event is modeled as Poisson,
    # where rate independent at each time t:
    #
    #   rate ~ Gamma(shape, rate)
    "eventRate" = list("type" = "gamma", "shape" = 5, "rate" = 1/3),  # expected amount of NE, wait time

    # Behavior of anomalous periods of time is modeled by a binary process z(t),
    # which indicates presence of an event. Probability distribution over z(t) is
    # defined to be Markov in time, with following transition probability matrix:
    #        _              _
    #        | 1-z0    z1   |
    #   M =  |              |
    #        |  z0    1-z1  |
    #        -              -
    # where Beta distribution is used for z0 and z1 priors.
    #
    # the length of each time period between events is geometric with expected value 1/(1 − z00),
    # expected length of event 1/(1 - z11) 
    #
    # http://stats.stackexchange.com/questions/47771/what-is-the-intuition-behind-beta-distribution
    "z0" = list("type" = "beta", "shape1" = 44, "shape2" = 4),
    "z1" = list("type" = "beta", "shape1" = 35, "shape2" = 13)
    )

z0 = rbeta(1, priors$z0$shape1, priors$z0$shape2)
z1 = rbeta(1, priors$z1$shape1, priors$z1$shape2)

transitionMatrix = matrix(c(1 - z0, z1, z0, 1 - z1), 2, 2)

intervalCount = timeIntervalCount$day*timeIntervalCount$timeOfDay
iterationCount = Niter + Nburn

# Traffic is modeled using nonhomogeneous Poisson distribution, with event rate varying depending on day/time.
eventRate = matrix(0, intervalCount, iterationCount)

# Event/no event (0/1) at each day/time interval for all iterations. The average across all iterations gives us
# the probability of an event at each day/time interval.
eventData = matrix(0, intervalCount, iterationCount)
# we consider at least 1 entries if there is no event
normalCountData = pmax(countData, 1)

estimateNormalPoissonRate = function(normalCountData, priors, timeIntervalCount) {
  length = length(normalCountData)
  
  normalEventRate = matrix(0, timeIntervalCount$timeOfDay, timeIntervalCount$day)
  
  baseEventRate = rgamma(1, shape = sum(normalCountData) + priors$baseRate$shape, rate = length + priors$baseRate$rate)
  
  # estimation of "day of week" effect.
  obs = matrix(normalCountData, timeIntervalCount$timeOfDay, timeIntervalCount$day)
  
  dayOfWeekEffect = matrix(0, 1, timeIntervalCount$dayOfWeek) 
  
  for (weekDay in 1:timeIntervalCount$dayOfWeek) {
    columnIndex =  seq(weekDay, timeIntervalCount$day, 7)
    entriesByDay = sum(normalCountData[columnIndex]) + priors$dayEffect[weekDay]
    dayOfWeekEffect[1, weekDay] = rgamma(1, entriesByDay, rate = 1)
  }
  dayOfWeekEffect = dayOfWeekEffect/mean(dayOfWeekEffect)
  
  # estimation of "time of day" effect.
  timeOfDayEffect = matrix(0, timeIntervalCount$timeOfDay, timeIntervalCount$dayOfWeek)
  
  for (weekDay in 1:timeIntervalCount$dayOfWeek) {
    columnIndex = seq(weekDay, timeIntervalCount$day, 7)
    
    for (timeInt in 1:timeIntervalCount$timeOfDay) {
      entriesByDayTimeInt = sum(normalCountData[timeInt, columnIndex]) + priors$timeOfDayEffect[timeInt]
      timeOfDayEffect[timeInt, weekDay] = rgamma(1, entriesByDayTimeInt, rate = 1)
    }
  }
  meanDayRates = t(matrix(colMeans(timeOfDayEffect), timeIntervalCount$dayOfWeek, timeIntervalCount$timeOfDay))
  timeOfDayEffect = timeOfDayEffect/meanDayRates
  
  # combining into "normal" event rate.
  for (weekday in 1:timeIntervalCount$dayOfWeek) {
    columnIndex = seq(weekday, timeIntervalCount$day, 7)
    for (timeInt in 1:timeIntervalCount$timeOfDay) { 
      normalEventRate[timeInt, columnIndex] =
        baseEventRate*dayOfWeekEffect[weekday]*timeOfDayEffect[timeInt, weekday]
    } 
  }
  return(normalEventRate)
}

# Based on given the periodic Poisson mean λ(t) and the transition probability matrix,
# draw a sample sequence Z(t) using a variant of the forward–backward algorithm.
#
# http://www.cs.rochester.edu/u/james/CSC248/Lec11.pdf
detectEvents = function(transitionMatrix, countData, normalRate, priors, uniqueDayCount) {
  intervalCount = timeIntervalCount$timeOfDay*uniqueDayCount
  
  eventData  = matrix(0, intervalCount, 1)
  normalCountData = matrix(0, intervalCount, 1)
  eventCountData = matrix(0, intervalCount, 1)
  
  # initial state where system at the very beginning, randomly defined
  initialState = c(1, 0)
  
  priorState = transitionMatrix %*% initialState
  
  # This probability calculated based on observed data N, using normal rates
  # for state = no event and state = event
  likelihoodProb = likelihood(countData, normalRate, priors, timeIntervalCount)
  
  posteriorProb = matrix(0, intervalCount, 2)
  posteriorProbBack = matrix(0, intervalCount, 2)
  
  # P(Qt = State Event/NoEvent at time t, observed sequence N)
  # Could be rewrite using chain rule 
  #  P(Qt produces = State Event/NoEvent at time i, observed sequence N up from normalCountData to Nt) 
  #                  *P(observed sequence N from t to T | Qt produces = State Event/NoEvent at time t, observed sequence N up from normalCountData to Nt)
  # Which, because the Markov assumption says that the future output depends only on the current state,
  #   P(Qt produces = State Event/NoEvent at time i, observed sequence N up from normalCountData to Nt) 
  #                  *P(observed sequence N from t to T | Qt produces = State Event/NoEvent at time t)
  
  # forward iteration
  # P(Qt produces = State Event/NoEvent at time i, observed sequence N up from normalCountData to Nt) 
  # Then the forward probability for State Event/NoEvent  at time t, at
  # is the probability that an HMM will output a sequence N and end in State Event/No event
  
  # Computing forward probabilities using probabilities of 
  # observed N based on normal rates for state = no event and state = event 
  
  # initialization of start parameter to get probability of N at 
  # the beginning to be in state = ne event or state = event
  posteriorProb[1, ] = likelihoodProb[1,] * t(priorState)/sum(likelihoodProb[1,] * t(priorState))
  
  # Computing probabilities for given N being in state = non event/event for the next time intervals
  # prob at time 2 (state = event) = (prob at time 1 (no event) * P(No Event|Event) + prob at time 1 (event) * P(Event|Event)) * P(N|Event)
  # prob at time 2 (state = no event) = (prob at time 1 (no event) * P(No Event| No Event) + prob at time 1 (event) * P(Event|No Event)) * P(N|No Event)
  for (time in 2:intervalCount) {
    prob = likelihoodProb[time,] * t(transitionMatrix%*%posteriorProb[time - 1,])
    posteriorProb[time, ] = prob/sum(prob)
  }
  
  # bt(i) = Sj Ai,j * Prob(ot+1 | Sj) * bt+1(j)
  # initialization of start parameter to get probability of N at 
  # the beginning to be in state = ne event or state = event
  posteriorProbBack[intervalCount, ] = likelihoodProb[intervalCount, ] * t(priorState)/sum(likelihoodProb[intervalCount,] * t(priorState))
  
  # Computing probabilities for given N being in state = non event/event for the next time intervals
  # prob at time 2 (state = event) = (prob at time 1 (no event) * P(No Event|Event) + prob at time 1 (event) * P(Event|Event)) * P(N|Event)
  # prob at time 2 (state = no event) = (prob at time 1 (no event) * P(No Event| No Event) + prob at time 1 (event) * P(Event|No Event)) * P(N|No Event)
  for (time in (intervalCount - 1): 1) {
    prob = likelihoodProb[time,] * t(transitionMatrix%*%posteriorProb[time + 1,])
    posteriorProbBack[time, ] = prob/sum(prob)
  }
  
  # mix together
  denom = sum(posteriorProb[intervalCount, ])
  mixedProb = posteriorProb * posteriorProbBack/denom
  
  # Estimate normalCountData, eventCountData
  
  for (time in 1:intervalCount) { 
    stateMaxProb = which.max(mixedProb[time,])
    eventData[time] = if (stateMaxProb == 1) 0 else 1
  }

  # Computing eventData, normalCountData, eventCountData
  # Based on maximum probability infer state = Event//No event
  for (time in 1:intervalCount) {
    count = countData[time]
    rate = normalRate[time]
    
    mostProbableState = which.max(mixedProb[time, ])
    
    if (mostProbableState == 1) {
      eventData[time] = 0
      normalCountData[time] = count
      eventCountData[time] = 0    
    } else {
      eventData[time] = 1
      probEvent = max(mixedProb[time, ])
      
      # normalCountData
      if (count != 0) {
        x = seq(count, 0)
        # normalCountData ~ poisson * negative binomial
        dp = dpois(x, rate) * dnbinom(rev(x), priors$eventRate$shape, priors$eventRate$rate/(1 + priors$eventRate$rate))

        dpLog = log(dp)
        expDPLog = exp(dpLog - max(dpLog))
        p = expDPLog/sum(expDPLog)
        p = rev(p)
        indexes = which(cumsum(p) >= probEvent)
        if (length(indexes) > 0) {
          normalCountData[time] = min(indexes) - 1
        } else {
          normalCountData[time] = 0
        }
        eventCountData[time] = countData[time] - normalCountData[time] 
      }
    }
  }
  return(list("eventData" = eventData, "normalCountData" = normalCountData, "eventCountData" = eventCountData))
}

estimateEventTransitionMatrix = function(eventData, priors) {
  events = countEvents(eventData)
  
  z0 = rbeta(1,
             events$eventStartCount + priors$z0$shape1,
             events$normalDuration - events$eventStartCount + priors$z0$shape2) 

  z1 = rbeta(1,
             events$eventStopCount + priors$z1$shape1,
             events$eventDuration - events$eventStopCount + priors$z1$shape2) 
  
  transitionMatrix = matrix(c(1 - z0, z0, z1, 1 - z1), 2, 2)
  
  return (transitionMatrix)
}

# evaluate p(event count|rate, transition matrix)
evalEventCountGivenRateAndTransitionMatrix = function(countData, normalRate, transitionMatrix, priors) { 
  intervalCount = timeIntervalCount$timeOfDay*timeIntervalCount$day
  initialState = c(1, 0)
  
  priorState = transitionMatrix%*%initialState
  
  likelihoodProb = likelihood(countData, normalRate, priors, timeIntervalCount)
  
  posteriorProb = matrix(0, intervalCount, 2)
  
  posterior = likelihoodProb[1,]*t(priorState)
  posteriorProb[1,] = posterior/sum(posterior)
  logprob = log(posterior)

  for (time in 2:intervalCount) {
    prob = likelihoodProb[time,] * t(transitionMatrix%*%posteriorProb[time - 1, ])
    posteriorProb[time, ] = prob/sum(prob)
    logprob = logprob + log(prob)
  }
  return (logprob)
}

# evaluate p(transition matrix|event data)
evalTransitionMatrixGivenEvents = function(transitonStateMatrix, eventData, priors) { 
  events = countEvents(eventData)
  
  z0 = transitonStateMatrix[2,1]
  z1 = transitonStateMatrix[1,2]

  logp0 = log(pbeta(z0,
                    events$eventStartCount + priors$z0$shape1,
                    events$normalDuration - events$eventStartCount + priors$z0$shape2)) 

  logp1 = log(pbeta(z1,
                    events$eventStopCount + priors$z1$shape1,
                    events$eventDuration - events$eventStopCount + priors$z1$shape2)) 
  
  return(logp0 + logp1)
}

# likelihood function p(N|z)
likelihoodOfDataGivenEvents = function(countData, eventRate, eventData, priors) {
  intervalCount = length(countData)
  logp = 0
  for (time in 1:intervalCount) {
    count = countData[time]
    rate  = eventRate[time]
    event = eventData[time]
    
    if (event == 0) { # No Event
      logp = logp + log(dpois(count, rate))
    } else {          # Event
      x = seq(0, count)
      logp = logp + log(sum(dpois(x, rate) * dnbinom(rev(x), priors$eventRate$shape, priors$eventRate$rate/(1 + priors$eventRate$rate))))
    }
  }
  return (logp)
}

# Evaluate log probability of event rate based on obtained data
eventLogProbabilityGivenBaseRate = function(eventRate, normalCountData, priors, timeIntervalCount) {
  intervalCount = length(normalCountData)
  
  eventRate = matrix(eventRate, timeIntervalCount$timeOfDay, timeIntervalCount$day)
  
  eventRate0 = mean(eventRate)
  
  # Estimation of "day of week" effect.
  obs = matrix(normalCountData, timeIntervalCount$timeOfDay, timeIntervalCount$day)
  
  weekdays = matrix(0, 1, timeIntervalCount$dayOfWeek) 
  entriesByDay =  matrix(0, 1, timeIntervalCount$dayOfWeek) 
  
  for (weekDay in 1:timeIntervalCount$dayOfWeek) {
    columnIndex =  seq(weekDay, timeIntervalCount$day, 7)
    weekdays[1, weekDay] = mean(eventRate[columnIndex]/eventRate0)
    entriesByDay[1, weekDay] = sum(normalCountData[columnIndex]) + priors$dayEffect[weekDay]
  }
  
  # Estimation of "time of day" effect.
  timeIntRates = matrix(0, timeIntervalCount$timeOfDay, timeIntervalCount$dayOfWeek)
  entriesByDayTimeInt = matrix(0, timeIntervalCount$timeOfDay, timeIntervalCount$dayOfWeek)
  
  for (weekDay in 1:timeIntervalCount$dayOfWeek) {
    columnIndex = seq(weekDay, timeIntervalCount$day, 7)
    
    for (timeInt in 1:timeIntervalCount$timeOfDay) {
      entriesByDayTimeInt[timeInt, weekDay]  = sum(normalCountData[timeInt, columnIndex]) + priors$timeOfDayEffect[timeInt]
      timeIntRates[timeInt, weekDay] = sum(eventRate[timeInt, columnIndex])/eventRate0/weekdays[1, weekDay]
    }
  }
  
  # log of probability based on this eventRate and rates
  logprob = 0
  logprob = logprob + log(pgamma(eventRate0,
                                 sum(sum(normalCountData)) + priors$baseRate$shape,
                                 rate = length(normalCountData) + priors$baseRate$rate))
  logprob = logprob + log(dirpdf(weekdays/timeIntervalCount$dayOfWeek, entriesByDay))
  
  for (i in 1:timeIntervalCount$dayOfWeek) {
    p = dirpdf(timeIntRates[, i]/timeIntervalCount$timeOfDay, entriesByDayTimeInt[, i])

    if (p != 0) {
      logprob = logprob + log(p)
    }
  }
  return(logprob)
}

dirpdf = function(X, alpha) {
  a = 1
  g = 1
  for (i in 1:length(X)) {
    a = a * X[i]^(alpha[i] - 1)
    g = g * lgamma(alpha[i])
  }
  return(g * a/sum(alpha))
}

likelihood = function(countData, normalRate, priors, timeIntervalCount) {
  # State 1 No event
  #
  # Probability to see such Ns, based on non event normalRate
  probState1 = dpois(countData, normalRate) 
  
  # State 2 Event
  #
  # NE is modulated by Poisson Process P(z(t)N; γ(t)) with rate γ(t) ∼ Γ(γ; aE, bE),
  # simplified to NBin(N; aE, bE/(1 + bE))
  probState2 = matrix(0, timeIntervalCount$timeOfDay, timeIntervalCount$day)

  for (t in 1:timeIntervalCount$timeOfDay) {
    for (d in 1:timeIntervalCount$day) {
      x = seq(countData[t, d], 0)
      dp = dpois(x, normalRate[t, d])
      db = dnbinom(rev(x), priors$eventRate$shape, priors$eventRate$rate/(1 + priors$eventRate$rate))
      probState2[t, d] = sum(dp*db)
    }
  }
  probMatrix = matrix(0, timeIntervalCount$timeOfDay*timeIntervalCount$day, 2)
  probMatrix[, 1] = probState1
  probMatrix[, 2] = probState2
  return(probMatrix)
}

countEvents = function(eventProb) {
  count = length(eventProb)

  baseValue = eventProb[1:(count - 1)]
  nextValue = eventProb[2:count]

  eventStart = which(baseValue == 0 & nextValue == 1)
  eventStop  = which(baseValue == 1 & nextValue == 0)

  eventOn  =  which(baseValue == 0)
  eventOff =  which(baseValue == 1)
  
  return(list("eventStartCount" = length(eventStart),
              "eventStopCount"  = length(eventStop),
              "normalDuration"  = length(eventOff),
              "eventDuration"   = length(eventOn))) 
}


# Transition Probability Matrix shows probability for transition from one state (no event) to another (event),
# or staying in the same state.
#
# Matrices for all iterations.
transitionMatrices = array(0, c(2, 2, iterationCount)) 

# Estimated values of event/no event counts for all iterations.
eventCountMatrix  = matrix(0, intervalCount, iterationCount)
normalCountMatrix = matrix(0, intervalCount, iterationCount)

likelihoodOfDataGivenEventsMatrix = matrix(0, 2, iterationCount)

for (iter in 1:(Niter + Nburn)) {
  # Estimation of the Poisson rate of a particular time and day by averaging the observed counts
  # on similar days (e.g., Mondays) at the same time, i.e., the maximum likelihood estimate.
  # It is base Poisson rate for no event day/time interval.
  normalRate = estimateNormalPoissonRate(normalCountData, priors, timeIntervalCount) 
  
  # Detect an event of increased activity when the observed count is sufficiently different than
  # the average, as indicated by having low probability under the estimated Poisson distribution.
  events = detectEvents(transitionMatrix, countData, normalRate, priors, timeIntervalCount$day)
  normalCountData = matrix(events$normalCountData, timeIntervalCount$timeOfDay, timeIntervalCount$day)
  
  transitionMatrix = estimateEventTransitionMatrix(events$eventData, priors)
  
  # Computing log probability for obtained results
  likelihoodOfDataGivenEventsMatrix[, iter] = likelihoodOfDataGivenEvents(countData, normalRate, events$eventData, priors)
  
  eventRate[, iter] = normalRate
  eventData[, iter] = events$eventData

  transitionMatrices[,, iter] = transitionMatrix
  normalCountMatrix[, iter] = events$normalCountData
  eventCountMatrix[, iter] = events$eventCountData

  # estimates the marginal likelihood of the data using the samples
  transitionMatrixAcc = matrix(0, 2, 2)
  eventRateAcc   = matrix(0, intervalCount, 1)
  normalCountAcc = matrix(0, intervalCount, 1)
  eventDataAcc   = matrix(0, intervalCount, 1)

  for (e in 1:iter) {
    transitionMatrixAcc = transitionMatrixAcc + transitionMatrices[, , e]

    eventRateAcc   = eventRateAcc   + eventRate[, e]
    normalCountAcc = normalCountAcc + normalCountMatrix[, e]
    eventDataAcc   = eventDataAcc   + eventCountMatrix[, e]
  }
  
  transitionMatrixSt = transitionMatrixAcc/iter
  eventRateAcc   = matrix(eventRateAcc/iter,   timeIntervalCount$timeOfDay, timeIntervalCount$day)
  normalCountAcc = matrix(normalCountAcc/iter, timeIntervalCount$timeOfDay, timeIntervalCount$day) 
  
  eventDataAcc = eventDataAcc/iter
  
  log1 = eventLogProbabilityGivenBaseRate(eventRateAcc, normalCountAcc, priors, timeIntervalCount) 
  log2 = evalTransitionMatrixGivenEvents(transitionMatrixSt, eventDataAcc, priors)
  log3 = evalEventCountGivenRateAndTransitionMatrix(countData, eventRateAcc, transitionMatrixSt, priors) 
  
  logPresult = log1 + log2 + log3
}

# Plots
#t = ts(mixedProb[, 2])
Nts = ts(matrix(countData, intervalCount, 1))
normalRateTS = ts(matrix(normalRate, intervalCount, 1))

par(mfrow = c(2, 1))

# Plot normal/actual counts for the last few days
plot.ts(Nts[2409:3556])
lines(normalRateTS[2409:3556])


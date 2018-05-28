load("digits.RData")
num.class = dim(training.data)[1] # Number of classes
num.training = dim(training.data)[2] # Number of training data per class
d = prod(dim(training.data)[3:4]) # Dimension of each training image (rowsxcolumns)
num.test = dim(test.data)[2] # Number of test data
dim(training.data) = c(num.class * num.training, d) # Reshape training data to 2d matrix
dim(test.data) = c(num.class * num.test, d) # Same for test.
training.label = rep(0:9, num.training) # Labels of training data.
test.label = rep(0:9, num.test) # Labels of test data

## Calculate log-likelihood and gamma

sl = function(x,mu,pi,N){
  l = matrix(0,nrow = N,ncol = M)
  for(i in seq(N)){
    for(m in seq(M)){l[i,m] = pi[m] * prod(mu[m,]^x[i,]) * prod((1 - mu[m,])^(1 - x[i,]))}
  }
  max0 = max(log(l))
  gamma0 = exp(log(l) - max0) / rowSums(exp(log(l) - max0)) # Gamma is the weight of specific likelihood to the sum of likelihood
  return(list(l = l, gamma0 = gamma0))
}

## Calculate mu

smiu = function(gamma,x){
  mu = matrix(0, nrow = M, ncol = D)
  for(m in seq(M)){
    for (j in seq(D)){mu[m, j] = (1 + sum(gamma[,m] * x[,j])) / (2 + sum(gamma[, m]))}
  }
  return(mu)
}

## Calculate pi

spi = function(gamma){
  pi = vector(length = M)
  for(m in seq(M)){pi[m] = (1 + sum(gamma[, m])) / (M + N)}
  return(pi)
}

## EM algorithm

ema = function(dif,class,dataset,plot=logical())
{
  index = rep(class + 1,N) + 10 * rep(0:(N - 1),each = 1) # Add indices to a particular class
  data = dataset[index, ] # Pick out the data
  muem = matrix(0,nrow = M,ncol = D); piem = rep(0,M)
  logl = vector(); count = 0
  
  ##Initial value (First iteration)
  group = as.factor(sample(seq(M), dim(data)[1], replace = T))
  gamma_init = model.matrix(~ group - 1) # Construct 0,1 matrix
  munew = smiu(gamma_init,data); pinew = spi(gamma_init) # Update new miu and pi values
  dist = sum((pinew-piem)^2) + sum((munew-muem)^2)
  
  ##Iteration
  while(dist > dif){
    muem = munew; piem = pinew
    lnew = sl(data, muem, piem, N)$l; gammanew = sl(data, muem, piem, N)$gamma0 # Update gamma and log-likelihood (Step Expectation)
    munew = smiu(gammanew, data); pinew = spi(gammanew) # Update miu and pi (Step Maximization)
    dist = sum((pinew - piem)^2)+sum((munew - muem)^2) # Calculate step size
    
    ss = gammanew * log(lnew)
    loglik = sum(ss) + sum(log(6*muem*(1-muem))) + log(prod(1:(2*M-1))*prod(piem)) # Calculate log-lik
    logl = c(logl,loglik) # Record log-lik
    count=count+1 # Record number of steps
  }
  
  ##Plot (transfer boolean values to black or white in a picture)
  if(plot == T)
  {
    plt = t(muem)
    dim(plt) = c(sqrt(D),sqrt(D),M)
    for(i in 1:M){
      image(t(1 - round(plt[,,i],3))[,20:1], col=gray(seq(0, 1, length.out=256)),axes=FALSE, asp=1)
    }
  }
  #Report the final outcome
  return(list(mu = round(muem,2),pi = piem,iterations = count,log_likelihood = logl))
}

## Calculating the test error

emerror=function(dataset)
{
  # Build EM model from traning data
  for (i in 0:9)
  {
    assign(paste("out",i,sep=""), ema(dif,i,training.data,plot=F))
  }
  ratio0 = matrix(0,10,10); ratio1 = matrix(0,10,10)
  for(class1 in 0:8){
    for(class2 in ((class1+1):9)){
      miu1 = get(paste("out", class1, sep = ""))$mu
      miu2 = get(paste("out", class2, sep = ""))$mu
      pi1 = get(paste("out", class1, sep = ""))$pi
      pi2 = get(paste("out", class2, sep = ""))$pi
      index10 = rep(class1 + 1, N0) + 10 * rep(0:(N0 - 1), each = 1)
      index20 = rep(class2 + 1, N0) + 10 * rep(0:(N0 - 1), each = 1)
      test1 = test.data[index10,]
      test2 = test.data[index20,]
      
      # Calculate and compare log-likelihood
      l11=sl(test1, miu1, pi1, N0)$l 
      l12=sl(test1, miu2, pi2, N0)$l
      l21=sl(test2, miu1, pi1, N0)$l
      l22=sl(test2, miu2, pi2, N0)$l
      
      ##likelihood standard
      error11 = length(which(t(l11%*%pi1) - t(l12%*%pi2) < 0))
      error12 = length(which(t(l22%*%pi2) - t(l21%*%pi1) < 0))
      
      ##max standard
      error01 = length(which(apply(l11,1,max) - apply(l12,1,max) < 0))
      error02 = length(which(apply(l22,1,max) - apply(l21,1,max) < 0))
      
      ##building matrices
      ratio0[class1 + 1,class2 + 1]=(error01 + error02)/(2 * N0)
      ratio1[class1 + 1,class2 + 1]=(error11 + error12)/(2 * N0)
    }
  }
  
  ratio0[lower.tri(ratio0)] = t(ratio0)[lower.tri(ratio0)] # Symmetric matrix
  ratio1[lower.tri(ratio1)] = t(ratio1)[lower.tri(ratio1)] # Symmetric matrix
  
  row.names(ratio0) = seq(0,9,1); colnames(ratio0) = seq(0,9,1)
  row.names(ratio1) = seq(0,9,1); colnames(ratio1) = seq(0,9,1)
  return(list(MaxLikelihood_Standard = ratio0, MixtureLikelihood_Standard = ratio1))
}

## Experiment

D = d
N = num.training
N0 = num.test
dif = 10^-6

M = 2
result2 = ema(10^-6,2,training.data, plot = T)
cat("Number of iterations:",result2$iterations,"\n")
result2$log_likelihood
mu2 = result2$mu; pi2 = result2$pi

M = 3
result3 = ema(10^-6,2,training.data, plot = T)
cat("Number of iterations:",result3$iterations,"\n")
result3$log_likelihood
mu3 = result3$mu; pi3 = result3$pi

M = 5
result5 = ema(10^-6,2,training.data, plot = T)
cat("Number of iterations:",result5$iterations,"\n")
result5$log_likelihood
mu5 = result5$mu; pi5 = result5$pi

M = 8
result8 = ema(10^-6,2,training.data, plot = T)
cat("Number of iterations:", result8$iterations, "\n")
result8$log_likelihood
mu8 = result8$mu; pi8 = result8$pi

## Final result
result = list(mu2 = mu2, pi2 = pi2, mu3 = mu3, pi3 = pi3,
            mu5 = mu5, pi5 = pi5, mu8 = mu8, pi8 = pi8)

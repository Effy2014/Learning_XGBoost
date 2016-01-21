require(xgboost)
setwd("/Users/XW/Desktop/Influencers in Social Networks")
train = read.csv("train.csv", header = TRUE)
test = read.csv("test.csv", header = TRUE)
y = train[,1]
train = as.matrix(train[,-1])
test = as.matrix(test)
#colnames(train)
#train[1,]
#feature engineering <y, A, B> => <1-y, B, A>
new.train = cbind(train[, 12:22], train[, 1:11])
train = rbind(train, new.train)
y = c(y, 1-y)
x = rbind(train, test)
calcRatio = function(dat, i, j, lambda = 1){
    (dat[, i] + lambda)/(dat[, j] + lambda)
}
#followers/following
A.follow.ratio = calcRatio(x, 1, 2)
B.follow.ratio = calcRatio(x, 12, 13)
#mentions received/sent
A.mention.ratio = calcRatio(x, 4, 6)
B.mention.ratio = calcRatio(x, 15, 17)
#retweets received/sent
A.retweet.ratio = calcRatio(x, 5, 7)
B.retweet.ratio = calcRatio(x, 16, 18)
#followers/post
A.follow.post = calcRatio(x, 1, 8)
B.follow.post = calcRatio(x, 12, 19)
#retweets received/posts
A.retweet.post = calcRatio(x, 5, 8)
B.retweet.post = calcRatio(x, 16, 19)
#mentions received/posts
A.mention.post = calcRatio(x, 4, 8)
B.mention.post = calcRatio(x, 15, 19)

#combine the features into dataset
x = cbind(x[,1:11], A.follow.ratio, A.mention.ratio, A.retweet.ratio, 
          A.follow.post, A.mention.post, A.retweet.post,
          x[,12:22], B.follow.ratio, B.mention.ratio, B.retweet.ratio,
          B.follow.post, B.mention.post, B.retweet.post)

#compare the difference between A and B
#XGBoost is scale invariant 
AB.diff = x[,1:17] - x[,18:34]
x = cbind(x, AB.diff)
train = x[1:nrow(train),]
test = x[-(1:nrow(train)),]
#modeling in XGBoost
set.seed(1024)
#cv.res = xgb.cv(data = train, nfold = 3, label = y, nrounds = 100, verbose = TRUE,
#                objective = 'binary:logistic', eval_metric = "auc")
#setting verbose = TRUE will see the result, showing overfitting 
#calculating AUC minus the standard deviation and choose the iteration with the largest value
#decrease eta and increase nround
set.seed(1024)
cv.res = xgb.cv(data = train, nfold = 3, label = y, nrounds = 3000,
                objective = 'binary:logistic', eval_metric = 'auc',
                eta = 0.005, gamma = 1, lambda = 3, nthread = 8, max_depth =4,
                min_child_weight = 1, verbose = F, subsample = 0.8,
                colsample_bytree = 0.8)
bestRound = which.max(as.matrix(cv.res)[,3] - as.matrix(cv.res)[,4])
cv.res[bestRound, ]
#train the model 
set.seed(1024)
bst = xgboost(data = train, nfold = 3, label = y, nrounds = 3000, 
              objective = 'binary:logistic', eval_metric = 'auc',
              eta = 0.005, gamma = 1, lambda = 3, nthread = 8, max_depth =4,
              min_child_weight = 1, verbose = F, subsample = 0.8,
              colsample_bytree = 0.8)
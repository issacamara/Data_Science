library(tree)
library(ISLR)
library(MASS)
library(randomForest)
library(gbm)
library(glmnet)
########################### Random Forest ###########################

set.seed(1)
train <- sample(1:nrow(Boston), nrow(Boston) / 2)
Boston.train <- Boston[train, -14]
Boston.test <- Boston[-train, -14]
Y.train <- Boston[train, 14]
Y.test <- Boston[-train, 14]
rf.boston1 <- randomForest(Boston.train, y = Y.train, xtest = Boston.test, ytest = Y.test, mtry = ncol(Boston) - 1, ntree = 500)
rf.boston2 <- randomForest(Boston.train, y = Y.train, xtest = Boston.test, ytest = Y.test, mtry = (ncol(Boston) - 1) / 2, ntree = 500)
rf.boston3 <- randomForest(Boston.train, y = Y.train, xtest = Boston.test, ytest = Y.test, mtry = sqrt(ncol(Boston) - 1), ntree = 500)
plot(1:500, rf.boston1$test$mse, col = "green", type = "l", xlab = "Number of Trees", ylab = "Test MSE", ylim = c(10, 19))
lines(1:500, rf.boston2$test$mse, col = "red", type = "l")
lines(1:500, rf.boston3$test$mse, col = "blue", type = "l")
legend("topright", c("m = p", "m = p/2", "m = sqrt(p)"), col = c("green", "red", "blue"), cex = 1, lty = 1)

########################### Regression Tree ###########################
set.seed(1)

train <- sample(1:nrow(Carseats), nrow(Carseats) / 2)
Carseats.train <- Carseats[train,]
Carseats.test <- Carseats[-train,]
Y.train <- Carseats.test[train, 1]
Y.test <- Carseats.test[-train, 1]
tree.Carseats <- tree(formula = Sales ~ ., data = Carseats.train)
summary(tree.Carseats)
plot(tree.Carseats)
text(tree.Carseats, pretty = 0)
yhat=predict(tree.Carseats,newdata=Carseats.test)
mean((yhat-Carseats.test$Sales)^2)

cv.carseats <- cv.tree(tree.Carseats)
plot(cv.carseats$size, cv.carseats$dev, type = "b")
tree.min <- which.min(cv.carseats$dev)
points(tree.min, cv.carseats$dev[tree.min], col = "red", cex = 2, pch = 20)

prune.carseats <- prune.tree(tree.Carseats, best = 8)
plot(prune.carseats)
text(prune.carseats, pretty = 0)

yhat <- predict(prune.carseats, newdata = Carseats.test)
mean((yhat - Carseats.test$Sales)^2)

bag.carseats <- randomForest(Sales ~ ., data = Carseats.train, mtry = 10, ntree = 500, importance = TRUE)
yhat.bag <- predict(bag.carseats, newdata = Carseats.test)
mean((yhat.bag - Carseats.test$Sales)^2)

importance(bag.carseats)

rf.carseats <- randomForest(Sales ~ ., data = Carseats.train, mtry = 3, ntree = 500, importance = TRUE)
yhat.rf <- predict(rf.carseats, newdata = Carseats.test)
mean((yhat.rf - Carseats.test$Sales)^2)

importance(rf.carseats)
########################### OJ data set ###########################
set.seed(1)
train <- sample(1:nrow(OJ), 800)
OJ.train <- OJ[train,]
OJ.test <- OJ[-train,]

tree.OJ <- tree(formula = Purchase ~ ., data = OJ.train)
summary(tree.OJ)

tree.OJ

plot(tree.OJ)
text(tree.OJ, pretty = 0)

pred=predict(tree.OJ,newdata=OJ.test,type = "class")

cv.OJ <- cv.tree(tree.OJ, FUN = prune.misclass)
cv.OJ
plot(cv.OJ$size, cv.OJ$dev, type = "b", xlab = "Tree size", ylab = "Deviance")

prune.OJ <- prune.misclass(tree.OJ, best = 5)
plot(prune.OJ)
text(prune.OJ, pretty = 0)

summary(tree.OJ)
summary(prune.OJ)
prune.pred <- predict(prune.OJ, OJ.test, type = "class")
table(prune.pred, OJ.test$Purchase)

########################### Hitters data set ###########################
Hitters <- na.omit(Hitters)
Hitters$Salary <- log(Hitters$Salary)
set.seed(1)
train <- 1:200
Hitters.train <- Hitters[train,]
Hitters.test <- Hitters[-train,]
pows <- seq(-10, -0.2, by = 0.1)
lambdas <- 10^pows
train.err <- rep(NA, length(lambdas))
for (i in 1:length(lambdas)) {
  boost.Hitters <- gbm(Salary ~ ., data = Hitters.train, distribution = "gaussian", n.trees = 1000, shrinkage = lambdas[i])
  pred.train <- predict(boost.Hitters, Hitters.train, n.trees = 1000)
  train.err[i] <- mean((pred.train - Hitters.train$Salary)^2)
}
plot(lambdas, train.err, type = "b", xlab = "Shrinkage values", ylab = "Training MSE")

set.seed(1)
test.err <- rep(NA, length(lambdas))
for (i in 1:length(lambdas)) {
  boost.Hitters <- gbm(Salary ~ ., data = Hitters.train, distribution = "gaussian", n.trees = 1000, shrinkage = lambdas[i])
  yhat <- predict(boost.Hitters, Hitters.test, n.trees = 1000)
  test.err[i] <- mean((yhat - Hitters.test$Salary)^2)
}
plot(lambdas, test.err, type = "b", xlab = "Shrinkage values", ylab = "Test MSE")

fit1 <- lm(Salary ~ ., data = Hitters.train)
pred1 <- predict(fit1, Hitters.test)
mean((pred1 - Hitters.test$Salary)^2)

x <- model.matrix(Salary ~ ., data = Hitters.train)
x.test <- model.matrix(Salary ~ ., data = Hitters.test)
y <- Hitters.train$Salary
fit2 <- glmnet(x, y, alpha = 0)
pred2 <- predict(fit2, s = 0.01, newx = x.test)
mean((pred2 - Hitters.test$Salary)^2)

boost.Hitters <- gbm(Salary ~ ., data = Hitters.train, distribution = "gaussian", n.trees = 1000, shrinkage = lambdas[which.min(test.err)])
summary(boost.Hitters)

set.seed(1)
bag.Hitters <- randomForest(Salary ~ ., data = Hitters.train, mtry = 19, ntree = 500)
yhat.bag <- predict(bag.Hitters, newdata = Hitters.test)
mean((yhat.bag - Hitters.test$Salary)^2)
########################### Caravan data set ###########################
Caravan$Purchase <- ifelse(Caravan$Purchase == "Yes", 1, 0)
train <- 1:1000
Caravan.train <- Caravan[train,]
Caravan.test <- Caravan[-train,]

set.seed(1)
boost.Caravan <- gbm(Purchase~., data=Caravan.train, distribution = "gaussian",n.trees = 1000, shrinkage = 0.01)
summary(boost.Caravan)
pred.probs <- predict(boost.Caravan, Caravan.test, n.trees = 1000, type = "response")
pred <- ifelse(pred.probs > 0.2, 1, 0)
table(Caravan.test$Purchase, pred)

########################### Weekly data set ###########################
set.seed(1)
train <- sample(nrow(Weekly), nrow(Weekly) / 2)
Weekly$Direction <- ifelse(Weekly$Direction == "Up", 1, 0)
Weekly.train <- Weekly[train, ]
Weekly.test <- Weekly[-train, ]

logit.fit <- glm(Direction ~ . - Year - Today, data = Weekly.train, family = "binomial")
logit.probs <- predict(logit.fit, newdata = Weekly.test, type = "response")
logit.pred <- ifelse(logit.probs > 0.5, 1, 0)
table(Weekly.test$Direction, logit.pred)

boost.fit <- gbm(Direction ~ . - Year - Today, data = Weekly.train, distribution = "bernoulli", n.trees = 5000)
boost.probs <- predict(boost.fit, newdata = Weekly.test, n.trees = 5000)
boost.pred <- ifelse(boost.probs > 0.5, 1, 0)
table(Weekly.test$Direction, boost.pred)

bag.fit <- randomForest(Direction ~ . - Year - Today, data = Weekly.train, mtry = 6)

bag.probs <- predict(bag.fit, newdata = Weekly.test)
bag.pred <- ifelse(bag.probs > 0.5, 1, 0)
table(Weekly.test$Direction, bag.pred)

rf.fit <- randomForest(Direction ~ . - Year - Today, data = Weekly.train, mtry = 2)

rf.probs <- predict(rf.fit, newdata = Weekly.test)
rf.pred <- ifelse(rf.probs > 0.5, 1, 0)
table(Weekly.test$Direction, rf.pred)
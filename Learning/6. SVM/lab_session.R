library(ISLR)
library(e1071)
library(MASS)
library(class)
################################# SVM #################################
set.seed(1)
X1 <- rnorm(100)
X2 <- 2 * X1^2 + 5 + rnorm(100)

class <- sample(100, 50)


X2[class] <- X2[class] + 3
X2[-class] <- X2[-class] - 3
plot(X1[class], X2[class], col = "red", xlab = "X1", ylab = "X2", ylim = c(-6, 30))
points(X1[-class], X2[-class], col = "blue")

Y <- rep(0,100)
Y[class] <- 1

data <- data.frame(X1,X2,Y=as.factor(Y))

train <- sample(100, 50)
data.train <- data[train, ]
data.test <- data[-train, ]
svm.linear <- svm(Y ~ ., data = data.train, kernel = "linear", cost = 10)
plot(svm.linear, data.train)

table(predict = predict(svm.linear, data.train), truth = data.train$Y)

svm.poly <- svm(Y ~ ., data = data.train, kernel = "polynomial", cost = 10)
plot(svm.poly, data.train)
table(predict = predict(svm.poly, data.train), truth = data.train$Y)

svm.radial <- svm(Y ~ ., data = data.train, kernel = "radial", gamma = 1, cost = 10)
plot(svm.radial, data.train)
table(predict = predict(svm.radial, data.train), truth = data.train$Y)

plot(svm.linear, data.test)
table(predict = predict(svm.linear, data.test), truth = data.test$Y)

plot(svm.poly, data.test)
table(predict = predict(svm.poly, data.test), truth = data.test$Y)

plot(svm.radial, data.test)
table(predict = predict(svm.radial, data.test), truth = data.test$Y)

################################# Logistic Regression #################################

dev.off()
set.seed(1)
x1=runif(500)-0.5
x2=runif(500)-0.5
y=1*(x1^2-x2^2 > 0)

plot(x1, x2, xlab = "X1", ylab = "X2", col = (3 - y), pch = (5 - y))

data <- data.frame(x1,x2,y)
train <- sample(500, 400)
data.train <- data[train, ]
data.test <- data[-train, ]


lr.fit <- glm(y~., data=data, family=binomial);
summary(lr.fit);
probs.lr <- predict(lr.fit, data, type="response");
pred.lr <- rep(0,length(probs.lr));
pred.lr[probs.lr>0.5] <- 1;


plot(x1, x2, xlab = "X1", ylab = "X2", col = (3 - pred.lr), pch = (5 - pred.lr))

logitnl.fit <- glm(y ~ poly(x1, 2) + poly(x2, 2) + I(x1 * x2), family = "binomial")

probs.lr <- predict(logitnl.fit, type="response");
pred.lr <- rep(0, data, length(probs.lr));
pred.lr[probs.lr>0.5] <- 1;
plot(x1, x2, xlab = "X1", ylab = "X2", col = (3 - pred.lr), pch = (5 - pred.lr))


plot(x1[train], x2[train], xlab = "X1", ylab = "X2", col = (3 - pred.lr[train]), pch = (5 - pred.lr[train]))

data <- data.frame(x1,x2,y=as.factor(y))

svm.linear <- svm(y ~ x1 + x2, data = data, kernel = "linear", cost = 0.01)
pred <- predict(svm.linear, data);
plot(x1[pred==0], x2[pred==0], xlab = "X1", ylab = "X2", col = "blue")
points(x1[pred==1], x2[pred==1], col = "red")
plot(svm.linear, data)


data <- data.frame(x1,x2,y=as.factor(y))

svm.radial <- svm(y ~ x1 + x2, data = data, kernel = "radial", gamma = 1)
pred <- predict(svm.radial, data);
plot(x1[pred==0], x2[pred==0], xlab = "X1", ylab = "X2", col = 2, pch=2)
points(x1[pred==1], x2[pred==1], col = 3,pch=3)
plot(svm.radial, data.train)

################################# Barely separable classes #################################

set.seed(1)
x1 <- runif(500, 0, 90)
y1 <- runif(500, x1 + 10, 100)
x1_noise <- runif(50, 20, 80)
y1_noise <- 5/4 * (x1_noise - 10) + 0.1

x0 <- runif(500, 10, 100)
y0 <- runif(500, 0, x0 - 10)
x0_noise <- runif(50, 20, 80)
y0_noise <- 5/4 * (x0_noise - 10) - 0.1

class1 <- seq(1, 550)
class2 <- seq(551, 1100)
x <- c(x1, x1_noise, x0, x0_noise)
y <- c(y1, y1_noise, y0, y0_noise)

plot(x[class1], y[class1], col = "blue", pch = "+", ylim = c(0, 100))
points(x[class2], y[class2], col = "red", pch = "*")

z <- rep(1,550)
z <- c(z,rep(0,550))
z <- as.factor(z)
data <- data.frame(x,y,z)
train <- sample(1100, 800)
data.train <- data[train, ]
data.test <- data[-train, ]
cost <- c(0.001, 0.1, 1, 5, 10, 50, 100, 1000)


set.seed(2)
tune.out <- tune(svm, z ~ ., data = data, kernel = "linear", ranges = list(cost = cost))
summary(tune.out)

accuracy <- rep(NA,length(cost))
for(i in 1:length(cost)){
  svm.fit <- svm(z ~ x + y, data = data.train, kernel = "linear", cost = cost[i])
  pred <- predict(svm.fit, data.train);
  accuracy[i] <- mean(pred==data.train$z)
}
plot(accuracy)

################################# Auto dataset #################################

y <- ifelse(Auto$mpg > median(Auto$mpg),yes = 1, no = 0) 
y <- as.factor(y)
set.seed(1)
cost <- c(0.001, 0.1, 1, 10, 100, 1000)
gamma <- c()
degree <- c()
Auto <- data.frame(Auto, y)
tune.out <- tune(svm, y ~ ., data = Auto, kernel = "linear", ranges = list(cost = cost))
summary(tune.out)

set.seed(1)
tune.out <- tune(svm, y ~ ., data = Auto, kernel = "polynomial", ranges = list(cost = c(0.01, 0.1, 1, 5, 10, 100), degree = c(2, 3, 4)))
summary(tune.out)

set.seed(1)
tune.out <- tune(svm, y ~ ., data = Auto, kernel = "radial", ranges = list(cost = c(0.01, 0.1, 1, 5, 10, 100), gamma = c(0.01, 0.1, 1, 5, 10, 100)))
summary(tune.out)

svm.linear <- svm(y ~ ., data = Auto, kernel = "linear", cost = 1)
svm.poly <- svm(y ~ ., data = Auto, kernel = "polynomial", cost = 100, degree = 2)
svm.radial <- svm(y ~ ., data = Auto, kernel = "radial", cost = 100, gamma = 0.01)
plotpairs = function(fit) {
  for (name in names(Auto)[!(names(Auto) %in% c("mpg", "mpglevel", "name"))]) {
    plot(fit, Auto, as.formula(paste("mpg~", name, sep = "")))
  }
}
plotpairs(svm.linear)
plotpairs(svm.poly)
plotpairs(svm.radial)

################################# OJ dataset #################################


library(ISLR);
library(boot);
library(MASS);
attach(Default);
set.seed(1);
glm.fit <- glm(default ~ income + balance, data = Default, family = "binomial");
summary(glm.fit);
train <- sample(dim(Default)[1], dim(Default)[1] / 2);
glm.fit <- glm(default ~ income + balance, data = Default, family = "binomial", subset = train);
summary(glm.fit);
probs <- predict(glm.fit, newdata = Default[-train, ], type = "response");
glm.pred <- rep("No", length(probs));
glm.pred[probs > 0.5] <- "Yes";
table(glm.pred, default[-train]);
mean(glm.pred!=default[-train]);
#################################################
train <- sample(dim(Default)[1], dim(Default)[1] / 2);
glm.fit <- glm(default ~ income + balance + student, data = Default, family = "binomial", subset = train);
summary(glm.fit);
probs <- predict(glm.fit, newdata = Default[-train, ], type = "response");
glm.pred <- rep("No", length(probs));
glm.pred[probs > 0.5] <- "Yes";
table(glm.pred, default[-train]);
mean(glm.pred!=default[-train]);
##################################################################################################

set.seed(1);
train <- sample(dim(Default)[1], dim(Default)[1] / 2);
glm.fit <- glm(default ~ income + balance, data = Default, family = "binomial", subset = train);
summary(glm.fit);
boot.fn <- function(data, index) {
    fit <- glm(default ~ income + balance, data = data, family = "binomial", subset = index)
    return (coef(fit))
}
boot(Default, boot.fn, 1000);
##################################################################################################
set.seed(1)
attach(Weekly)
glm.fit <- glm(Direction ~ Lag1 + Lag2, data = Weekly, family = "binomial");
summary(glm.fit);
error <- rep(0, dim(Weekly)[1])
for (i in 1:dim(Weekly)[1]) {
    fit <- glm(Direction ~ Lag1 + Lag2, data = Weekly[-i, ],  family = "binomial")
    pred.up <- predict.glm(fit, Weekly[i, ], type = "response") > 0.5
    true.up <- Weekly[i, ]$Direction == "Up"
    if (pred.up != true.up)
        error[i] <- 1;
}
mean(error);
##################################################################################################

set.seed(1);
y <- rnorm(100);
x <- rnorm(100);
y <- x - 2 * x^2 + rnorm(100);
plot(x, y);

XY <- data.frame(x, y);
glm.fit1 <- glm(y ~ x);
cv.glm(XY, glm.fit)$delta[1];

glm.fit2 <- glm(y ~ poly(x, 2));
cv.glm(XY, glm.fit2)$delta[1];


glm.fit3 <- glm(y ~ poly(x, 3));
cv.glm(Data, glm.fit3)$delta[1];


glm.fit4 <- glm(y ~ poly(x, 4));
cv.glm(Data, glm.fit4)$delta[1];
##################################################################################################
attach(Boston);
mu.hat <- mean(medv);
sd(medv);
se.hat <- sd(medv) / sqrt(dim(Boston)[1]);
set.seed(1);

boot.fn <- function(data, index) {
	mu <- mean(data[index])
    return (mu)
}
boot(medv,boot.fn,1000);

boot.fn <- function(data, index) {
	mu <- mean(data[index])
	se <- sd(data[index]) / sqrt(length(index));
    return (c(mu-2*se,mu+2*se))
}
boot(medv,boot.fn,1000);
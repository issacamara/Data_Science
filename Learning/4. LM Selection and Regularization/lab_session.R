library(ISLR)
library(leaps)
library(glmnet)
library(pls)
########################### Best Subset selection ###########################
set.seed(1)
X <- rnorm(100);
b0 <- 0.5
b1 <- 1
b2 <- 1.5
b3 <- 2
eps <- rnorm(100)
Y <- b0 + b1*X + b2*X^2 + b3*X^3 + eps;
data.full <- data.frame(x=X,y=Y);
attach(data.full)
regfit.full <- regsubsets(y ~ x + I(x^2) + I(x^3) + I(x^4) + I(x^5) + I(x^6) 
                          + I(x^7) + I(x^8) + I(x^9) + I(x^10), 
                          data = data.full, nvmax = 10)
regfit.summary <- summary(regfit.full)

par(mfrow = c(2, 2))
plot(regfit.summary$cp, xlab = "Number of variables", ylab = "C_p", type = "l")
points(which.min(regfit.summary$cp), 
       regfit.summary$cp[which.min(regfit.summary$cp)], 
       col = "red", cex = 2, pch = 20)

plot(regfit.summary$bic, xlab = "Number of variables", ylab = "BIC", type = "l")
points(which.min(regfit.summary$bic), 
       regfit.summary$bic[which.min(regfit.summary$bic)], 
       col = "red", cex = 2, pch = 20)

plot(regfit.summary$adjr2, xlab = "Number of variables", 
     ylab = "Adjusted R^2", type = "l")
points(which.max(regfit.summary$adjr2), 
       regfit.summary$adjr2[which.max(regfit.summary$adjr2)], 
       col = "red", cex = 2, pch = 20)
dev.off()
########################### Forward selection ###########################
regfit.forward <- regsubsets(y ~ x + I(x^2) + I(x^3) + I(x^4) + I(x^5) + I(x^6) 
                          + I(x^7) + I(x^8) + I(x^9) + I(x^10), 
                          data = data.full, nvmax = 10, method = "forward")
regfit.summary <- summary(regfit.forward)

par(mfrow = c(2, 2))
plot(regfit.summary$cp, xlab = "Number of variables", ylab = "C_p", type = "l")
points(which.min(regfit.summary$cp), 
       regfit.summary$cp[which.min(regfit.summary$cp)], 
       col = "red", cex = 2, pch = 20)

plot(regfit.summary$bic, xlab = "Number of variables", ylab = "BIC", type = "l")
points(which.min(regfit.summary$bic), 
       regfit.summary$bic[which.min(regfit.summary$bic)], 
       col = "red", cex = 2, pch = 20)

plot(regfit.summary$adjr2, xlab = "Number of variables", 
     ylab = "Adjusted R^2", type = "l")
points(which.max(regfit.summary$adjr2), 
       regfit.summary$adjr2[which.max(regfit.summary$adjr2)], 
       col = "red", cex = 2, pch = 20)
dev.off()
########################### Backward selection ###########################
regfit.backward <- regsubsets(y ~ x + I(x^2) + I(x^3) + I(x^4) + I(x^5) + I(x^6) 
                          + I(x^7) + I(x^8) + I(x^9) + I(x^10), 
                          data = data.full, nvmax = 10, method = "backward")
regfit.summary <- summary(regfit.backward)

par(mfrow = c(2, 2))
plot(regfit.summary$cp, xlab = "Number of variables", ylab = "C_p", type = "l")
points(which.min(regfit.summary$cp), 
       regfit.summary$cp[which.min(regfit.summary$cp)], 
       col = "red", cex = 2, pch = 20)

plot(regfit.summary$bic, xlab = "Number of variables", ylab = "BIC", type = "l")
points(which.min(regfit.summary$bic), 
       regfit.summary$bic[which.min(regfit.summary$bic)], 
       col = "red", cex = 2, pch = 20)

plot(regfit.summary$adjr2, xlab = "Number of variables", 
     ylab = "Adjusted R^2", type = "l")
points(which.max(regfit.summary$adjr2), 
       regfit.summary$adjr2[which.max(regfit.summary$adjr2)], 
       col = "red", cex = 2, pch = 20)
dev.off()
########################### LASSO ###########################
xmat <- model.matrix(y ~ x + I(x^2) + I(x^3) + I(x^4) + I(x^5) + I(x^6) + I(x^7) + I(x^8) + I(x^9) + I(x^10), data = data.full)[, -1]
cv.lasso <- cv.glmnet(xmat, Y, alpha = 1)
plot(cv.lasso)

best_lambda <- cv.lasso$lambda.min
fit.lasso <- glmnet(xmat, Y, alpha = 1)
predict(fit.lasso, s = best_lambda, type = "coefficients")[1:11, ]

b7 <- 7
Y <- b0 + b7 * X^7 + eps
data.full <- data.frame(y = Y, x = X)
regfit.full <- regsubsets(y ~ x + I(x^2) + I(x^3) + I(x^4) + I(x^5) + I(x^6) + I(x^7) + I(x^8) + I(x^9) + I(x^10), data = data.full, nvmax = 10)
reg.summary <- summary(regfit.full)
par(mfrow = c(2, 2))
plot(reg.summary$cp, xlab = "Number of variables", ylab = "C_p", type = "l")
points(which.min(reg.summary$cp), reg.summary$cp[which.min(reg.summary$cp)], col = "red", cex = 2, pch = 20)
plot(reg.summary$bic, xlab = "Number of variables", ylab = "BIC", type = "l")
points(which.min(reg.summary$bic), reg.summary$bic[which.min(reg.summary$bic)], col = "red", cex = 2, pch = 20)
plot(reg.summary$adjr2, xlab = "Number of variables", ylab = "Adjusted R^2", type = "l")
points(which.max(reg.summary$adjr2), reg.summary$adjr2[which.max(reg.summary$adjr2)], col = "red", cex = 2, pch = 20)

xmat <- model.matrix(y ~ x + I(x^2) + I(x^3) + I(x^4) + I(x^5) + I(x^6) + I(x^7) + I(x^8) + I(x^9) + I(x^10), data = data.full)[, -1]
cv.lasso <- cv.glmnet(xmat, Y, alpha = 1)
fit.lasso <- glmnet(xmat, Y, alpha = 1)
predict(fit.lasso, s = best_lambda, type = "coefficients")[1:11, ]
dev.off()
########################### College dataset ###########################
data("College")
set.seed(1)
train <- sample(c(TRUE,FALSE), nrow(College),rep=TRUE)
test <- (!train)
fit.lm <- lm(Apps ~ ., data = College[train,])
pred.lm <- predict(fit.lm, College[test,])
mean((pred.lm - College[test,]$Apps)^2)

train.mat <- model.matrix(Apps ~ ., data = College[train,])
test.mat <- model.matrix(Apps ~ ., data = College[test,])
grid <- 10 ^ seq(4, -2, length = 100)
fit.ridge <- glmnet(train.mat, College[train,]$Apps, alpha = 0, lambda = grid)
cv.ridge <- cv.glmnet(train.mat, College[train,]$Apps, alpha = 0, lambda = grid)
bestlambda.ridge <- cv.ridge$lambda.min
bestlambda.ridge
pred.ridge <- predict(fit.ridge, s = bestlambda.ridge, newx = test.mat)
mean((pred.ridge - College[test,]$Apps)^2)

fit.lasso <- glmnet(train.mat, College[train,]$Apps, alpha = 1, lambda = grid, thresh = 1e-12)
cv.lasso <- cv.glmnet(train.mat, College[train,]$Apps, alpha = 1, lambda = grid, thresh = 1e-12)
bestlambda.lasso <- cv.lasso$lambda.min
bestlambda.lasso
pred.lasso <- predict(fit.lasso, s = bestlambda.lasso, newx = test.mat)
mean((pred.lasso - College[test,]$Apps)^2)

fit.pcr <- pcr(Apps ~ ., data = College[train,], scale = TRUE, validation = "CV")
validationplot(fit.pcr, val.type = "MSEP")
pred.pcr <- predict(fit.pcr, College[test,], ncomp = 10)
mean((pred.pcr - College[test,]$Apps)^2)
dev.off()

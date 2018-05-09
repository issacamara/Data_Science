simple_linear_regression = function(){
	lm.fit1 <- lm(mpg~horsepower, data=Auto);
	summary(lm.fit1);
	predict(lm.fit1,data.frame(horsepower=98), interval="confidence");
	predict(lm.fit1,data.frame(horsepower=98), interval="prediction");
	plot(Auto$horsepower, Auto$mpg, main = "Scatterplot of mpg vs. horsepower", xlab = "horsepower", ylab = "mpg", col = "blue");
	abline(lm.fit1, col = "red");
	par(mfrow = c(2, 2));
	plot(lm.fit1);
}

multiple_linear_regression1 = function(){
	pairs(Auto);
	cor(Auto[-which(names(Auto)=="name")])
	lm.fit2 <- lm(mpg~.-names, data=Auto);
	summary(lm.fit2);
	predict(lm.fit2,data.frame(horsepower=98), interval="prediction");
	plot(Auto$horsepower, Auto$mpg, main = "Scatterplot of mpg vs. horsepower", xlab = "horsepower", ylab = "mpg", col = "blue");
	abline(lm.fit2, col = "red");
	par(mfrow = c(2, 2));
	plot(lm.fit2);
	lm.fit3 <- lm(mpg ~ cylinders * displacement+displacement * weight, data = Auto);
	summary(lm.fit3);
	par(mfrow = c(2, 2));
	plot(log(Auto$horsepower), Auto$mpg);
	plot(sqrt(Auto$horsepower), Auto$mpg);
	plot((Auto$horsepower)^2, Auto$mpg)

}

multiple_linear_regression2 = function(){
	lm.fit4 <- lm(Sales~Price+Urban+US, data=Carseats);
	summary(lm.fit4);
	par(mfrow = c(2, 2));
	plot(lm.fit4);
}

t_statistic = function(){
	set.seed (1);
	x <- rnorm(100);
	y <- 2*x+rnorm(100);
	lm.fit5 <- lm(y~x+0);
	summary(lm.fit5);
	lm.fit5 <- lm(x~y+0);
	summary(lm.fit5);
	n <- length(x);
	t <- sqrt(n - 1)*(x %*% y)/sqrt(sum(x^2) * sum(y^2) - (x %*% y)^2);
	lm.fit5 <- lm(x~y);
	summary(lm.fit5);
	lm.fit5 <- lm(y~x);
	summary(lm.fit5);
}
lin_reg_without_intercept = function(){
	#x^2!=y^2
	set.seed(1);
	x <- 1:100;
	y <- 2*x + rnorm(100);
	lm.fit6 <- lm(y~x + 0);
	lm.fit6$coefficients;
	lm.fit6 <- lm(x~y + 0);
	lm.fit6$coefficients;
	#x^2=y^2
	x <- 1:100;
	y <- 100:1;
	lm.fit6 <- lm(y~x+0);
	lm.fit6$coefficients;
	lm.fit6 <- lm(x~y+0);
	lm.fit6$coefficients;
}

lin_reg_simulation = function(){
	set.seed(1);
	x <- rnorm(100);
	eps <- rnorm(100,sd=sqrt(0.25));
	y <- 0.5*x-1+eps;
	lm.fit7 <- lm(y~x);
	summary(lm.fit7);
	plot(x, y);
	abline(lm.fit7, col = "red");
	abline(-1, 0.5, col = "blue");
	legend("topleft", c("Least square", "Regression"), col = c("red", "blue"), lty = c(1, 1));
	lm.fit7 <- lm(y~x+I(x^2));
	summary(lm.fit7);
	confint(lm.fit7);
	############################################## Less noise
	set.seed(1);
	x <- rnorm(100);
	eps <- rnorm(100,sd=sqrt(0.01));
	y <- 0.5*x-1+eps;
	lm.fit7 <- lm(y~x);
	plot(x, y);
	abline(lm.fit7, col = "red");
	abline(-1, 0.5, col = "blue");
	legend("topleft", c("Least square", "Regression"), col = c("red", "blue"), lty = c(1, 1));
	confint(lm.fit7);
	############################################## More noise
	set.seed(1);
	x <- rnorm(100);
	eps <- rnorm(100,sd=1);
	y <- 0.5*x-1+eps;
	lm.fit7 <- lm(y~x);
	plot(x, y);
	abline(lm.fit7, col = "red");
	abline(-1, 0.5, col = "blue");
	legend("topleft", c("Least square", "Regression"), col = c("red", "blue"), lty = c(1, 1));
	confint(lm.fit7);
}

collinearity = function(){
	set.seed(1);
	x1 <- runif(100);
	x2 <- 0.5*x1+rnorm(100)/10
	y <- 2+2*x1+0.3*x2+rnorm(100);
	cor(x1,x2);
	plot(x1,x2);
	lm.fit8 <- lm(y~x1+x2);
	summary(lm.fit8);
	lm.fit8 <- lm(y~x1);
	summary(lm.fit8);
	lm.fit8 <- lm(y~x2);
	summary(lm.fit8);
	x1 <- c(x1, 0.1);
	x2 <- c(x2, 0.8);
	y <- c(y, 6);
	lm.fit8 <- lm(y~x1+x2);
	summary(lm.fit8);
	par(mfrow = c(2, 2));
	plot(lm.fit8);
	lm.fit8 <- lm(y~x1);
	summary(lm.fit8);
	lm.fit8 <- lm(y~x2);
	summary(lm.fit8);


}
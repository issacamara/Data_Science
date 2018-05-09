library(ISLR);
library(MASS);
library(class);
names(Weekly);
dim(Weekly);
summary(Weekly);
pairs(Weekly);
cor(Smarket[,-9]);
attach(Weekly);
plot(Volume);

# 10
	# Logistic Regression
glm.fit <- glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume, data=Weekly, family=binomial);
summary(glm.fit);
probs.glm <- predict(glm.fit, type="response");
pred.glm <- rep("Down",length(probs.glm));
pred.glm[probs.glm>0.5] <- "Up";
table(pred.glm, Direction);

mean(pred.glm==Direction);

train <- (Year < 2009);
Weekly.2009_2010 <- Weekly[!train, ];
Direction.20092010 <- Direction[!train];
glm.fit2 <- glm(Direction ~ Lag2, data = Weekly, family = binomial, subset = train)
summary(glm.fit2);
probs.glm2 <- predict(glm.fit2, Weekly.2009_2010, type="response");
pred.glm2 <- rep("Down",length(probs.glm2));
pred.glm2[probs.glm2>0.5] <- "Up";
table(pred.glm2, Direction.20092010);	

mean(pred.glm2==Direction.20092010);	
	
	# LDA

lda.fit <- lda(Direction~Lag2,data=Weekly);
summary(lda.fit);
probs.lda <- predict(lda.fit, Weekly.2009_2010, type="response");
pred.lda <- rep("Down",length(probs.lda$class));
pred.lda[probs.lda$posterior>0.5] <- "Up";
table(pred.lda, Direction.20092010);
	
	# QDA
	
qda.fit <- qda(Direction~Lag2,data=Weekly, subset=train);
summary(qda.fit);
probs.qda <- predict(qda.fit, Weekly.2009_2010,type="response");
pred.qda <- rep("Down",length(probs.qda$class));
pred.qda[probs.qda$posterior>0.5] <- "Up";
table(pred.qda, Direction.20092010);	
	
	# KNN
test.X =cbind(Weekly$Lag2[train]);
train.X = cbind(Weekly$Lag2[train]);
test.X =cbind(Weekly$Lag2[!train]);
train.Direction = cbind(Weekly$Direction[train]);
knn.pred=knn(train.X,test.X,train.Direction ,k=1);
levels(knn.pred)[1] <- "Down";
levels(knn.pred)[2] <- "Up";
table(knn.pred,Direction.20092010);

	# High - Low Mileage
mpg01 <- rep(1,length(mpg));
mpg01[mpg<median(mpg)] <- 0;
Auto <- data.frame(Auto, mpg01);
cor(Auto[,-9]);
pairs(Auto[,-9]);
boxplot(cylinders ~ mpg01, data = Auto, main = "Cylinders vs mpg01");
boxplot(displacement ~ mpg01, data = Auto, main = "Displacement vs mpg01");
boxplot(horsepower ~ mpg01, data = Auto, main = "Horsepower vs mpg01");
boxplot(weight ~ mpg01, data = Auto, main = "Weight vs mpg01");
boxplot(acceleration ~ mpg01, data = Auto, main = "Acceleration vs mpg01");
boxplot(year ~ mpg01, data = Auto, main = "Year vs mpg01");

train <- year %% 2 == 0;
Auto.train <- Auto[train,];
Auto.test <- Auto[!train,];
mpg01.test <- mpg01[!train];
mpg01.train <- mpg01[train];

lda.fit <- lda(mpg01~ cylinders + displacement + horsepower + weight + acceleration + year,
data=Auto, subset = train);
summary(lda.fit);
pred.lda <- predict(lda.fit, Auto.test, type="response");
table(pred.lda$class, mpg01.test);
mean(pred.lda$class == mpg01.test);

qda.fit <- qda(mpg01~ cylinders + displacement + horsepower + weight + acceleration + year,
data=Auto, subset = train);
summary(qda.fit);
pred.qda <- predict(qda.fit, Auto.test, type="response");
table(pred.qda$class, mpg01.test);
mean(pred.qda$class == mpg01.test);

glm.fit <- glm(mpg01~ cylinders + displacement + horsepower + weight + acceleration + year,
data=Auto, subset = train);
summary(glm.fit);
probs.glm <- predict(glm.fit, Auto.test, type="response");
pred.glm <- rep(0,length(probs.glm));
pred.glm[probs.glm>0.5] <- 1;
table(pred.glm, mpg01.test);	
mean(pred.glm == mpg01.test);
train.X <- cbind(cylinders, weight, displacement, horsepower)[train, ];
test.X <- cbind(cylinders, weight, displacement, horsepower)[!train, ];
train.mpg01 <- mpg01[train];
set.seed(1);
pred.knn <- knn(train.X, test.X, train.mpg01, k = 1);
table(pred.knn, mpg01.test);
mean(pred.knn==mpg01.test);

pred.knn <- knn(train.X, test.X, train.mpg01, k = 10);
table(pred.knn, mpg01.test);
mean(pred.knn==mpg01.test);

pred.knn <- knn(train.X, test.X, train.mpg01, k = 100);
table(pred.knn, mpg01.test);
mean(pred.knn==mpg01.test);

Power = function(x,n){
	res <- x^n;
	return (res);
}

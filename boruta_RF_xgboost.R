# Libraries
library(Boruta)
library(mlbench)
library(caret)
library(randomForest)
library(tidyr)
data <- read.csv("C:/Users/varun/Downloads/oec.csv", header=TRUE, dec=",", comment.char="#")

data$ListsPlanetIsOn<-recode(data$ListsPlanetIsOn, "c('Confirmed planets','Confirmed planets, Planets in binary systems, P-type, Planets in globular clusters','Planets in binary systems, S-type, Confirmed planets')='Confirmed Planets';
                            c('Controversial', 'Kepler Objects of Interest') = 'Controversial'")

# Data
data$ListsPlanetIsOn <- factor(c(data$ListsPlanetIsOn))
data[is.na(data)] <- 0
data[2:24] <- as.numeric(as.factor(data[2:24]))

data[1]<-as.character(as.factor(data[1]))
str(data)

summary(data)


# Feature Selection
set.seed(111)
boruta <- Boruta(ListsPlanetIsOn ~., data = data, doTrace = 2, maxRuns = 500)
print(boruta)
plot(boruta, las = 2, cex.axis = 0.7)
plotImpHistory(boruta)


# Data Partition
set.seed(222)
ind <- sample(2, nrow(data), replace = T, prob = c(0.6, 0.4))
train <- data[ind==1,]
test <- data[ind==2,]
View(train)
# Random Forest Model
set.seed(333)
rf60 <- randomForest(ListsPlanetIsOn ~., data = train)

# Prediction & Confusion Matrix - Test
p <- predict(rf60, test)
confusionMatrix(p, test$ListsPlanetIsOn)




data_variables <- as.matrix(train[,-25])
data_label <- train[,"ListsPlanetIsOn"]
data_matrix <- xgb.DMatrix(data = data_variables, label = data_label)

numberOfClasses <- length(unique(train$ListsPlanetIsOn))
xgb_params <- list("objective" = "multi:softmax",eta=0.04,gamma=0.6,max_depth=10,eval_metric="mlogloss","num_class" = numberOfClasses)
xgbcv <- xgb.cv( params = xgb_params,
                 data = data_matrix,
                 nrounds = 400, nfold = 10, showsd = T, stratified = T,
                 print.every.n = 10,
                 early.stop.round = 20, 
                 maximize = F)
nround    <- xgbcv$best_iteration # number of XGBoost rounds
cv.nfold  <- 10

# Fit cv.nfold * cv.nround XGB models and save OOF predictions
bst_model <- xgb.train(params = xgb_params,
                       data = data_matrix,
                       nrounds = nround)

test_matrix<-xgb.DMatrix(data = as.matrix(test))
predictionsXGBoost<-predict(bst_model,newdata=test_matrix)
table(predictionsXGBoost,temptest$ListsPlanetIsOn)

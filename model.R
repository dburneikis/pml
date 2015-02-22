library(caret)
library(dplyr)
library(randomForest)

##Load files into R
#download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile="training.csv")
#download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile="testing.csv")
train <- read.csv("training.csv", na.strings=c("","NA"))
test <- read.csv("testing.csv", na.strings=c("","NA"))

##Create index of names that include Euler angles (roll, pitch, yaw) and raw accelerometer, gyroscope,
##and magnetometer readings. In other words, exclude all "calculated" variables.
names <- names(train)
roll <- grep("^roll", names)
pitch <- grep("^pitch", names)
yaw <- grep("^yaw", names)
accel <- grep("^accel", names)
gyro <- grep("^gyro", names)
magnet <- grep("^magnet", names)
variable_index <- sort(c(2:7, 160, roll, pitch, yaw, accel, gyro, magnet))
cleanTrain <- train[,variable_index]

##Convert all measurement variable classes to numeric
cleanTrain[,7:54] <- sapply(cleanTrain[,7:54], as.character)
cleanTrain[,7:54] <- sapply(cleanTrain[,7:54], as.numeric)

##Construct a correlation matrix and remove variables that correlate to other variables in the data with a
##cutoff threshold of 0.75
correlationMatrix <- cor(cleanTrain[,7:54])
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=.75)
redundantVars <- colnames(correlationMatrix)[highlyCorrelated]
remove <- which(colnames(cleanTrain) %in% redundantVars)
cleanTrain2 <- select(cleanTrain, -remove)

##Remove summary rows
cleanTrain3 <- filter(cleanTrain2, new_window=="no")
cleanTrain4 <- select(cleanTrain3, 1:3, 7:39)

#Model***
rows <- sample(nrow(cleanTrain3), size=5000, replace=FALSE)
rows2 <- sample(nrow(cleanTrain3), size=5000, replace=FALSE)
rows3 <- sample(nrow(cleanTrain4), size=5000, replace=FALSE)
rows4 <- sample(nrow(cleanTrain4), size=5000, replace=FALSE)
#model <- randomForest(classe ~., data=cleanTrain3[rows,], ntree=20, importance=TRUE, proximity=TRUE)
#model2 <- randomForest(classe ~., data=cleanTrain3[rows,], ntree=50, importance=TRUE, proximity=TRUE)
#model3 <- randomForest(classe ~., data=cleanTrain3[rows,], ntree=100, importance=TRUE, proximity=TRUE)
model4 <- randomForest(classe ~., data=cleanTrain4[rows3,], ntree=100, importance=TRUE, proximity=TRUE)

##Test
cleanTest <- test[,variable_index]
cleanTest[,7:54] <- sapply(cleanTest[,7:54], as.character)
cleanTest[,7:54] <- sapply(cleanTest[,7:54], as.numeric)
cleanTest2 <- select(cleanTest, -remove)
cleanTest3 <- filter(cleanTest2, new_window=="no")
cleanTest4 <- select(cleanTest3, 1:3, 7:39)

prediction <- predict(model4, cleanTrain4[rows,])
confusionMatrix(prediction, cleanTrain4[rows,36])

##Write answers
#pml_write_files = function(x){
#  n = length(x)
#  for(i in 1:n){
#    filename = paste0("problem_id_",i,".txt")
#    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
#  }
#}


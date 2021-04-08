rm(list = ls())
setwd("D:/Studies/Big Data Analytics/Project")
library(dplyr)

# The data was already split in train and test when downloaded from Kaggle
Airline.survey <- read.csv("Airline_survey.csv")

# Removing first 2 columns as sr.no & id are not relevant for analysis.
Airline.survey <- Airline.survey[(-c(1,2))]
glimpse(Airline.survey)

# check for missing values
sum(is.na(Airline.survey))

# We have 393 missing values in a dataframe of 129880 rows. 
# Since the magnitude of missing values is very low compared to total dataset, we can drop the rows with na values
Airline.survey <- na.omit(Airline.survey)

# visualizing satisfaction across customer segments
library(ggplot2)
ggplot(Airline.survey, aes(x= satisfaction,  group=1)) + 
  geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
  labs(y = "Percent of passengers", fill="Satisfaction") +
  scale_y_continuous(labels = scales::percent)

ggplot(Airline.survey, aes(x= satisfaction,  group=Type.of.Travel)) + 
  geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") +
  geom_text(aes( label = scales::percent(..prop..),
                 y= ..prop.. ), stat= "count", vjust = -.5) +
  labs(y = "Percent of passengers", fill="Satisfaction") +
  facet_grid(~Type.of.Travel) +
  scale_y_continuous(labels = scales::percent)

ggplot(Airline.survey, aes(x= satisfaction,  group=Customer.Type)) + 
  geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") +
  geom_text(aes( label = scales::percent(..prop..),
                 y= ..prop.. ), stat= "count", vjust = -.5) +
  labs(y = "Percent of passengers", fill="Satisfaction") +
  facet_grid(~Customer.Type) +
  scale_y_continuous(labels = scales::percent)


# Data transformation
Airline.survey$satisfaction <- ifelse(Airline.survey$satisfaction == "satisfied",1,0)
library(caret)
dmy <- dummyVars(" ~ .", data = Airline.survey, fullRank = T)
Airline.survey <- data.frame(predict(dmy, newdata = Airline.survey))

glimpse(Airline.survey)

# Finding correlation
library(corrplot)
Airline.survey.cor = cor(Airline.survey)
corrplot(Airline.survey.cor, type = "upper")

for (i in 1:nrow(Airline.survey.cor)){
  correlations <-  which((Airline.survey.cor[i,] > 0.8) & (Airline.survey.cor[i,] != 1))
  
  if(length(correlations)> 0){
    print(colnames(Airline.survey)[i])
    print(correlations)
  }
}

# Departure and Arrival Delay in minutes have high correlation. Dropping Departure Delay in minutes column
Airline.survey = subset(Airline.survey, select = -c(Departure.Delay.in.Minutes))


pcs <- prcomp(Airline.survey[,-23], scale = TRUE)
summary(pcs)
PoV <- pcs$sdev^2/sum(pcs$sdev^2)
sum(PoV[1:15])

scores.cor <- pcs$x
data_pcs<-data.frame(Airline.survey[23],scores.cor[,1:15])
train.index <- sample(nrow(data_pcs), nrow(data_pcs)*0.7) 
train.df <- data_pcs[train.index,]
valid.df <- data_pcs[-train.index,]

#logistic regression
library(caret)
logit_spam <- glm(as.factor(satisfaction) ~ ., data = train.df, family = "binomial")
logit_spam_pred_prob <- predict(logit_spam, valid.df,type = "response")
logit_spam_pred <- ifelse(logit_spam_pred_prob>0.5,1,0)
confusionMatrix(as.factor(logit_spam_pred),as.factor(valid.df$satisfaction))

#Naive Bayes
library(e1071)
nb_spam <- naiveBayes(as.factor(satisfaction)~.,data=train.df)
nb_spam_pred_prob <- predict(nb_spam, valid.df, type="raw")
nb_spam_pred <- predict(nb_spam, valid.df)
confusionMatrix(as.factor(nb_spam_pred), as.factor(valid.df$satisfaction))

#KNN
set.seed(123)
library(kknn) 
kknn.train <- train.kknn(satisfaction ~ ., data = train.df, kmax = 7, distance = 2, kernel = "epanechnikov")
kknn.pred.prob <- predict(kknn.train, newdata = valid.df)
kknn.pred <- ifelse(kknn.pred.prob>0.5,1,0)
confusionMatrix(as.factor(kknn.pred), as.factor(valid.df$satisfaction))

##Neural Network
library(neuralnet) 
n<-names(train.df)
f<-as.formula(paste("satisfaction~",paste(n[!n %in% "satisfaction"],
                               collapse="+")))
nn<-neuralnet(f,data=train.df,hidden = c(4,2),linear.output = FALSE, stepmax = 1e7)
plot(nn)

preds.valid<-compute(nn,valid.df[,-c(1)])
preds.valid.class<-ifelse(preds.valid$net.result>0.5,1,0)
confusionMatrix(as.factor(preds.valid.class),as.factor(valid.df$satisfaction))

# Ensemble Using Averaging
En_Avg_Prob<-(nb_spam_pred_prob[,2] + kknn.pred.prob + logit_spam_pred_prob)/3
En_Avg_Class<-ifelse(En_Avg_Prob>0.5,1,0)
confusionMatrix(as.factor(En_Avg_Class),as.factor(valid.df$satisfaction))


library(tidyverse)
library(ggthemes)
library(corrplot)
library(GGally)
library(DT)
library(caret)
# install.packages("e1071")
# library(e1071)
library(ROSE)


loan = read_csv("D:\\loan.csv" , na = "")
colnames(loan)

loan = loan %>%
  select(loan_status , loan_amnt , int_rate , grade , emp_length , home_ownership , 
         annual_inc , term)

sapply(loan , function(x) sum(is.na(x)))

loan = loan %>%
  filter(!is.na(annual_inc) , 
         !(home_ownership %in% c('NONE' , 'ANY')) , 
         emp_length != 'n/a')

loan %>%
  count(loan_status) %>%
  ggplot(aes(x = reorder(loan_status , desc(n)) , y = n , fill = n)) + 
    geom_col() + 
    coord_flip() + 
    labs(x = 'Loan Status' , y = 'Count')

loan = loan %>%
  mutate(loan_outcome = ifelse(loan_status %in% c('Charged Off' , 'Default'), 
        1, 
        ifelse(loan_status == 'Fully Paid' , 0 , 'No info')
        ))

barplot(table(loan$loan_outcome) , col = 'lightblue')

loan2 = loan %>%
  select(-loan_status) %>%
  filter(loan_outcome %in% c(0 , 1))

ggplot(loan2 , aes(x = grade , y = int_rate , fill = grade)) + 
  geom_boxplot() + 
  theme_igray() + 
  labs(y = 'Interest Rate' , x = 'Grade')

table(loan2$grade , factor(loan2$loan_outcome , c(0 , 1) , c('Fully Paid' , 'Default')))

ggplot(loan2 , aes(x = grade , y = ..count.. , fill = factor(loan_outcome , c(1 , 0) , c('Default' , 'Fully Paid')))) + 
  geom_bar() + 
  theme(legend.title = element_blank())

ggplot(loan2[sample(244179 , 10000) , ] , aes(x = annual_inc , y = loan_amnt , color = int_rate)) +
  geom_point(alpha = 0.5 , size = 1.5) + 
  geom_smooth(se = F , color = 'darkred' , method = 'loess') +
  xlim(c(0 , 300000)) + 
  labs(x = 'Annual Income' , y = 'loan Ammount' , color = 'Interest Rate')

# Split dataset 
loan2$loan_outcome = as.numeric(loan2$loan_outcome)
idx = sample(dim(loan2)[1] , 0.75*dim(loan2)[1] , replace = F)
trainset = loan2[idx , ]
testset = loan2[-idx , ]

# Fit logistic regression
glm_model = train(
  form = loan_outcome ~ .,
  data = trainset,
  trControl = trainControl(method = "cv", number = 5),
  method = "glm",
  family = "binomial")
summary(glm_model)

# Prediction on test set
preds <- as.data.frame(predict(glm_model , testset))
preds <- ifelse(preds$`predict(glm_model, testset)` > .5, 1, 0)

# Density of probabilities
ggplot(data.frame(preds) , aes(preds$`predict(glm_model, testset)`)) + 
  geom_density(fill = 'lightblue' , alpha = 0.4) +
  labs(x = 'Predicted Probabilities on test set')

k = 0
accuracy = c()
sensitivity = c()
specificity = c()
for(i in seq(from = 0.01 , to = 0.5 , by = 0.01)){
  k = k + 1
  preds_binomial = ifelse(preds > i , 1 , 0)
  confmat = table(testset$loan_outcome , preds_binomial)
  accuracy[k] = sum(diag(confmat)) / sum(confmat)
  sensitivity[k] = confmat[1 , 1] / sum(confmat[ , 1])
  specificity[k] = confmat[2 , 2] / sum(confmat[ , 2])
}

threshold = seq(from = 0.01 , to = 0.5 , by = 0.01)

data = data.frame(threshold , accuracy , sensitivity , specificity)
head(data)

# Gather accuracy , sensitivity and specificity in one column
ggplot(gather(data , key = 'Metric' , value = 'Value' , 2:4) , 
       aes(x = threshold , y = Value , color = Metric)) + 
  geom_line(size = 1.5)

preds.for.50 = ifelse(preds > 0.5 , 1 , 0)
confusion_matrix_50 = confusionMatrix(as.factor(preds.for.50), as.factor(testset$loan_outcome))
confusion_matrix_50

library(pROC)

# Area Under Curve
auc(roc(testset$loan_outcome , preds$`predict(glm_model, testset)`))

# Plot ROC curve
plot.roc(testset$loan_outcome , preds$`predict(glm_model, testset)` , main = "Confidence interval of a threshold" , percent = TRUE , 
         ci = TRUE , of = "thresholds" , thresholds = "best" , print.thres = "best" , col = 'blue')

#Use different sampling techniques to account for imbalance dataset
#Oversampling
over.samp <- ovun.sample(loan_outcome ~ ., data = trainset, p = 0.5, seed = 1, method = "over")$data

#Undersampling
under.samp <- ovun.sample(loan_outcome ~ ., data = trainset, p = 0.5, seed = 1, method = "under")$data

#Train the model on the oversampled dataset
glm_over = train(
  form = loan_outcome ~ .,
  data = over.samp,
  method = "glm",
  family = "binomial")

pred.over <- as.data.frame(predict(glm_over, over.samp))
pred.over <- ifelse(pred.over$`predict(glm_over, over.samp)` > .5, 1, 0)

confusionMatrix(as.factor(pred.over), as.factor(over.samp$loan_outcome))

# Area Under Curve
auc(roc(over.samp$loan_outcome , pred.over))


#Train the model on the undersampled dataset
glm_under = train(
  form = loan_outcome ~ .,
  data = under.samp,
  method = "glm",
  family = "binomial")

pred.under <- as.data.frame(predict(glm_under, under.samp))
pred.under <- ifelse(pred.under$`predict(glm_under, under.samp)` > .5, 1, 0)

confusionMatrix(as.factor(pred.under), as.factor(under.samp$loan_outcome))

# Area Under Curve
auc(roc(under.samp$loan_outcome , pred.under))

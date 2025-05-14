
#### train, test, validate, evaluate  several ML models for classification of "signs of life" data
#### used for predicting the presence status of inhabitants (present/absent in/from the country)
#### this code was used for our paper in SJIAOS linked in the readme-information of this repository


library(RODBC) 
library(DBI)

library(caret)    
library(pROC)
library(dplyr)
library(adabag)

library(magrittr)

library(cluster)
library(factoextra)

library(ggplot2)
library(iml)
###------------------------------ PART 0: data ---------------------------------------------------###

channel <- odbcDriverConnect("DRIVER=SQL Server; SERVER=ZZZZZ")

df<-sqlQuery(channel,"select * from XXXXX", stringsAsFactors=FALSE ) 


########## rename some variables: English names ##################
names(df)
names(df)[names(df) == "born_skoli"] <- "children_in_school"
names(df)[names(df) == "n_breytinga_skra"] <- "changes_in_register"
names(df)[names(df) == "sex"] <- "gender"
names(df)[names(df) == "n_skoli"] <- "adults_in_school"

### varaibles
cols_no_icelanders <- c('carOwner','ID','income_increase_max','citizenship','children','carOwner','ID', 'time_in_iceland', 'income_increase_2yr')
cols_no_foreigners <- c('carOwner','ID','income_increase_max','citizenship','children','ever_abroad','carOwner','ID','student_abroad', 'income_increase_2yr')
### factor variables
factor_cols<-c('Presence','married','homeOwner','gender','region')   


###------ separate by citizenship:classifiers are run separately -----------------

#### ------- foreigners
library(dplyr)
ddf <- df %>% filter(citizenship==0) %>%
          select(-any_of(matches(cols_no_foreigners)))  %>% 
          mutate(across(all_of(factor_cols), as.factor), across(-all_of(factor_cols), as.numeric))  %>%
          mutate(homeOwner = ifelse(homeOwner==2,1,homeOwner))

numeric_cols <- ddf %>%   
                select_if(is.numeric) %>% colnames()


# #### --------- Icelanders
ddf <- df %>% filter(citizenship==1) %>%
          select(-any_of(matches(cols_no_icelanders)))  %>%
          mutate(across(factor_cols, as.factor),
         across(-factor_cols, as.numeric))  %>% mutate(homeOwner = ifelse(homeOwner==2,1,homeOwner))

 

###----- descriptive and exploratory stage--------------------------------###


###----- compare distributions of attributes over classes (defined by status) ------------------

## normalised income, over whole population ----------------
# summary(df$income)
# ggplot(df, aes(x=income, colour=as.factor(Presence)) ) + 
#   geom_density(alpha=0.1)  +
#   labs(x="income") +
#   theme(legend.position = "bottom") +
#   scale_colour_discrete(name="presence", breaks=c('0','1'), labels=c("Absence", "Presence"))+
#   xlim(-3, 3)

## age, over the entire population ---------------
ggplot(df, aes(x=age, colour=as.factor(Presence))) + 
  geom_density(alpha=0.1 , size=0.9)  +
  labs(x= "age") +
  theme(legend.position = "bottom") +
  scale_colour_discrete(name="Presence", breaks=c('0','1'), labels=c("Absent", "Present"))

## over foreigners
ggplot2::ggplot(ddf, aes(x=time_in_iceland, colour=as.factor(Presence)) ) +
  geom_density(alpha=0.1, size=0.9)  +
  labs(x= "time in Iceland (foreign citizens)") +
  theme(legend.position = "bottom") +
  scale_colour_discrete(name="Presence", breaks=c(0,1), labels=c("Absent", "Present"))

# ggplot2::ggplot(ddf, aes(x=months_worked, colour=as.factor(Presence)) ) +
#   geom_density(alpha=0.1)  +
#   labs(x= "months worked (foreign citizens)") +
#   theme(legend.position = "bottom") +
#   scale_colour_discrete(name="Presence", breaks=c(0,1), labels=c("Absent", "Present"))
# 


###-------- check clustering as an unsupervised ML verification ---------------
ddf_cluster0 <- ddf %>%
  select_if(is.numeric)  # select numeric columns %>%
  
ddf_cluster <- preProcess(ddf_cluster0, method = c("knnImpute", "zv","center", "scale"))$data
 
### pam-algorithm 
#pamddf <- pam(ddf_cluster, 2)-  
#summary(pamddf)
#plot(pamddf)
# Visualize differently, much easier
#plot_pam <- 
#factoextra::fviz_cluster(pamddf)


### km clustering algorithm ------------------------------------
set.seed(123)
km.res <- kmeans(ddf_cluster, 2)
# Visualize ### (cannot label all points, since too busy, but most)
plot_kmeans <- 
  factoextra::fviz_cluster(km.res, data = ddf_cluster,
                           ellipse.type = "convex",
                           palette = "jco",
                           repel = TRUE,
                           ggtheme = ggplot2::theme_minimal())
   
ggpubr::ggpar(plot_kmeans, title='',
              legend = "bottom" ,
              legend.title ="Clusters of present (1) and absent (2) individuals:")
  
  
km.res$size 
    #### most informative: (cluster 1) 994 and (cluster 2) 354 for Foreigners;
    #### (cluster 1 10721) and (cluster 2) 5641 for Icelanders
km.res$centers
km.res$betweens
km.res$tot.withinss


###----------correlations --------------------------------------------
DataExplorer::plot_correlation(
  data=ddf, 
  type = "all",
  maxcat = 50L, cor_args = list(),
  title = NULL,
  theme_config = list(legend.position ="bottom", 
                      axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)))



###-------------- Part 0.bis Split data into training and (extra-)testing -------------------------------###

set.seed(565)
index0 <- sample(1:nrow(ddf),round(0.7*nrow(ddf)))
training<- ddf[index0,]
training$Presence<-as.factor(training$Presence)
testing <- ddf[-index0,]
testing$Presence<-as.factor(testing$Presence)
length(training[,1])

## sizes
# training %>% group_by(Presence) %>% summarise(n = n())
# testing %>% group_by(Presence) %>% summarise(n = n())


###---------------------------- Part 1: One model, first choice: Random forest -----------------------------------### 

## "legitimate" R-level-names for factors needed by caret 

training <- training %>%
  mutate(across(where(is.factor), ~factor(paste0('x', as.character(.)))))

testing <- testing %>%
  mutate(across(where(is.factor), ~factor(paste0('x', as.character(.)))))

preProc <- preProcess(training, method = c("knnImpute", "zv","center", "scale"))

training <- predict(preProc, training)
testing  <- predict(preProc, testing)

summary(training$Presence)
summary(testing$Presence)

###---------- train, tune, evaluate--------------------------

ctrl <- trainControl(
  method = "repeatedcv",
  repeats = 3,  ##  for speed  ## but 7, when estimating variability
  n=10,  ## 10 for speed
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions=T
)

training1 <- training %>% select(-Presence)


##-------- for very imbalanced samples -------------
######  the following up/down-sampling is needed only when the classes are very imbalanced indeed 
###### (e.g. tens of times smaller/bigger):

## for foreigners, we would do up-sampling 
###  ctrl$sampling <- "up"  #### due to imbalanced samples. I checked, the performance improves

## for icelanders, we would do down-sampling
##ctrl$sampling <- "down"  #### due to imbalanced samples, but this way the performance improves
##--------------------------------------------------------



### for the random forest algorithm---------------
model_0_Fit <- caret::train(
  ##Presence ~ .,
  x = training1,
  y= training$Presence,
  method = "rf",   
  preProc = c("center", "scale"),
  tuneLength = 15,
  ntree=50,           ###for fast runs; otherwise  ntree=500, etc.
  trControl = ctrl,   ### note the new argument: how to search for best model
  metric = "ROC"      ### for measuring performance
)

model_0_Fit

model_0_Fit$finalModel  ##----------------

model_0_Fit$pred  ##--------------


## comparing with and without up/down-sampling: was done e.g. with:--------------
roc(model_0_Fit$pred$obs, model_0_Fit$pred$x1)$auc  
## this is at thres_prob=50%


### we can do the following run for multiple cutoff probability values ---------
### , but might have bias still (see need of resampling, etc)
probs <- seq(0.0, 1.0, by = 0.01)
ths <- thresholder(model_0_Fit,
                   threshold = probs,
                   final = TRUE,
                   statistics = "all")
head(ths, 20)

### choosing the criterion for the "best" threshold probab value ---------------
ths_withPE <- ths %>%
  mutate(prob = probs) %>%
  mutate(PopError = 1- abs
          ( Sensitivity*Prevalence - Specificity*(1-Prevalence) )
  ) 
head(ths_withPE)

thr_1 <- ths_withPE  %>% 
         mutate(Perfomance=Sensitivity) %>% 
         select(prob, Perfomance) %>%
         mutate(what="Sensitivity")

thr_2 <- ths_withPE %>%  
  mutate(Perfomance=Specificity) %>% 
  select(prob, Perfomance) %>%
  mutate(what="Specificity")

thr_3 <- ths_withPE %>%  
  mutate(Perfomance=PopError) %>% 
  select(prob, Perfomance) %>%
  mutate(what="Population_Error")

thr_5 <- ths_withPE %>% 
  mutate(Perfomance=Precision) %>% 
  select(prob, Perfomance) %>%
  mutate(what="Precision")

thr_6 <- ths_withPE %>%  
  mutate(Perfomance=Accuracy) %>% 
  select(prob, Perfomance) %>%
  mutate(what="Accuracy")

thr_7 <- ths_withPE %>%  
  mutate(Perfomance=Kappa) %>% 
  select(prob, Perfomance) %>%
  mutate(what="Kappa")

thr_8 <- ths_withPE %>%  
  mutate(Perfomance=J) %>% 
  select(prob, Perfomance) %>%
  mutate(what="J")

thr_9 <- ths_withPE %>%  
  mutate(Perfomance=F1) %>% 
  select(prob, Perfomance) %>%
  mutate(what="F1")

thr <- rbind(thr_1, thr_2) %>% rbind(thr_3) %>% 
      rbind(thr_5) %>% rbind(thr_6) %>% 
      rbind(thr_7) %>% rbind(thr_8) %>% rbind(thr_9) 

thr <- thr %>% mutate(Metric=what) %>%  filter(Metric !="Precision")
 

### nice plot of all performance measures varying with the classification probability threshold--------------
ggplot(thr, aes(x = prob, y = Perfomance)) + 
  geom_line(aes(color = Metric), size=0.9) +   
  ylab("Performance measures") +
  xlab("probability threshold")
  

## repeat this with other models (see below): model_2_Fit, ...---------(to make a function for this)-------------
rf_default <- model_0_Fit
selectedIndices <- rf_default$pred$mtry == rf_default$bestTune[1,1]
library(pROC)
selectedIndices <- model_0_Fit$pred$mtry == model_0_Fit$bestTune[1,1]

## roc-function will select the best (== when sensitivity + specificity is max) threshold automatically here
roc1 <- roc(rf_default$pred$obs[selectedIndices], rf_default$pred$x1[selectedIndices], print.thres='best',print.auc =T,ci=T)
roc1
ci1<- ci.thresholds(roc1)

proc0 <- 
  plot(roc1, print.thres="best")   
#plot(ci1)


######## grid plot of all proc0, proc4, ...


### NOTE: thresholds= probab. values (cutoffs) for classifying cases into class 1 and class 2;
### the option "best" in function "roc" selects the threshold which gives the point on the curve with
###  highest sum (sensitivity + specificity) !

#### alternative WAY, shows point with max sum of sens and spec
##
# roc1a <- roc(rf_default$finalModel$pred$obs, rf_default$finalModel$pred$x1, print.thres='best',print.auc =T,ci=T)
# roc1a
# ci1a <- ci.thresholds(roc1a)
# plot(roc1a, print.thres="best")   ### ok ### ###########################################
#plot(ci1a)


# ci1b <- ci.thresholds(roc1,
#               thresholds=probs,
#               boot.n=10000, conf.level=0.9, stratified=FALSE)
#plot(roc1, print.thres="best")
#plot(ci1b)

# roc1
# roc1a


###--- more options for roc curves ----------
# You can use coords to plot for instance a sensitivity + specificity vs. cut-off diagram
# plot(specificity + sensitivity ~ threshold,
#      coords(roc1, "all", transpose = FALSE),
#      type = "l", log="x",
#      subset = is.finite(threshold))
# 
# # Plot the Precision-Recall curve
# plot(precision ~ recall,
#      coords(roc1, "all", ret = c("recall", "precision"), transpose = FALSE),
#      type="l", ylim = c(0, 100))
# 
# # Alternatively plot the curve with TPR and FPR instead of SE/SP
# # (identical curve, only the axis change)
# plot(tpr ~ fpr,
#      coords(roc1, "all", ret = c("tpr", "fpr"), transpose = FALSE),
#      type="l")
# 
# 

#------- old style plots, now with confidence bands-----------------

se_sp <- ci1$sensitivity %>% 
  as.data.frame()  %>%  
  cbind(as.data.frame(ci1$specificity) ) 

se_sp$cutoff <- row.names(se_sp)
colnames(se_sp) <- c('sens_low','sens_median','sens_high','spec_low','spec_median','spec_high','cutoff')
se_sp <- se_sp[-1,]

##Find the row where sens_median is better than 0.985
specific_row <- se_sp %>% 
                filter(abs(sens_median - 0.985) < 1e-3)
## Extract the corresponding specificity
specific_spec <-specific_row$spec_median
 
cutoff_critic <- as.numeric(specific_row$cutoff[1])  ### actually for the other class would be 1-cutoff ---------------


## plotting performance metrics as a function of probability threshold, with Confidence Bands- -----------***------------------------

ggplot(data = se_sp) +
  
  geom_line(aes(x=1.0-as.numeric(cutoff), y=sens_median, group=1), size=.7, color="blue") +
  
  xlab("probability threshold") +
  ylab("Sensitivity and Specificity with uncertainty bands") +
  
  geom_ribbon(aes(x=1-as.numeric(cutoff), ymin=sens_low, ymax=sens_high, group=1), alpha=0.2, fill="blue") +
  geom_line(aes(x=1-as.numeric(cutoff), y=spec_median, group=1), size=.7, color="red") +
  geom_ribbon(aes(x=1-as.numeric(cutoff), ymin=spec_low, ymax=spec_high, group=1), alpha=0.2, fill="red") +
  geom_hline(yintercept=0.98, linetype="dashed", color="black") +
  geom_vline(xintercept=1-as.numeric(cutoff_critic), linetype="dashed", color="black")  ## +

###--------- end old style plots ------------


# rf_default$results # standard deviation and average of the metrics
# rf_default$finalModel
# rf_default$pred
# plot(rf_default)
# predict(rf_default,testing,class='prob')

###--------------------------------------------------------------------------------


### ---------------------- Variability of results, for this model--------------

## if only one model, then:
 resamps0 <- rf_default$resample
 ss0 <- summary(resamps0)  ### (ROC; Sens, Spec: median, min, 1st Qu., ... etc)
 ss0


###---------------------------------------------------------------------------------------###
###-----------------------PART 2: multiple models and comparing them ---------------------###
###---------------------------------------------------------------------------------------### 

### collect also the roc-plots of all models and make one grid plot
 
### fit another model: Bayes-glm :::::::::::::::::::::::::::::::::::::::
# note: this does not require tunning  since embedded, but it knows what to do

model_2_Fit <- caret::train(
  x = training1,
  y= training$Presence,
  method = "bayesglm",   ###---------------------------------------***--------------
  preProc = c("center", "scale"),
  tuneLength = 15,
  trControl = ctrl,   ### note the new argument: how to search for best model
  metric ="ROC"      ### for measuring performance
)

model_2_Fit
## ggplot(model_Fit) ## no need for this one since no tunning
# predicting on test data
model_Classes <- predict(model_2_Fit, newdata = testing)
str(model_Classes)
model_Probs <- predict(model_2_Fit, newdata = testing, type = "prob")
head(model_Probs)


## more models: ---------------------------

### logistic regression model -------------------------------
model_3_Fit <- caret::train(
   x = training1,
   y= training$Presence,
    trControl = ctrl,  
   metric ="ROC" ,
    tuneLength = 15,
    method = "glm",
    family = "binomial"  
)

model_3_Fit

model_Classes <- predict(model_3_Fit, newdata = testing)
str(model_Classes)
model_Probs <- predict(model_3_Fit, newdata = testing, type = "prob")
head(model_Probs)


## neural network model --------- (longer run) ----------------
# model_4_Fit <- caret::train(
#   x = training1,
#   y= training$Presence,
#   trControl = ctrl,
#   tuneLength = 15,
#   method="nnet"  
# )
# 
# model_4_Fit

## check it ------------------------
# trellis.par.set(caretTheme())
# plot(model_4_Fit)
# trellis.par.set(caretTheme())
# #plot(model_4_Fit, metric = "Kappa")
# ggplot(model_4_Fit)
# predict(model_4_Fit, newdata = head(testing))
# #predict(model_4_Fit, newdata = head(testing), type = "prob")
# # visualize more
# trellis.par.set(caretTheme())
# densityplot(model_4_Fit, pch = "|")
# ------------------------------------

# model_Classes <- predict(model_4_Fit, newdata = testing)
# str(model_Classes)
# model_Probs <- predict(model_4_Fit, newdata = testing, type = "prob")
# head(model_Probs)


### may add many more models ...............

### method = 'rpart' ## classification tree --------------------------
model_5_Fit <- caret::train(
  x = training1,
  y= training$Presence,
  trControl = ctrl,  
  ## tuneLength = 15,
  method="rpart"  
)

model_5_Fit
summary(model_5_Fit)

model_Classes <- predict(model_5_Fit, newdata = testing)
str(model_Classes)
model_Probs <- predict(model_5_Fit, newdata = testing, type = "prob")
head(model_Probs)



##----------------- comparison of all models ---------------------------------##

### compare models ----------------------------------------------------------------
resamps <- resamples(list(model_0 = model_0_Fit,
                          model_2 = model_2_Fit, 
                          model_3 = model_3_Fit ,
                          model_5 = model_5_Fit #,
                         # model_4 = model_4_Fit
                         ))
summary(resamps)

### comparison of models that works: --------------------
diffs <- diff(resamps)
summary(diffs)
trellis.par.set(theme1)
bwplot(diffs, layout = c(5, 1))  

### more comparisons ------------------------------------------------

testing1 <- testing %>% select(-Presence)

predict(model_0_Fit$finalModel, newdata = testing1, type="response")[1:5]
predict(model_0_Fit, newdata = testing)[1:5]

models <- list(model_0 = model_0_Fit,
               model_2 = model_2_Fit, 
               model_3 = model_3_Fit ,
               model_5 = model_5_Fit #,
              # model_4= model_4_Fit
            )


testPred <- predict(models, newdata = testing1)
## check a few lines:
lapply(testPred, function(x) x[1:5])

predValues <- extractPrediction(models, testX = testing1, testY = testing$Presence)

testValues <- subset(predValues, dataType == "Test")
head(testValues)
table(testValues$model)
nrow(tesxtDescr)

probValues <- extractProb(models,
                          testX = testing1, testY = testing$Presence)
testProbs <- subset(probValues, dataType == "Test")
str(testProbs)
plotClassProbs(testProbs)  ###---------------################ Compar_predictedProbabValues_on_testing_data ###################


###--- comparing performance measures

r0 <- resampleHist(model_0_Fit, main="Random Forest")
r2 <- resampleHist(model_2_Fit, main="Bayesian_log_reg")
r3 <- resampleHist(model_3_Fit, main="log_reg")
#r4 <- resampleHist(model_4_Fit , main="Neural Network")
r5 <- resampleHist(model_5_Fit , main="classif_tree")

gridExtra::grid.arrange(r0, r2, r3, nrow =  3)  ###----------------#############Compar__ROC__Sens__Spec_all ###############



###-------------------------------------------Part 3.-----------------------------------------------
###-------------------- Interpretation of all models with comparison compare  ------------------###

library('iml')
X <- training %>% select(-Presence) 

predictor_0 <- Predictor$new(model_0_Fit, data=X, y=training$Presence)
imp_0 <-FeatureImp$new(predictor_0, loss="ce")

# effects, ex: one of the features
# eff <- FeatureEffect$new(predictor_0, feature="income") 
# eff
# eff$plot()

## surrogate tree  ###***************** find correct interpretation "!!!!!!!!!!******************
tree_0 <- TreeSurrogate$new(predictor_0, maxdepth = 3)
plot(tree_0)
plot(tree_0$tree)

## may also predict with this tree: *******
##head(tree$predict(...))


### the other models ---------------------------------
predictor_2 <- Predictor$new(model_2_Fit, data=X, y=training$Presence)
imp_2 <-FeatureImp$new(predictor_2, loss="ce")

predictor_3 <- Predictor$new(model_3_Fit, data=X, y=training$Presence)
imp_3 <-FeatureImp$new(predictor_3, loss="ce")

#predictor_4 <- Predictor$new(model_4_Fit, data=X, y=training$Presence)
#imp_4 <-FeatureImp$new(predictor_4, loss="ce")

predictor_5 <- Predictor$new(model_5_Fit, data=X, y=training$Presence)
imp_5 <-FeatureImp$new(predictor_5, loss="ce")


# plot output for all
p0 <- plot(imp_0) + ggtitle("Random Forest") + xlab("Feature importance")  ###-----------
## p3 <- plot(imp_3) + ggtitle("log_reg") + xlab("Feature importance")
 p2 <- plot(imp_2) + ggtitle("Bayesian log_reg") + xlab("Feature importance")  ###---------
## p5 <- plot(imp_5) + ggtitle("classif_tree") + xlab("Feature importance")


gridExtra::grid.arrange(p0, p2, nrow = 2)   ###--------------------##############Compar_imp###################

#### ------------------------------------------


### more examples of tree surrogates
tree_0 <- TreeSurrogate$new(predictor_0, maxdepth = 3)
plot(tree_0)
plot(tree_0$tree)
# 
tree_2 <- TreeSurrogate$new(predictor_2, maxdepth = 3)
plot(tree_2)
plot(tree_2$tree)

tree_3 <- TreeSurrogate$new(predictor_3, maxdepth = 3)
plot(tree_3)
plot(tree_3$tree)


## tree4: not applicable 

###-------------------------------------------------------------------------
###================= end multiple models======================================###







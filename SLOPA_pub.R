
## content of this script, for solving:
## a classification problem of demography which is
## to predict the status (in/out) country of registered people
## as a function of their multiple attributes as
## observed in many databsaes/registers (see abstract paper)

## it shows also the evolution of solutions
## from simplest models/packages to most recent ones

# steps:
# get data
# explore and check clusters
# part 1: use several packages for fitting many/ensemble classifiers
# part 2: use stacks-package, for fitting stacked models
# part 3: use h20 package, for fitting stacked models 
#         and for automated stack selection
# part 4: interpretability, e.g. by using iml package
# part 5: measuring uncertainty



## get data
# query the sql-database and extract data, from R ----------------
# get real data of old Census, call it df
# for example, if using a linux computer:
#library(odbc)
#library(DBI)
# con_dev <- dbConnect(odbc(),
#                      Driver = "FreeTDS",
#                      Server = "myserver",
#                      port=1433,  
#                      Database = "mydatabase",
#                      UID = "myusername",
#                      PWD = rstudioapi::askForPassword("Database password")
# )
# res<- dbSendQuery(con_dev, "SELECT * FROM mydatabase")
# df <- dbFetch(res) 

#df <- ....
# delay in de-registration variable deltat has been recoded



#------------------- Part 0.  Exploration ------------------------


## most general correlation plots: for any type of var

# main attributes (of immigrants): age, salary/pension at t0 and one year before,
# time since arrival in Iceland, status in employment, 
# last and before last year's changes in registers

ddf <- dplyr::select(df, age, laun0, laun1, pension0, pension1, 
                     life, stada, brt, brt1, deltat)

DataExplorer::plot_correlation(
                 data=ddf, 
                 type = "all",
                 maxcat = 50L, cor_args = list(),
                 title = NULL,
                theme_config = list(legend.position ="bottom"))

# may also run a more complete exploration:
# DataExplorer::create_report(ddf)

#--- Clustering, seen as part of exploration 

library(cluster)
library(factoextra)

# find optimal number of clusters
dff <- scale(ddf)  #-------- useful also later 

### factoextra::fviz_nbclust(dff, kmeans, method = "gap_stat")
# compute and visualise
set.seed(123)
km.res <- kmeans(dplyr::sample_frac(data.frame(dff), 0.001), 2, nstart = 25)
# Visualize #
plot_kmeans <- 
  factoextra::fviz_cluster(km.res, data = dplyr::sample_frac(data.frame(dff), 0.001),
                           ellipse.type = "convex",
                           palette = "jco",
                           repel = TRUE,
                           ggtheme = ggplot2::theme_minimal())

# compare with PAM clustering
# Compute PAM
pam.res <- cluster::pam(dplyr::sample_frac(data.frame(dff), 0.001), 2)
# Visualize
#plot_pam <- 
factoextra::fviz_cluster(pam.res)


#----------- Part 1: Multiple models , using caret -------------------


#---  classification trees, for easy interpretation 
library(caret)
sol.classifTree <- train(as.factor(deltat) ~ ., 
                         data=ddf, 
                         method="rpart", 
                         trControl = trainControl(method = "cv"))

sol.classifTree

summary(sol.classifTree)

# try fancy plotting, it works
suppressMessages(library(rattle))
fancyRpartPlot(sol.classifTree$finalModel, caption="")


index0 <- sample(1:nrow(ddf),round(0.9*nrow(ddf)))
training<- ddf[index0,]
testing <- ddf[-index0,]

# predictions ***
 sol.pred = predict(sol.classifTree, newdata = testing)
# checks
table(sol.pred, testing$deltat)

# for faster experiments
set.seed(150)
small <- dplyr::sample_frac(data.frame(ddf), 0.05)  
# data is scaled already; sampling for speed
scaledd <-scale(small) 

# Cross validation examples 
library(caret)
set.seed(111)
inTraining <- createDataPartition(small$deltat, p = .75, list = FALSE)
training <- small[ inTraining,]
testing  <- small[-inTraining,]

fitControl <- trainControl(
       ## 10-fold CV
       method = "repeatedcv",
       number = 10,
       ## repeated ten times
        repeats = 10)

# neural net model ---------
set.seed(123)
nnetfit <- train(deltat ~ ., data = training, 
                 method = "nnet", 
                 trControl = fitControl,
                 verbose = FALSE)
nnetfit

trellis.par.set(caretTheme())
plot(nnetfit) 

trellis.par.set(caretTheme())
#plot(nnetfit, metric = "Kappa")
ggplot(nnetfit)

predict(nnetfit, newdata = head(testing))
#predict(nnetfit, newdata = head(testing), type = "prob")

# visualize more
trellis.par.set(caretTheme())
densityplot(nnetfit, pch = "|")


# comparing models ----------------------------
# see https://topepo.github.io/caret/model-training-and-tuning.html


# fit more models, for example:
# random forest ------------------
rf <- train(deltat~., 
                    data=training, 
                    method='rf',
                    trControl=fitControl)
print(rf)

# then compare them ---------------

resamps <- resamples(list(nn=nnetfit,rf = rf ))
resamps

summary(resamps)
splom(resamps)

trellis.par.set(caretTheme())
dotplot(resamps, metric = "ROC")

trellis.par.set(theme1)
xyplot(resamps, what = "BlandAltman")

difValues
summary(difValues)
trellis.par.set(caretTheme())
dotplot(difValues)


#------------------
## more models are easily added, see the list at:

# http://topepo.github.io/caret/models-clustered-by-tag-similarity.html 

## and ensembles/stacks as well


#---------------------

# we then discovered the new R-package, which has common author with caret :), 
# it works even more efficiently, so:


#----- PART 2: using stacking of models with stacks-R-package -----------

# ensemble models, for multiple, same type of models (see forests, etc)
# ....

# stacking models, as shown at: -----------------------------------
# https://cran.r-project.org/web/packages/stacks/vignettes/classification.html

# setup SOL data: -----------

SOL <- ddf     ## or a smaller sample, like data frame called small

# models ----------

set.seed(1)

SOL_split <- initial_split(SOL)
SOL_train <- training(SOL_split)
SOL_test  <- testing(SOL_split)

folds <- rsample::vfold_cv(SOL_train, v = 5)

XXX <- deltat  

SOL_rec <- 
  recipe(XXX ~ ., data = SOL_train) %>%
  step_dummy(all_nominal(), -XXX) %>%
  step_zv(all_predictors())

SOL_wflow <- 
  workflow() %>% 
  add_recipe(SOL_rec)

ctrl_grid <- control_stack_grid()

# random forest in stacks, this time --------

rand_forest_spec <- 
  rand_forest(
    mtry = tune(),
    min_n = tune(),
    trees = 500
  ) %>%
  set_mode("classification") %>%
  set_engine("ranger")

rand_forest_wflow <-
  SOL_wflow %>%
  add_model(rand_forest_spec)

rand_forest_res <- 
  tune_grid(
    object = rand_forest_wflow, 
    resamples = folds, 
    grid = 10,
    control = ctrl_grid
  )

# neural networks in stacks, this time --------------

nnet_spec <-
  mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) %>%
  set_mode("classification") %>%
  set_engine("nnet")

nnet_rec <- 
  SOL_rec %>% 
  step_normalize(all_predictors())

nnet_wflow <- 
  SOL_wflow %>%
  add_model(nnet_spec)

nnet_res <-
  tune_grid(
    object = nnet_wflow, 
    resamples = folds, 
    grid = 10,
    control = ctrl_grid
  )


# stacking the models now ---------

SOL_model_st <- 
  # initialize the stack
  stacks() %>%
  # add candidate members
  add_candidates(rand_forest_res) %>%
  add_candidates(nnet_res) %>%
  # determine how to combine their predictions
  blend_predictions() %>%
  # fit the candidates with nonzero stacking coefficients
  fit_members()

SOL_model_st

theme_set(theme_bw())
autoplot(SOL_model_st)
autoplot(SOL_model_st, type = "members")
autoplot(SOL_model_st, type = "weights")

collect_parameters(SOL_model_st, "rand_forest_res")

SOL_pred <-
  SOL_test %>%
  bind_cols(predict(SOL_model_st, ., type = "prob"))

yardstick::roc_auc(
  SOLs_pred,
  truth = reflex,
  contains(".pred_")
)


# check each model's predictions  ------------
SOL_pred <-
  SOL_test %>%
  select(reflex) %>%
  bind_cols(
    predict(
      SOL_model_st,
      SOL_test,
      type = "class",
      members = TRUE
    )
  )

SOL_pred

# and

map_dfr(
  setNames(colnames(SOL_pred), colnames(SOL_pred)),
  ~mean(SOL_pred$reflex == pull(SOL_pred, .x))
) %>%
  pivot_longer(c(everything(), -reflex))


#-------------------- Part 3: using h2o package --------------------


# h2o and automatic learners-----------new stage!--------------
 library(h2o)
  #h2o.init()

# see https://bradleyboehmke.github.io/HOML/stacking.html#stacking-existing  
# which shows (in Chapter 15) that:

#"Leo Breiman, known for his work on classification and regression trees 
# and random forests, formalized stacking in his 1996 paper on Stacked 
# Regressions (Breiman 1996b). Although the idea originated in (Wolpert 1992) 
# under the name “Stacked Generalizations”, the modern form of stacking that 
# uses internal k-fold CV was Breiman’s contribution.
#However, it wasn’t until 2007 that the theoretical background for stacking 
# was developed, and also when the algorithm took on the cooler name, 
# Super Learner (Van der Laan, Polley, and Hubbard 2007). 
# Moreover, the authors illustrated that super learners will learn an optimal 
# combination of the base learner predictions and will typically perform as 
# well as or better than any of the individual models that make up the 
# stacked ensemble. Until this time, the mathematical reasons for why stacking 
# worked were unknown and stacking was considered a black art."


## several models
#...

## stacking models, and remember:
# the biggest gains are usually produced when 
# we are stacking base learners that have 
# high variability, and uncorrelated, predicted values ****


## data
sol <- dff  
set.seed(123)  
split <- initial_split(sol, strata = "deltat")  # check strata need?
sol_train <- training(split)
sol_test <- testing(split)

# check consistency of categorical levels 
#...

# Get response and feature names
Y <- deltat ##"Sale_Price"
X <- setdiff(names(sol_train), Y)


## automatic learners, directly! -----------
auto_ml <- h2o.automl(
  x = X, y = Y, training_frame = train_h2o, nfolds = 5,
  max_runtime_secs = 60 * 120, max_models = 50,
  keep_cross_validation_predictions = TRUE, sort_metric = "RMSE", seed = 123,
  stopping_rounds = 50, stopping_metric = "RMSE", stopping_tolerance = 0
)


auto_ml@leaderboard %>% 
  as.data.frame() %>%
  dplyr::select(model_id, rmse) %>%
  dplyr::slice(1:25)


# and still experimenting with stacked models and evaluation ...


# -------------------- Part 4: interpretability of models ----------------------
# use measures like partial dependence plots, individual conditional expectation, 
# accumulated local effects, feature interaction, shapley values, etc 
# (see iml - package for example)

# reference for interpretability:
# https://christophm.github.io/interpretable-ml-book/ 


# ------------------- Part 5: measuring uncertainty of classifiers --------------
# ...(to add)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------





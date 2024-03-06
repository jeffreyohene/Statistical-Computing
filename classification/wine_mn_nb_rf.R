# CLASSIFICATION
# Building Simple Classifiers

# Data source: https://archive.ics.uci.edu/dataset/109/wine
# Create model to classify wine type based on other variables in the wine dataset
# Algorithms used: Multinomial Logistic Regression, Naive Bayes & Random Forest


# According to data source, here is the data dictionary:
# class
# Alcohol
# Malicacid
# Ash
# Alcalinity_of_ash
# Magnesium
# Total_phenols
# Flavanoids
# Nonflavanoid_phenols
# Proanthocyanins
# Color_intensity
# Hue	Feature
# 0D280_0D315_of_diluted_wines
# Proline


# SCRIPT DEPENDENCIES
library(e1071)
library(nnet)
library(randomForest)
library(ggplot2)
library(corrplot)

options(scipen = 999)
set.seed(7)

# DATA IMPORT

# import dataset
wine <- read.csv("data/wine.data")

# inspect Structure and Variables
dim(wine)
str(wine)
summary(wine)


# DATA PREPROCESSING

# define the column names
column_names <- c("class", "alcohol", "malicacid",
                  "ash", "alcalinity_of_ash",
                  "magnesium", "total_phenols", "flavanoids",
                  "nonflavanoid_phenols", "proanthocyanins", "color_intensity",
                  "hue", "of_diluted_wines", "proline")

# replace unclear column names with descriptive ones
colnames(wine) <- column_names

# data type conversions

# convert class variable to categorical
wine$class <- factor(wine$class)
table(wine$class)

# summarize dataset again
summary(wine)


# DATA VISALIZATION

# histograms

# alcohol
ggplot(wine,
       aes(
         x = alcohol
       )) +
  geom_histogram(col="black",fill="gold") +
  labs(
    title = "Distribution of Alcohol Percentage",
    x="Alcohol(%)",
    y="# of Wine Bottles"
  ) +
  theme_minimal()


# maliacid
ggplot(wine,
       aes(
         x = malicacid
       )) +
  geom_histogram(col="black",fill="gold") +
  labs(
    title = "Distribution of Malicacid",
    x="Malicacid",
    y="# of Wine Bottles"
  ) +
  theme_minimal()

# ash
ggplot(wine,
       aes(
         x = ash
       )) +
  geom_histogram(col="black",fill="gold") +
  labs(
    title = "Distribution of Ash",
    x="Ash",
    y="# of Wine Bottles"
  ) +
  theme_minimal()

# alcalinity of ash
ggplot(wine,
       aes(
         x = alcalinity_of_ash
       )) +
  geom_histogram(col="black",fill="gold") +
  labs(
    title = "Distribution of Alcalinity of Ash",
    x="Ash Alcalinity",
    y="# of Wine Bottles"
  ) +
  theme_minimal()

# magnesium
ggplot(wine,
       aes(
         x = magnesium
       )) +
  geom_histogram(col="black",fill="gold") +
  labs(
    title = "Distribution of Magnesium",
    x="Magnesium",
    y="# of Wine Bottles"
  ) +
  theme_minimal()

# total phenols
ggplot(wine,
       aes(
         x = total_phenols
       )) +
  geom_histogram(col="black",fill="gold") +
  labs(
    title = "Distribution of Total Phenols",
    x="Total Phenols",
    y="# of Wine Bottles"
  ) +
  theme_minimal()

# flavanoids
ggplot(wine,
       aes(
         x = flavanoids
       )) +
  geom_histogram(col="black",fill="gold") +
  labs(
    title = "Distribution of Flavanoids",
    x="Flavanoid",
    y="# of Wine Bottles"
  ) +
  theme_minimal()

# non-flavanoid phenols
ggplot(wine,
       aes(
         x = nonflavanoid_phenols
       )) +
  geom_histogram(col="black",fill="gold") +
  labs(
    title = "Distribution of Non-Flavanoid Phenols",
    x="Flavanoid",
    y="# of Wine Bottles"
  ) +
  theme_minimal()

# proanthocyanins
ggplot(wine,
       aes(
         x = proanthocyanins
       )) +
  geom_histogram(col="black",fill="gold") +
  labs(
    title = "Distribution of Proanthocyanins",
    x="Proanthocyanin",
    y="# of Wine Bottles"
  ) +
  theme_minimal()

# color intensity
ggplot(wine,
       aes(
         x = color_intensity
       )) +
  geom_histogram(col="black",fill="gold") +
  labs(
    title = "Distribution of Color Intensity",
    x="Color Intensity",
    y="# of Wine Bottles"
  ) +
  theme_minimal()

# hue
ggplot(wine,
       aes(
         x = hue
       )) +
  geom_histogram(col="black",fill="gold") +
  labs(
    title = "Distribution of Wine Hue",
    x="Hue",
    y="# of Wine Bottles"
  ) +
  theme_minimal()

# of diluted wines
ggplot(wine,
       aes(
         x = of_diluted_wines
       )) +
  geom_histogram(col="black",fill="gold") +
  labs(
    title = "Distribution of % of Diluted Wines",
    x="Dilution(%)",
    y="# of Wine Bottles"
  ) +
  theme_minimal()

# proline
ggplot(wine,
       aes(
         x = proline
       )) +
  geom_histogram(col="black",fill="gold") +
  labs(
    title = "Distribution of Proline",
    x="Proline",
    y="# of Wine Bottles"
  ) +
  theme_minimal()

# Boxplots
ggplot(wine,
       aes(
         x = class,
         y = alcohol,
         fill=class
       )) +
  geom_boxplot() +
  coord_flip() +
  labs(
    title = "Alcohol Distribution of Different Wine Types",
    x ="Wine Type",
    y="Alcohol (%)"
  ) +
  theme_minimal()


# MODEL BUILDING

# check for missing values and duplicate rows in dataset
missing_data <- colSums(is.na(wine))
missing_data

# check for duplicate rows
duplicate_rows <- wine[duplicated(wine), ]
nrow(duplicate_rows)

# calculate correlation matrix
correlation_matrix <- cor(wine[, -1])  # Exclude the response variable (class)

# visualize correlation matrix using corrplot
corrplot(correlation_matrix, method = "color")

# split data into training and testing sets with the ratio of 70:30

# randomly sample 70% of rows in dataset
sampled.indices <- sample(1:nrow(wine), nrow(wine) * 0.7)

# subset wine dataset for rows in sampled.indices and assign to train
# for test, we will select rows that were not in sampled.indices
train.data <- wine[sampled.indices, ]
test.data <- wine[-sampled.indices, ]


# MULTINOMIAL LOGISTIC REGRESSION MODEL
mn.model <- multinom(class ~ .,
                 data = train.data)

summary(mn.model)

# make predictions
predictions_mn.model <- predict(mn.model,
                                 newdata = test.data)

cm_mn.model <- table(
  actual = test.data$class,
  model_predictions = predictions_mn.model
)

cm_mn.model

# prediction accuracy
acc_mn.model <- sum(diag(cm_mn.model))/sum(cm_mn.model)
acc_mn.model <- round((acc_mn.model * 100), 2)


# NAIVE BAYES CLASSIFIER
nb.model <- naiveBayes(class ~.,
                       data = train.data)

summary(nb.model)

# make ppredictions withh test data
predictions_nb.model <- predict(nb.model,
                                newdata =test.data)

# confusion matrix
cm_nb.model <- table(
  actual = test.data$class,
  model_predictions= predictions_nb.model
)

cm_nb.model

# prediction accuracy
acc_nb.model <- sum(diag(cm_nb.model))/sum(cm_nb.model)
acc_nb.model <- round((acc_nb.model * 100), 2)


# RANDOM FOREST CLASSIFIER
rf.model <- randomForest(class ~ .,
                         data = train.data)

rf.model

# predictions with test data
predictions_rf.model <- predict(rf.model,
                                newdata = test.data)

# confusion matrix
cm_rf.model <- table(
  actual=test.data$class,
  model_predictions=predictions_rf.model
)

cm_rf.model

# prediction accuracy
acc_rf.model <- sum(diag(cm_rf.model))/sum(cm_rf.model)
acc_rf.model <- round((acc_rf.model*100),2)

# variables with highest predictive power
importance(rf.model)
varImpPlot(rf.model) # color_intensity, flavanoids, proline and alcohol were variables with highest predictive power


# PREDICTION EVALUATION OF THE 3 MODELS
acc_mn.model
acc_nb.model
acc_rf.model

# Our random forest classifier had highest prediction accuracy with 98.15%.
# The naive bayes classifier recorded a 96.3% prediction accuracy rate
# The multinomial logistic regression classifier recorded a 92.59% prediction accuracy rate

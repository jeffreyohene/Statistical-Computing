# CLASSIFICATION
# A Classifier to Predict Income Categories from US Census data

# Data Source: https://archive.ics.uci.edu/dataset/2/adult
# Classify Income Types based on demographic data
# Algorithms used: Random Forest, Logistic Regression, Decision Trees,
#                  Naive Bayes, Support Vector Classification

# Variables
# age
# workclass
# fnlwgt
# education
# education-num
# marital-status
# occupation
# relationship
# race
# sex
# capital-gain
# capital-loss
# hours-per-week
# native-country
# income

# WORKFLOW
# 1. Download dataset remotely and unzip to project directory
# 2. Data preprocessing
# 3. Exploratory data analysis
# 4. Splitting into training and testing datasets
# 5. Random Forest Classifier and Evaluation
# 6. Logistic Regression and Evaluation
# 7. Decision Trees and Evalluation
# 8. Naive Bayes and Evaluation
# 9. Support Vector Classification and Evaluation
# 10. Model Performance Comparison

# script dependencies
library(randomForest)
library(e1071)
library(ggplot2)
library(dplyr)

# data source
url <- "https://archive.ics.uci.edu/static/public/2/adult.zip"

# download destination
destination <- "adult.zip"

# directory to save extracted components
extract_dir <- "data"

# download data
download.file(url, destination)

# check if the data files are downloaded
if (file.exists(destination)) {
  # if it doesn't exist, create new directory
  if (!file.exists(extract_dir)) {
    dir.create(extract_dir)
  }

  # unzip the file
  unzip(destination, exdir = extract_dir)

  # list the files in the extraction directory to confirm
  list.files(extract_dir)
} else {
  cat("Download failed.")
}

# import dataset
adult <- read.csv("data/adult.data")


# create a vector of variable names
variables <- c("age", "workclass", "fnlwgt",
               "education", "education_num", "marital_status",
               "occupation", "relationship", "race",
               "sex", "capital_gain", "capital_loss",
               "hours_per_week", "native_country", "income")

# assign variable names to dataset
names(adult) <- variables

# inspect Structure and Variables
dim(adult)
str(adult)
summary(adult)

# variable data type conversions

# categorical variable conversions
adult$workclass <- factor(adult$workclass)
adult$education <- factor(adult$education)
adult$marital_status <- factor(adult$marital_status)
adult$occupation <- factor(adult$occupation)
adult$relationship <- factor(adult$relationship)
adult$race <- factor(adult$race)
adult$native_country <- factor(adult$native_country)

# binary variable conversions

# trim whitespace before converting to binary variable
adult$sex <- stringr::str_trim(adult$sex)
adult$income <- stringr::str_trim(adult$income)

# convert sex and income to binary variables
adult$sex <- ifelse(adult$sex == "Male", 1, 0)
adult$income <- ifelse(adult$income == ">50K", 1, 0)

# convert sex and income to factors
adult$sex <- factor(adult$sex)
adult$income <- factor(adult$income)

# EXPLORATORY DATA ANALYSIS

# age
ggplot(adult,
       aes(
         x = age
       )) +
  geom_histogram(fill="#006466",col="grey5",bins=25) +
  labs(
    title = "Distribution of Ages of Census Participants",
    x = "Age",
    y = "# of Adults"
  ) +
  theme_minimal()


# workclass
workclass <- adult %>%
  group_by(workclass) %>%
  summarize(
    total_workers = n()
  )

# plot workclass
ggplot(workclass,
       aes(
         x = reorder(workclass, total_workers),
         y = total_workers
       )) +
  geom_col(fill="#006466", col="grey5") +
  coord_flip()+
  labs(
    title="Census Participants by Work Class",
    x = "Work Class",
    y = "# of People") +
  theme_minimal()

# education
education <- adult %>%
  group_by(education) %>%
  summarise(
    total = n()
  ) %>%
  arrange(desc(education))

education

# plot education
ggplot(education,
       aes(
         x = reorder(education, total),
         y = total
       )) +
  geom_col(fill="#006466",
           col="grey5") +
  coord_flip() +
  labs(
    title = "Education Level of Census Participants",
    x = "Education Level",
    y="# of Adults"
  ) +
  theme_minimal()

# marital status
marital_status <- adult %>%
  group_by(marital_status) %>%
  summarise(
    total = n()
  ) %>%
  arrange(desc(marital_status))

marital_status




# occupation
occupation <- adult %>%
  group_by(occupation) %>%
  summarise(
    total = n()
  ) %>%
  arrange(desc(occupation))

occupation





# relationship
relationship <- adult %>%
  group_by(relationship) %>%
  summarise(
    total = n()
  ) %>%
  arrange(desc(relationship))

relationship




# race
race <- adult %>%
  group_by(race) %>%
  summarise(
    total = n()
  ) %>%
  arrange(desc(race))

race

# sex
sex_df <- adult
sex_df$sex <- ifelse(sex_df$sex == 1, "Male", "Female")

ggplot(sex_df,
       aes(
         x = sex
       )) +
  geom_bar(fill="#006466",col="grey5") +
  labs(
    title="Sex of Census Participants",
    x="Sex",
    y="# of Adults"
  ) +
  theme_minimal()

# capital gain
ggplot(adult,
       aes(
         x = capital_gain
       )) +
  geom_histogram(fill="#006466", col="grey5", bins=25) +
  labs(
    title="Distribution of Capital Gain Among Census Participants",
    x="Capital Gain",
    y="# of Adults"
  ) +
  theme_minimal()

# capital loss
ggplot(adult,
       aes(
         x = capital_loss
       )) +
  geom_histogram(fill="#006466", col="grey5", bins=25) +
  labs(
    title="Distribution of Capital Loss Among Census Participants",
    x="Capital Loss",
    y="# of Adults"
  ) +
  theme_minimal()


# hours per week
ggplot(adult,
       aes(
         x = hours_per_week
       )) +
  geom_histogram(fill="#006466", col="grey5", bins=25) +
  labs(
    title="Distribution of Working Hours Among Census Participants",
    x="Hours Worked",
    y="# of Adults"
  ) +
  theme_minimal()


# native country




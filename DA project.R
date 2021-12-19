if(!require(tidyverse)) 
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) 
  install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) 
  install.packages("data.table", repos = "http://cran.us.r-project.org")
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", 
                             readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)

colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))
movies
movielens <- left_join(ratings, movies, by = "movieId")

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, 
                                  times = 1, p = 0.1, list = FALSE)

edx <- movielens[-test_index,]
temp <- movielens[test_index,]

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")


removed <- anti_join(temp, validation)

edx <- rbind(edx, removed)
head(edx)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)

train_set <- edx[-test_index,]
temp <- edx[test_index,]
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)

#2.2 Data Exploration
str(edx)
dim(edx)
head(edx)

#2.2.1 Genres

edx %>% group_by(genres) %>% 
  summarise(n=n()) %>%
  head()


tibble(count = str_count(edx$genres, fixed("|")), genres = edx$genres) %>% 
  group_by(count, genres) %>%
  summarise(n = n()) %>%
  arrange(-count) %>% 
  head()

# 2.2.2 Date

library(lubridate)
tibble(`Initial Date` = date(as_datetime(min(edx$timestamp), origin="1970-01-01")),
       `Final Date` = date(as_datetime(max(edx$timestamp), origin="1970-01-01"))) %>%
  mutate(Period = duration(max(edx$timestamp)-min(edx$timestamp)))

if(!require(ggthemes)) 
  install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(scales)) 
  install.packages("scales", repos = "http://cran.us.r-project.org")
edx %>% mutate(year = year(as_datetime(timestamp, origin="1970-01-01"))) %>%
  ggplot(aes(x=year)) +
  geom_histogram(color = "white") + 
  ggtitle("Rating Distribution Per Year") +
  xlab("Year") +
  ylab("Number of Ratings") +
  scale_y_continuous(labels = comma) + 
  theme_economist()


edx %>% mutate(date = date(as_datetime(timestamp, origin="1970-01-01"))) %>%
  group_by(date, title) %>%
  summarise(count = n()) %>%
  arrange(-count) %>%
  head(10)


#2.2.3 Ratings

edx %>% group_by(rating) %>% summarize(n=n())

edx %>% group_by(rating) %>% 
  summarise(count=n()) %>%
  ggplot(aes(x=rating, y=count)) + 
  geom_line() +
  geom_point() +
  scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x))) +
  ggtitle("Rating Distribution", subtitle = "Higher ratings are prevalent.") + 
  xlab("Rating") +
  ylab("Count") +
  theme_economist()

# 2.2.4 Movies

edx %>% group_by(movieId) %>%
  summarise(n=n()) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "white") +
  scale_x_log10() + 
  ggtitle("Distribution of Movies", 
          subtitle = "The distribution is almost symetric.") +
  xlab("Number of Ratings") +
  ylab("Number of Movies") + 
  theme_economist()

# 2.2.5 Users

edx %>% group_by(userId) %>%
  summarise(n=n()) %>%
  arrange(n) %>%
  head()

edx %>% group_by(userId) %>%
  summarise(n=n()) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "white") +
  scale_x_log10() + 
  ggtitle("Distribution of Users", 
          subtitle="The distribution is right skewed.") +
  xlab("Number of Ratings") +
  ylab("Number of Users") + 
  scale_y_continuous(labels = comma) + 
  theme_economist()

users <- sample(unique(edx$userId), 100)
edx %>% filter(userId %in% users) %>%
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% 
  select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")
title("User x Movie Matrix")


# 2.3 Data Cleaning

train_set <- train_set %>% select(userId, movieId, rating, title)
test_set  <- test_set  %>% select(userId, movieId, rating, title)

# ------------------------- 3 Results ----------------------------

# 3.1 Model Evaluation Functions

# Define Mean Absolute Error (MAE)
MAE <- function(true_ratings, predicted_ratings){
  mean(abs(true_ratings - predicted_ratings))
}

# Define Mean Squared Error (MSE)
MSE <- function(true_ratings, predicted_ratings){
  mean((true_ratings - predicted_ratings)^2)
}

# Define Root Mean Squared Error (RMSE)
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# 3.2 Random Prediction

set.seed(4321, sample.kind = "Rounding")

# Create the probability of each rating
p <- function(x, y) mean(y == x)
rating <- seq(0.5,5,0.5)

# Estimate the probability of each rating with Monte Carlo simulation
B <- 10^3
M <- replicate(B, {
  s <- sample(train_set$rating, 100, replace = TRUE)
  sapply(rating, p, y= s)
})
prob <- sapply(1:nrow(M), function(x) mean(M[x,]))

# Predict random ratings
y_hat_random <- sample(rating, size = nrow(test_set), 
                       replace = TRUE, prob = prob)

# Create a table with the error results
result <- tibble(Method = "Project Goal", RMSE = 0.8649, MSE = NA, MAE = NA)
result <- bind_rows(result, 
                    tibble(Method = "Random prediction", 
                           RMSE = RMSE(test_set$rating, y_hat_random),
                           MSE  = MSE(test_set$rating, y_hat_random),
                           MAE  = MAE(test_set$rating, y_hat_random)))

result



# 3.3.1 Initial Prediction

# Mean of observed values
mu <- mean(train_set$rating)

# Update the error table  
result <- bind_rows(result, 
                    tibble(Method = "Mean", 
                           RMSE = RMSE(test_set$rating, mu),
                           MSE  = MSE(test_set$rating, mu),
                           MAE  = MAE(test_set$rating, mu)))
# Show the RMSE improvement  
result


#3.3.2 Include Movie Effect (bi)

# Movie effects (bi)
bi <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
head(bi)

bi %>% ggplot(aes(x = b_i)) + 
  geom_histogram(bins=10, col = I("black")) +
  ggtitle("Movie Effect Distribution") +
  xlab("Movie effect") +
  ylab("Count") +
  scale_y_continuous(labels = comma) + 
  theme_economist()




# Predict the rating with mean + bi  
y_hat_bi <- mu + test_set %>% 
  left_join(bi, by = "movieId") %>% 
  .$b_i

# Calculate the RMSE  
result <- bind_rows(result, 
                    tibble(Method = "Mean + bi", 
                           RMSE = RMSE(test_set$rating, y_hat_bi),
                           MSE  = MSE(test_set$rating, y_hat_bi),
                           MAE  = MAE(test_set$rating, y_hat_bi)))

# Show the RMSE improvement  
result


# 3.3.3 Include User Effect (bu)

# User effect (bu)
bu <- train_set %>% 
  left_join(bi, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Prediction
y_hat_bi_bu <- test_set %>% 
  left_join(bi, by='movieId') %>%
  left_join(bu, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

# Update the results table
result <- bind_rows(result, 
                    tibble(Method = "Mean + bi + bu", 
                           RMSE = RMSE(test_set$rating, y_hat_bi_bu),
                           MSE  = MSE(test_set$rating, y_hat_bi_bu),
                           MAE  = MAE(test_set$rating, y_hat_bi_bu)))

# Show the RMSE improvement  
result




train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(color = "black") + 
  ggtitle("User Effect Distribution") +
  xlab("User Bias") +
  ylab("Count") +
  scale_y_continuous(labels = comma) + 
  theme_economist()





# 3.3.4 Evaluating the model result

train_set %>% 
  left_join(bi, by='movieId') %>%
  mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual))) %>%  
  slice(1:10)


titles <- train_set %>% 
  select(movieId, title) %>% 
  distinct()

bi %>% 
  inner_join(titles, by = "movieId") %>% 
  arrange(-b_i) %>% 
  select(title) %>%
  head()

bi %>% 
  inner_join(titles, by = "movieId") %>% 
  arrange(b_i) %>% 
  select(title) %>%
  head()



# Number of ratings for 10 best movies:

train_set %>% 
  left_join(bi, by = "movieId") %>%
  arrange(desc(b_i)) %>% 
  group_by(title) %>% 
  summarise(n = n()) %>% 
  slice(1:10)


train_set %>% count(movieId) %>% 
  left_join(bi, by="movieId") %>% 
  arrange(desc(b_i)) %>% 
  slice(1:10) %>% 
  pull(n)





# 3.4 Regularization

regularization <- function(lambda, trainset, testset){
  
  # Mean
  mu <- mean(trainset$rating)
  
  # Movie effect (bi)
  b_i <- trainset %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  
  # User effect (bu)  
  b_u <- trainset %>% 
    left_join(b_i, by="movieId") %>%
    filter(!is.na(b_i)) %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  
  # Prediction: mu + bi + bu  
  predicted_ratings <- testset %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    filter(!is.na(b_i), !is.na(b_u)) %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, testset$rating))
}

# Define a set of lambdas to tune
lambdas <- seq(0, 10, 0.25)

# Tune lambda
rmses <- sapply(lambdas, 
                regularization, 
                trainset = train_set, 
                testset = test_set)

# Plot the lambda vs RMSE
tibble(Lambda = lambdas, RMSE = rmses) %>%
  ggplot(aes(x = Lambda, y = RMSE)) +
  geom_point() +
  ggtitle("Regularization", 
          subtitle = "Pick the penalization that gives the lowest RMSE.") +
  theme_economist()

# ----------------------------------------------------------------



# We pick the lambda that returns the lowest RMSE.
lambda <- lambdas[which.min(rmses)]

# Then, we calculate the predicted rating using the best parameters 
# achieved from regularization.  
mu <- mean(train_set$rating)

# Movie effect (bi)
b_i <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

# User effect (bu)
b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

# Prediction
y_hat_reg <- test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Update the result table
result <- bind_rows(result, 
                    tibble(Method = "Regularized bi and bu", 
                           RMSE = RMSE(test_set$rating, y_hat_reg),
                           MSE  = MSE(test_set$rating, y_hat_reg),
                           MAE  = MAE(test_set$rating, y_hat_reg)))

# Regularization made a small improvement in RMSE.  
result




# 3.5 Matrix Factorization


train_data <- train_set %>% 
  select(userId, movieId, rating) %>% 
  spread(movieId, rating) %>% 
  as.matrix()

gc()
memory.limit(9999999999)
fit <-lm(Y ~ X)
gc()


if(!require(recosystem)) 
  install.packages("recosystem", repos = "http://cran.us.r-project.org")
set.seed(123, sample.kind = "Rounding") # This is a randomized algorithm

# Convert the train and test sets into recosystem input format
train_data <-  with(train_set, data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating     = rating))
test_data  <-  with(test_set,  data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating     = rating))

# Create the model object
r <-  recosystem::Reco()

# Select the best tuning parameters
opts <- r$tune(train_data, opts = list(dim = c(10, 20, 30), 
                                       lrate = c(0.1, 0.2),
                                       costp_l2 = c(0.01, 0.1), 
                                       costq_l2 = c(0.01, 0.1),
                                       nthread  = 4, niter = 10))

# Train the algorithm  
r$train(train_data, opts = c(opts$min, nthread = 4, niter = 20))



# Calculate the predicted values  
y_hat_reco <-  r$predict(test_data, out_memory())
head(y_hat_reco, 10)


result <- bind_rows(result, 
                    tibble(Method = "Matrix Factorization - recosystem", 
                           RMSE = RMSE(test_set$rating, y_hat_reco),
                           MSE  = MSE(test_set$rating, y_hat_reco),
                           MAE  = MAE(test_set$rating, y_hat_reco)))
result


# 3.6 Final Validation

# 3.6.1 Linear Model With Regularization

mu_edx <- mean(edx$rating)

# Movie effect (bi)
b_i_edx <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_edx)/(n()+lambda))

# User effect (bu)
b_u_edx <- edx %>% 
  left_join(b_i_edx, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu_edx)/(n()+lambda))

# Prediction
y_hat_edx <- validation %>% 
  left_join(b_i_edx, by = "movieId") %>%
  left_join(b_u_edx, by = "userId") %>%
  mutate(pred = mu_edx + b_i + b_u) %>%
  pull(pred)

# Update the results table
result <- bind_rows(result, 
                    tibble(Method = "Final Regularization (edx vs validation)", 
                           RMSE = RMSE(validation$rating, y_hat_edx),
                           MSE  = MSE(validation$rating, y_hat_edx),
                           MAE  = MAE(validation$rating, y_hat_edx)))

# Show the RMSE improvement
result

# Top 10 best movies

validation %>% 
  left_join(b_i_edx, by = "movieId") %>%
  left_join(b_u_edx, by = "userId") %>% 
  mutate(pred = mu_edx + b_i + b_u) %>% 
  arrange(-pred) %>% 
  group_by(title) %>% 
  select(title) %>%
  head(10)


# Top 10 worst movies

validation %>% 
  left_join(b_i_edx, by = "movieId") %>%
  left_join(b_u_edx, by = "userId") %>% 
  mutate(pred = mu_edx + b_i + b_u) %>% 
  arrange(pred) %>% 
  group_by(title) %>% 
  select(title) %>%
  head(10)



#3.6.2 Matrix Factorization

set.seed(1234, sample.kind = "Rounding")

# Convert 'edx' and 'validation' sets to recosystem input format
edx_reco <-  with(edx, data_memory(user_index = userId, 
                                   item_index = movieId, 
                                   rating = rating))
validation_reco  <-  with(validation, data_memory(user_index = userId, 
                                                  item_index = movieId, 
                                                  rating = rating))

# Create the model object
r <-  recosystem::Reco()

# Tune the parameters
opts <-  r$tune(edx_reco, opts = list(dim = c(10, 20, 30), 
                                      lrate = c(0.1, 0.2),
                                      costp_l2 = c(0.01, 0.1), 
                                      costq_l2 = c(0.01, 0.1),
                                      nthread  = 4, niter = 10))

# Train the model
r$train(edx_reco, opts = c(opts$min, nthread = 4, niter = 20))


# Calculate the prediction
y_hat_final_reco <-  r$predict(validation_reco, out_memory())

# Update the result table
result <- bind_rows(result, 
                    tibble(Method = "Final Matrix Factorization - recosystem", 
                           RMSE = RMSE(validation$rating, y_hat_final_reco),
                           MSE  = MSE(validation$rating, y_hat_final_reco),
                           MAE  = MAE(validation$rating, y_hat_final_reco)))

# Show the RMSE improvement
result 


# Top 10 best movies

tibble(title = validation$title, rating = y_hat_final_reco) %>%
  arrange(-rating) %>% 
  group_by(title) %>% 
  select(title) %>%
  head(10)

# Top 10 worst movies

tibble(title = validation$title, rating = y_hat_final_reco) %>%
  arrange(rating) %>% 
  group_by(title) %>% 
  select(title) %>%
  head(10)
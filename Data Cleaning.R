train_set <- train_set %>% select(userId, movieId, rating)
test_set  <- test_set  %>% select(userId, movieId, rating)
train_set
test_set 
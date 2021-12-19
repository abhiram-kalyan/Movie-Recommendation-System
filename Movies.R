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
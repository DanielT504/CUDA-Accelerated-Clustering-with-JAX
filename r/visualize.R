library(ggplot2)

# Load clustering results
results <- read.csv("cluster_results.csv")

# Plot clusters
ggplot(results, aes(x = X1, y = X2, color = as.factor(Cluster))) +
  geom_point() +
  labs(title = "k-Means Clustering Results", x = "Feature 1", y = "Feature 2", color = "Cluster") +
  theme_minimal()
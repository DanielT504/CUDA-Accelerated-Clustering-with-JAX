library(ggplot2)

# Load clustering results
data <- read.csv("C:/Users/User/Desktop/Not backed up/CUDA-Accelerated-Clustering-with-JAX/r/cluster_results.csv")

# Create a scatter plot of the clusters
plot <- ggplot(data, aes(x = X1, y = X2)) +
  geom_hex(bins = 50) +
  labs(title = "Cluster Density Visualization") +
  theme_minimal()

# Use the same absolute path for the output directory
output_dir <- "C:/Users/User/Desktop/Not backed up/CUDA-Accelerated-Clustering-with-JAX/r"
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

# Save the plot
output_plot <- file.path(output_dir, "cluster_plot.png")
ggsave(output_plot, plot, width = 7, height = 7, dpi = 300)
print(paste("Plot saved to:", normalizePath(output_plot)))

print(paste("Cluster visualization saved to", output_plot))

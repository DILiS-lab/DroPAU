library(tidyverse)
library(ggplot2)
library(cowplot)
library(magick)

theme_set(theme_bw(base_size = 26))

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

input = ggdraw() +
  draw_image(magick::image_read("data/mnist_plus/lrp/attribution_mean_1520_a1_b0_extended2_input_250.png", density = 600)) +
  ggtitle("Input") +
  theme(plot.margin = unit(c(0, 0, 0, 0), "cm"))
lrp.1 = ggdraw() +
  draw_image(magick::image_read("data/mnist_plus/lrp/attribution_mean_1520_a1_b0_extended2_250.png", density = 600)) +
  ggtitle("Mean Explanation") +
  theme(plot.margin = unit(c(0, 0, 0, 0), "cm"))
lrp.2 = ggdraw() +
  draw_image(magick::image_read("data/mnist_plus/lrp/attribution_variance_1520_a1_b0_extended2.png", density = 600)) +
  ggtitle("Variance Explanation") +
  theme(plot.margin = unit(c(0, 0, 0, 0), "cm"),)

plot_grid(input, lrp.1, lrp.2,
          nrow=1,
          labels = c("Input", "Mean Explanation", "Uncertainty Explanation"),
          label_size = 26,
          label_x = c(0.25, -0.1, -0.3)) +
  theme(plot.margin = unit(c(0, 0, 0., 0), "cm"))


############## LOAD LIBRARIES ##############
library(tidyverse)
library(ggplot2)
library(cowplot)
library(magick)

theme_set(theme_bw(base_size = 26))

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

lrp <- read_csv("data/mnist_plus/double_path/mnist_plus_lrp_zennit_double_path_a1_b0_extended2.csv")
shap <- read_csv("data/mnist_plus/double_path/mnist_plus_shap_500_double_path_extended2.csv")
infoshap <- read_csv("data/mnist_plus/double_path/mnist_plus_infoshap_500_double_path_extended2.csv")
clue <- read_csv("data/mnist_plus/double_path/mnist_plus_clue_double_path_extended2.csv")
ig <- read_csv("data/mnist_plus/double_path/mnist_plus_ig_double_path_extended2.csv")
process_df <- function(df, round_digits=2){
  df |>
  filter(metric != "iou_scores") |>
  select(-gt_uncertainty) |>
  group_by(metric, heatmap, mask) |>
  summarise(mean = mean(value),
            sd = sd(value)) |>
  mutate(mask = if_else(mask == "var_mask", "Uncertainty\nMask", "Mean\nMask"),
         heatmap = if_else(heatmap == "var_heatmap", "Uncertainty", "Mean"),
         metric = if_else(metric == "mass_accuracies", "RMA", "RRA"),
         mask = factor(mask, levels = c("Uncertainty\nMask", "Mean\nMask")),
         text = paste0(round(mean,round_digits), "\n± ", round(sd,round_digits)))
}

create_plot <- function(data, title, show_y_label=FALSE, show_facet_label=FALSE){
  p <- ggplot(data, aes(x=heatmap, y=mask, fill = mean)) +
    scale_fill_gradient(limits=c(0,1), low="#88CCEE", high = "#332288") +
    # scale_color_steps2(limits=c(0.5,0.97), low="black", high="white", midpoint = 0.75) +
    scale_y_discrete(position = "right") +
    scale_x_discrete(position = "top") +
    geom_tile() +
    geom_text(aes(label=text), size=8, color="white") +
    facet_grid(rows = vars(metric), switch="y") +
    ggtitle(title) +
    theme_void() +
    theme(legend.key = element_blank(), 
          strip.background = element_blank(),
          strip.text = element_blank(), 
          axis.text.x = element_text(size=22),
          legend.position = "None",
          plot.title = element_text(size=26, hjust = 0.5, face="bold")) 
  if (show_y_label == TRUE){
    p <- p +
      theme(axis.text.y = element_text(size=22, angle = 90))
  }
  if(show_facet_label == TRUE){
    p <- p +
      theme(strip.text = element_text(size=26, face="bold", angle = 90, vjust=0))
  }
  return(p)
}

lrp_result_mean <- process_df(lrp) |> filter(heatmap != "Uncertainty")
lrp_result <- process_df(lrp) |> filter(heatmap != "Mean")
ig_result <- process_df(ig) |> filter(heatmap != "Mean")
shap_result <- process_df(shap) |> filter(heatmap != "Mean")
infoshap_result <- process_df(infoshap) |> filter(heatmap != "Mean")
clue_result <- process_df(clue) |> filter(heatmap != "Mean") 

test <- expression(underline(bold("VFA-LRP")))

lrp_plot_mean <- create_plot(lrp_result_mean, "LRP", show_y_label = F, show_facet_label = T)
lrp_plot <- create_plot(lrp_result, test, show_y_label = F, show_facet_label = F)
ig_plot <- create_plot(ig_result, "VFA-IG", show_y_label = F, show_facet_label = F)
shap_plot <- create_plot(shap_result, "VFA-SHAP", show_y_label = F, show_facet_label = F)
infoshap_plot <- create_plot(infoshap_result, "InfoSHAP", show_y_label = F, show_facet_label = F)
clue_plot <- create_plot(clue_result, "CLUE*", show_y_label = T, show_facet_label = F) 
# + theme(legend.position = "right",
#                                                                                                legend.key.height = unit(2.8, "cm"),
#                                                                                                legend.key.width = unit(1, "cm"),
#                                                                                                legend.title = element_blank(),
#                                                                                                legend.text = element_text(size=14))


(final <- plot_grid(lrp_plot_mean, lrp_plot, ig_plot, shap_plot, infoshap_plot, clue_plot,
          nrow = 1,
          rel_widths = c(0.68, 0.6, 0.6, 0.6, 0.6, 0.725)))

ggsave("mnist_plus_heatmap_v4.pdf", final, width=14, height=7, device=cairo_pdf)
 

############## LOAD LIBRARIES ##############
library(tidyverse)
library(ggplot2)
library(cowplot)

theme_set(theme_bw(base_size = 26))

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

##############  LIPSCHITZ ROBUSTNESS BOXPLOT PER DATASET ##############

process_lipschitz <- function(df, dataset){
  return(df |>
           mutate(dataset = dataset) |>
           pivot_longer(cols=-dataset, names_to = "method") |>
           mutate(method = if_else(method == "varx", "VFA-SHAP",
                                   if_else(method == "varx_ig", "VFA-IG",
                                           if_else(method == "varx_lrp", "VFA-LRP",
                                                   if_else(method == "clue", "CLUE",
                                                           if_else(method == "infoshap", "InfoSHAP", "NONE"))))),
                  method = factor(method, levels=c( "VFA-IG",  "VFA-LRP", "VFA-SHAP", "InfoSHAP", "CLUE"))))
}


lip_combined <- process_lipschitz(read_csv("data/lipschitz/red_wine_out_lipschitz_fixed.csv"), "Red Wine") |>
  union(process_lipschitz(read_csv("data/lipschitz/ailerons_out_lipschitz_fixed.csv"), "Ailerons")) |>
  union(process_lipschitz(read_csv("data/lipschitz/synthetic_out_lipschitz_fixed.csv"), "Synthetic")) |>
  union(process_lipschitz(read_csv("data/lipschitz/lsat_out_lipschitz_fixed.csv"), "LSAT")) |>
  mutate(dataset = factor(dataset, levels=c("Red Wine", "Ailerons", "LSAT", "Synthetic")))


(lipschitz <- ggplot(lip_combined, aes(x=dataset, y=value, fill=method, color=method)) +
  geom_boxplot(alpha=0.5) +
  scale_y_continuous(trans = "log10") +
  ylab("Lipshitz Estimate") +
  xlab("Dataset") + 
  guides(fill = guide_legend(ncol= 1, title = "Method"), color = guide_legend(ncol = 1, title = "Method")) +
  scale_fill_manual(values=c("#88CCEE", "#CC6677", "#44AA99", "#DDCC77", "#882255")) +
  scale_color_manual(values=c("#88CCEE", "#CC6677", "#44AA99", "#DDCC77", "#882255")) +
  theme_bw() +
  theme(
    # legend.position = "top", 
    legend.text= element_text(size=14), 
    legend.title = element_blank(),
    legend.box.margin=margin(0,0,0,-5),
    axis.text.x = element_text(size=14),
    axis.title = element_text(size=16),
    plot.margin = unit(c(0.05, 0.05, 0, 0.05), "cm")))


ggsave("Lipschitz_robustness_v4.pdf", lipschitz, width=10, height=3.75, device=cairo_pdf)

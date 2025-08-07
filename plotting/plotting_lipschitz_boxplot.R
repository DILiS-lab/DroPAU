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


lip_combined <- process_lipschitz(read_csv("data/lipschitz/red_wine_out_lipschitz_fixed_new.csv"), "Red Wine") |>
  union(process_lipschitz(read_csv("data/lipschitz/ailerons_out_lipschitz_fixed_new.csv"), "Ailerons")) |>
  union(process_lipschitz(read_csv("data/lipschitz/synthetic_out_lipschitz_fixed_new.csv"), "Synthetic")) |>
  # union(process_lipschitz(read_csv("data/lipschitz/synthetic_mixed_5_out_lipschitz_fixed_new.csv"), "Syn. Mixed")) |>
  union(process_lipschitz(read_csv("data/lipschitz/lsat_out_lipschitz_fixed_new.csv"), "LSAT")) |>
  mutate(dataset = factor(dataset, levels=c("Red Wine", "Ailerons", "LSAT", "Synthetic"))) #, "Syn. Mixed")))


(lipschitz <- ggplot(lip_combined, aes(x=dataset, y=value, fill=method, color=method)) +
    geom_boxplot(alpha=0.5) +
    scale_y_continuous(trans = "log10") +
    ylab("Lipschitz Estimate") +
    xlab("Dataset") + 
    guides(fill = guide_legend(ncol= 3, title = "Method"), color = guide_legend(ncol = 5, title = "Method")) +
    scale_fill_manual(values=c("#88CCEE", "#CC6677", "#44AA99", "#DDCC77", "#882255")) +
    scale_color_manual(values=c("#88CCEE", "#CC6677", "#44AA99", "#DDCC77", "#882255")) +
    theme_bw() +
    theme(
      legend.position = "top",
      legend.text= element_text(size=7), 
      legend.title = element_blank(),
      legend.key.size = unit(0.5,"cm"),
      legend.box.margin=margin(0,0,-10,-5),
      axis.text = element_text(size=7),
      axis.title = element_text(size=10),
      plot.margin = unit(c(0.05, 0.05, 0, 0.05), "cm")))


ggsave("Lipschitz_robustness_v9_new_fixed.pdf", lipschitz, width=3.5, height=2.5, units="in", device=cairo_pdf)


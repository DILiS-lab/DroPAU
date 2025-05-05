############## LOAD LIBRARIES ##############
library(tidyverse)
library(ggplot2)
library(cowplot)
library(magick)

theme_set(theme_bw(base_size = 26))

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
experiment <- "importances_mult_syn_base"
path <- paste0("data/", experiment, "/")
if (experiment == "importances_syn_mixed") {
  colors <- c("#888888", "#332288", "#CC6677")
} else {
  colors <- c("#888888", "#CC6677")
}
############## VFA vs CLUE ############## 
process_df <- function(df){
  return(df |>
           arrange(desc(feature_importance)) |>
           head(15) |>
           mutate(noise_feature = if_else(grepl("noise", feature_name, fixed = TRUE), "Noise feature", if_else(grepl("mixed", feature_name, fixed = TRUE), "Mixed feature", "Mean feature")),
                  feature_name = if_else(noise_feature %in% c("noise", "mixed"), paste0("**", feature_name, "**"),feature_name)))
}

plot_exp <- function(df, rank_acc, mass_acc, title, y_side="left", x_side="bottom"){
  return(ggplot(df, aes(x=feature_importance, y=reorder(feature_name, feature_importance), fill=noise_feature)) +
           ggtitle(title) +
           geom_bar(stat = "identity", width = 0.6) +
           scale_y_discrete(position = y_side) + 
           scale_x_continuous(position = x_side) + 
           scale_fill_manual(values= colors) +
           xlab("Feature Importance") +
           ylab("") +
           geom_label(
             label=paste0("GRA: ", rank_acc, "\n GMA: ", mass_acc), 
             size=3.33,
             x=max(df$feature_importance) / 2, #- max(df$feature_importance) * 0.15,
             y=2.95,
             label.padding = unit(0.5, "lines"), # Rectangle size around label
             label.size = NA,
             color = "black",
             fill="white",
             alpha=0.9,
           ) +
           theme(
             # text = element_text(family="Arial"),
             plot.title = element_text(size=12, hjust = 0.5, vjust = -4),
             axis.title.y = element_text(size=12),
             axis.ticks = element_blank(),
             legend.position = "none",
             axis.text.y.left = element_blank(),
             axis.text.y.right = element_blank(),
             panel.grid.major.y = element_blank(),
             plot.margin = unit(c(0, 0, 0, 0), "cm"))
  )
}


clue_high <- read_csv(paste0(path, "CLUE_highU_importances_n_40000_s_2.00_n_test_1500_n_exp_200_run_0.csv")) 
clue_low <- read_csv(paste0(path, "CLUE_lowU_importances_n_40000_s_2.00_n_test_1500_n_exp_200_run_0.csv")) 
clue_random <- read_csv(paste0(path, "CLUE_randomU_importances_n_40000_s_2.00_n_test_1500_n_exp_200_run_0.csv"))

varx_high <- read_csv(paste0(path, "VarX_highU_importances_n_40000_s_2.00_n_test_1500_n_exp_200_run_0.csv")) 
varx_low <- read_csv(paste0(path, "VarX_lowU_importances_n_40000_s_2.00_n_test_1500_n_exp_200_run_0.csv")) 
varx_random <- read_csv(paste0(path, "VarX_randomU_importances_n_40000_s_2.00_n_test_1500_n_exp_200_run_0.csv")) 

varxig_high <- read_csv(paste0(path, "VarXIG_highU_importances_n_40000_s_2.00_n_test_1500_n_exp_200_run_0.csv")) 
varxig_low <- read_csv(paste0(path, "VarXIG_lowU_importances_n_40000_s_2.00_n_test_1500_n_exp_200_run_0.csv")) 
varxig_random <- read_csv(paste0(path, "VarXIG_randomU_importances_n_40000_s_2.00_n_test_1500_n_exp_200_run_0.csv")) 

varxlrp_high <- read_csv(paste0(path, "VarXLRP_highU_importances_n_40000_s_2.00_n_test_1500_n_exp_200_run_0.csv")) 
varxlrp_low <- read_csv(paste0(path, "VarXLRP_lowU_importances_n_40000_s_2.00_n_test_1500_n_exp_200_run_0.csv")) 
varxlrp_random <- read_csv(paste0(path, "VarXLRP_randomU_importances_n_40000_s_2.00_n_test_1500_n_exp_200_run_0.csv")) 

infoshap_high <- read_csv(paste0(path, "infoshap_highU_importances_n_40000_s_2.00_n_test_1500_n_exp_200_run_0.csv"))
infoshap_low <- read_csv(paste0(path, "infoshap_lowU_importances_n_40000_s_2.00_n_test_1500_n_exp_200_run_0.csv"))
infoshap_random <- read_csv(paste0(path, "infoshap_randomU_importances_n_40000_s_2.00_n_test_1500_n_exp_200_run_0.csv"))


global_metrics <- read_csv(paste0(path, "noise_feature_global_loc_metrics.csv"))

h_varx <- global_metrics |> filter(model=="VarX", unc=="high")
r_varx <- global_metrics |> filter(model=="VarX", unc=="random")
l_varx <- global_metrics |> filter(model=="VarX", unc=="low")

h_varxig <- global_metrics |> filter(model=="VarXIG", unc=="high")
r_varxig <- global_metrics |> filter(model=="VarXIG", unc=="random")
l_varxig <- global_metrics |> filter(model=="VarXIG", unc=="low")

h_varxlrp <- global_metrics |> filter(model=="VarXLRP", unc=="high")
r_varxlrp <- global_metrics |> filter(model=="VarXLRP", unc=="random")
l_varxlrp <- global_metrics |> filter(model=="VarXLRP", unc=="low")

h_infoshap <- global_metrics |> filter(model=="infoshap", unc=="high")
r_infoshap <- global_metrics |> filter(model=="infoshap", unc=="random")
l_infoshap <- global_metrics |> filter(model=="infoshap", unc=="low")

h_clue <- global_metrics |> filter(model=="CLUE", unc=="high")
r_clue <- global_metrics |> filter(model=="CLUE", unc=="random")
l_clue <- global_metrics |> filter(model=="CLUE", unc=="low")

l_clue$precision.mean

format_and_round <- function(mean, standard_deviation){
  paste0(format(round(mean, 2), nsmall = 2)," ± ",format(round(standard_deviation, 2), nsmall = 2))
}


p_clue_high <- plot_exp(process_df(clue_high), format_and_round(h_clue$precision.mean, h_clue$precision.std), format_and_round(h_clue$mass_accuracy.mean, h_clue$mass_accuracy.std), "") + ylab("CLUE") + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank(), axis.title.y=element_text(vjust=-3.95))
p_clue_low <- plot_exp(process_df(clue_low), format_and_round(l_clue$precision.mean, l_clue$precision.std), format_and_round(l_clue$mass_accuracy.mean, l_clue$mass_accuracy.std),  "CLUE for lowest uncertainty instances")  + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank())
p_clue_random <- plot_exp(process_df(clue_random), format_and_round(r_clue$precision.mean, r_clue$precision.std), format_and_round(r_clue$mass_accuracy.mean, r_clue$mass_accuracy.std),  "CLUE for random instances")  + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank())

p_varx_high <- plot_exp(process_df(varx_high), format_and_round(h_varx$precision.mean, h_varx$precision.std), format_and_round(h_varx$mass_accuracy.mean, h_varx$mass_accuracy.std),  "") + ylab("VFA-SHAP") + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank(), axis.title.y=element_text(vjust=-3.95))
p_varx_low <- plot_exp(process_df(varx_low), format_and_round(l_varx$precision.mean, l_varx$precision.std), format_and_round(l_varx$mass_accuracy.mean, l_varx$mass_accuracy.std),  "Variance attribution for lowest uncertainty instances")  + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank())
p_varx_random <-plot_exp(process_df(varx_random), format_and_round(r_varx$precision.mean, r_varx$precision.std), format_and_round(r_varx$mass_accuracy.mean, r_varx$mass_accuracy.std),  "Variance attribution for random instances")  + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank())

p_varxig_high <- plot_exp(process_df(varxig_high), format_and_round(h_varxig$precision.mean, h_varxig$precision.std), format_and_round(h_varxig$mass_accuracy.mean, h_varxig$mass_accuracy.std),  "Highest\nuncertainty")  + ylab("VFA-IG") + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), axis.title.y=element_text(vjust=-3.95))
p_varxig_low <- plot_exp(process_df(varxig_low), format_and_round(l_varxig$precision.mean, l_varxig$precision.std), format_and_round(l_varxig$mass_accuracy.mean, l_varxig$mass_accuracy.std),  "Lowest\nuncertainty")  + theme(axis.title.x = element_blank(), axis.text.x = element_blank())
p_varxig_random <-plot_exp(process_df(varxig_random), format_and_round(r_varxig$precision.mean, r_varxig$precision.std), format_and_round(r_varxig$mass_accuracy.mean, r_varxig$mass_accuracy.std),  "\nRandom")  + theme(axis.title.x = element_blank(), axis.text.x = element_blank())

p_varxlrp_high <- plot_exp(process_df(varxlrp_high), format_and_round(h_varxlrp$precision.mean, h_varxlrp$precision.std), format_and_round(h_varxlrp$mass_accuracy.mean, h_varxlrp$mass_accuracy.std),  "")  + ylab("VFA-LRP") + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank(), axis.title.y=element_text(vjust=-3.95))
p_varxlrp_low <- plot_exp(process_df(varxlrp_low), format_and_round(l_varxlrp$precision.mean, l_varxlrp$precision.std), format_and_round(l_varxlrp$mass_accuracy.mean, l_varxlrp$mass_accuracy.std),  "Variance attribution for lowest uncertainty instances")  + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank())
p_varxlrp_random <-plot_exp(process_df(varxlrp_random), format_and_round(r_varxlrp$precision.mean, r_varxlrp$precision.std), format_and_round(r_varxlrp$mass_accuracy.mean, r_varxlrp$mass_accuracy.std),  "Variance attribution for random instances")  + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank())

p_infoshap_high <- plot_exp(process_df(infoshap_high), format_and_round(h_infoshap$precision.mean, h_infoshap$precision.std), format_and_round(h_infoshap$mass_accuracy.mean, h_infoshap$mass_accuracy.std),  "")  + ylab("InfoSHAP") + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank(), axis.title.y=element_text(vjust=-3.95))
p_infoshap_low <- plot_exp(process_df(infoshap_low), format_and_round(l_infoshap$precision.mean, l_infoshap$precision.std), format_and_round(l_infoshap$mass_accuracy.mean, l_infoshap$mass_accuracy.std),  "InfoSHAP for lowest uncertainty instances")  + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank())
p_infoshap_random <-plot_exp(process_df(infoshap_random), format_and_round(r_infoshap$precision.mean, r_infoshap$precision.std), format_and_round(r_infoshap$mass_accuracy.mean, r_infoshap$mass_accuracy.std),  "InfoSHAP for random instances")  + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank())

legend <- get_legend(
  p_clue_high + theme(legend.position = "top", legend.text= element_text(size=10), legend.title = element_blank()) + guides(fill = guide_legend(nrow = 1, title = "Feature Type")))

p_clue_high + theme(legend.position = "top")
grid <- plot_grid(p_varxig_high, p_varxig_random, p_varxig_low,
                  p_varxlrp_high, p_varxlrp_random, p_varxlrp_low,  
                  p_varx_high, p_varx_random,  p_varx_low,
                  p_infoshap_high, p_infoshap_random,  p_infoshap_low,
                  p_clue_high, p_clue_random, p_clue_low,
                  ncol=3, 
                  labels = c("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"), 
                  label_y = c(0.85, 0.85, 0.85, 1.075, 1.075, 1.075, 1.075, 1.075, 1.075, 1.075, 1.075, 1.075, 1.075, 1.075, 1.075),
                  label_x = c(0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0, 0.03, 0.03),
                  label_size = 16,
                  rel_heights = c(1.27, 1, 1, 1, 1),
                  rel_widths = c(1, 1, 1))


(vfa_vs_baselines <- plot_grid(legend, grid, nrow=2, rel_heights = c(0.05, 1)) +
  theme(plot.margin = unit(c(-1.75, 0.05, 0, -0.25), "cm")))


# shap.1 = ggdraw() +
#   draw_image(magick::image_read_pdf(paste0(path, "variance_output.pdf"), density = 600)) +
#   theme(plot.margin = unit(c(0, 0, 0, 0), "cm"))
# shap.2 = ggdraw() +
#   draw_image(magick::image_read_pdf(paste0(path, "mean_output.pdf"), density = 600)) +
#   theme(plot.margin = unit(c(0, 0, 0, 0), "cm"))
# 
# left <- plot_grid(shap.1, shap.2, ncol=1,  
#           labels = c("A", "B"), 
#           label_size = 26,
#           rel_heights = c(1, 1.1),
#           label_x = c(-0.04, -0.04))
# 
# 
# (final <- plot_grid(left, vfa_vs_baselines,
#           ncol=2,
#           label_size = 26,
#           labels = c("", "C"),
#           label_x = c(0, -0.01),
#           rel_widths = c(0.25, 1)))


# ggsave(paste0(experiment, "_experiment_v10.pdf"), final, width=12, height=4.75, device=cairo_pdf)
  
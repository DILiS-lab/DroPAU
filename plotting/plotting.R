############## LOAD LIBRARIES ##############
library(tidyverse)
library(ggplot2)
library(cowplot)
library(extrafont)
library(magick)

loadfonts()
theme_set(theme_bw(base_size = 26))

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

############## FIGURE 1 DISTRIBUTIONS ############## 
x <- seq(-4, 4, length=100)
y <- dnorm(x)

df = tibble(x=x, y=y)
sd <- tibble(x=x[c(35,66)],y=y[c(35,66)])

ggplot(df, aes(x=x, y=y)) +
  geom_hline(yintercept = 0, linewidth=8) +
  geom_vline(xintercept = 0, linewidth=10) +
  geom_line(linewidth=10, color="#44AA99") +
  geom_line(sd, mapping=aes(x=x, y=y),linewidth=10, arrow = arrow(length=unit(1,"cm"), ends="both", type = "closed"), color="#AA4499") +
  theme_void() +
  theme(plot.margin = margin(1,1,1,1, "cm"))

ggplot(df, aes(x=x, y=y)) +
  geom_line(linewidth=10, color="white") +
  geom_hline(yintercept = 0, linewidth=8) +
  geom_vline(xintercept = 0, linewidth=10, color="#44AA99") +
  theme_void() +
  theme(plot.margin = margin(1,1,1,1, "cm"))
  


############## VFA vs CLUE ############## 
process_df <- function(df){
  return(df |>
           arrange(desc(feature_importance)) |>
           head(15) |>
           mutate(noise_feature = if_else(grepl("noise", feature_name, fixed = TRUE), "Noise feature", "Mean feature"),
                  feature_name = if_else(noise_feature == "noise", paste0("**", feature_name, "**"),feature_name)))
}

plot_exp <- function(df, rank_acc, mass_acc, title, y_side="left", x_side="bottom"){
  return(ggplot(df, aes(x=feature_importance, y=reorder(feature_name, feature_importance), fill=noise_feature)) +
           ggtitle(title) +
           geom_bar(stat = "identity", width = 0.6) +
           scale_y_discrete(position = y_side) + 
           scale_x_continuous(position = x_side) + 
           scale_fill_manual(values=c("#888888", "#CC6677")) +
           xlab("Feature Importance") +
           ylab("") +
           geom_label(
             label=paste0("GRA: ", rank_acc, "\n GMA: ", mass_acc), 
             size=2.75,
             x=max(df$feature_importance) / 2, #- max(df$feature_importance) * 0.15,
             y=4.7,
             label.padding = unit(0.5, "lines"), # Rectangle size around label
             label.size = NA,
             color = "black",
             fill="white",
             alpha=0.9,
           ) +
           theme(
             # text = element_text(family="Arial"),
                 plot.title = element_text(size=12, hjust = 0.5),
                 axis.title.y = element_text(size=12),
                 axis.ticks = element_blank(),
                 legend.position = "none",
                 axis.text.y.left = element_blank(),
                 axis.text.y.right = element_blank(),
                 panel.grid.major.y = element_blank(),
                 plot.margin = unit(c(0, 0, 0, 0), "cm"))
         )
}


clue_high <- read_csv("data/importances/CLUE_highU_importances_n_40000_s_2.00_n_test_1500_n_exp_200.csv") 
clue_low <- read_csv("data/importances/CLUE_lowU_importances_n_40000_s_2.00_n_test_1500_n_exp_200.csv") 
clue_random <- read_csv("data/importances/CLUE_randomU_importances_n_40000_s_2.00_n_test_1500_n_exp_200.csv")

varx_high <- read_csv("data/importances/VarX_highU_importances_n_40000_s_2.00_n_test_1500_n_exp_200.csv") 
varx_low <- read_csv("data/importances/VarX_lowU_importances_n_40000_s_2.00_n_test_1500_n_exp_200.csv") 
varx_random <- read_csv("data/importances/VarX_randomU_importances_n_40000_s_2.00_n_test_1500_n_exp_200.csv") 

varxig_high <- read_csv("data/importances/VarXIG_highU_importances_n_40000_s_2.00_n_test_1500_n_exp_200.csv") 
varxig_low <- read_csv("data/importances/VarXIG_lowU_importances_n_40000_s_2.00_n_test_1500_n_exp_200.csv") 
varxig_random <- read_csv("data/importances/VarXIG_randomU_importances_n_40000_s_2.00_n_test_1500_n_exp_200.csv") 

varxlrp_high <- read_csv("data/importances/VarXLRP_highU_importances_n_40000_s_2.00_n_test_1500_n_exp_200.csv") 
varxlrp_low <- read_csv("data/importances/VarXLRP_lowU_importances_n_40000_s_2.00_n_test_1500_n_exp_200.csv") 
varxlrp_random <- read_csv("data/importances/VarXLRP_randomU_importances_n_40000_s_2.00_n_test_1500_n_exp_200.csv") 

infoshap_high <- read_csv("data/importances/infoshap_highU_importances_n_40000_s_2.00_n_test_1500_n_exp_200.csv")
infoshap_low <- read_csv("data/importances/infoshap_lowU_importances_n_40000_s_2.00_n_test_1500_n_exp_200.csv")
infoshap_random <- read_csv("data/importances/infoshap_randomU_importances_n_40000_s_2.00_n_test_1500_n_exp_200.csv")


global_metrics <- read_csv("data/importances/noise_feature_global_loc_metrics.csv")

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

h_varx$precision

p_clue_high <- plot_exp(process_df(clue_high), format(round(h_clue$precision, 2), nsmall = 2), format(round(h_clue$mass_accuracy, 2), nsmall = 2), "CLUE for highest uncertainty instances") + ylab("CLUE\n\n") + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank())
p_clue_low <- plot_exp(process_df(clue_low), format(round(l_clue$precision, 2), nsmall = 2), format(round(l_clue$mass_accuracy, 2), nsmall = 2),  "CLUE for lowest uncertainty instances")  + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank())
p_clue_random <- plot_exp(process_df(clue_random), format(round(r_clue$precision, 2), nsmall = 2), format(round(r_clue$mass_accuracy, 2), nsmall = 2),  "CLUE for random instances")  + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank())

p_varx_high <- plot_exp(process_df(varx_high), format(round(h_varx$precision, 2), nsmall = 2), format(round(h_varx$mass_accuracy, 2), nsmall = 2),  "Variance attribution for highest uncertainty instances")  + ylab("VFA-SHAP\n\n") + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank())
p_varx_low <- plot_exp(process_df(varx_low), format(round(l_varx$precision, 2), nsmall = 2), format(round(l_varx$mass_accuracy, 2), nsmall = 2),  "Variance attribution for lowest uncertainty instances")  + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank())
p_varx_random <-plot_exp(process_df(varx_random), format(round(r_varx$precision, 2), nsmall = 2), format(round(r_varx$mass_accuracy, 2), nsmall = 2),  "Variance attribution for random instances")  + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank())

p_varxig_high <- plot_exp(process_df(varxig_high), format(round(h_varxig$precision, 2), nsmall = 2), format(round(h_varxig$mass_accuracy, 2), nsmall = 2),  "Highest uncertainty")  + ylab("VFA-IG\n\n") + theme(axis.title.x = element_blank(), axis.text.x = element_blank())
p_varxig_low <- plot_exp(process_df(varxig_low), format(round(l_varxig$precision, 2), nsmall = 2), format(round(l_varxig$mass_accuracy, 2), nsmall = 2),  "Lowest uncertainty")  + theme(axis.title.x = element_blank(), axis.text.x = element_blank())
p_varxig_random <-plot_exp(process_df(varxig_random), format(round(r_varxig$precision, 2), nsmall = 2), format(round(r_varxig$mass_accuracy, 2), nsmall = 2),  "Random")  + theme(axis.title.x = element_blank(), axis.text.x = element_blank())

p_varxlrp_high <- plot_exp(process_df(varxlrp_high), format(round(h_varxlrp$precision, 2), nsmall = 2), format(round(h_varxlrp$mass_accuracy, 2), nsmall = 2),  "Variance attribution for highest uncertainty instances") + ylab("VFA-LRP\n\n") + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank())
p_varxlrp_low <- plot_exp(process_df(varxlrp_low), format(round(l_varxlrp$precision, 2), nsmall = 2), format(round(l_varxlrp$mass_accuracy, 2), nsmall = 2),  "Variance attribution for lowest uncertainty instances")  + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank())
p_varxlrp_random <-plot_exp(process_df(varxlrp_random), format(round(r_varxlrp$precision, 2), nsmall = 2), format(round(r_varxlrp$mass_accuracy, 2), nsmall = 2),  "Variance attribution for random instances")  + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank())

p_infoshap_high <- plot_exp(process_df(infoshap_high), format(round(h_infoshap$precision, 2), nsmall = 2), format(round(h_infoshap$mass_accuracy, 2), nsmall = 2),  "InfoSHAP for highest uncertainty instances") + ylab("InfoSHAP\n\n") + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank())
p_infoshap_low <- plot_exp(process_df(infoshap_low), format(round(l_infoshap$precision, 2), nsmall = 2), format(round(l_infoshap$mass_accuracy, 2), nsmall = 2),  "InfoSHAP for lowest uncertainty instances")  + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank())
p_infoshap_random <-plot_exp(process_df(infoshap_random), format(round(r_infoshap$precision, 2), nsmall = 2), format(round(r_infoshap$mass_accuracy, 2), nsmall = 2),  "InfoSHAP for random instances")  + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), plot.title = element_blank())


legend <- get_legend(
  p_clue_high + theme(legend.position = "top", legend.text= element_text(size=10), legend.title = element_blank()) + guides(fill = guide_legend(nrow = 1, title = "Feature Type")))


 grid <- plot_grid(p_varxig_high, p_varxig_random, p_varxig_low,
                  p_varxlrp_high, p_varxlrp_random, p_varxlrp_low,
                  p_varx_high, p_varx_random, p_varx_low,
                  p_infoshap_high, p_infoshap_random, p_infoshap_low,
                  p_clue_high, p_clue_random, p_clue_low,
                  nrow=5, 
                  labels = c("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"), 
                  label_y = c(0.8, 0.8, 0.8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                  label_x = c(0.1, 0, 0, 0.1, 0, 0, 0.1, 0, 0, 0.1, 0, 0, 0.1, 0, 0),
                  label_size = 16,
                  rel_heights = c(1.3, 1, 1, 1, 1),
                  rel_widths = c(1.1, 1, 1))
 
 plot_grid(legend, grid, nrow=2, rel_heights = c(0.05, 1)) +
  theme(plot.margin = unit(c(0, 0.05, 0, 0), "cm"))


############## SYNTHETIC EXPERIMENT SHAP ############## 

shap.1 = ggdraw() +
  draw_image(magick::image_read_pdf("data/shap_randomU.pdf", density = 600)) +
  theme(plot.margin = unit(c(-20, 2, 0, 0), "cm"))
shap.2 = ggdraw() +
  draw_image(magick::image_read_pdf("data/shap_randomU_mean.pdf", density = 600)) +
  theme(plot.margin = unit(c(-20, 0, 0, 2), "cm"))

plot_grid(shap.1, shap.2, ncol=2,  
          labels = c("A", "B"), 
          label_size = 26,
          label_x = c(0, 0.09))

  
############## SYNTHETIC EXPERIMENT CALIBRATION & UNCERTAINTY QUALITY EVAL ############## 

calibration <- read_csv("data/synth_experiment_calibration_data.csv")

p1 = ggplot(calibration, aes(x=pms, y=empirical_frequency)) +
  geom_abline(slope=1, linetype="dashed", linewidth=1.5, color="#888888") +
  geom_line(linewidth=2, color="#CC6677") +
  geom_point(size=4, shape=17, color="#CC6677") +
  xlab("Probability level") +
  ylab("Empirical frequency") 

uncertainty_pred <- read_csv("data/synth_experiment_qual_uncertainty.csv") |>
  select("Predicted uncertainty" = cummulative_sq_error)
baseline_dist_mean <- read_csv("data/synth_experiment_qual_baseline.csv") |>
  select("Distance from mean" = cummulative_sq_error)
baseline_noise_feature <- read_csv("data/synth_experiment_qual_ground_truth_std.csv") |>
  select("Noise standard deviation" = cummulative_sq_error)


data_pas <- uncertainty_pred |>
  mutate(index = 1:n(),
         perc = index/n()) |>
  bind_cols(baseline_dist_mean) |>
  bind_cols(baseline_noise_feature) |>
  pivot_longer(-c("index", "perc"))


p2 <- ggplot(data_pas, aes(x=perc, y=value, color=name)) +
  geom_point(size=2) +
  xlab("Quantile of data") +
  ylab("Mean Squared Error (MSE)") +
  scale_x_reverse() +
  scale_color_manual(values = c("#88CCEE", "#117733", "#CC6677")) +
  guides(colour = guide_legend(override.aes = list(size=8))) +
  theme(legend.position = c(0.3,0.903),
        legend.title = element_blank())


plot_grid(p1, p2, ncol=2,  
          labels = c("A", "B"), 
          label_size = 40)


############## SYNTHETIC EXPERIMENT NOISE vs DATASET SIZE ############## 

heat_map = read_csv("data/noise_vs_dataset_size.csv")
 
ggplot(heat_map, aes(x=as.factor(n), y=as.factor(noise_scalers), fill=precision)) +
  geom_tile() +
  geom_text(aes(label=precision), size=12, color="white") +
  xlab("Dataset size (n)") +
  ylab("Noise scalar (\u03B1)") +
  scale_x_discrete(labels = function(x) format(as.numeric(x), scientific = TRUE)) +
  scale_fill_gradient(low="#6699CC", high="#332288") +
  theme(legend.position = "none")
 

##############  AGE DETECTION UNCERTAINTY QUALITY EVAL ############## 

theme_set(theme_bw(base_size = 40))

uncertainty <- read_csv("data/age_detection_qual_uncertainty.csv") |>
  select("Predicted uncertainty" = cummulative_sq_error)
predicted_age <- read_csv("data/age_detection_qual_baseline.csv") |>
  select("Predicted age" = cummulative_sq_error)


data <- uncertainty |>
  mutate(index = 1:n(),
         perc = index/n()) |>
  bind_cols(predicted_age) |>
  pivot_longer(-c("index", "perc"))

ggplot(data, aes(x=perc, y=value, color=name)) +
  geom_point(size=2) +
  xlab("Quantile of data") +
  ylab("Mean Squared Error (MSE)") +
  scale_x_reverse() +
  scale_color_manual(values = c("#88CCEE", "#CC6677")) +
  guides(colour = guide_legend(override.aes = list(size=8))) +
  theme(legend.position = "top",
        legend.title = element_blank())




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


lip_combined <- process_lipschitz(read_csv("data/lipschitz/red_wine_1_out_lipschitz.csv"), "Red Wine") |>
  union(process_lipschitz(read_csv("data/lipschitz/ailerons_1_out_lipschitz.csv"), "Ailerons")) |>
  union(process_lipschitz(read_csv("data/lipschitz/synthetic_out_lipschitz.csv"), "Synthetic")) |>
  union(process_lipschitz(read_csv("data/lipschitz/lsat_out_lipschitz.csv"), "LSAT")) |>
  mutate(dataset = factor(dataset, levels=c("Red Wine", "Ailerons", "LSAT", "Synthetic")))


ggplot(lip_combined, aes(x=dataset, y=value, fill=method)) +
  geom_boxplot() +
  scale_y_continuous(trans = "log10") +
  ylab("Lipshitz Estimate") +
  xlab("Dataset") + 
  guides(fill = guide_legend(nrow = 1, title = "Method")) +
  scale_fill_manual(values=c("#88CCEE", "#CC6677", "#44AA99", "#DDCC77", "#882255")) +
  theme_bw() +
  theme(
    legend.position = "top", 
    legend.text= element_text(size=14), 
    legend.title = element_blank(),
    axis.text.x = element_text(size=14),
    axis.title = element_text(size=16),
    plot.margin = unit(c(0, 0.05, 0, 0), "cm"))
  
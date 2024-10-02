library(cumstats)
library(tidyverse)
library(ggplot2)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


predictions <-  read_csv("test_results_variance_feature_attribution_mivolo_checkpoint.csv") |>
  select(age_pred, age_var, age_target) |>
  mutate(mean_pred = mean(age_pred),
         distance_from_mean = abs(mean_pred-age_pred))

get_removal_for_column <- function(df, column){
  
  temp <- df |> arrange(!!sym(column)) |>
    mutate(se = (age_pred-age_target)^2,
           ae = abs(age_pred-age_target),
           mse = cummean(se),
           mae = cummean(ae),
           medae = cummedian(ae),
           medse = cummedian(se),
           rmse = sqrt(mse),
           mean_var = cummean(!!sym(column)))
  
  
  temp$idx = nrow(predictions):1
  
  return(
    temp |> 
    arrange(desc(!!sym(column))) |>
    mutate(quant = (nrow(temp)-idx)/nrow(temp),
           column=column)
    )
  
}


test <- bind_rows(get_removal_for_column(predictions, "age_var"),
                  get_removal_for_column(predictions, "distance_from_mean")) |>
  mutate(column=if_else(
    column=="age_var", 
    "Predicted Uncertainty", 
    if_else(
      column=="noise_std_test", 
      "Noise Standard Deviation", 
      "Distance From Mean")
  )
  )



# ----- this is where the function should end (after this is plotting)

ggplot() +
  geom_point(data=test, aes(x=quant, y=rmse, color=column), size=3, alpha=0.4) +
  geom_point(data=test, aes(x = quant - 999999, y = rmse - 999999, colour = column)) +
  ylim(0, max(test$rmse)) +
  scale_x_reverse() +
  xlim(1, 0) +
  scale_color_manual(values = c("#332288", "#44AA99")) +
  theme_bw() +
  xlab("Uncertainty Quantile") +
  ylab("Root Mean Squared Error") +
  theme(legend.position = "top",
        legend.title =element_blank(),
        axis.title = element_text(size=18),
        axis.text = element_text(size=16),
        legend.text = element_text(size=16))



ggsave("age_detection_remove_uncertainty.pdf", plot = , width=10, height=7, device=cairo_pdf)

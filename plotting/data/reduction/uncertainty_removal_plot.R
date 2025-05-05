library(cumstats)
library(tidyverse)
library(ggplot2)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


# synthetic_qual_baseline
# synthetic_mixed_5_qual_baseline
filename <- "synthetic_mixed_5_qual_baseline"

predictions <-  read_csv(paste0(filename, ".csv")) 

get_removal_for_column <- function(df, column){
  
  temp <- df |> arrange(!!sym(column)) |>
    mutate(se = (y_preds-response)^2,
           ae = abs(y_preds-response),
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


test <- bind_rows(get_removal_for_column(predictions, "y_uncertainty"),
          get_removal_for_column(predictions, "noise_std_test"),
          get_removal_for_column(predictions, "distance_from_mean")) |>
  mutate(column=if_else(
    column=="y_uncertainty", 
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
  scale_color_manual(values = c("#332288", "#AA4499", "#44AA99")) +
  theme_bw() +
  xlab("Uncertainty Quantile") +
  ylab("Root Mean Squared Error") +
  theme(legend.position = "top",
        legend.title =element_blank(),
        axis.title = element_text(size=18),
        axis.text = element_text(size=16),
        legend.text = element_text(size=16))



ggsave(paste0(filename, "_remove_uncertainty.pdf"), plot = , width=10, height=7, device=cairo_pdf)

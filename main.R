source("functions.R")

main <- function() {
  pfvc <- read_csv("data/benchmark_pfvc.csv")
  data <- group_by(pfvc, ptid) %>% do(datum = make_datum(.))
  train <- data[["datum"]]

  set.seed(1)

  model <- fit_model(train)
  xgrid <- seq(0, 20, 0.5)
  Xgrid <- model$basis(xgrid)
  matplot(xgrid, Xgrid %*% model$param$B)

  inferences <- lapply(train, apply_model, model)
  posteriors <- combine(inferences, "posterior", .a = "rbind")
  subtypes <- data_frame(ptid = data$ptid, subtype = apply(posteriors, 1, which.max))
  pfvc <- left_join(pfvc, subtypes, "ptid")

  p <- ggplot(pfvc) + xlim(0, 20) + ylim(0, 120)
  p <- p + geom_line(aes(years_seen_full, pfvc, group = ptid))
  p <- p + facet_wrap(~ subtype)
  print(p)
}

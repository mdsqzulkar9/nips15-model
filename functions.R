library(readr)
library(dplyr)
library(mvtnorm)
library(nnet)
library(splines)


fit_model <- function(train, opts = numeric()) {
  num_subtypes <- get_option(opts, "num_subtypes", 6)
  num_coef     <- get_option(opts, "num_coef", 6)
  degree       <- get_option(opts, "degree", 2)
  v_const      <- get_option(opts, "v_const", 16.0)
  v_ou         <- get_option(opts, "v_ou", 25.0)
  l_ou         <- get_option(opts, "l_ou", 3.0)
  lambda       <- get_option(opts, "lambda", 1e-1)
  max_iter     <- get_option(opts, "max_iter", 25)
  tol          <- get_option(opts, "tol", 1e-4)

  ## Create the basis function.
  knots <- quantile_knots(combine(train, "x", .a=c), num_coef, degree)
  basis <- make_basis(knots, degree)
  precision <- lambda * penalty(basis)

  ## Create the covariance kernel.
  kernel <- make_kernel(v_const, v_ou, l_ou)

  ## Create the log-likelihood functions.
  logliks <- lapply(train, make_loglik, basis, kernel)

  ## Initialize the parameters
  ytrain <- combine(train, "y", .a = "c")
  nq <- num_subtypes + 2
  b0 <- quantile(ytrain, seq(0, 1, length.out = nq))[-c(1, nq)]
  B0 <- t(matrix(rev(b0), ncol = num_coef, nrow = num_subtypes))
  m0 <- rep(1, num_subtypes) / num_subtypes

  param <- run_em(logliks, B0, m0, precision, max_iter, tol)

  list(basis = basis, kernel = kernel, param = param)
}


run_em <- function(logliks, B0, m0, precision, max_iter, tol) {
  ss <- lapply(logliks, do.call, list(B = B0, m = m0))
  ll <- combine(ss, "obs_logl", .r = "+")
  msg("Init LL=%.04f", ll)

  for (iter in 1:max_iter) {
    B <- fit_coefficients(ss, precision)
    m <- fit_marginal(ss)
    ss <- lapply(logliks, do.call, list(B = B, m = m))

    ll_old <- ll
    ll <- combine(ss, "obs_logl", .r = "+")
    delta <- (ll - ll_old) / abs(ll_old)
    msg("%04d LL=%.04f, Convergence=%.04f", iter, ll, delta)
    if (delta < tol) break
  }

  list(B = B, m = m)
}


fit_coefficients <- function(ss, precision) {
  num_subtypes <- length(ss[[1]][["posterior"]])
  num_coef <- nrow(ss[[1]][["XX"]])
  B <- matrix(NA, num_coef, num_subtypes)

  for (i in 1:num_subtypes) {
    XX <- precision
    Xy <- numeric(num_coef)
    for (j in seq_along(ss)) {
      wj <- ss[[j]][["posterior"]][i]
      XX <- XX + wj * ss[[j]][["XX"]]
      Xy <- Xy + wj * ss[[j]][["Xy"]]
    }
    B[, i] <- solve(XX, Xy)
  }

  B
}


fit_marginal <- function(ss) {
  Z <- combine(ss, "posterior", .a = "rbind")
  X <- combine(ss, "features", .a = "rbind")
  capture.output(m <- multinom(Z ~ ., X))
  m
}


make_loglik <- function(datum, basis, kernel) {
  basis <- match.fun(basis)
  kernel <- match.fun(kernel)

  x <- datum[["x"]]
  y <- datum[["y"]]
  features <- datum[["features"]]
  df_feats <- as.data.frame(as.list(features))

  X <- basis(x)
  K <- kernel(x)

  XX <- t(X) %*% solve(K, X)

  lp_subtype <- function(m) {
    if (is.vector(m)) {
      p <- m
    } else {
      p <- predict(m, df_feats, type = "probs")
    }
    log(c(p))
  }

  lp_markers <- function(yhat) {
    dmvnorm(y, yhat, K, log = TRUE)
  }

  loglik <- function(B, m) {
    ll <- lp_subtype(m)
    for (i in 1:ncol(B)) {
      ll[i] <- ll[i] + lp_markers(c(X %*% B[, i]))
    }
    list(observed = logsumexp(ll), complete = ll)
  }

  suffstat <- function(B, m) {
    logl <- loglik(B, m)
    list(
      obs_logl  = logl[["observed"]]
    , posterior = exp(logl[["complete"]] - logl[["observed"]])
    , features  = df_feats
    , XX = XX
    , Xy = t(X) %*% c(solve(K, y))
    )
  }

  suffstat
}


make_basis <- function(knots, degree) {
  nk <- length(knots)
  boundary_knots <- knots[ c(1, nk)]
  interior_knots <- knots[-c(1, nk)]
  degree <- force(degree)

  function(x) {
    bs(x
     , knots = interior_knots
     , Boundary.knots = boundary_knots
     , degree = degree
     , intercept = TRUE
     )
  }
}


num_bases <- function(basis) {
  e <- environment(basis)
  with(e, length(interior_knots) + degree + 1)
}


penalty <- function(basis) {
  n <- num_bases(basis)
  D <- diag(1.0, n)
  D <- diff(D)
  t(D) %*% D
}


quantile_knots <- function(x, num_coef, degree) {
  num_interior <- num_coef - degree - 1
  num_knots <- num_interior + 2
  unname(quantile(x, seq(0, 1, length.out = num_knots)))
}


uniform_knots <- function(x, num_coef, degree) {
  num_interior <- num_coef - degree - 1
  num_knots <- num_interior + 2
  seq(min(x), max(x), length.out = num_knots)
}


make_kernel <- function(v_const, v_ou, l_ou) {
  v_const <- force(v_const)
  v_ou <- force(v_ou)
  l_ou <- force(l_ou)

  function(x1, x2) {
    if (missing(x2)) {
      n <- length(x1)
      matrix(v_const, n, n) +
        ou_covariance(outer(x1, x1, "-"), v_ou, l_ou) +
        diag(1.0, n)
    } else {
      n1 <- length(x1)
      n2 <- length(x2)
      matrix(v_const, n1, n2) +
        ou_covariance(outer(x1, x2, "-"), v_ou, l_ou)
    }
  }
}


ou_covariance <- function(d, v, l) {
  v * exp(- abs(d) / l)
}


make_datum <- function(patient_tbl) {
  list(
    ptid     = patient_tbl[["ptid"]][1]
  , x        = patient_tbl[["years_seen_full"]]
  , y        = patient_tbl[["pfvc"]]
  , features = model.matrix(~ female+afram+aca+scl-1, patient_tbl)[1, ]
  )
}


### --------------------------------------------------------------------------
### Utility functions.


combine <- function(structs, field, .reduce = NULL, .apply = NULL) {
  vals <- lapply(structs, "[[", field)
  if (!is.null(.reduce)) {
    return(Reduce(match.fun(.reduce), vals))
  }
  if (!is.null(.apply)) {
    return(do.call(match.fun(.apply), vals))
  }
  vals
}


logsumexp <- function(x) {
  m <- max(x)
  m + log(sum(exp(x - m)))
}


msg <- function(m, ...) {
  message(sprintf(m, ...))
}


err <- function(e, ...) {
  stop(sprintf(e, ...), call. = FALSE)
}


get_option <- function(opts, key, default) {
  if (key %in% names(opts)) {
    opts[[key]]
  } else {
    if (missing(default)) {
      err("Option %s is not specified and has no default.", key)
    } else {
      default
    }
  }
}

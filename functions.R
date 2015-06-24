library(readr)
library(dplyr)
library(ggplot2)
library(mvtnorm)
library(nnet)
library(splines)


apply_model <- function(datum, model, xnew) {
  loglik <- make_loglik(datum, model$basis, model$kernel)
  ll <- loglik[["loglik"]](model$param$b, model$param$B, model$param$m)
  if (!missing(xnew)) {
    yhat_new <- predict_means(xnew, datum, model)
    ycov_new <- model$kernel(xnew)

    if (length(datum[["x"]]) > 0) {
      yhat_old <- predict_means(datum[["x"]], datum, model)
      for (i in 1:ncol(yhat_new)) {
        yres <- c(datum[["y"]] - yhat_old[, i])
        gpp <- gp_posterior(xnew, datum[["x"]], yres, model$kernel)
        yhat_new[, i] <- yhat_new[, i] + gpp$mean
        ycov_new <- gpp$covariance
      }
    }
    output <- list(
      likelihood = ll$observed
    , posterior  = ll$posterior
    , ynew_hat   = yhat_new
    , ynew_cov   = ycov_new
    )

  } else {
    output <- list(
      likelihood = ll$observed
    , posterior  = ll$posterior
    , ynew_hat   = NULL
    , ynew_cov   = NULL
    )
  }
  output
}


predict_means <- function(x, datum, model) {
  pop_X <- t(replicate(length(x), datum[["pop_feat"]]))
  sub_X <- model$basis(x)
  c(pop_X %*% model$param$b) + sub_X %*% model$param$B
}


fit_model <- function(train, opts = numeric()) {
  num_subtypes <- get_option(opts, "num_subtypes", 8)
  num_coef     <- get_option(opts, "num_coef", 6)
  degree       <- get_option(opts, "degree", 2)
  xlo          <- get_option(opts, "xlo", NULL)
  xhi          <- get_option(opts, "xhi", NULL)
  v_const      <- get_option(opts, "v_const", 16.0)
  v_ou         <- get_option(opts, "v_ou", 36.0)
  l_ou         <- get_option(opts, "l_ou", 2.0)
  lambda       <- get_option(opts, "lambda", 1e-1)
  max_iter     <- get_option(opts, "max_iter", 25)
  tol          <- get_option(opts, "tol", 1e-4)

  ## Create the basis function.
  if (!is.null(xlo) && !is.null(xhi)) {
    ## knots <- quantile_knots(combine(train, "x", .a = c), num_coef, degree, c(xlo, xhi))
    knots <- uniform_knots(combine(train, "x", .a = c), num_coef, degree, c(xlo, xhi))
  } else {
    ## knots <- quantile_knots(combine(train, "x", .a=c), num_coef, degree)
    knots <- uniform_knots(combine(train, "x", .a=c), num_coef, degree)
  }
  basis <- make_basis(knots, degree)
  precision <- lambda * penalty(basis)

  ## Create the covariance kernel.
  kernel <- make_kernel(v_const, v_ou, l_ou)

  ## Create the log-likelihood functions.
  logliks <- lapply(train, make_loglik, basis, kernel)

  ## Initialize the parameters
  b0 <- numeric(length(train[[1]][["pop_feat"]]))
  ytrain <- combine(train, "y", .a = "c")
  nq <- num_subtypes + 2
  Bq <- quantile(ytrain, seq(0, 1, length.out = nq))[-c(1, nq)]
  B0 <- t(matrix(rev(Bq), ncol = num_coef, nrow = num_subtypes))
  m0 <- rep(1, num_subtypes) / num_subtypes

  param <- run_em(logliks, b0, B0, m0, precision, max_iter, tol)

  list(basis = basis, kernel = kernel, param = param)
}


run_em <- function(logliks, b0, B0, m0, precision, max_iter, tol) {
  N <- length(logliks)
  ith_logl <- function(i) logliks[[i]][["loglik"]]
  ith_b_ss <- function(i) logliks[[i]][["pop_suffstat"]]
  ith_B_ss <- function(i) logliks[[i]][["sub_suffstat"]]
  ith_m_ss <- function(i) logliks[[i]][["marg_suffstat"]]

  do_estep <- function(b, B, m) {
    all_logl <- lapply(1:N, function(i) ith_logl(i)(b, B, m))
    list(
      b = b, B = B, m = m
    , total_logl = combine(all_logl, "observed", .reduce = "+")
    , posteriors = combine(all_logl, "posterior")
    )
  }

  do_mstep <- function(estep) {
    pst <- estep[["posteriors"]]

    m_ss <- lapply(1:N, function(i) ith_m_ss(i)(pst[[i]]))
    m    <- fit_marg(m_ss)

    b <- estep$b
    B <- estep$B

    for (iter in 1:max_iter) {
      old_b <- b
      old_B <- B

      b_ss <- lapply(1:N, function(i) ith_b_ss(i)(B, pst[[i]]))
      b    <- fit_pop_coef(b_ss)

      for (j in 1:ncol(B)) {
        Bj_ss  <- lapply(1:N, function(i) ith_B_ss(i)(b, j, pst[[i]]))
        B[, j] <- fit_sub_coef(Bj_ss, precision)
      }

      db <- frobenius(b - old_b)
      dB <- frobenius(B - old_B)
      if (iter %% 10 == 0) {
        msg("M-step loop (%04d): db=%.03f, dB=%.03f", iter, db, dB)
      }
      if (mean(c(db, dB)) < tol) {
        break
      }
    }

    list(b = b, B = B, m = m)
  }

  estep <- do_estep(b0, B0, m0)
  msg("Init LL=%.04f", estep[["total_logl"]])

  for (iter in 1:max_iter) {
    new_param <- do_mstep(estep)
    b <- new_param[["b"]]
    B <- new_param[["B"]]
    m <- new_param[["m"]]

    ll_old <- estep[["total_logl"]]
    estep <- do_estep(b, B, m)
    delta <- (estep[["total_logl"]] - ll_old) / abs(ll_old)
    msg("%04d LL=%.04f, Convergence=%.04f", iter, estep[["total_logl"]], delta)
    if (delta < tol) break
  }

  list(b = b, B = B, m = m)
}


fit_pop_coef <- function(ss) {
  XX <- combine(ss, "XX", .reduce = "+")
  Xy <- combine(ss, "Xy", .reduce = "+")
  c(solve(XX, Xy))
}


fit_sub_coef <- function(ss, precision) {
  XX <- precision + combine(ss, "XX", .reduce = "+")
  Xy <- combine(ss, "Xy", .reduce = "+")
  c(solve(XX, Xy))
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


fit_marg <- function(ss) {
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
  pop_feat <- datum[["pop_feat"]]
  sub_feat <- datum[["sub_feat"]]

  K <- kernel(x)

  if (length(x) > 0) {
    pop_X <- t(replicate(length(x), pop_feat))
    pop_XX <- t(pop_X) %*% solve(K, pop_X)
    sub_X <- basis(x)
    sub_XX <- t(sub_X) %*% solve(K, sub_X)
  } else {
    pop_X <- NULL
    pop_XX <- NULL
    sub_X <- NULL
    sub_XX <- NULL
  }

  sub_feat_df <- as.data.frame(as.list(sub_feat))

  lp_subtype <- function(m) {
    if (is.vector(m)) {
      p <- m
    } else {
      p <- predict(m, sub_feat_df, type = "probs")
    }
    log(c(p))
  }

  lp_markers <- function(yhat) {
    dmvnorm(y, yhat, K, log = TRUE)
  }

  loglik <- function(b, B, m) {
    ll <- lp_subtype(m)
    if (length(x) > 0) {
      for (i in 1:ncol(B)) {
        yhat <- c(pop_X %*% b + sub_X %*% B[, i])
        ll[i] <- ll[i] + lp_markers(yhat)
      }
    }
    obs <- logsumexp(ll)
    pos <- exp(ll - obs)
    list(observed = obs, complete = ll, posterior = pos)
  }

  b_suffstat <- function(B, posterior) {
    yhat <- sub_X %*% B %*% posterior
    list(
      XX = pop_XX
    , Xy = t(pop_X) %*% solve(K, y - yhat)
    )
  }

  B_suffstat <- function(b, z, posterior) {
    yhat <- pop_X %*% b
    list(
      XX = posterior[z] * sub_XX
    , Xy = posterior[z] * t(sub_X) %*% solve(K, y - yhat)
    )
  }

  m_suffstat <- function(posterior) {
    list(
      posterior = posterior
    , features  = sub_feat_df
    )
  }

  list(
    loglik        = loglik
  , pop_suffstat  = b_suffstat
  , sub_suffstat  = B_suffstat
  , marg_suffstat = m_suffstat
  )
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


quantile_knots <- function(x, num_coef, degree, boundaries) {
  num_interior <- num_coef - degree - 1
  num_knots <- num_interior + 2
  knots <- unname(quantile(x, seq(0, 1, length.out = num_knots)))
  if (!missing(boundaries)) {
    knots[1] <- boundaries[1]
    knots[num_knots] <- boundaries[2]
  }
  knots
}


uniform_knots <- function(x, num_coef, degree, boundaries) {
  num_interior <- num_coef - degree - 1
  num_knots <- num_interior + 2
  if (missing(boundaries)) {
    seq(min(x), max(x), length.out = num_knots)
  } else {
    seq(boundaries[1], boundaries[2], length.out = num_knots)
  }
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


gp_posterior <- function(xnew, x, y, cov_fn) {
  K11 <- cov_fn(xnew)
  K12 <- cov_fn(xnew, x)
  K22 <- cov_fn(x)

  yhat <- c(K12 %*% solve(K22, y))
  ycov <- K11 - K12 %*% solve(K22, t(K12))

  list(mean = yhat, covariance = ycov)
}


make_datum <- function(patient_tbl) {
  list(
    ptid     = patient_tbl[["ptid"]][1]
  , x        = patient_tbl[["years_seen_full"]]
  , y        = patient_tbl[["pfvc"]]
  , sub_feat = model.matrix(~ female+afram+aca+scl-1, patient_tbl)[1, ]
  , pop_feat = model.matrix(~ female+afram-1, patient_tbl)[1, ]
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


frobenius <- function(x) {
  norm(as.matrix(x), "F")
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

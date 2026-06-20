cran <- "https://cloud.r-project.org"
needed <- c("mgcv", "gamlss", "gamlss.dist")

# Install to user library if needed
user_lib <- Sys.getenv("R_LIBS_USER")
if (nzchar(user_lib) && !dir.exists(user_lib)) dir.create(user_lib, recursive = TRUE)

to_install <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]

if (length(to_install) == 0) {
  cat("All packages already installed.\n")
} else {
  cat("Installing:", paste(to_install, collapse = ", "), "\n")
  install.packages(to_install, repos = cran)
}

cat("\nVersions:\n")
for (pkg in needed) {
  v <- tryCatch(as.character(packageVersion(pkg)), error = function(e) "MISSING")
  cat(sprintf("  %-12s %s\n", pkg, v))
}

# Smoke test the gamlss API used in simulate_r.R / benchmark_r.R.
# NOTE: predict.gamlss(..., se.fit=TRUE) silently downgrades to a plain vector
# when newdata is supplied (warning: "se.fit = TRUE is not supported for new
# data values at the moment"). simulate_r.R's try_metrics_gamlss() already
# handles this — coverage will be reported as NA for gamdist.
suppressPackageStartupMessages(library(gamlss))
set.seed(1)
df  <- data.frame(x = runif(200))
df$y <- sin(2*pi*df$x) + rnorm(200)
m    <- gamlss(y ~ pb(x), data = df, family = NO(),
               control = gamlss.control(trace = FALSE))
p    <- suppressWarnings(predict(m, what = "mu",
              newdata = data.frame(x = c(0.25, 0.75)),
              type = "response", se.fit = TRUE, data = df))
fit <- if (is.list(p)) p$fit    else p
se  <- if (is.list(p)) p$se.fit else rep(NA_real_, length(fit))
cat(sprintf("\nSmoke test OK: fit=%s  se=%s  (NA se is expected on newdata)\n",
            paste(round(fit, 3), collapse=","),
            paste(ifelse(is.na(se), "NA", round(se, 3)), collapse=",")))

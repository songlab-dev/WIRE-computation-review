cran_repo <- "https://cloud.r-project.org"

if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes", repos = cran_repo)
}

required_versions <- c(
  survival = "3.8-6",
  glmnet = "4.1-10",
  dplyr = "1.2.1",
  readr = "2.1.5",
  randomForestSRC = "3.6.2",
  gbm = "2.2.2",
  survivalmodels = "0.1.191",
  reticulate = "1.46.0"
)

for (package in names(required_versions)) {
  target <- required_versions[[package]]
  installed <- if (requireNamespace(package, quietly = TRUE)) {
    as.character(packageVersion(package))
  } else {
    NA_character_
  }
  matches <- !is.na(installed) &&
    package_version(installed) == package_version(target)
  if (!matches) {
    remotes::install_version(
      package,
      version = target,
      repos = cran_repo,
      upgrade = "never"
    )
  }
}

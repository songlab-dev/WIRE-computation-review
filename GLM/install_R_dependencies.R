#!/usr/bin/env Rscript
options(repos=c(CRAN=Sys.getenv("CRAN_REPO","https://cloud.r-project.org")),timeout=600)
args <- commandArgs(trailingOnly=TRUE)
if (any(!args %in% "--reference")) stop("Usage: Rscript install_R_dependencies.R [--reference]")
reference <- "--reference" %in% args
if (getRversion() < "4.3.0") stop("Use R >= 4.3.0")
if (reference && getRversion() != "4.4.0") stop("Reference package installation requires R 4.4.0; load that module or omit --reference for a new environment")
# fastglm 0.1.2 uses variadic Rcpp::List::create in ancillary model code.
# Its declared Imports floor is insufficient for old Rcpp releases.
if (!requireNamespace("Rcpp",quietly=TRUE) || packageVersion("Rcpp") < "1.1.0") install.packages("Rcpp")
pkgs <- c("Matrix","MASS","jsonlite","glmnet","biglm","speedglm","fastglm")
if (reference) {
  filearg <- grep("^--file=",commandArgs(trailingOnly=FALSE),value=TRUE)[1]
  root <- dirname(normalizePath(sub("^--file=","",filearg)))
  pins <- read.csv(file.path(root,"environment","R-packages.csv"),colClasses="character")
  if (!requireNamespace("remotes",quietly=TRUE)) install.packages("remotes")
  for (i in seq_len(nrow(pins))) {
    p <- pins$package[i]; version <- pins$version[i]
    if (!requireNamespace(p,quietly=TRUE) || packageVersion(p) != version)
      remotes::install_version(p,version=version,upgrade="never",dependencies=NA)
    if (packageVersion(p) != version) stop(paste("Could not install recorded version:",p,version))
  }
}
for (p in pkgs) if (!requireNamespace(p,quietly=TRUE)) install.packages(p)
if (packageVersion("fastglm") < "0.1.2") install.packages("fastglm")
stopifnot(packageVersion("fastglm") >= "0.1.2")
versions <- setNames(vapply(pkgs,function(p)as.character(packageVersion(p)),character(1)),pkgs)
write.csv(data.frame(package=names(versions),version=unname(versions)),"R_packages_installed.csv",row.names=FALSE)
print(versions)

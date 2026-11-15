#!/usr/bin/env Rscript
# Same benchmark protocol, restricted to the R implementations.
a <- commandArgs(trailingOnly=FALSE)
p <- sub("^--file=", "", a[grepl("^--file=",a)][1])
script <- file.path(dirname(normalizePath(p)),"run_benchmark.py")
args <- c(commandArgs(trailingOnly=TRUE),"--methods","R_stats","R_fastglm","R_speedglm","R_biglm","R_glmnet")
status <- system2(Sys.getenv("WIRE_PYTHON","python3"),c(shQuote(script),vapply(args,shQuote,character(1))))
quit(status=status)

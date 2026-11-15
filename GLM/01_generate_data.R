#!/usr/bin/env Rscript
# Shared cross-language generator replaces the former RDS/CSV dual export.
a <- commandArgs(trailingOnly=FALSE)
p <- sub("^--file=", "", a[grepl("^--file=",a)][1])
script <- file.path(dirname(normalizePath(p)),"01_generate_data.py")
status <- system2(Sys.getenv("WIRE_PYTHON","python3"),c(shQuote(script),vapply(commandArgs(trailingOnly=TRUE),shQuote,character(1))))
quit(status=status)

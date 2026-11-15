#!/usr/bin/env Rscript
# Export the published time/RSS tables; use --results RUN [--out TABLE_DIR].
a <- commandArgs(trailingOnly=FALSE)
p <- sub("^--file=", "", a[grepl("^--file=",a)][1])
root <- dirname(normalizePath(p))
script <- file.path(root,"export_tables.py")
python <- Sys.getenv("WIRE_PYTHON",if(file.exists(file.path(root,".venv/bin/python"))) file.path(root,".venv/bin/python") else "python3")
status <- system2(python,c(shQuote(script),vapply(commandArgs(trailingOnly=TRUE),shQuote,character(1))))
quit(status=status)

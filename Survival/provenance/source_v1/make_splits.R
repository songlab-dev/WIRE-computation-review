# Native survivalmodels validation sampling, replayed by its unmodified interface.
args <- commandArgs(TRUE)
m <- read.csv(args[1])
dir.create(args[2], recursive=TRUE, showWarnings=FALSE)
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
for (i in seq_len(nrow(m))) {
  set.seed(m$training_seed[i])
  idx <- sample(seq_len(m$n[i] * 0.8), m$n[i] * 0.2)
  write.table(idx, file.path(args[2], paste0(m$dataset_id[i], ".txt")),
              row.names=FALSE, col.names=FALSE)
}

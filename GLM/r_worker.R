#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(Matrix);library(jsonlite)})
args <- commandArgs(trailingOnly=TRUE)
task <- fromJSON(args[match("--task",args)+1],simplifyVector=TRUE)
cfg <- task$config
meta <- fromJSON(task$dataset)
read_num <- function(path,n) readBin(path,"double",n=n,size=8,endian="little")
read_x <- function(path,n,p) {
  if (grepl("\\.mtx$",path)) return(as(readMM(path),"CsparseMatrix"))
  matrix(read_num(path,n*p),nrow=n,ncol=p)
}
x <- read_x(meta$files$x,meta$n,meta$p)
y <- read_num(meta$files$y,meta$n)
foldid <- readBin(meta$files$foldid,"integer",n=meta$n,size=4,endian="little")
lambda <- read_num(meta$files$lambdas,cfg$lambda_count)
phase <- function(value) {
  tmp <- paste0(task$phase_file,".tmp")
  pid <- if (file.exists("/proc/self")) as.integer(Sys.readlink("/proc/self")) else Sys.getpid()
  write_json(list(phase=value,pid=pid),tmp,auto_unbox=TRUE);file.rename(tmp,task$phase_file)
}

fit_unpen <- function() {
  meth <- task$method
  # Matrix construction belongs to the measured fitting expression in both languages.
  if (meth == "R_stats") {
    xx <- cbind(1,as.matrix(x))
    fit <- stats::glm.fit(xx,y,family=poisson(),control=glm.control(epsilon=cfg$solver_tol,maxit=cfg$max_iter))
    return(list(coef=unname(fit$coefficients[-1]),intercept=unname(fit$coefficients[1]),
                native_converged=fit$converged,native_iterations=fit$iter))
  }
  if (meth == "R_fastglm") {
    xx <- cbind(1,x)
    fit <- fastglm::fastglm(xx,y,family=poisson(),method=3L,firth=FALSE,
                           tol=cfg$solver_tol,maxit=as.integer(cfg$max_iter))
    return(list(coef=unname(fit$coefficients[-1]),intercept=unname(fit$coefficients[1]),
                native_converged=fit$converged,native_iterations=fit$iter))
  }
  if (meth == "R_speedglm") {
    xx <- cbind(1,x)
    fit <- speedglm::speedglm.wfit(y,xx,intercept=TRUE,family=poisson(),
                       sparse=inherits(x,"sparseMatrix"),method="Cholesky",
                       acc=cfg$solver_tol,maxit=cfg$max_iter)
    return(list(coef=unname(fit$coefficients[-1]),intercept=unname(fit$coefficients[1]),
                native_converged=fit$convergence,native_iterations=fit$iter))
  }
  if (meth == "R_biglm") {
    # Data remain resident; this measures chunked fitting, not disk streaming.
    # Sparse inputs are converted one chunk at a time, not globally densified.
    pos <- 1L; names_x <- paste0("x",seq_len(meta$p))
    provider <- function(reset=FALSE) {
      if (reset) {pos <<- 1L;return(invisible(NULL))}
      if (pos > meta$n) return(NULL)
      idx <- pos:min(meta$n,pos+cfg$chunk_size-1L);pos <<- max(idx)+1L
      d <- as.data.frame(as.matrix(x[idx,,drop=FALSE]));names(d) <- names_x
      d$y <- y[idx];d
    }
    formula <- reformulate(names_x,response="y")
    fit <- biglm::bigglm(formula,provider,family=poisson(),maxit=cfg$max_iter,
                        tolerance=cfg$solver_tol,quiet=TRUE)
    b <- coef(fit)
    return(list(coef=unname(b[-1]),intercept=unname(b[1]),
                native_converged=fit$converged,native_iterations=fit$iterations))
  }
  stop("Unknown R method: ",meth)
}

fit_pen <- function() {
  old <- glmnet::glmnet.control();on.exit(do.call(glmnet::glmnet.control,old),add=TRUE)
  glmnet::glmnet.control(fdev=0,devmax=1)
  path <- function(xx,yy) {
    fit <- glmnet::glmnet(xx,yy,family="poisson",alpha=1,lambda=lambda,
                 standardize=FALSE,intercept=TRUE,thresh=cfg$glmnet_thresh,
                 maxit=as.integer(cfg$max_iter*1000))
    if (fit$jerr != 0 || length(fit$lambda) != length(lambda)) stop("glmnet path failed or truncated")
    list(b=as.matrix(fit$beta),a=fit$a0,passes=fit$npasses)
  }
  scores <- matrix(0,nrow=cfg$cv_folds,ncol=length(lambda))
  for (k in 0:(cfg$cv_folds-1)) {
    tr <- foldid != k;va <- !tr;f <- path(x[tr,,drop=FALSE],y[tr])
    pred <- exp(sweep(as.matrix(x[va,,drop=FALSE] %*% f$b),2,f$a,"+"))
    yy <- matrix(y[va],nrow=sum(va),ncol=length(lambda))
    term <- -yy+pred;positive <- yy > 0
    term[positive] <- term[positive]+yy[positive]*log(yy[positive]/pred[positive])
    scores[k+1,] <- colSums(2*term)
  }
  cv <- colSums(scores)/length(y)
  if (any(!is.finite(cv))) stop("Non-finite CV scores")
  j <- which.min(cv);f <- path(x,y)
  list(coef=unname(f$b[,j]),intercept=unname(f$a[j]),
       lambda_selected=lambda[j],lambda_index=j-1L,cv_deviance=cv[j],
       cv_scores=cv,native_converged=NULL,native_iterations=f$passes,
       lambda_boundary=j %in% c(1L,length(lambda)))
}

fitter <- if (task$method == "R_glmnet") fit_pen else fit_unpen
result <- list(status="error",timings=numeric(),warnings=character(),warmup_warnings=character())
capture <- function(fun,warm=FALSE) withCallingHandlers(fun(),warning=function(w) {
  if (warm) result$warmup_warnings <<- c(result$warmup_warnings,conditionMessage(w))
  else result$warnings <<- c(result$warnings,conditionMessage(w))
  invokeRestart("muffleWarning")
})
tryCatch({
  phase("warmup");warm <- capture(fitter,TRUE);rm(warm);invisible(gc())
  phase("fit");last <- NULL
  for (i in seq_len(task$repeats)) {
    last <- NULL;invisible(gc());start <- Sys.time()
    last <- capture(fitter);elapsed <- as.numeric(difftime(Sys.time(),start,units="secs"))
    result$timings <- c(result$timings,elapsed)
  }
  phase("evaluation");result <- c(result,last);result$status <- "ok"
},error=function(e) {result$error <<- conditionMessage(e);phase("error")})
result$warnings <- unique(result$warnings)
write_json(result,task$result_file,auto_unbox=TRUE,null="null",na="null",digits=16)
phase("done")

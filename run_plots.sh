conda activate checkspred

doit () {
  echo ${1}
  nohup nice Rscript plot_results.R ${1} > logs/${1}_plot.log 
}

doit equal &
doit notequal &

conda activate checkspred

doit () {
  nohup python models.py ${1} > logs/${1}.log && Rscript plot_results.R ${1}
}

doit equal &
doit notequal &

conda activate checkspred

doit () {
  nohup nice python -m sklearnex models.py ${1} > logs/${1}.log && Rscript plot_results.R ${1}
}

rm out/equalized/*.csv
rm out/not_equalized/*.csv

doit equal &
doit notequal &

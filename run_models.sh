conda activate checkspred

doit () {
  #nohup nice python -m sklearnex models.py ${1} > logs/${1}.log && Rscript plot_results.R ${1} > logs/${1}_plot.log 
  echo ${1}
  nohup nice Rscript plot_results.R ${1} > logs/${1}_plot.log 
}

#nohup nice python prepare_data.py > logs/prepare_data.out
#rm out/equalized/*.csv
#rm out/not_equalized/*.csv

doit equal &
doit notequal &

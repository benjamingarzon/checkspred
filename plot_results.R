rm(list = ls())
library(ggplot2)
library(dplyr)
library(stringr)

WIDTH = 10.24
HEIGHT = 8.68
DPI = 300
setwd("~/checkspred")

OUT_PATH = "./out/not_equalized/"
#OUT_PATH = "./out/equalized/" 

MODEL_RESULTS = 'y_data_S2_ridge_gc_8.csv'  # select model of interest to plot predictions and missingness

read_file = function(myfile){
  data = read.csv(file.path(OUT_PATH, myfile))
  myfile = (myfile %>% str_split("\\."))[[1]][1]
  data$model_type = (myfile %>% str_split("_"))[[1]][4]
  data$descriptor = (myfile %>% str_split("_"))[[1]][5]
  
  return(data)
}

################################################################################
# plot scores
################################################################################
scores_files = list.files(OUT_PATH,  pattern = glob2rx("scores*"))
print(scores_files)

# check results
scores_data = bind_rows(lapply(scores_files, read_file), .id = 'index')
scores_data$r2[ scores_data$r2<0 ] = 0

scores.mean = scores_data %>% group_by(target, check_type, model_type, descriptor) %>% 
  summarise(r2.mean = mean(r2, na.rm=T), r2.sd = sd(r2, na.rm=T),
    mse.mean = mean(mse, na.rm=T), mse.sd = sd(mse, na.rm=T)) %>% filter(check_type != "P5")

myplot.r2score = ggplot(scores.mean, aes(x = descriptor, y = r2.mean, 
  ymax = r2.mean + r2.sd, ymin = r2.mean - r2.sd, fill = target)) + 
  geom_col(position=position_dodge()) + 
#  ylim(-0.4, 0.8) + 
  geom_errorbar(position=position_dodge()) + 
  ylab(expression(R^2)) +
  xlab('Model') +
  facet_grid( check_type ~  model_type) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), legend.position="bottom", legend.title=element_blank())
print(myplot.r2score)
ggsave(file.path(OUT_PATH, "./figs/r2scores.png"), plot = myplot.r2score, dpi = DPI, width = WIDTH, height = HEIGHT)

myplot.msescore = ggplot(scores.mean, aes(x = descriptor, y = mse.mean, 
  ymax = mse.mean + mse.sd, ymin = mse.mean - mse.sd, fill = target)) + 
  geom_col(position=position_dodge()) + 
  geom_errorbar(position=position_dodge()) + 
  ylab('MSE') +
  xlab('Model') +
  facet_grid( check_type ~  model_type) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), legend.position="bottom", legend.title=element_blank())
print(myplot.msescore)
ggsave(file.path(OUT_PATH, "./figs/MSEscores.png"), plot = myplot.msescore, dpi = DPI, width = WIDTH, height = HEIGHT)

################################################################################
# plot n
################################################################################
n_files = list.files(OUT_PATH,  pattern = glob2rx("n_data*"))
n_data = bind_rows(lapply(n_files, read_file), .id = 'index')

myplot.n = ggplot(n_data, aes(x = descriptor, y = nsub, fill = target)) + 
  geom_col(position=position_dodge()) + 
  ylab('Number of subjects') +
  xlab('Model') +
  facet_grid( check_type ~  model_type) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), legend.position="bottom", legend.title=element_blank())
print(myplot.n)
ggsave(file.path(OUT_PATH, "./figs/n.png"), plot = myplot.n, dpi = DPI, width = WIDTH, height = HEIGHT)

################################################################################
# plot correlations between true and predicted and distribution for missing
################################################################################
Y_FILE = file.path(OUT_PATH, MODEL_RESULTS)
y_data = read.csv(Y_FILE)

myplot.y = ggplot(y_data, aes(x = y, y = ypred)) + 
  geom_point(size = 0.3) + 
  facet_wrap(. ~ target) + 
  ylab('Predicted ability') +
  xlab('True ability') 
print(myplot.y)
ggsave(file.path(OUT_PATH, "./figs/true_predicted.png"), plot = myplot.y, dpi = DPI, width = WIDTH, height = HEIGHT)

myplot.miss = ggplot(y_data, aes(x = as.factor(missing), y = y-ypred, fill = missing)) + 
  geom_violin() + 
  facet_wrap(. ~ target) + 
  ylab('Difference between true and predicted') +
  xlab('Missing') + theme(legend.position = 'none')
print(myplot.miss)
ggsave(file.path(OUT_PATH, "./figs/missing.png"), plot = myplot.miss, dpi = DPI, width = WIDTH, height = HEIGHT)


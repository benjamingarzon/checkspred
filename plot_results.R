rm(list = ls())
library(ggplot2)
library(ggpubr)
library(dplyr)
library(stringr)
library(reshape2)

# fig links
# GM4 n
# missing xgb
# gmS as main?
# calculate relative increases
# compare use cases - enough adaptive data?
# refine xgb

args = commandArgs(trailingOnly=TRUE)
if (length(args) == 0 | args[1] == 'notequal') {
  OUT_PATH = "./out/not_equalized/"
  
} else {
  OUT_PATH = "./out/equalized/" 
}

#OUT_PATH = "./out/equalized/" 

domain = c(
  'gender'='Demographics',
  'motherTongue'='Demographics',
  'mean_dles'='German',
  'mean_dsif'='German',
  'mean_ehoe'='English',
  'mean_eles'='English',
  'mean_mfur'='Math',
  'mean_mgfd'='Math',
  'mean_mzuv'='Math')

base_size = 25
theme_set(
  theme_bw(
    base_size = base_size)  #base_family = "",  #base_line_size = base_size/22,  #base_rect_size = base_size/22)
  )

WIDTH = 10.24
HEIGHT = 8.68
DPI = 1200
LABEL_SIZE = 28
setwd("~/checkspred")


MODEL_RESULTS.1 = 'y_data_S2_ridge_gM_4_all.csv'  # select model of interest to plot predictions and missingness
MODEL_RESULTS.2 = 'y_data_S2_ridge_gc_9_all.csv'  # select model of interest to plot predictions and missingness
IMPORTANCE_RESULTS = 'importances_S2_xgb_gM_4_all.csv' 

read_file = function(myfile){
  data = read.csv(file.path(OUT_PATH, myfile))
  myfile = (myfile %>% str_split("\\."))[[1]][1]
  data$model_type = (myfile %>% str_split("_"))[[1]][4]
  data$descriptor = (myfile %>% str_split("_"))[[1]][5]
  
  return(data)
}

# read files
scores_files = list.files(OUT_PATH,  pattern = glob2rx("scores*_all.csv"))
n_files = list.files(OUT_PATH,  pattern = glob2rx("n_data*_all.csv"))
best_params_files = list.files(OUT_PATH,  pattern = glob2rx("best_params*_all.csv"))

descriptors.1.ridge = c("gl", "gL", "gm", "gM", "gMs", "gMp", "gMI")
descriptors.1.xgb = c("gM")
descriptors.2 = c("gc", "gMc")


################################################################################
# plot scores
################################################################################

print(scores_files)

# check results
scores_data = bind_rows(lapply(scores_files, read_file), .id = 'index')
scores_data$r2[ scores_data$r2<0 ] = 0

# remove wle
scores_data = scores_data %>% mutate(target = str_split_i(target, "_", 2))

scores.mean = scores_data %>% group_by(target, check_type, model_type, descriptor) %>% 
  summarise(r2.mean = mean(r2, na.rm=T), r2.sd = sd(r2, na.rm=T),
    mse.mean = mean(mse, na.rm=T), mse.sd = sd(mse, na.rm=T)) 


scores.mean.1 = scores.mean %>% filter(
      (model_type == "ridge" & descriptor %in% descriptors.1.ridge)|
      (model_type == "xgb" & descriptor %in% descriptors.1.xgb)) #check_type != "P5",

scores.mean.2 = scores.mean %>% filter(model_type == "ridge", descriptor %in% descriptors.2) #check_type != "P5",

MAX_R2 = max(c(scores.mean.1$r2.mean, scores.mean.2$r2.mean))*1.1

myplot.r2score.1 = ggplot(scores.mean.1, aes(x = factor(descriptor, levels = descriptors.1.ridge), 
  y = r2.mean, ymax = r2.mean + r2.sd, ymin = r2.mean - r2.sd, fill = target)) + 
  geom_col(position=position_dodge()) + 
  geom_errorbar(position=position_dodge()) + 
  ylab(expression(R^2)) +
  xlab('Model') +
  coord_cartesian(ylim = c(0, MAX_R2)) +
  facet_grid( check_type ~ model_type, scales="free_x", space = "free_x") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), legend.position="bottom", legend.title=element_blank())
print(myplot.r2score.1)
ggsave(file.path(OUT_PATH, "./figs/r2scores.png"), plot = myplot.r2score.1, dpi = DPI, width = WIDTH, height = HEIGHT)
# 
# myplot.msescore.1 = ggplot(scores.mean.1, aes(x = descriptor, y = mse.mean, 
#   ymax = mse.mean + mse.sd, ymin = mse.mean - mse.sd, fill = target)) + 
#   geom_col(position=position_dodge()) + 
#   geom_errorbar(position=position_dodge()) + 
#   ylab('MSE') +
#   xlab('Model') +
#   facet_grid( check_type ~ model_type, scales="free_x", space = "free_x") +
#   theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), legend.position="bottom", legend.title=element_blank())
# print(myplot.msescore.1)
# ggsave(file.path(OUT_PATH, "./figs/MSEscores_ridge.png"), plot = myplot.msescore.1, dpi = DPI, width = WIDTH, height = HEIGHT)
# 

myplot.r2score.2 = ggplot(scores.mean.2, aes(x = factor(descriptor, levels = descriptors.2), y = r2.mean, 
  ymax = r2.mean + r2.sd, ymin = r2.mean - r2.sd, fill = target)) + 
  geom_col(position=position_dodge()) + 
  geom_errorbar(position=position_dodge()) + 
  ylab(expression(R^2)) +
  xlab('Model') +
  coord_cartesian(ylim = c(0, MAX_R2)) +
  facet_grid( check_type ~ model_type, scales="free_x", space = "free_x") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), legend.position="bottom", legend.title=element_blank())
print(myplot.r2score.2)
ggsave(file.path(OUT_PATH, "./figs/r2scores_checks.png"), plot = myplot.r2score.2, dpi = DPI, width = WIDTH, height = HEIGHT)


################################################################################
# plot n
################################################################################

n_data = bind_rows(lapply(n_files, read_file), .id = 'index')

n_data.1 = n_data %>% filter(
  (model_type == "ridge" & descriptor %in% descriptors.1.ridge)|
    (model_type == "xgb" & descriptor %in% descriptors.1.xgb))

n_data.2 = n_data %>% filter(model_type == "ridge", descriptor %in% descriptors.2) 

MAX_N = max(c(n_data.1$nsub, n_data.2$nsub))
myplot.n.1 = ggplot(n_data.1, aes(x = factor(descriptor, levels = descriptors.1.ridge), 
  y = nsub, fill = target)) + 
  geom_col(position=position_dodge()) + 
  ylab('Number of subjects') +
  xlab('Model') +
  ylim(0, MAX_N) + 
  facet_grid( check_type ~  model_type, scales="free_x", space = "free_x") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), legend.position="none", legend.title=element_blank())
print(myplot.n.1)
ggsave(file.path(OUT_PATH, "./figs/n.png"), plot = myplot.n.1, dpi = DPI, width = WIDTH, height = HEIGHT)

myplot.n.2 = ggplot(n_data.2, aes(x = factor(descriptor, levels = descriptors.2), y = nsub, fill = target)) + 
  geom_col(position=position_dodge()) + 
  ylab('Number of subjects') +
  xlab('Model') +
  ylim(0, MAX_N) + 
  facet_grid( check_type ~  model_type, scales="free_x", space = "free_x") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), legend.position="none", legend.title=element_blank())
print(myplot.n.2)
ggsave(file.path(OUT_PATH, "./figs/n_checks.png"), plot = myplot.n.2, dpi = DPI, width = WIDTH, height = HEIGHT)

################################################################################
# plot correlations between true and predicted and distribution for missing
################################################################################
Y_FILE = file.path(OUT_PATH, MODEL_RESULTS.1)
y_data = read.csv(Y_FILE)

y_data = y_data %>% mutate(target = str_split_i(target, "_", 2))

myplot.y = ggplot(y_data, aes(x = y, y = ypred)) + 
  geom_point(data = y_data %>% sample_n(5000), size=0.01, col ="black") + 
  geom_density_2d_filled(alpha = 0.8, bins=50) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", linewidth = 0.5) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  facet_wrap(. ~ target) + 
  ylab('Predicted ability') +
  xlab('True ability') +
  theme(legend.position='none', panel.grid = element_blank())
print(myplot.y)
myplot.y.1 = myplot.y 

Y_FILE = file.path(OUT_PATH, MODEL_RESULTS.2)
y_data = read.csv(Y_FILE)

y_data = y_data %>% mutate(target = str_split_i(target, "_", 2))

myplot.y = ggplot(y_data, aes(x = y, y = ypred)) + 
  geom_point(data = y_data %>% sample_n(5000), size = 0.01, col ="black") + 
  geom_density_2d_filled(alpha = 0.8, bins=50) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", size = 0.5) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  facet_wrap(. ~ target) + 
  ylab('Predicted ability') +
  xlab('True ability') +
  theme(legend.position='none', panel.grid = element_blank())
print(myplot.y)
myplot.y.2 = myplot.y

ggsave(file.path(OUT_PATH, "./figs/true_predicted.png"), plot = myplot.y, dpi = DPI, width = WIDTH, height = HEIGHT)

myplot.miss = ggplot(y_data, aes(x = as.factor(missing), y = y-ypred, fill = missing)) + 
  geom_violin() + 
  facet_wrap(. ~ target) + 
  ylab('Difference between true and predicted') +
  xlab('Missing') + theme(legend.position = 'none')
print(myplot.miss)
ggsave(file.path(OUT_PATH, "./figs/missing.png"), plot = myplot.miss, dpi = DPI, width = WIDTH, height = HEIGHT)

################################################################################
# plot feature importances 
################################################################################

importances = read.csv(file.path(OUT_PATH, IMPORTANCE_RESULTS))
importances = importances %>% mutate(target = str_split_i(target, "_", 2))
importances = melt(importances, id.vars = "target", variable.name = "feature")

importances.mean = importances %>% group_by(target, feature) %>% 
  summarise(imp.mean = mean(value, na.rm=T), imp.sd = sd(value, na.rm=T)) 

importances.mean$domain = as.factor(domain[importances.mean$feature])

myplot.importance = ggplot(importances.mean, aes(x = feature, 
  y = imp.mean, 
  ymax = imp.mean + imp.sd, 
  ymin = imp.mean - imp.sd, fill = domain)) + 
  geom_col() + 
  geom_errorbar(width=0.2, size=0.5) + 
  coord_flip() +
  facet_grid(. ~ target) + 
  ylab('Feature importance') +
  xlab('Feature') +
  theme(legend.position="bottom", legend.title=element_blank(), axis.text.x = element_text(size = 10))

print(myplot.importance)

ggsave(file.path(OUT_PATH, "./figs/importance.png"), plot = myplot.importance, dpi = DPI, width = WIDTH, height = HEIGHT*0.7)

################################################################################
# plot parameters
################################################################################

print(best_params_files)

# check results
best_params_data = bind_rows(lapply(best_params_files, read_file), .id = 'index')
best_params.melt = melt(best_params_data, id = c("index", "model_type", "descriptor"), na.rm=T, variable.name = "parameter")
myplot.best_params = ggplot(best_params.melt, aes(x = value, fill = descriptor)) + 
   geom_histogram(alpha = 0.5) + 
   ylab('Frequency') +
   xlab('Parameter value') +
   facet_grid( model_type + descriptor ~ parameter, scales = "free_x") +
   theme(legend.position="bottom", legend.title=element_blank())
print(myplot.best_params)
ggsave(file.path(OUT_PATH, "./figs/best_params.png"), plot = myplot.best_params, dpi = DPI, width = WIDTH, height = HEIGHT)


################################################################################
# compose final figures
################################################################################
myplot.models = ggarrange(
  myplot.r2score.1, myplot.r2score.2,
  myplot.n.1, myplot.n.2,  
  labels = c("A", "C", "B", "D"), ncol = 2, nrow = 2, heights = c(2, 1), widths = c(2,1), 
  font.label=list(size=LABEL_SIZE))

ggsave(file.path(OUT_PATH, "./figs/model_comparison.png"), 
  plot = myplot.models, dpi = DPI, 
  width = WIDTH*2, height = HEIGHT*2)

myplot.scatter = ggarrange(
  myplot.y.1, myplot.y.2,
  labels = c("A", "B"), ncol = 2, nrow = 1, 
  font.label=list(size=LABEL_SIZE))

ggsave(file.path(OUT_PATH, "./figs/true_predicted.png"), 
  plot = myplot.scatter, dpi = DPI, 
  width = WIDTH*2, height = HEIGHT)
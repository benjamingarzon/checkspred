rm(list = ls())
library(ggplot2)
library(ggpubr)
library(dplyr)
library(stringr)
library(reshape2)


args = commandArgs(trailingOnly=TRUE)
#args[1] == 'equal'
if (length(args) == 0 | args[1] == 'notequal') {
  EQUALIZED = F
  OUT_PATH = "./out/not_equalized/"
  
} else {
  EQUALIZED = T
  OUT_PATH = "./out/equalized/" 
}

#selected_scales = c('dles', 'ehoe', 'mfur', 'mgfd')
selected_scales = c('dles', 'mfur')

domain = c(
  'gender'='Demographics',
  'mother_tongue'='Demographics',
  'years_from_start'='Use',
  'previous_sessions'='Use',
  'frequency'='Use',
  'mean_dles'='German',
  'mean_dsif'='German',
  'mean_ehoe'='English',
  'mean_eles'='English',
  'mean_esif'='English',
  'mean_fhoe'='French',
  'mean_fles'='French',
  'mean_fsif'='French',
  'mean_mfur'='Math',
  'mean_mgfd'='Math',
  'mean_mzuv'='Math',
  'dles'='German',
  'dsif'='German',
  'ehoe'='English',
  'eles'='English',
  'esif'='English',
  'fhoe'='French',
  'fles'='French',
  'fsif'='French',
  'mfur'='Math',
  'mgfd'='Math',
  'mzuv'='Math'
)

target_labels =  c('dles' = expression(dles[SA]),
                   'dsif' = expression(dsif[SA]),
                   'ehoe' = expression(ehoe[SA]),
                   'eles' = expression(eles[SA]),
                   'fhoe' = expression(fhoe[SA]),
                   'fles' = expression(fles[SA]),
                   'mfur' = expression(mfur[SA]),
                   'mgfd' = expression(mgfd[SA]),
                   'mzuv' = expression(mzuv[SA]))

target_labels2 =  c('dles' = 'dles[SA]',
  'dsif' = 'dsif[SA]',
  'ehoe' = 'ehoe[SA]',
  'eles' = 'eles[SA]',
  'fhoe' = 'fhoe[SA]',
  'fles' = 'fles[SA]',
  'mfur' = 'mfur[SA]',
  'mgfd' = 'mgfd[SA]',
  'mzuv' = 'mzuv[SA]')

feature_labels =  c('gender'='gender',
  'mother_tongue'= 'mother tongue',
  'years_from_start'= 'years from start',
  'previous_sessions'= 'previous sessions',
  'frequency'= 'frequency',
  'mean_dles'= expression(mean~dles[FA]),
  'mean_dsif'= expression(mean~dsif[FA]),
  'mean_ehoe'= expression(mean~ehoe[FA]),
  'mean_eles'= expression(mean~eles[FA]),
  'mean_esif'= expression(mean~esif[FA]),
  'mean_fhoe'= expression(mean~fhoe[FA]),
  'mean_fles'= expression(mean~fles[FA]),
  'mean_fsif'= expression(mean~fsif[FA]),
  'mean_mfur'= expression(mean~mfur[FA]),
  'mean_mgfd'= expression(mean~mgfd[FA]),
  'mean_mzuv'= expression(mean~mzuv[FA]))

  
base_size = 24
theme_set(
  theme_bw(
    base_size = base_size)  #base_family = "",  #base_line_size = base_size/22,  #base_rect_size = base_size/22)
)

myPalette = c("red", "blue")

WIDTH = 7.2
HEIGHT = 6.8
DPI = 1000
LABEL_SIZE = 28
SCATTER_SAMPLE = 500
CUTS = c(-7, -5, -3, -1, 1, 3, 5, 7)
CUT_LABELS = c(-6, -4, -2, 0, 2, 4, 6)

CUTS = seq(-6.5, 6.5) 
CUT_LABELS = seq(-6, 6) 

setwd("~/checkspred")

# s2 or 3
IMPORTANCE_RESULTS = 'importances_S2_xgb_gM_4_all.csv' 
#IMPORTANCE_RESULTS = 'importances_S2_ridge_gM_4_all.csv' 
MODEL_RESULTS.1 = 'y_data_S2_ridge_gM_4_all.csv'  # select model of interest to plot predictions and missingness
MODEL_RESULTS.2 = 'y_data_S2_ridge_gc_9_all.csv'  # select model of interest to plot predictions and missingness
MODEL_X.1 = 'X_data_S2_ridge_gM_4_all.csv'
MODEL_X.2 = 'X_data_S2_ridge_gc_9_all.csv'


read_file = function(myfile){
  data = read.csv(file.path(OUT_PATH, myfile))
  myfile = (myfile %>% str_split("\\."))[[1]][1]
  data$model_type = (myfile %>% str_split("_"))[[1]][4]
  data$descriptor = (myfile %>% str_split("_"))[[1]][5]
  
  return(data)
}

rsquared = function(y, ypred){
  model = lm(ypred ~ y)
  return(summary(model)$r.squared)
}

firstup <- function(x) {
  substr(x, 1, 1) <- toupper(substr(x, 1, 1))
  return(x[1])
}

pretty_features = function(labels){
  labels = gsub('_', ' ', labels)
  #labels = sapply(labels, firstup)
  return(labels)
}


# read files
scores_files = list.files(OUT_PATH,  pattern = glob2rx("scores*_all.csv"))
n_files = list.files(OUT_PATH,  pattern = glob2rx("n_data*_all.csv"))
best_params_files = list.files(OUT_PATH,  pattern = glob2rx("best_params*_all.csv"))

descriptors.1.ridge = c("g", "gl", "gL", "gm", "gM", "gMs", "gMp", "gMI")
descriptors.1.xgb = c("gM")
descriptors.1.enet = descriptors.1.ols = descriptors.1.xgb
descriptors.2 = c("gc", "gMc")
grade_labels = c('S2' = 'Gr. 8', 'S3' = 'Gr. 9', 'P3' = 'Gr. 3', 'P5' = 'Gr. 5')

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
    mse.mean = mean(mse, na.rm=T), mse.sd = sd(mse, na.rm=T)) %>% mutate(grade = grade_labels[check_type]) 

scores.mean.1 = scores.mean %>% filter(
#    (model_type == "ols" & descriptor %in% descriptors.1.ols)|
#    (model_type == "enet" & descriptor %in% descriptors.1.enet)|
    (model_type == "ridge" & descriptor %in% descriptors.1.ridge & check_type != "P5")|
    (model_type == "xgb" & descriptor %in% descriptors.1.xgb & check_type != "P5")) 

scores.mean.2 = scores.mean %>% filter(model_type == "ridge", descriptor %in% descriptors.2 & check_type != "P5") #check_type != "P5",
scores.ranges = scores.mean %>% group_by(grade, check_type, model_type, descriptor) %>% 
  summarize(var.min = round(min(r2.mean)*100), var.max = round(max(r2.mean)*100))

print(scores.ranges %>% filter(descriptor %in% c('gM', 'gc') , model_type %in% c('ridge', 'xgb') )%>% arrange(descriptor, model_type, check_type))

MAX_R2 = max(c(scores.mean.1$r2.mean, scores.mean.2$r2.mean))*1.1
MAX_R2.1 = max(scores.mean.1$r2.mean)*1.1
MAX_R2.2 = max(scores.mean.2$r2.mean)*1.1

myplot.r2score.1 = ggplot(scores.mean.1, aes(x = factor(descriptor, levels = descriptors.1.ridge), 
  y = r2.mean, ymax = r2.mean + r2.sd, ymin = r2.mean - r2.sd, fill = target)) + 
  geom_col(position=position_dodge()) + 
  geom_errorbar(position=position_dodge()) + 
  ylab(expression(R^2)) +
  xlab('Model') +
  coord_cartesian(ylim = c(0, MAX_R2.1)) +
  facet_grid( grade ~ model_type, scales="free_x", space = "free_x") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), legend.position="none", legend.title=element_blank())
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
  coord_cartesian(ylim = c(0, MAX_R2.2)) +
  facet_grid( grade ~ model_type, scales="free_x", space = "free_x") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), legend.position="none", legend.title=element_blank())
print(myplot.r2score.2)
ggsave(file.path(OUT_PATH, "./figs/r2scores_checks.png"), plot = myplot.r2score.2, dpi = DPI, width = WIDTH, height = HEIGHT)


################################################################################
# plot n
################################################################################

n_data = bind_rows(lapply(n_files, read_file), .id = 'index')
n_data$nsub = log(n_data$nsub, 10)
# remove wle
n_data = n_data %>% mutate(target = str_split_i(target, "_", 2)) 

n_data = n_data %>% mutate(grade = grade_labels[check_type])
n_data.1 = n_data %>% filter(
#  (model_type == "ols" & descriptor %in% descriptors.1.ols)|
#  (model_type == "enet" & descriptor %in% descriptors.1.enet)|
  (model_type == "ridge" & descriptor %in% descriptors.1.ridge & check_type != "P5")|
  (model_type == "xgb" & descriptor %in% descriptors.1.xgb & check_type != "P5"))

n_data.2 = n_data %>% filter(model_type == "ridge", descriptor %in% descriptors.2 & check_type != "P5") 

MAX_N = max(c(n_data.1$nsub, n_data.2$nsub))
MAX_N.1 = max(n_data.1$nsub)
MAX_N.2 = max(n_data.2$nsub)

myplot.n.1 = ggplot(n_data.1, aes(x = factor(descriptor, levels = descriptors.1.ridge), 
  y = nsub, fill = target)) + 
  geom_col(position=position_dodge()) + 
  ylab(expression(paste(log[10], "(sample size)"))) +
  xlab('Model') +
  ylim(0, MAX_N.1) + 
  facet_grid( grade ~  model_type, scales="free_x", space = "free_x") +
  scale_fill_discrete(labels = target_labels) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), legend.position="bottom", legend.title=element_blank())
print(myplot.n.1)
ggsave(file.path(OUT_PATH, "./figs/n.png"), plot = myplot.n.1, dpi = DPI, width = WIDTH, height = HEIGHT)

myplot.n.2 = ggplot(n_data.2, aes(x = factor(descriptor, levels = descriptors.2), y = nsub, fill = target)) + 
  geom_col(position=position_dodge()) + 
  ylab(expression(paste(log[10], "(sample size)"))) +
  xlab('Model') +
  ylim(0, MAX_N.2) + 
  facet_grid( grade ~  model_type, scales="free_x", space = "free_x") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), legend.position="bottom") +
  theme(legend.text = element_blank(), legend.title = element_blank()) +
  guides(fill=guide_legend(override.aes=list(fill=NA)))
print(myplot.n.2)
ggsave(file.path(OUT_PATH, "./figs/n_checks.png"), plot = myplot.n.2, dpi = DPI, width = WIDTH, height = HEIGHT)

if (!EQUALIZED){
################################################################################
# plot correlations between true and predicted and distribution for missing
################################################################################
Y_FILE = file.path(OUT_PATH, MODEL_RESULTS.1)
y_data = read.csv(Y_FILE)

X_FILE = file.path(OUT_PATH, MODEL_X.1)
X_data = read.csv(X_FILE)
y_data = cbind(y_data, X_data[, c('gender', 'mother_tongue', 'frequency', 'previous_sessions', 'years_from_start')])

y_data = y_data %>% 
  mutate(target = str_split_i(target, "_", 2),
         y.cut = cut(y, CUTS, labels=CUT_LABELS),
         gender_label = as.factor(ifelse(gender == 1, 'Male', 'Female')),
         mother_tongue_label = as.factor(ifelse(mother_tongue == 1, 'German', 'Other')),
         err = (ypred - y), err2 = err^2 ) 

n_folds = length(unique(y_data$fold))
y_data.scales = y_data %>% filter(target %in% selected_scales)
y_sample = y_data.scales %>% group_by(target, fold) %>% sample_n(SCATTER_SAMPLE%/%(n_folds))

myplot.y.1 = ggplot(y_data.scales, aes(x = y, y = ypred)) + 
  geom_point(data = y_sample, size=0.01, col ="black") + 
  geom_density_2d_filled(alpha = 0.8, bins=50) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", linewidth = 0.5) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  facet_wrap(. ~ target) + 
  xlim(-3, 3) + 
  ylim(-3, 3) + 
  ylab('Predicted Ability') +
  xlab('Observed Ability') +
  theme(legend.position='none', panel.grid = element_blank())
print(myplot.y.1)

myplot.level.1 = ggplot(y_data.scales, aes(x=y, y=ypred-y)) + 
  geom_point(data = y_sample, size = 0.01, col = "black") + 
  geom_density_2d_filled(alpha = 0.8, bins=50) +
  geom_abline(intercept = 0, slope = 0, linetype = "dashed", linewidth = 0.5) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  facet_wrap(. ~ target) + 
  xlim(-3, 3) + 
  ylim(-3, 3) + 
  ylab('Predicted Ability - Observed Ability') +
  xlab('Observed Ability') +
  theme(legend.position='none', panel.grid = element_blank())
print(myplot.level.1)

# gender
rsquared_data.gender = y_data %>% group_by(gender_label, target, check_type, fold) %>% 
  summarise(r2 = rsquared(y, ypred))

rsquared.gender.mean = rsquared_data.gender %>% group_by(gender_label, target, check_type) %>% 
  summarise(r2.mean = mean(r2, na.rm=T), r2.sd = sd(r2, na.rm=T)) %>% mutate(grade = grade_labels[check_type]) 

myplot.gender.1 = ggplot(rsquared.gender.mean, aes(x=target, fill=gender_label, 
  y=r2.mean, ymax=r2.mean + r2.sd, 
  ymin=r2.mean - r2.sd)) + 
  geom_col(position=position_dodge()) + 
  geom_errorbar(position=position_dodge()) +
  scale_fill_manual(values=myPalette) + 
  ylab(expression(R^2)) +
  xlab('Competence Domain') +
  labs(fill = "Gender") +
  theme(legend.position='bottom', panel.grid = element_blank())
print(myplot.gender.1)


# mother tongue
n.mother_tongue = y_data %>% filter(y.cut %in% seq(-2, 2)) %>% group_by(mother_tongue_label, target, check_type, fold) %>% 
  summarise(n_obs = n()) 
MIN_OBS = min(n.mother_tongue$n_obs)

rsquared_data.mother_tongue = y_data %>% group_by(mother_tongue_label, target, check_type, fold) %>% 
  #sample_n(MIN_OBS) %>% 
  summarise(r2 = rsquared(y, ypred))

rsquared.mother_tongue.mean = rsquared_data.mother_tongue %>% group_by(mother_tongue_label, target, check_type) %>% 
  summarise(r2.mean = mean(r2, na.rm=T), r2.sd = sd(r2, na.rm=T)) %>% mutate(grade = grade_labels[check_type]) 

myplot.mother_tongue.1 = ggplot(rsquared.mother_tongue.mean, aes(x=target, fill=mother_tongue_label, 
  y = r2.mean, ymax = r2.mean + r2.sd, 
  ymin = r2.mean - r2.sd)) + 
  geom_col(position=position_dodge()) + 
  geom_errorbar(position=position_dodge()) +
  scale_fill_manual(values=myPalette) + 
  ylab(expression(R^2)) +
  xlab('Competence Domain') +
  labs(fill = "Mother Tongue") +
  theme(legend.position='bottom', panel.grid = element_blank())
print(myplot.mother_tongue.1)

# gender and ability
rsquared_data.gender = y_data %>% group_by(gender_label, target, check_type, fold, y.cut) %>% 
  summarise(r2 = rsquared(y, ypred))

rsquared.gender.mean = rsquared_data.gender %>% group_by(gender_label, target, check_type, y.cut) %>% 
  summarise(r2.mean = mean(r2, na.rm=T), r2.sd = sd(r2, na.rm=T)) %>% mutate(grade = grade_labels[check_type]) 

myplot.gender.abil.1 = ggplot(rsquared.gender.mean %>% filter(y.cut %in% seq(-2, 2)), 
  aes(x=target, fill=gender_label, 
  y = r2.mean, ymax = r2.mean + r2.sd, 
  ymin = r2.mean - r2.sd)) + 
  geom_col(position=position_dodge()) + 
  geom_errorbar(position=position_dodge()) +
  scale_fill_manual(values=myPalette) + 
  ylab(expression(R^2)) +
  xlab('Competence Domain') +
  facet_grid(. ~ y.cut ) + 
  theme(legend.position='none', panel.grid = element_blank(), axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
print(myplot.gender.abil.1)

# mother tongue and ability
n.mother_tongue = y_data %>% filter(y.cut %in% seq(-2, 2)) %>% group_by(mother_tongue_label, target, check_type, fold, y.cut) %>% 
  summarise(n_obs = n()) 
MIN_OBS = min(n.mother_tongue$n_obs)

rsquared_data.mother_tongue = y_data %>% filter(y.cut %in% seq(-2, 2)) %>% group_by(mother_tongue_label, target, check_type, fold, y.cut) %>% 
  sample_n (MIN_OBS)%>% 
  summarise(r2 = rsquared(y, ypred))

rsquared.mother_tongue.mean = rsquared_data.mother_tongue %>% group_by(mother_tongue_label, target, check_type, y.cut) %>% 
  summarise(r2.mean = mean(r2, na.rm=T), r2.sd = sd(r2, na.rm=T)) %>% mutate(grade = grade_labels[check_type]) 

myplot.mother_tongue.abil.1 = ggplot(rsquared.mother_tongue.mean %>% filter(y.cut %in% seq(-2, 2)), 
  aes(x=target, fill=mother_tongue_label, 
    y = r2.mean, ymax = r2.mean + r2.sd, 
    ymin = r2.mean - r2.sd)) + 
  geom_col(position=position_dodge()) + 
  geom_errorbar(position=position_dodge()) +
  scale_fill_manual(values=myPalette) + 
  ylab(expression(R^2)) +
  xlab('Competence Domain') +
  facet_grid(. ~ y.cut ) + 
  theme(legend.position='none', panel.grid = element_blank(), axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
print(myplot.mother_tongue.abil.1)

myplot.mother_tongue.abil.1 = ggplot(rsquared.mother_tongue.mean %>% filter(y.cut %in% seq(-2, 2)), 
  aes(x=y.cut, fill=mother_tongue_label, 
    y = r2.mean, ymax = r2.mean + r2.sd, 
    ymin = r2.mean - r2.sd)) + 
  geom_col(position=position_dodge()) + 
  geom_errorbar(position=position_dodge()) +
  scale_fill_manual(values=myPalette) + 
  ylab(expression(R^2)) +
  xlab('Competence Domain') +
  facet_grid(. ~ target ) + 
  theme(legend.position='none', panel.grid = element_blank(), axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
print(myplot.mother_tongue.abil.1)

# check the effect of different factors on error

# covariates = c("gender", "mother_tongue", 
#                "frequency" , "years_from_start", 
#                "previous_sessions")

covariates = c("gender", "mother_tongue")

pvalues = NULL
coefs = NULL
for (correct in c("Corrected", "Raw")){
for (ff in seq(0, 9)){
for (cc in covariates){
y_data$covariate = scale(unlist(y_data[cc]))
y_data$y.1 = scale(y_data$y)
y_data$y.2 = scale(y_data$y)^2
for (tt in unique(y_data$target)){
  print(paste(cc, tt))
  #lmfactors = lmer(log(err2) ~ 1 + y.1 + y.2 + covariate + (y.1+y.2|fold), 
  #data = subset(y_data, target == tt))
  if (correct == "Corrected"){
  lmfactors = lm(err ~ 1 + covariate + poly(y.1, 2), # + y.2, 
    data = subset(y_data, target == tt & fold == ff))
  } else {
    lmfactors = lm(err ~ 1 + covariate, 
      data = subset(y_data, target == tt & fold == ff))
  }
  
  coefs = rbind(coefs, c(
    correct,
    ff,
    cc, 
    tt,
    summary(lmfactors)$coefficients['covariate', 't value']))
  pvalues = rbind(pvalues, c(
    correct,
    ff,
    cc, 
    tt,
    summary(lmfactors)$coefficients['covariate', 'Pr(>|t|)']))
}
}
}
}
pvalues = as.data.frame(pvalues)
coefs = as.data.frame(coefs)
colnames(pvalues) = colnames(coefs) = c('correct', 'fold', 'covar', 'target', 'value')
pvalues$value = as.numeric(pvalues$value)
coefs$value = as.numeric(coefs$value)

coefs.mean = coefs %>% group_by(correct, covar, target) %>% 
  summarise(value.mean = mean(value), value.sd = sd(value))
pvalues$coef = coefs$value

pvalues.mean = pvalues %>% arrange(fold) %>% group_by(correct, covar, target) %>% 
  summarise(value.mean = mean(coef), pvalue = value[which.min(abs(coef-value.mean))])  
coefs.mean$pvalue.adj = p.adjust(pvalues.mean$pvalue, method = 'fdr')
coefs.mean$sig = ifelse(coefs.mean$pvalue.adj < 0.05, '*', '')
myplot.error = ggplot(coefs.mean, aes(x=target, y=value.mean, fill=correct, 
  ymax=value.mean + value.sd, ymin=value.mean-value.sd, label=sig)) + 
  geom_col(position=position_dodge()) + 
  geom_errorbar(position=position_dodge()) +
  geom_text(size=6) +
  facet_wrap(. ~ pretty_features(covar)) +
  scale_fill_manual(values=myPalette) + 
  xlab('Competence Domain') +
  ylab('t statistic') +
  theme(legend.position='bottom', panel.grid = element_blank(), legend.title=element_blank())
print(myplot.error)

# for the non-natives, the model overpredicted their ability, makes a larger error as a group
if (F){
# get the fold corresponding to the median effect
dd = subset(y_data, target == "dles" & fold == 9)
lmfactors = lm(err ~ 1 + poly(scale(y), degree=2) + mother_tongue, 
dd)
lmfactors0 = lm(err ~ 1 + mother_tongue, 
  dd)
print(summary(lmfactors))
print(summary(lmfactors0))
dd$res = lm(err ~ 1 + poly(scale(y), degree=2), dd)$residuals
plot(dd$y, dd$err)
boxplot(dd$gender, dd$y)
par(mfrow=c(2,2))
boxplot(y ~ mother_tongue_label, data=dd, ylim=c(-2,2), xlab='')
boxplot(err ~ mother_tongue_label, data=dd, ylim=c(-2,2), xlab='')
boxplot(res ~ mother_tongue_label, data=dd, ylim=c(-2,2), xlab='')
plot(dd$y, dd$err)
plot(dd$y, dd$err, col = dd$mother_tongue+2)
plot(dd$y, dd$res, col = dd$mother_tongue+2)
myplot.y.mother_tongue = ggplot(y_data %>% filter(target=="dles"), 
  aes(x = y, y = ypred, col=mother_tongue_label)) + 
  geom_point(size = 0.1) + 
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", linewidth = 0.5) +
  facet_wrap(. ~ target) + 
  xlim(-3, 3) + 
  ylim(-3, 3) + 
  scale_color_manual(values=myPalette) + 
  ylab('Predicted Ability') +
  xlab('Observed Ability') +
  theme(legend.position='bottom', panel.grid = element_blank(), legend.title=element_blank())
print(myplot.y.mother_tongue)

print(table(dd$mother_tongue)/sum(table(dd$mother_tongue)))


}
stophere
# myplot.gender.1 = ggplot(y_data, aes(x=gender_label, fill=gender_label, y=ypred-y)) + 
#   #geom_point(data = y_data %>% sample_n(SCATTER_SAMPLE), size = 0.01, col = "black") + 
#   geom_abline(intercept = 0, slope = 0, linetype = "dashed", linewidth = 0.5) +
#   geom_violin(alpha=0.8) + 
#   facet_wrap(. ~ target) + 
#   ylab('Predicted Ability - True Ability') +
#   xlab('Gender') +
#   theme(legend.position='none', panel.grid = element_blank())
# print(myplot.gender.1)
# 
# myplot.mother_tongue.1 = ggplot(y_data, aes(x=mother_tongue_label, fill=mother_tongue_label, y=ypred-y)) + 
#   #geom_point(data = y_data %>% sample_n(SCATTER_SAMPLE), size = 0.01, col = "black") + 
#   geom_abline(intercept = 0, slope = 0, linetype = "dashed", linewidth = 0.5) +
#   geom_violin(alpha=0.8) + 
#   facet_wrap(. ~ target) + 
#   ylab('Predicted Ability - True Ability') +
#   xlab('Mother Tongue') +
#   theme(legend.position='none', panel.grid = element_blank())
# print(myplot.mother_tongue.1)

Y_FILE = file.path(OUT_PATH, MODEL_RESULTS.2)
y_data = read.csv(Y_FILE)
X_FILE = file.path(OUT_PATH, MODEL_X.2)
X_data = read.csv(X_FILE)
y_data = cbind(y_data, X_data[, c('gender', 'mother_tongue', 'frequency', 'previous_sessions', 'years_from_start')])

y_data = y_data %>% 
  mutate(target = str_split_i(target, "_", 2),
    y.cut = cut(y, CUTS, labels=CUT_LABELS),
    gender_label = as.factor(ifelse(gender == 1, 'Male', 'Female')),
    mother_tongue_label = as.factor(ifelse(mother_tongue == 1, 'German', 'Other'))) 

n_folds = length(unique(y_data$fold))
y_data.scales = y_data %>% filter(target %in% selected_scales)
y_sample = y_data.scales %>% group_by(target, fold) %>% sample_n(SCATTER_SAMPLE%/%(n_folds))

myplot.y.2 = ggplot(y_data.scales, aes(x = y, y = ypred)) + 
  geom_point(data = y_sample, size = 0.01, col ="black") + 
  geom_density_2d_filled(alpha = 0.8, bins=50) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", linewidth = 0.5) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  facet_wrap(. ~ target) + 
  xlim(-3, 3) + 
  ylim(-3, 3) + 
  ylab('Predicted Ability') +
  xlab('Observed Ability') +
  theme(legend.position='none', panel.grid = element_blank())
print(myplot.y.2)

myplot.level.2 = ggplot(y_data.scales, aes(x = y, y=ypred-y)) + 
  geom_point(data = y_sample, size = 0.01, col = "black") + 
  geom_density_2d_filled(alpha = 0.8, bins=50) +
  geom_abline(intercept = 0, slope = 0, linetype = "dashed", linewidth = 0.5) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  facet_wrap(. ~ target) + 
  xlim(-3, 3) + 
  ylim(-3, 3) + 
  ylab('Predicted Ability - Observed Ability') +
  xlab('Observed Ability') +
  theme(legend.position='none', panel.grid = element_blank())
print(myplot.level.2)

rsquared_data.gender = y_data %>% group_by(gender_label, target, check_type, fold) %>% 
  summarise(r2 = rsquared(y, ypred))

rsquared.gender.mean = rsquared_data.gender %>% group_by(gender_label, target, check_type) %>% 
  summarise(r2.mean = mean(r2, na.rm=T), r2.sd = sd(r2, na.rm=T)) %>% mutate(grade = grade_labels[check_type]) 

myplot.gender.2 = ggplot(rsquared.gender.mean, aes(x=target, fill=gender_label, 
  y=r2.mean, ymax=r2.mean + r2.sd, 
  ymin=r2.mean - r2.sd)) + 
  geom_col(position=position_dodge()) + 
  geom_errorbar(position=position_dodge()) +
  scale_fill_manual(values=myPalette) + 
  ylab(expression(R^2)) +
  xlab('Competence Domain') +
  theme(legend.position='none', panel.grid = element_blank())
print(myplot.gender.2)

rsquared_data.mother_tongue = y_data %>% group_by(mother_tongue_label, target, check_type, fold) %>% 
  summarise(r2 = rsquared(y, ypred))

rsquared.mother_tongue.mean = rsquared_data.mother_tongue %>% group_by(mother_tongue_label, target, check_type) %>% 
  summarise(r2.mean = mean(r2, na.rm=T), r2.sd = sd(r2, na.rm=T)) %>% mutate(grade = grade_labels[check_type]) 

myplot.mother_tongue.2 = ggplot(rsquared.mother_tongue.mean, aes(x=target, fill=mother_tongue_label, 
  y = r2.mean, ymax = r2.mean + r2.sd, 
  ymin = r2.mean - r2.sd)) + 
  geom_col(position=position_dodge()) + 
  geom_errorbar(position=position_dodge()) +
  scale_fill_manual(values=myPalette) + 
  ylab(expression(R^2)) +
  xlab('Competence Domain') +
  theme(legend.position='none', panel.grid = element_blank())
print(myplot.mother_tongue.2)

myplot.miss = ggplot(y_data, aes(x = as.factor(missing), y = y-ypred, fill = missing)) + 
  geom_violin() + 
  facet_wrap(. ~ target) + 
  ylab('Predicted Ability - True Ability') +
  xlab('Missing') + theme(legend.position = 'none')
print(myplot.miss)
ggsave(file.path(OUT_PATH, "./figs/missing.png"), plot = myplot.miss, dpi = DPI, width = WIDTH, height = HEIGHT)

myplot.miss = ggplot(y_data, aes(x = as.factor(missing), y = abs(y-ypred), fill = missing)) + 
  geom_violin() + 
  facet_wrap(. ~ target) + 
  ylab('Predicted Ability - True Ability') +
  xlab('Missing') + theme(legend.position = 'none')
print(myplot.miss)
ggsave(file.path(OUT_PATH, "./figs/missing.png"), plot = myplot.miss, dpi = DPI, width = WIDTH, height = HEIGHT)

# myplot.level = ggplot(y_data, aes(y, y=abs(y-ypred), col=target)) + 
#   geom_point(size=1) + 
#   facet_wrap(. ~ target) + 
#   ylab('Absolute difference between true and predicted Ability') +
#   xlab('True Ability') + theme(legend.position = 'none')

}
################################################################################
# plot feature importances 
################################################################################

importances = read.csv(file.path(OUT_PATH, IMPORTANCE_RESULTS))
importances = importances %>% mutate(target = str_split_i(target, "_", 2))
importances = melt(importances, id.vars = "target", variable.name = "feature")

importances.mean = importances %>% group_by(target, feature) %>% 
  summarise(imp.mean = mean(value, na.rm=T), imp.sd = sd(value, na.rm=T))

importances.mean$domain = factor(domain[as.character(importances.mean$feature)])
importances.mean$target_domain = factor(domain[as.character(importances.mean$target)], 
  levels= c('German', 'English', 'French', 'Math'))
importances.mean$target_label = paste0(importances.mean$target, '[SA]')
#levels(importances.mean$feature) = pretty_features(levels(importances.mean$feature))
#importances.mean$feature = pretty_features(importances.mean$feature)

myplot.importance = ggplot(importances.mean, aes(x = feature, 
  y = imp.mean, 
  ymax = imp.mean + imp.sd, 
  ymin = imp.mean - imp.sd, fill = domain)) + 
  geom_col() + 
  geom_errorbar(width=0.2, linewidth=0.5) + 
  #coord_flip() +
  scale_x_discrete(labels = feature_labels) + 
  facet_grid( target_domain + target_label ~ ., labeller = label_parsed) + 
  ylab('Feature importance') +
  xlab('Feature') +
  theme(legend.position="bottom", legend.title=element_blank(), axis.text.x = element_text(size = 20, angle = 90, vjust = 0.5, hjust=1))

print(myplot.importance)

ggsave(file.path(OUT_PATH, "./figs/importance.png"), plot = myplot.importance, dpi = DPI, 
  width=WIDTH*2, 
  height = HEIGHT*2.6)

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
  labels = c("A", "C", "B", "D"), ncol = 2, nrow = 2, heights = c(2, 1.3), widths = c(2, 1), 
  font.label=list(size=LABEL_SIZE))

ggsave(file.path(OUT_PATH, "./figs/model_comparison.png"), 
  plot = myplot.models, dpi = DPI, 
  width = WIDTH*2, height = HEIGHT*2.5)

if (!EQUALIZED){
  
# myplot.scatter = ggarrange(
#     myplot.y.1, myplot.y.2,
#     myplot.level.1, myplot.level.2,
#     myplot.gender.1, myplot.mother_tongue.1,
#     labels = c("A", "B", "C", "D", "E", "F"), ncol = 2, nrow = 3, 
#     font.label=list(size=LABEL_SIZE))
# 
# ggsave(file.path(OUT_PATH, "./figs/true_predicted.png"), 
#     plot = myplot.scatter, dpi = DPI, 
#     width = WIDTH*2, height = HEIGHT*3)
  
myplot.scatter = ggarrange(
  myplot.y.1, myplot.y.2,
  myplot.level.1, myplot.level.2,
  labels = c("A", "B", "C", "D"), ncol = 2, nrow = 2,
  font.label=list(size=LABEL_SIZE))


ggsave(file.path(OUT_PATH, "./figs/true_predicted.png"),
  plot = myplot.scatter, dpi = DPI,
  width = WIDTH*2, height = HEIGHT*2)

myplot.row1 = ggarrange(
  myplot.gender.1, myplot.mother_tongue.1,
  labels = c("A", "B"), ncol = 2, nrow = 1, 
  font.label=list(size=LABEL_SIZE))

myplot.scatter = ggarrange(
  myplot.row1,
  myplot.gender.abil.1, myplot.mother_tongue.1,
  labels = c("", "C", "D"), ncol = 1, nrow = 3, 
  font.label=list(size=LABEL_SIZE))

ggsave(file.path(OUT_PATH, "./figs/biases.png"), 
  plot = myplot.scatter, dpi = DPI, 
  width = WIDTH*2, height = HEIGHT*3)

}
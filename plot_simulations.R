rm(list = ls())
library(ggplot2)
library(ggpubr)
library(dplyr)
library(stringr)
library(reshape2)

SIMUL_PATH = "./out/simulations.csv"

base_size = 20
theme_set(
  theme_bw(
    base_size = base_size)  #base_family = "",  #base_line_size = base_size/22,  #base_rect_size = base_size/22)
)

myPalette = c("red", "blue")

WIDTH = 7.2
HEIGHT = 6.8
DPI = 1000
LABEL_SIZE = 28
LEGEND_TEXT_SIZE = 18
LEGEND_TITLE_SIZE = 20
POINT_SIZE = 5
LINE_SIZE = 1
setwd("~/checkspred")
OUT_PATH = "~/checkspred"

simul.data = read.csv(SIMUL_PATH)[-1]
simul.data$group = paste(simul.data$noise, simul.data$nmodels)
simul.data$noise_label = paste0("Simulated Data (noise std=", as.character(simul.data$noise), ")")
simul.data$noise_label[simul.data$noise == 0] = "Actual Data" 
simul.data$noise_label = as.factor(simul.data$noise_label)
simul.data = subset(simul.data, use_RFE == 0)

myplot.size = ggplot(simul.data %>%filter(model=='OLS', nmodels == 1), 
  aes(x=nsamples, y=rel_size*100, col=nmodels, shape=noise_label, lty=noise_label, group=group)) + 
  ylab('Relative Dataset Size (%)') + 
  geom_point(size=POINT_SIZE, col='black') + 
  geom_line(linewidth=LINE_SIZE, col='black') + 
#  facet_grid(. ~ model) + 
  xlab('Number of Samples per Bin') + 
  theme(legend.position="none")

print(myplot.size)

myplot.r2 = ggplot(simul.data, 
  aes(x=nsamples, y=r2, col=nmodels, shape=noise_label, lty=noise_label, group=group)) + 
  ylab(expression(R^2)) + 
  geom_point(size=POINT_SIZE) + 
  geom_line(linewidth=LINE_SIZE) + 
  facet_grid(. ~ model) + 
  xlab('Number of Samples per Bin') + 
  theme(legend.position="none")

print(myplot.r2)

myplot.slope = ggplot(simul.data, 
  aes(x=nsamples, y=slope, col=nmodels, shape=noise_label, lty=noise_label, group=group)) + 
  ylab("Regression Slope") + 
  geom_point(size=POINT_SIZE) + 
  geom_line(linewidth=LINE_SIZE) + 
  facet_grid(. ~ model) + 
  xlab('Number of Samples per Bin') + 
  theme(legend.position="none")

print(myplot.slope)

myplot.compare = ggplot(simul.data, 
  aes(x=slope, y=r2, col=nmodels, shape=noise_label, lty=noise_label, group=group)) + 
  ylab(expression(R^2)) + 
  xlab('Regression Slope') + 
  geom_point(size=POINT_SIZE) + 
  geom_line(linewidth=LINE_SIZE) + 
  facet_grid(. ~ model) + 
  labs(col="Number of Models Averaged", lty="", shape="") + 
  guides(lty=guide_legend(nrow=length(unique(simul.data$noise_label)), byrow=TRUE),
         shape=guide_legend(nrow=length(unique(simul.data$noise_label)), byrow=TRUE),
         col = guide_colourbar(title.position = "top", barwidth=18, title.hjust=0)) +
  theme(legend.position="bottom", 
    legend.title.align = 1,
    legend.title = element_text(size=LEGEND_TITLE_SIZE),    
    legend.text = element_text(size=LEGEND_TEXT_SIZE),
    legend.key.width = unit(5, "line")
  )

print(myplot.compare)

myplot.simulations = ggarrange(
  myplot.size, myplot.slope, myplot.r2, myplot.compare,
  labels = c("A", "B", "C", "D"), ncol = 1, nrow = 4, 
  font.label=list(size=LABEL_SIZE))

print(myplot.simulations)
ggsave(file.path(OUT_PATH, "./figs/final/Simulations.png"), 
  plot = myplot.simulations, dpi = DPI, 
  width = WIDTH*3, height = HEIGHT*4)

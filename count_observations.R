
rm(list = ls())
library(dplyr)
MS_PATH = './data/abilities_2pl.rda'
CHECKS_PATH = './data/CDW_data_forILD_2022-10-25.Rds'
DEMO_PATH = '/home/garben/mindsteps/data/mindsteps/dd_2023_Jan_valid_parameters.rds'

load(MS_PATH)
checks = readRDS(CHECKS_PATH)
demo = readRDS(DEMO_PATH)

# responses
print("Checks")
print(paste("Observations:", nrow(checks)))
print(paste("Students:", length(unique(checks$studentId))))
#print(paste("Items:", length(unique(demo$code))))

print("Mindsteps")
print(paste("Observations:", nrow(demo)))
print(paste("Students:", length(unique(demo$studentId))))
print(paste("Items:", length(unique(demo$code))))

print("Abilities")
print(paste("Observations:", nrow(abilities)))
print(paste("Students:", length(unique(abilities$studentId))))


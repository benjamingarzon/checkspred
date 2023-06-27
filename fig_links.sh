#!/bin/sh
cd figs/final
ln -s ~/checkspred/out/not_equalized/figs/model_comparison.png R2Imbalanced.png
ln -s ~/checkspred/out/equalized/figs/model_comparison.png R2Balanced.png
ln -s ~/checkspred/out/not_equalized/figs/true_predicted.png True_predicted.png
ln -s ~/checkspred/out/not_equalized/figs/missing.png Missing.png
ln -s ~/checkspred/out/equalized/figs/importance.png FeatureImportance.png
ln -s ~/checkspred/out/not_equalized/figs/biases.png Biases.png
#ln -s ~/checkspred/out/not_equalized/figs/level.png LevelError.png
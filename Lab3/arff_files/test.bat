@echo off
title This is your first batch script!
echo Welcome to batch scripting!
pause

cd results

FOR %%F IN (act_set_1.arff act_set_2.arff act_set_3.arff act_set_3_selected_features.arff  val_set_1.arff val_set_2.arff val_set_3.arff  val_set_3_selected_features.arff ) DO (
	cd %%F
	test
	cd ..
)

cd ..
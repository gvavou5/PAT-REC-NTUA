@echo off
title This is your first batch script!
echo Welcome to batch scripting!


mkdir results
cd results

FOR %%F IN (act_set_3_selected_features.arff val_set_3_selected_features.arff ) DO (
	mkdir %%F
	cd %%F
		FOR %%L IN (0.1 0.4 0.7 0.9) DO (
			java -classpath C:/"Program Files"/weka-3-7/weka.jar weka.classifiers.functions.MultilayerPerceptron -t ../../%%F -o -x 5 -L %%L -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a > layers_1_ratio_%%L.txt
		)
	cd ..
)

FOR %%F IN (act_set_3_selected_features.arff val_set_3_selected_features.arff ) DO (
	mkdir %%F
	cd %%F
		FOR %%L IN (0.1 0.4 0.7 0.9) DO (
			java -classpath C:/"Program Files"/weka-3-7/weka.jar weka.classifiers.functions.MultilayerPerceptron -t ../../%%F -o -x 5 -L %%L -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a,a > layers_2_ratio_%%L.txt
		)
	cd ..
)

FOR %%F IN (act_set_3_selected_features.arff val_set_3_selected_features.arff ) DO (
	mkdir %%F
	cd %%F
		FOR %%L IN (0.1 0.4 0.7 0.9) DO (
			java -classpath C:/"Program Files"/weka-3-7/weka.jar weka.classifiers.functions.MultilayerPerceptron -t ../../%%F -o -x 5 -L %%L -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a,a,a,a > layers_4_ratio_%%L.txt
		)
	cd ..
)

FOR %%F IN (act_set_3_selected_features.arff val_set_3_selected_features.arff ) DO (
	mkdir %%F
	cd %%F
		FOR %%L IN (0.1 0.4 0.7 0.9) DO (
			java -classpath C:/"Program Files"/weka-3-7/weka.jar weka.classifiers.functions.MultilayerPerceptron -t ../../%%F -o -x 5 -L %%L -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a,a,a,a,a,a > layers_6_ratio_%%L.txt
		)
	cd ..
)

FOR %%F IN (act_set_3_selected_features.arff val_set_3_selected_features.arff ) DO (
	mkdir %%F
	cd %%F
		FOR %%L IN (0.1 0.4 0.7 0.9) DO (
			java -classpath C:/"Program Files"/weka-3-7/weka.jar weka.classifiers.functions.MultilayerPerceptron -t ../../%%F -o -x 5 -L %%L -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a,a,a,a,a,a,a,a > layers_8_ratio_%%L.txt
		)
	cd ..
)

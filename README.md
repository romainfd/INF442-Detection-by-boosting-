# INF442 : Detection by boosting
The subject is available at the root of the project.
The aim of the project is to train weak classifiers to later group them to boost the performances.

In a first part, we compute locally the summed-area table (https://en.wikipedia.org/wiki/Summed-area_table) of an image in O(row x col) with dynamic programming.
Then, we compute in parallel all the caracteristics of the image (all the subsquares of the image).
With these, we train one weak classifier *h_i* per rank of caracteristic *i* with 2 weights : (if *w_i1 x X_i + w_i2 >= 0*, where *X_i* is the i-th caracteristic of the image we are trying to classify, the image is declared positive by the weak classifier). To train them we use K images and we update the weights with a factor proportional to epsilon and the error according to the perceptron method (Frank Rosenblatt. The perceptron: A probabilistic model for information storage and organi-
zation in the brain. In Psychological Review, volume 65, pages 386{408, 1958. or see https://en.wikipedia.org/wiki/Perceptron#Learning_algorithm).
Once we have train these weak classifiers, we try to combine them with a linear combinaison to focus on the important caracteristics. To do so we use the Adaboost algorithm (Yoav Freund and Robert E. Schapire. A decision-theoretic generalization of on-line learning
and an application to boosting. 1997.) We iterate N times on n images and each time we choose in parallel the best weak classifier and add it to our final classifier after decreasing its effectiveness (lowering the weight of the images it classified well, to select other weak classifiers).

To try our program, you can do
- Change the *imageFolder* adress in boostingDetection.cpp to match your computer folder (it must have 'test' and 'app' (to learn) repos with 'neg' and 'pos' subfolders containing the images).
- Then open a command line and write
```
make
LD_LIBRARY_PATH=/usr/local/opencv-3.4.1/lib64 /usr/local/openmpi-3.0.1/bin/mpirun -np 4 boostingDetection 2000 0.00000003475 200 20 0 1
```
*You may have to change the two first paths to match your computer set up*
- You can display more informations with an optional additionnal parameter (at the end):
1 displays informations on the available images, the results of the prediction of the weak classifiers in each loop and the new weights of the classifiers
2 displays the caracteristics of an image of each subloop in the K-loop, the relative evolution of w1 and w2 to see they are converging and the best classifier in each N-loop
3 displays the variations made to the weights, the list of the relative evolutions to plot over time on Python and other informations.
4 displays some debug informations.
- For instance, to see all the availble informations (this might be very long as it prints some big matrices), you can do 
```
LD_LIBRARY_PATH=/usr/local/opencv-3.4.1/lib64 /usr/local/openmpi-3.0.1/bin/mpirun -np 4 boostingDetection 2000 0.00000003475 200 20 0 1 4
```
- The other parameters match those of the algorithms explained above :
```
boostingDetection K epsilon n N theta normalisation printDebug
```
where theta is a threshold for our final classifier and normalisation multiplies all the caracteristics to make them smaller and independant of the size of the image (if proportionnal to this size).

- This configuration gives us a F-score of 0.444 on our image database.
```
LD_LIBRARY_PATH=/usr/local/opencv-3.4.1/lib64 /usr/local/openmpi-3.0.1/bin/mpirun -np 4 boostingDetection 2000 0.1 200 20 0 0.001 1
```


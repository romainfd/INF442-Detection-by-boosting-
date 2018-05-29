//============================================================================
// Name        : util.cpp
// Author      : Antoine Grosnit and Romain Fouilland
// Version     :
// Copyright   : Work of Antoine Grosnit and Romain Fouilland
// Description : PI INF442 : detection by "boosting"
//============================================================================
#include <iostream>
#include <vector>
#include "img_processing.h"
#include "math.h"
#include "dirent.h"
using namespace std;

// TO USE to_string
#include <string>
#include <sstream>
template <typename T>
std::string toString(T val)
{
    std::stringstream stream;
    stream << val;
    return stream.str();
}

// infinity
double infinity = 1E+30;

const int deltaSize = 4;
const int minSize = 8;
const string imageFolder = "/usr/local/INF442-2018/P5/";
const string repo = "app";
const string repo2 = "test";

// Parameters set in the args
double normalisation = 1;
int printDebug = 0;

// Affichage d'une matrice de façon alignée pour des valeurs entre -9 et 99
void displayMatrix(vector<vector<int> >& matrix) {
	// we only use the reference to avoid a useless copy
	for (unsigned int i = 0; i < matrix.size(); i++) {
		for (unsigned int j = 0; j < matrix[0].size(); j++) {
			if (matrix[i][j] >= 0 and matrix[i][j] < 20) {
				cout << " "; // pour tout avoir aligné pour des valeurs entre -9 et 99
			}
			cout << matrix[i][j] << " ";
		}
		cout << endl;
	}
}

// Renvoie une matrice vide de la taille demandée
vector<vector<int> > emptyMatrix(int& l, int& c) {
	vector<vector<int> > matrix;
	for (int i = 0; i < l; i++) {
		vector<int> ligne(c); // une ligne de c cases = 0
		matrix.push_back(ligne);
	}
	return matrix;
}

// Calcule l'image intégrale de l'image fournie en paramètre avec la relation de récurrence fournie
vector<vector<int> > imageIntegrale(cv::Mat& image) {
	// on crée 2 matrices vides pour contenir les sommes sur les colonnes et les sommes intégrales
	int r = image.rows;
	int c = image.cols;
	vector<vector<int> > s = emptyMatrix(r, c);
	vector<vector<int> > I = emptyMatrix(r, c);

	// initialisation de la premiere colonne
	I[0][0] = (int)image.at<uchar>(0,0);
	s[0][0] = (int)image.at<uchar>(0,0);
	for (unsigned int y = 1; y < r; y++) {
		s[y][0] = s[y-1][0] + (int)image.at<uchar>(y,0); // comme pour x>=1 pour s
		I[y][0] = s[y][0];
	}
	for (unsigned int x = 1; x < c; x++) {
		// initialisation de la première ligne
		s[0][x] = (int)image.at<uchar>(0,x);
		I[0][x] = I[0][x-1] + (int)image.at<uchar>(0,x); // comme pour y>=1 pour I

		for (unsigned int y = 1; y < r; y++) {
			s[y][x] = s[y-1][x] + (int)image.at<uchar>(y,x);
			I[y][x] = I[y][x-1] + s[y][x];
		}
	}
	return I;
}

// test de la fonction de calcul de l'image intégrale en fournissant en entrée un vector<vector<int> > de test
vector<vector<int> > imageIntegraleTest(vector<vector<int> >& img) {
	// on crée 2 matrices vides pour contenir les sommes sur les colonnes et les sommes intégrales
	int l = img.size();
	int c = img[0].size();
	vector<vector<int> > s = emptyMatrix(l, c);
	vector<vector<int> > I = emptyMatrix(l, c);

	// initialisation de la premiere colonne
	I[0][0] = img[0][0];
	s[0][0] = img[0][0];
	for (unsigned int y = 1; y < img.size(); y++) {
		s[y][0] = s[y-1][0] + img[y][0]; // comme pour x>=1 pour s
		I[y][0] = s[y][0];
	}
	for (unsigned int x = 1; x < img[0].size(); x++) {
		// initialisation de la première ligne
		s[0][x] = img[0][x];
		I[0][x] = I[0][x-1] + img[0][x]; // comme pour y>=1 pour I

		for (unsigned int y = 1; y < img.size(); y++) {
			s[y][x] = s[y-1][x] + img[y][x];
			I[y][x] = I[y][x-1] + s[y][x];
		}
	}
	return I;
}

//int testQ1Old() {
//	// To test locally, we initiate an small image
//	vector<vector<int> > image;
//	vector<int> ligne1{ 1,  1, -1};
//	vector<int> ligne2{-1, -1,  1};
//	vector<int> ligne3{ 1, -1,  1};
//	image.push_back(ligne1);
//	image.push_back(ligne2);
//	image.push_back(ligne3);
//	cout<<"Matrice de base:"<<endl;
//	displayMatrix(image);
//
//	// TRAITEMENT DE L'IMAGE
//	// Question 1 : calcul de l'image intégrale en 1 seul parcours
//	vector<vector<int> > I = imageIntegraleTest(image);
//	cout<<"Image intégrale:"<<endl;
//	displayMatrix(I);
//}

// To test if the beginning of the integral image matches the given img_processing and q_test results : OK !
void testQ1() {
	cv::Mat image = cv::imread(imageFolder+repo+"/neg/im0.jpg", cv::IMREAD_GRAYSCALE);
	vector<vector<int> > I = imageIntegrale(image);
	cout<<"Image intégrale:"<<endl;
	displayMatrix(I);
}

// Cette fonction calcule le nombre de caractéristiques calculées par un processeur de rang rank
int nbCaracts(int& rank, int& np, int& rows, int& cols) {
    int s = 0;
    for (unsigned int n = minSize + deltaSize*rank; n < rows; n += deltaSize * np ) {
		  s += ((rows-n) / deltaSize) * ((cols-n) / deltaSize);
	}
	return s;
}

// Cette fonction calcule le nombre de caractéristiques calculées au total
int nbCaractsTot(int rows, int cols) {
    int s = 0;
    for (unsigned int n = minSize; n < rows; n += deltaSize ) {
		  s += ((rows-n) / deltaSize) * ((cols-n) / deltaSize);
	}
	return s;
}

// Renvoie les données (x, y et taille) de la caractéristique d'indice ind dans le vecteur global des caractéristiques (elle cherche d'abord le bon processus avant de faire un parcours brutal)
vector<int> findCaract(int ind, int& np, int rows, int cols){ //renvoie [row, col, size] correspondant à l'emplacement de la caractéristique d'indice ind
	int rank = 0;
	int s = 0;
	vector<int> coord(3);
	while(ind >= s + nbCaracts(rank, np, rows, cols)){
		s += nbCaracts(rank, np, rows, cols);
		rank++;
	}
	ind -= s;
	// indice est l'indice dans le vecteur des caracts local (du processus rank)
	int count = 0;
	for (unsigned int n = minSize + deltaSize*rank; n < rows; n += deltaSize * np ) {
		for (unsigned int x = 0; x < rows - n; x += deltaSize) {
			for (unsigned int y = 0; y < cols - n; y += deltaSize) {
				if(ind == count) {
					coord[0] = x;
					coord[1] = y;
					coord[2] = n;
					return coord;
				}
				count++;
			}
		}
	}
}

// Renvoie les caractéristiques de l'image fournie centralisées chez le processus de rang ROOT
// le calcul est fait en parallèle (chaque processus calcule 1/np % des caractéristiques avant d'être regroupées chez le processus ROOT
vector<int> caract_mpi(vector<vector<int> >& I, int ROOT, int& rank, int& np) {
	if (printDebug >= 4) {
		printf("Processus %d is computing info for processus %d\n", rank, ROOT);
	}
	int rows = I.size();
	int cols = I[0].size();

	// MPI: Init
	MPI_Status status;

	// On calcule le nb d'image à calculer pour ce processeur afin d'initialiser un tableau à la bonne taille
	unsigned int nCar = nbCaracts(rank, np, rows, cols);
	int* results = new int[nCar];
    int* d1 = new int[nCar];
    int* d2 = new int[nCar];
    int h;
	unsigned int i = 0; // notre compteur
	for (unsigned int n = minSize + deltaSize*rank; n < rows; n += deltaSize * np ) {
		  for (unsigned int x = 0; x < rows - n; x += deltaSize) {
			  for (unsigned int y = 0; y < cols - n; y += deltaSize) {
				  results[i] = I[x+n][y+n] - I[x+n][y] - I[x][y+n] + I[x][y];
				  i++;
			}
		}
	}
	// on envoie tous les résultats à la racine pour qu'elle centralise tout
	vector<int> resultsGlobal; // renvoyé si ce n'est pas le processus en charge de centraliser l'image
	if (rank != ROOT) {
		MPI_Send(results, nCar, MPI_INT, ROOT, 0, MPI_COMM_WORLD);
	} else {
    	for (int i = 0; i < np; i++) {
			int* procResults = results; // si on est le bon proc, inutile de communiquer
			int procNCar = nbCaracts(i, np, rows, cols);
    		if (i != ROOT) {
				procResults = new int[procNCar];
				MPI_Recv(procResults, procNCar, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
    		}
    		resultsGlobal.insert(resultsGlobal.end(), procResults, procResults + procNCar);
    		if (i != ROOT) {
    			delete[] procResults;
    		}
    	}
    }
	delete[] results;
	return resultsGlobal;
}

// Permet l'affichage des len premières valeurs d'une ligne
template<typename T> void printline(string name, T* ligne, int len) {
	cout << name << ":\t";
	for (int i = 0; i < len; i++) {
		cout<<ligne[i]<<" ";
	}
	cout<<endl;
}

// Permet l'affichage des len premières valeurs d'un vector
template<typename T> void printlineVect(string name, vector<T>& ligne, int len) {
	cout<<name<<":\t";
	if (len > ligne.size()) {
		len = ligne.size(); // no influence outside because len is given by value
	}
	for (int i = 0; i < len; i++) {
		cout<<ligne[i]<<" ";
	}
	cout<<endl;
}

// Calcule la norme euclidienne d'un vecteur
double norm(vector<double>& w){
	double s = 0;
	for(int i = 0; i< w.size(); i++){
		s += w[i]*w[i];
	}
	return sqrt(s);
}

// Calcule la norme euclidienne d'une ligne
double normTab(double w[], int size){
	double s = 0;
	for(int i = 0; i< size; i++){
		s += w[i]*w[i];
	}
	return sqrt(s);
}

// Donne l'erreur d'un classifieur (0 si correct ou 1) en fonction de sa prédiction (valeur de h >< 0) et de la categorie attendue
int error(double h, bool cat) {
	if ((h >= 0 && cat) || (h < 0 && !cat)) {
		// classifie pos et pos ou classifie neg et neg
		return 0; // aucune erreur
	} else {
		return 1;
	}
}

// Renvoie la classification (de l'image de caracteristiques carsImg) donnée par le classificateur global F défini par les 2 vectors de poids w1 et w2 en comparant la valeur obtenue au seuil threshold
int classifyF(vector<double>& w1F, vector<double>& w2F, vector<int>& carsImg, double& threshold) {
	double value = 0;
	for (int c = 0; c < w1F.size(); c++) {
		value += w1F[c]*normalisation*carsImg[c] + w2F[c];
	}
	if (value >= threshold) {
		return 1;
	} else {
		return -1;
	}
}

int main(int argc, char** argv) {
	// ETAPE 1 du dossier repo (param global)
	// Récupération des images
	DIR *dir;
	struct dirent *ent;
	int nbNeg = 0;
	vector<string> images; // on aura nbNeg images neg au debut, suivi des positives
	if ((dir = opendir ((imageFolder+repo+"/neg/").c_str())) != NULL) {
		// on enlève . et .. qui ne nous intéressent pas
		readdir(dir);
		readdir(dir);
		  /* gets all the images in the directory */
		  while ((ent = readdir (dir)) != NULL) {
			  images.push_back(ent->d_name);
			  nbNeg++;
		  }
		  closedir (dir);
	} else {
		  /* could not open directory */
		  perror ("");
		  return EXIT_FAILURE;
	}
	if ((dir = opendir ((imageFolder+repo+"/pos/").c_str())) != NULL) {
		readdir(dir);
		readdir(dir);
		  /* gets all the images in the directory */
		  while ((ent = readdir (dir)) != NULL) {
			  images.push_back(ent->d_name);
		  }
		  closedir (dir);
	} else {
		  /* could not open directory */
		  perror ("");
		  return EXIT_FAILURE;
	}

	// On change la normalisation des images pour réduire l'influence des caractéristiques
	if (argc >= 7) {
		normalisation = atof(argv[6]);
	}
	// On gère l'affichage des infos dans la boucle
	if (argc >= 8) {
		printDebug = atoi(argv[7]);
	}

	// Question 1 : calcul de l'image intégrale en 1 seul parcours
	// testQ1();

	// Question 2
	MPI_Init(&argc, &argv);
	int np=0;
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	int rank=0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// Initialisation
	if (rank == 0) {
		cout << " ------------------------------------------------------------ " << endl;
		cout << " ------------------ LANCEMENT DU PROGRAMME ------------------ " << endl;
		cout << " ------------------------------------------------------------ " << endl;
		if (printDebug >= 1) {
			printf("We have %d images available (%.2f%% negative)\n", images.size(), (((double)nbNeg)/ (double) images.size()) *100);
		}
		if (printDebug >= 4) {
			printlineVect("Images", images, 20);
		}
	}

	// ETAPE 2 : ENTRAINEMENT DES CLASSIFIEURS FAIBLES
	// Etape 2.a : INITIALISATION
	//initialisation commune de la seed (pour que tous les processus aient les mêmes images)
	unsigned int seed = 0;
    srand(seed);
    unsigned int K = 0;
    double epsilon = 1;
    if (argc >= 3) {
    	K = atoi(argv[1]);
    	epsilon = atof(argv[2]);
    } else {
        if(rank==0)cerr << "Please run as: " << endl << "   " << argv[0] << " K epsilon" << endl;
        MPI_Finalize();
        return 0;
    }

    // Initialisation des vecteurs des poids des classifieurs
    int nbC = nbCaractsTot(92, 112); // déterminer le nombre de caractéristiques pour initialiser w1 et w2
    vector<double> w1(nbC);
    vector<double> w2(nbC);
    vector<double> formerw1(nbC);
    vector<double> formerw2(nbC);
    vector<double> cv1;
    vector<double> cv2;
    // variations dues à l'apprentissage sur l'image i
    double* delta1 = new double[nbC];
    double* delta2 = new double[nbC];
    // variations dues a l'apprentissage de tous nos processus en 1 boucle
    double* globD1 = new double[nbC];
    double* globD2 = new double[nbC];
    int category; // de l'image en cours de traitement
    // We initianilze the values of the coeffs of the classifiers (to 1 and 0) and the variations (to 0)
    for (int i = 0; i < nbC; i++){
        w1[i] = 1.;
        w2[i] = 0.;
        delta1[i] = 0.;
        delta2[i] = 0.;
    }

    // Etape 2.b : boucle d'apprentissage des classifieurs faibles
	for (int i = 0; i < K - (K%np); i++) {
        // On choisit d'abord le type (pos/neg) puis l'image
        // il faut la meme seed pour tous pour être sur qu'il ait la meme image
        string filename = imageFolder+repo+"/neg/im0.jpg";
        // on choisit au hasard notre image dans toutes celles dispos
        int imageRank = (rand() * images.size())/ RAND_MAX;
        if (imageRank < nbNeg) {
        	// on a une image négative !
            filename = imageFolder+repo+"/neg/" + images[imageRank];
            category = -1;
        } else {
            // choisir fichier au hasard dans la classe -1
            filename = imageFolder+repo+"/pos/" + images[imageRank];
            category = 1;
        }
        // on charge l'image choisie
		cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
		// Check for invalid input
		assert(!image.empty());
		// on calcule LOCALEMENT l'image intégrale
		vector<vector<int> > I = imageIntegrale(image);
		if (printDebug >= 3) {
			printf("Processus %d in loop %d is going to compute the caracts of the image %s of category %d\n", rank, i, images[imageRank].c_str(), category);
		}
		if (printDebug >= 4) {
			displayMatrix(I);
		}
		// On calcule toutes les caractéristiques et on les centralise dans le processus i%np
		vector<int> cars = caract_mpi(I, i%np, rank, np);
		// Update caracteristics
		if (rank == i%np) { // thanks to this condition cars is not empty
			//printlineVect("caracts", cars, 20);
			// printf("Taille attendue : %d et taille reelle : %d\n", nbC, cars.size());
			int h;
			int nbRight = 0;
			// on parcourt tout le vecteur caracts et on maj les wi avec les infos apportees par l'image
			for (int c = 0; c < nbC; c++) {
				// classification (pos = +1, neg = -1)
				  h = (w1[c]*normalisation*cars[c] + w2[c] >= 0) ? 1 : -1;
				  if (h == category) { nbRight++; }
				  // on compare notre classification a la categorie => eventuellement un delta si erreur
				  delta1[c]= epsilon * (h - category) * normalisation * cars[c];
				  delta2[c]= epsilon * (h - category);
			}
			if (printDebug >= 1) {
				printf("Processus %d in loop %d has centralized all the caracteristics for image %s of category %d and %.2f%% of classifiers made a correct prediction\n", rank, i, images[imageRank].c_str(), category, double(nbRight)/nbC * 100);
			}
			if (printDebug >= 3) {
				printline("Delta1", delta1, 20);
				printline("Delta2", delta2, 20);
				printf("Processus %d has computed the delta\n", rank);
			}
		}
		// pas fait avant la npè boucle puis toutes
        if((i > 0) && ((i+1)%np == 0)){
        	//une fois que tous les processus ont calculé une variation de w1 et w2 on met à jour w1 et w2 chez chacun avec AllReduce
			// on somme tous les apprentissages
			MPI_Allreduce (delta1, globD1, nbC, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce (delta2, globD2, nbC, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			for(int j = 0; j < nbC; j++){
				formerw1[j]=w1[j];
				formerw2[j]=w2[j];
				// on met à jour localement notre wi
				w1[j] -= globD1[j];
				w2[j] -= globD2[j];
			}
			if (rank + 1 == np && printDebug >= 2) {// c'est le dernier qui vient de centraliser les caracteristiques
				cout << "Ponderated caracts (for info):\t";
				for (int i = 0; i < 20; i++) {
					cout<<normalisation*cars[203*i]<<" ";
				}
				cout<<endl;
			}
			if (rank == 0) {
				double norm1 = norm(formerw1);
				if(norm1 != 0) {
					cv1.push_back(normTab(globD1,nbC)/norm1);
					if (printDebug >= 2) {
						printf("Evolution 1 : %f \n", normTab(globD1,nbC)/norm1);
					}
				} else {
					cv1.push_back(100);
				}
				double norm2 = norm(formerw2);
				if(norm2 != 0) {
					cv2.push_back(normTab(globD2,nbC)/norm2);
					if (printDebug >= 2) {
						printf("Evolution 2 : %f \n", normTab(globD2,nbC)/norm2);
					}
				} else {
					cv2.push_back(100);
				}
				if (printDebug >= 1) {
					printf("The %d-th loop has been made. w1 and w2 have been updated to :\n", i/np);
					printlineVect("w1", w1, 20);
					printlineVect("w2", w2, 20);
					cout << " ------------------------------------------------------------ " << endl;
				}
			}
        }
	}
	if (printDebug >= 3) {
		if(rank == 0){ //permet d'afficher l'évolution de ||w_(k+1) - w_k||/||w_k|| => pour l'exploiter avec le script Python
			cout<<"Evolution w1 : [";
			for(int i = 1; i < cv1.size()-1; i++){
				cout<< cv1[i]<<", ";
			}
			cout<< cv1[cv1.size()-1] <<"]"<<endl;
			cout<<"Evolution w2 : [";
			for(int i = 1; i < cv2.size()-1; i++){
				cout<< cv2[i]<<", ";
			}
			cout<< cv2[cv2.size()-1] << "]"<<endl;
		}
	}
	delete[] delta1;
	delete[] delta2;
	delete[] globD1;
	delete[] globD2;

	// QUESTION 3
	// We have in w1 and w2 the coefficients of our weak classifiers trained with the perceptron method.
	// We will now use the bossting method to improve the results of our final classifier.

	// To start with clean processus
	MPI_Barrier(MPI_COMM_WORLD);

	// We initialize our final classifier at 0
	vector<double> w1F(nbC);
	vector<double> w2F(nbC);

	// We get the number of images we want to train on and the number of steps.
	int n = 0;
	int N = 0;
    if (argc >= 5) {
    	n = atoi(argv[3]);
    	N = atoi(argv[4]);
    } else {
        if(rank==0)cerr << "Please run as: " << endl << "   " << argv[0] << " K epsilon n N" << endl;
        MPI_Finalize();
        return 0;
    }

	// Initialisation
	if (rank == 0) {
		cout << " ------------------------------------------------------------ " << endl;
		cout << " ------------------ LANCEMENT DU BOOSTING ------------------- " << endl;
		cout << " ------------------------------------------------------------ " << endl;
	}
	// we initiate the array for the N values of the coefficient of our final classifier
	vector<double> alphas(nbC, 0);

    // We find n random images in our database and we store their category.
    srand(0);
    vector<string> imagesBoosting(n);
    vector<bool> cat(n);
    int imgRank;
    for (unsigned int i = 0; i < n; i++) {
    	imgRank = (rand() * images.size())/ RAND_MAX;
    	imagesBoosting[i] = images[imgRank];
    	cat[i] = imgRank > nbNeg;
    }
    // Rq: it's actually useless to store all the name of the images and we could have directly calculated their caracs as we did for question 2 (ie merge with the next loop directly)

    // We compute all their caracteristics to iterate on all of them for all classifiers later
    // VERY HEAVY !!!
    int** carsN = new int*[n];
    for (int l = 0; l < n; l++) {
    	carsN[l] = new int[nbC];
    }
    string filename;
    for (int img = 0; img < n; img++) {
        // on charge l'image n° img
    	filename = imageFolder+repo+ (cat[img] ? "/pos/" : "/neg/") + imagesBoosting[img];
		cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
		// Check for invalid input
		assert(!image.empty());
		// on calcule LOCALEMENT l'image intégrale
		vector<vector<int> > I = imageIntegrale(image);

		// puis avec mpi on calcule son vecteur caract et le processus img%np centralise les résultats avant de les renvoyer à tout le monde
		vector<int> carsImg = caract_mpi(I, img%np, rank, np); // is empty if rank != img%np
		if (rank == img%np) { // on a les bonnes caracs que l'on stocke pour pouvoir les broadcast
			for (unsigned int c = 0; c < nbC; c++) {
				carsN[img][c] = carsImg[c];
			}
		}
		// Probablement facultatif (tant que l'on conserve la meme relation sur les classifieurs traités par un processus (ex: %np = rank))
		MPI_Bcast(carsN[img], nbC, MPI_INT, img%np, MPI_COMM_WORLD);
    }

    // We initialise their weigths to 1/n
    vector<double> weights(n, 1/double(n));

    MPI_Barrier(MPI_COMM_WORLD); // we wait for all the caracteristics to have be computed and be broadcasted to all processus
    if (rank == 0 && printDebug >= 3) {
    	cout << "All the caracteristics of our boosting images have been computed."<<endl;
        printline("carsN[0]",carsN[0], 20);
        printlineVect("weigths",weights, 20);
    }

    // We compute locally the best weak classifier
	struct { double value; int index; } in, out;
	for (int l = 0; l < N; l++) {
		in.value = infinity;
		double err;
		in.index = -1;
		for (int c = 0; c < nbC; c++) {
			// each processus only focuses on the c%np == rank classifiers
			if (c % np == rank) {
				err = 0;
				// he computes the weighted error for this classifier
				for (int img = 0; img < n; img++) {
					err += weights[img]*error(w1[c]*normalisation*carsN[img][c] + w2[c], cat[img]);
				}
				// and compares it to its localMin
				if (err < in.value) {
					in.value = err;
					in.index = c;
				}
			}
		}
		//cout << in.value << endl;
		//cout << in.index << endl;
		// once they all have tried all the classifiers they had to compute the error for, we use Allreduce with minloc to find the best classifier and update the weights and the finalClassifier everywhere
		MPI_Allreduce((void*) &in, (void*) &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

		int k = out.index; // the best classifier
		// We recompute the error because Allreduce doesn't work for the value...
		double errk = 0;
		for (int img = 0; img < n; img++) {
			errk += weights[img]*error(w1[k]*normalisation*carsN[img][k] + w2[k], cat[img]);
		}

		if (rank == 0 && printDebug >= 2) {
			printf("The best classifier for this step is %d with an error of %f\n", k, errk);
		}
		// We update the weights of the images and the global classifier in EACH processus thanks to the ALLreduce
		// Update of our global classifier
		double alpha = log((1-errk)/errk)/2;
		alphas[k] += alpha; // += because it could already have been taken before
		w1F[k] += alpha * w1[k];
		w2F[k] += alpha * w2[k];
		// Update of the image weights to reduce the effectiveness of the selected weak classifier
		double s = 0; // to normalize
		for (int img = 0; img < n; img++) {
			weights[img] *= exp(-(cat[img] ? 1 : -1)*alphas[k]*((w1[k]*normalisation*carsN[img][k] + w2[k] >= 0) ? 1 : -1));
			s += weights[img];
		}
		// Normalisation of the image weights
		for (int img = 0; img < n; img++) {
			weights[img] *= 1/s;
		}
	}
	delete[] carsN;

	// We get the number of images we want to train on and the number of steps.
	double theta = 0;
    if (argc >= 6) {
    	theta = atof(argv[5]);
    } else {
        if(rank==0)cerr << "Please run as: " << endl << "   " << argv[0] << " K epsilon n N theta" << endl;
        MPI_Finalize();
        return 0;
    }

	// We have our final classifier whih is a linear combination of the best weak classifiers of the perceptron method found with the boosting method in w1F, w2F
	// and alphas are the coefficient of the linear combination
	double threshold = 0;
	for (int c = 0; c < nbC; c++) {
		threshold += alphas[c];
	}
	threshold *= theta;

	// Etude des classifieurs récupérés
	if (rank == 0) {
		for (int c = 0; c < nbC; c++) {
			if (alphas[c] != 0) {
				vector<int> shape = findCaract(c, np, 92, 112);
				printf("La caract %d (w1 = %f, w2 = %f) part de (%d, %d) et est de taille (%d, %d) et est ponderee avec %.16f\n", c, w1[c], w2[c], shape[0], shape[1], shape[2], shape[2], alphas[c]);
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	// QUESTION 5 : TESTS OF THE FINAL CLASSIFIER
	if (rank == 0) {
		cout << " ------------------------------------------------------------ " << endl;
		cout << " ------------------- LANCEMENT DES TESTS -------------------- " << endl;
		cout << " ------------------------------------------------------------ " << endl;
	}

	DIR *dirT;
	struct dirent *entT;
	nbNeg = 0;
	vector<string> imagesTest; // on aura nbNeg images neg au debut, suivi des positives
	if ((dirT = opendir ((imageFolder+repo2+"/neg/").c_str())) != NULL) {
		// on enlève . et .. qui ne nous intéressent pas
		readdir(dirT);
		readdir(dirT);
		/* gets all the images in the directory */
		while ((ent = readdir (dirT)) != NULL) {
			imagesTest.push_back(ent->d_name);
			nbNeg++;
		}
		closedir (dirT);
	} else {
		/* could not open directory */
		perror ("");
		return EXIT_FAILURE;
	}
	if ((dirT = opendir ((imageFolder+repo2+"/pos/").c_str())) != NULL) {
		readdir(dirT);
		readdir(dirT);
		/* gets all the images in the directory */
		while ((entT = readdir (dirT)) != NULL) {
			imagesTest.push_back(entT->d_name);
		}
		closedir (dirT);
	} else {
		/* could not open directory */
		perror ("");
		return EXIT_FAILURE;
	}

	//On considere tp (True positive) tn (true negative) fp (false positive) fn (false negative) pour la série de test
	int tp = 0;
	int tn = 0;
	int fp = 0;
	int fn = 0;

	int tpLoc = 0;
	int tnLoc = 0;
	int fpLoc = 0;
	int fnLoc = 0;

	for (int imageRank = 0; imageRank < imagesTest.size(); imageRank++) {
		if (imageRank < nbNeg) {
			// on a une image négative
			filename = imageFolder + "test/neg/" + imagesTest[imageRank];
			category = -1;
		} else {
			// on a une image positive
			filename = imageFolder + "test/pos/" + imagesTest[imageRank];
			category = 1;
		}
		// on charge l'image choisie
		cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
		// Check for invalid input
		assert(!image.empty());
		// on calcule LOCALEMENT l'image intégrale
		vector<vector<int> > I = imageIntegrale(image);
		// On calcule toutes les caractéristiques et on les centralise dans le processus 0
		vector<int> cars = caract_mpi(I, imageRank%np, rank, np);
		// Update caracteristics
		if (rank == imageRank%np) { // thanks to this condition cars is not empty
			//printlineVect("caracts", cars, 20);
			// computing the category associated to cars with the F-classifier
			int result = classifyF(w1F, w2F, cars, threshold);
			if (category == 1){
				if (result == category){
					tpLoc += 1;
				} else {
					fnLoc += 1;
				}
			} else {
				if (result == category){
					tnLoc += 1;
				} else {
					fpLoc += 1;
				}
			}
		}
	}

	MPI_Reduce(&tpLoc, &tp, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&tnLoc, &tn, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&fpLoc, &fp, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&fnLoc, &fn, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	// We compute the confusion matrix
	if(rank == 0){
		double accuracy = (tp+tn)/imagesTest.size();
		double precision = tp/(double)(tp+fp);
		double recall = tp/(double)(tp+fn);
		double Fscore = 2 * precision * recall / (double)(precision + recall);
		double FPR = fp /(double) (fp + tn);
		double TPR = tp / (double)(tp + fn);
		cout<< "tp : " << tp << "\ntn: " << tn << "\nfn : " << fn << "\nfp : " << fp;
		cout<<"\ntotal : " << tp + tn + fn + fp << "\nNombre d'image dans test : "<< imagesTest.size();
		cout<< "\n[precision, recall, F-score, FPR, TPR] :\n[" << precision << ", " << recall << ", " << Fscore << ", " << FPR << ", " <<
				TPR << "]" << endl;
		if (printDebug >= 3) {
			printlineVect("alphas",alphas,20);
		}
	}

	MPI_Finalize();

	return 0;
}

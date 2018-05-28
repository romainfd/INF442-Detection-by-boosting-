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


const int deltaSize = 4;
const int minSize = 8;
const string repo = "app";

// EXECUTION
// make clean
// make q1
// LD_LIBRARY_PATH=/usr/local/opencv-3.4.1/lib64 /usr/local/openmpi-3.0.1/bin/mpirun -np 4 q1 1000
// /usr/local/INF442-2018/P5/test/neg/im0.jpg


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

vector<vector<int> > emptyMatrix(int& l, int& c) {
	vector<vector<int> > matrix;
	for (int i = 0; i < l; i++) {
		vector<int> ligne(c); // une ligne de c cases = 0
		matrix.push_back(ligne);
	}
	return matrix;
}

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
	cv::Mat image = cv::imread("/usr/local/INF442-2018/P5/"+repo+"/neg/im0.jpg", cv::IMREAD_GRAYSCALE);
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

vector<int> caract_mpi(vector<vector<int> >& I, int ROOT, int& rank, int& np) {
	//printf("Processus %d is computing info for processes %d\n", rank, ROOT);
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

void printline(string name, double* ligne, int len) {
	cout << name << ":\t";
	for (int i = 0; i < len; i++) {
		cout<<ligne[i]<<" ";
	}
	cout<<endl;
}

template<typename T> void printlineVect(string name, vector<T>& ligne, int len) {
	cout<<name<<":\t";
	for (int i = 0; i < len; i++) {
		cout<<ligne[i]<<" ";
	}
	cout<<endl;
}

int main(int argc, char** argv) {
	// Récupération des images
	DIR *dir;
	struct dirent *ent;
	int nbNeg;
	vector<string> images; // on aura nbNeg images neg au debut, suivi des positives
	if ((dir = opendir (("/usr/local/INF442-2018/P5/"+repo+"/neg/").c_str())) != NULL) {
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
	if ((dir = opendir (("/usr/local/INF442-2018/P5/"+repo+"/pos/").c_str())) != NULL) {
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
	// ON RECUPERE L'IMAGE
	// Check command line args count
//	if(argc!=2){
//		cerr << "Please run as: " << endl << "   " << argv[0] << " image_name.jpg" << endl;
//	}
	// Test if argv[1] is actual file
//	struct stat buffer;
//	if (stat(argv[1],&buffer) != 0) {
//		cerr << "Cannot stat " << argv[1] <<  endl;
//	}

	// TRAITEMENT DE L'IMAGE
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
		printf("We have %d images available (%.2f%% negative)\n", images.size(), (((double)nbNeg)/ (double) images.size()) *100);
		printlineVect("Images", images, 20);
	}



	//initialisation commune de la seed
	unsigned int seed = 0;
    srand(seed);
    unsigned int K = 0;
    if (argc >= 2) {
    	K = atoi(argv[1]);
    } else {
        if(rank==0)cerr << "Please run as: " << endl << "   " << argv[0] << " nbOfImagesUsed" << endl;
        MPI_Finalize();
        return 0;
    }
    double epsilon=1;
    int nbC = nbCaractsTot(92, 112); // déterminer le nombre de caractéristiques pour initialiser w1 et w2
    vector<double> w1(nbC);
    vector<double> w2(nbC);
    // variation due à l'apprentissage sur l'image i
    double* delta1 = new double[nbC];
    double* delta2 = new double[nbC];
    // variation due a l'apprentissage de tous nos processus en 1 boucle
    double* globD1 = new double[nbC];
    double* globD2 = new double[nbC];
    int category;
    //initialiser w1 et w2
    for (int i = 0; i < nbC; i++){
        w1[i] = 1.;
        w2[i] = 0.;
        delta1[i] = 0.;
        delta2[i] = 0.;
    }
	for (int i = 0; i < K; i++) {
		// pas fait avant la npè boucle
        if((i > 1) && (i%np == 0)){
        	//une fois que tous les processus ont calculé une variation de w1 et w2 on met à jour w1 et w2 chez chacun avec AllReduce
			// on somme tous les apprentissages
			MPI_Allreduce (delta1, globD1, nbC, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce (delta2, globD2, nbC, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			for(int j = 0; j < nbC; j++){
				// on met à jour localement notre wi
				w1[j] -= globD1[j];
				w2[j] -= globD2[j];
			}
			if (rank == 0) {
				printf("The %d-th loop has been made. w1 and w2 have been updated to :\n", i/np);
				printlineVect("w1", w1, 20);
				printlineVect("w2", w2, 20);
				cout << " ------------------------------------------------------------ " << endl;
			}
        }
        // On choisit d'abord le type (pos/neg) puis l'image
        // il faut la meme seed pour tous pour être sur qu'il ait la meme image
        string filename = "/usr/local/INF442-2018/P5/"+repo+"/neg/im0.jpg";
        // on choisit au hasard notre image dans toutes celles dispos
        int imageRank = (rand() * images.size())/ RAND_MAX;
        if (imageRank < nbNeg) {
        	// on a une image négative !
            filename = "/usr/local/INF442-2018/P5/"+repo+"/neg/" + images[imageRank];
            category = -1;
        } else {
            // choisir fichier au hasard dans la classe -1
            filename = "/usr/local/INF442-2018/P5/"+repo+"/pos/" + images[imageRank];
            category = 1;
        }
        // on charge l'image choisie
		cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
		// Check for invalid input
		assert(!image.empty());
		// on calcule LOCALEMENT l'image intégrale
		vector<vector<int> > I = imageIntegrale(image);
		// printf("Processus %d in loop %d is going to compute the caracts of the image %s of category %d\n", rank, i, images[imageRank].c_str(), category);
		//displayMatrix(I);
		// On calcule toutes les caractéristiques et on les centralise dans le processus i%np
		vector<int> cars = caract_mpi(I, i%np, rank, np);
		// Update caracteristics
		if (rank == i%np) { // thanks to this condition cars is not empty
			printlineVect("caracts", cars, 20);
			// printf("Taille attendue : %d et taille reelle : %d\n", nbC, cars.size());
			int h;
			int nbRight = 0;
			// on parcourt tout le vecteur caracts et on maj les wi avec les infos apportees par l'image
			for (int c = 0; c < nbC; c++) {
				// classification (pos = +1, neg = -1)
				  h = (w1[c]*cars[c] + w2[c] >= 0) ? 1 : -1;
				  if (h == category) { nbRight++; }
				  // on compare notre classification a la categorie => eventuellement un delta si erreur
				  delta1[c]= epsilon * (h - category) * cars[c];
				  delta2[c]= epsilon * (h - category);
			}
			printf("Processus %d in loop %d has centralized all the caracteristics for image %s of category %d and %f%% of classifiers made a correct prediction\n", rank, i, images[imageRank].c_str(), category, double(nbRight)/nbC * 100);
			printline("Delta1", delta1, 20);
			printline("Delta2", delta2, 20);
			//printf("Processus %d has computed the delta\n", rank);
		}
	}
	delete[] delta1;
	delete[] delta2;
	delete[] globD1;
	delete[] globD2;

	// QUESTION 3

	MPI_Finalize();
	return 0;
}

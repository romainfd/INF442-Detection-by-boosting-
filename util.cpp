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
using namespace std;

const int deltaSize = 4;
const int minSize = 8;

// EXECUTION
// LD_LIBRARY_PATH=/usr/local/opencv-3.4.1/lib64 /usr/local/openmpi-3.0.1/bin/mpirun -np 4 q1 /usr/local/INF442-2018/P5/test/neg/im0.jpg


// Affichage d'une matrice de façon alignée pour des valeurs entre -9 et 99
void displayMatrix(vector<vector<int> >& matrix) {
	// we only use the reference to avoid a useless copy
	for (unsigned int i = 0; i < matrix.size(); i++) {
		for (unsigned int j = 0; j < matrix[0].size(); j++) {
			if (matrix[i][j] >= 0 and matrix[i][j] < 10) {
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
	cv::Mat image = cv::imread("/usr/local/INF442-2018/P5/test/neg/im0.jpg", cv::IMREAD_GRAYSCALE);
	vector<vector<int> > I = imageIntegrale(image);
	cout<<"Image intégrale:"<<endl;
	displayMatrix(I);
}

// Cette fonction calcule le nombre de caractéristiques calculées par un processeur de rang rank
int nbCaracts(int& rank, int& rows, int& cols) {
	return 0;
}

vector<vector<int> > caract_mpi(vector<vector<int> >& I, int& ROOT) {
	int rows = I.size();
	int cols = I[0].size();

	// MPI: rank and process number
	int rank=0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int np=0;
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	// On calcule le nb d'image à calculer pour ce processeur afin d'initialiser un tableau à la bonne taille
	unsigned int nCar = nbCaracts(rank, rows, cols);
	int* results = new int[nCar];

	unsigned int i = 0; // notre compteur
	for (unsigned int n = minSize + deltaSize*rank; n < rows; n += deltaSize * np ) {
	  for (unsigned int m = minSize + deltaSize*rank; m < cols; m += deltaSize * np) {
		  for (unsigned int x = 0; x < rows - n; x++) {
			  for (unsigned int y = 0; y < cols - m; y++) {
				  results[i++] = I[x+n][y+m] - I[x+n][y] - I[x][y+m] - I[x][y];
			  }
		  }
	   }
	}
	// on envoie tous les résultats à la racine pour qu'elle centralise tout
	if (rank != ROOT) {
		MPI_Send(results, nCar, MPI_INT, ROOT, 0, MPI_COMM_WORLD);
	}
    if (rank == ROOT) {
    	vector<int> resultsGlobal;
    	for (int i = 0; i < np; i++) {
			int procNCar = nbCaracts(i, rows, cols)
    		if (i != ROOT) {
				int* procResults = new int[procNCar];
				MPI_Recv(procResults, procNCar, MPI_INT, i, 0, MPI_COMM_WORLD);
    		} else { // pas besoin d'envoi: récupérer les résultats locaux
    			int* procResults = results;
    		}
    		resultsGlobal.insert(resulsGlobal.end(), procResults, procResults + procNCar);
    		delete[] procResults;
    	}
    }
	delete[] results;
	return resultsGlobal;
}

int main(int argc, char** argv) {
	// ON RECUPERE L'IMAGE
	// Check command line args count
	if(argc!=2){
		cerr << "Please run as: " << endl << "   " << argv[0] << " image_name.jpg" << endl;
	}
	// Test if argv[1] is actual file
//	struct stat buffer;
//	if (stat(argv[1],&buffer) != 0) {
//		cerr << "Cannot stat " << argv[1] <<  endl;
//	}
	// Lit l'image
	cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
	// Check for invalid input
	assert(!image.empty());

	// TRAITEMENT DE L'IMAGE
	// Question 1 : calcul de l'image intégrale en 1 seul parcours
	// testQ1();

	// Question 2
	MPI_Init(&argc, &argv);
	int np=0;
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	for (int i = 0; i < 10; i++) {
		string filename = "/usr/local/INF442-2018/P5/test/neg/im"<<i<<".jpg";
		cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
		vector<vector<int> > I = imageIntegrale(image);
		caract_mpi(I, i % np);
	}
	MPI_Finalize();

	// C'est du MPI je pense...
	// Il y a 5*11*23 + Somme des carrés jusqu'à 22 (j'ai plus la formule de tete) carrés
	// Grosso modo à un facteur près c'est du N cube...

	// Question 3 et suivantes
	// MPI aussi...
	return 0;
}

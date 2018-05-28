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

template<typename T> void printline(string name, T* ligne, int len) {
	cout << name << ":\t";
	for (int i = 0; i < len; i++) {
		cout<<ligne[i]<<" ";
	}
	cout<<endl;
}

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

int error(double h, bool cat) {
	if ((h >= 0 && cat) || (h < 0 && !cat)) {
		// classifie pos et pos ou classifie neg et neg
		return 0; // aucune erreur
	} else {
		return 1;
	}
}

int classifyF(vector<double>& w1F, vector<double>& w2F, vector<int>& carsImg, double& threshold) {
	double value = 0;
	for (int c = 0; c < w1F.size(); c++) {
		value += w1F[c]*carsImg[c] + w2F[c];
	}
	if (value >= threshold) {
		return 1;
	} else {
		return -1;
	}
}

int main(int argc, char** argv) {
	// Récupération des images
	DIR *dir;
	struct dirent *ent;
	int nbNeg = 0;
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
    double epsilon = 1;
    if (argc >= 3) {
    	K = atoi(argv[1]);
    	epsilon = atof(argv[2]);
    	printf("Epsilon = %f\n", epsilon);
    } else {
        if(rank==0)cerr << "Please run as: " << endl << "   " << argv[0] << " K epsilon" << endl;
        MPI_Finalize();
        return 0;
    }
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
			printf("Processus %d in loop %d has centralized all the caracteristics for image %s of category %d and %.2f%% of classifiers made a correct prediction\n", rank, i, images[imageRank].c_str(), category, double(nbRight)/nbC * 100);
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
	vector<double> alphas(nbC);

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
    	filename = "/usr/local/INF442-2018/P5/"+repo+ (cat[img] ? "/pos/" : "/neg/") + imagesBoosting[img];
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
    if (rank == 0) {
    	cout << "All the caracteristics of our boosting images have been computed."<<endl;
        printline("carsN[0]",carsN[0], 20);
        printlineVect("weigths",weights, 20);
    }

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
					err += weights[img]*error(w1[c]*carsN[img][c] + w2[c], cat[img]);
				}
				// and compares it to its localMin
				if (err < in.value) {
					in.value = err;
					in.index = c;
				}
			}
		}
		cout << in.value << endl;
		cout << in.index << endl;
		// once they all have done all the classifiers they had to compute the error for, we use Allreduce with minloc to find the best classifier and update the weights and the finalClassifier everywhere
		MPI_Allreduce((void*) &in, (void*) &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

		int k = out.index; // the best classifier
		// We recompute the error because Allreduce doesn't work
		double errk = 0;
		for (int img = 0; img < n; img++) {
			errk += weights[img]*error(w1[k]*carsN[img][k] + w2[k], cat[img]);
		}

		if (rank == 0) {
			printf("the best classifier for this step is %d with an error of %f\n", k, errk);
		}
		// We update the weights of the images and the global classifier in EACH processus thanks to the ALLreduce
		double s = 0; // to normalize
		for (int img = 0; img < n; img++) {
			weights[img] *= exp(-(cat[img] ? 1 : -1)*alphas[l]*((w1[k]*carsN[img][k] + w2[k] >= 0) ? 1 : -1));
			s += weights[img];
		}
		// Normalisation of the image weights
		for (int img = 0; img < n; img++) {
			weights[img] *= 1/s;
		}
		// Update of our global classifier
		double alpha = log((1-errk)/errk)/2;
		alphas[k] += alpha; // += because it could already have been taken before
		w1F[k] += alpha * w1[k];
		w2F[k] += alpha * w2[k];
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

	// we can classify an image with classifyF(w1F, w2F, cars, threshold);
	// and we compute the cars of the images in Test with caracts_mpi in parallel
    // il faut la meme seed pour tous pour être sur qu'il ait la meme image
    filename = "/usr/local/INF442-2018/P5/test/neg/im509.jpg";
    // on choisit au hasard notre image dans toutes celles dispos
//    int imageRank = (rand() * images.size())/ RAND_MAX;
//    if (imageRank < nbNeg) {
//    	// on a une image négative !
//        filename = "/usr/local/INF442-2018/P5/"+repo+"/neg/" + images[imageRank];
//        category = -1;
//    } else {
//        // choisir fichier au hasard dans la classe -1
//        filename = "/usr/local/INF442-2018/P5/"+repo+"/pos/" + images[imageRank];
//        category = 1;
//    }
    // on charge l'image choisie
	cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
	// Check for invalid input
	assert(!image.empty());
	// on calcule LOCALEMENT l'image intégrale
	vector<vector<int> > I = imageIntegrale(image);
	// printf("Processus %d in loop %d is going to compute the caracts of the image %s of category %d\n", rank, i, images[imageRank].c_str(), category);
	//displayMatrix(I);
	// On calcule toutes les caractéristiques et on les centralise dans le processus i%np
	vector<int> cars = caract_mpi(I, 0, rank, np);
	if (rank == 0) {
		cout << classifyF(w1F, w2F, cars, threshold) << endl;
	}
	MPI_Finalize();
	return 0;
}

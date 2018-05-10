//============================================================================
// Name        : PI.cpp
// Author      : Antoine Grosnit and Romain Fouilland
// Version     :
// Copyright   : Work of Antoine Grosnit and Romain Fouilland
// Description : PI INF442 : detection by "boosting"
//============================================================================
#include <iostream>
#include <vector>
using namespace std;

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

vector<vector<int> > imageIntegrale(vector<vector<int> >& img) {
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
	// delete s; // on efface l'espace mémoire créé pour s
	// Faut-il delete chaque ligne ?
	return I;
}

int main(int argc, char** argv) {
	// To test locally, we initiate an small image
	vector<vector<int> > image;
	vector<int> ligne1{ 1,  1, -1};
	vector<int> ligne2{-1, -1,  1};
	vector<int> ligne3{ 1, -1,  1};
	image.push_back(ligne1);
	image.push_back(ligne2);
	image.push_back(ligne3);
	cout<<"Matrice de base:"<<endl;
	displayMatrix(image);

	// TRAITEMENT DE L'IMAGE
	// Question 1 : calculde l'image intégrale en 1 seul parcours
	vector<vector<int> > I = imageIntegrale(image);
	cout<<"Image intégrale:"<<endl;
	displayMatrix(I);

	// Question 2
	// C'est du MPI je pense...
	// Il y a 5*11*23 + Somme des carrés jusqu'à 22 (j'ai plus la formule de tete) carrés
	// Grosso modo à un facteur près c'est du N cube...

	// Question 3 et suivantes
	// MPI aussi...
	return 0;
}

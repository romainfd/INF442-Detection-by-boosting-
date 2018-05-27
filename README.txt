Quelques notes sur le  projet 5 : Détection par "boosting".

- Les données d’apprentissage, de développement, de test se trouvent sur chaque PC des salles informatique dans le répertoire /usr/local/INF442-2018/P5/ :

	$ tree -d /usr/local/INF442-2018/P5/
	/usr/local/INF442-2018/P5/
	├── app
	│   ├── neg
	│   └── pos
	├── dev
	│   ├── neg
	│   └── pos
	└── test
	    ├── neg
	    └── pos

- Pour éviter les blocages sur les questions de compilation ou de lecteur de jpg, dans sp.zip se trouve un exemple de projet (Makefile + code) qui illustre l'usage des dernières versions de OpenCV et OpenMPI.  Dans l'exemple plus bas, 4 processus lisent en parallèle une image et affichent les valeurs d'une colonne des pixels qui correspond à l'ID de copie de processus qui effectue la lecture.

$ make
/usr/local/openmpi-3.0.1/bin/mpic++ -c -O3 -I /usr/local/opencv-3.4.1/include img_processing.cpp
/usr/local/openmpi-3.0.1/bin/mpic++ -c -O3 -I /usr/local/opencv-3.4.1/include q_test.cpp
/usr/local/openmpi-3.0.1/bin/mpic++ -o q_test img_processing.o q_test.o -L /usr/local/opencv-3.4.1/lib64 -lopencv_core -lopencv_imgcodecs -lopencv_highgui

$ LD_LIBRARY_PATH=/usr/local/opencv-3.4.1/lib64 /usr/local/openmpi-3.0.1/bin/mpirun -np 4 q_test /usr/local/INF442-2018/P5/test/neg/im0.jpg 
0: 41 33 38 35 23 30 39 27 33 17 20 31 22 11 15 19 19 19 14 32 46 100 95 57 46 46 51 36 25 40 70 108 63 44 49 48 40 27 16 35 43 21 32 63 67 64 86 115 155 153 151 143 139 155 165 157 135 121 111 110 107 101 103 110 97 111 121 116 107 110 129 146 154 159 163 163 162 164 167 167 155 159 157 154 159 170 172 166 168 172 176 175 171 168 169 171 168 175 170 155 156 173 176 166 160 157 154 155 164 173 172 165 
1: 27 27 25 22 22 26 27 23 29 14 16 24 21 19 21 20 16 21 22 31 36 74 80 66 57 56 38 35 45 61 78 71 80 52 52 54 47 36 32 60 62 47 69 114 126 117 120 131 157 161 163 152 145 158 163 147 139 127 117 114 109 101 100 105 83 98 108 103 97 107 128 145 161 168 167 160 162 170 167 155 152 163 164 156 160 175 176 164 164 168 171 171 169 168 170 173 164 171 168 160 166 181 180 166 150 154 157 160 166 173 173 169 
2: 20 27 25 26 33 29 20 20 16 16 17 14 22 46 48 23 22 27 32 42 54 77 81 73 62 58 33 59 92 100 99 62 63 56 72 73 64 61 64 92 94 84 97 129 153 163 148 122 140 150 157 147 140 154 157 137 127 120 115 114 111 104 99 100 93 88 86 91 96 105 122 139 162 159 151 146 152 162 161 152 156 164 166 164 167 175 175 169 163 165 166 166 164 164 166 168 165 159 163 177 181 170 161 160 143 150 156 157 161 168 172 173 
3: 34 37 32 30 34 27 19 22 22 19 21 23 36 58 56 29 25 22 28 52 95 105 79 44 34 44 39 71 86 74 80 64 30 49 85 93 96 101 88 88 107 88 84 100 119 137 135 118 139 143 142 127 119 136 147 136 115 113 115 118 117 109 102 98 111 98 98 110 110 102 115 140 157 145 140 148 157 159 160 162 163 159 163 172 173 167 167 174 174 173 170 165 160 157 156 157 166 156 156 166 164 147 136 137 135 146 156 159 159 162 165 164 

- Si vous exécutez votre code sur l'ensemble des images soyez vigilants avec l'usage de mémoire vive -> vous allez traiter des gigaoctets, voir des dizaines de gigaoctets des données.

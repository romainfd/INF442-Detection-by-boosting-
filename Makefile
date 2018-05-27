# --- macros
CC=/usr/local/openmpi-3.0.1/bin/mpic++
CC_LOC=g++
CC_O=g++
CFLAGS= -O3 -I /usr/local/opencv-3.4.1/include
LIBS = -L /usr/local/opencv-3.4.1/lib64 -lopencv_core -lopencv_imgcodecs -lopencv_highgui

# --- targets
all: q_test

q1: util.cpp
	$(CC_LOC) -o q1 util.cpp

q2: q1 util_mpi.o
	$(CC) -o q2 util.o util_mpi.o

q_test: img_processing.o q_test.o 
	$(CC) -o q_test img_processing.o q_test.o $(LIBS)

%.o: %.cpp
	$(CC) -c $(CFLAGS) $<
       
# --- remove binary and executable files
clean:
	rm -f q_test *.o

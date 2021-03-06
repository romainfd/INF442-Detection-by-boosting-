# --- macros
CC=/usr/local/openmpi-3.0.1/bin/mpic++
CFLAGS= -O3 -I /usr/local/opencv-3.4.1/include
LIBS = -L /usr/local/opencv-3.4.1/lib64 -lopencv_core -lopencv_imgcodecs -lopencv_highgui

# --- targets
all: boostingDetection.o img_processing.o
	$(CC) -o boostingDetection img_processing.o boostingDetection.o $(LIBS)

q_test: img_processing.o q_test.o 
	$(CC) -o q_test img_processing.o q_test.o $(LIBS)

boosting: boostingDetection.o img_processing.o
	$(CC) -o boostingDetection img_processing.o boostingDetection.o $(LIBS)
	
%.o: %.cpp
	$(CC) -c $(CFLAGS) $<
       
# --- remove binary and executable files
clean:
	rm -f q_test boostingDetection *.o

#include <sys/stat.h>
#include "img_processing.h"

using namespace cv;

int main(int argc, char** argv)
{
  // MPI: rank and process number 
  MPI_Init(&argc, &argv);
  int rank=0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int p=0;    MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Check command line args count
  if(argc!=2){
    if(rank==0)cerr << "Please run as: " << endl << "   " << argv[0] << " image_name.jpg" << endl;

    MPI_Finalize();
    return 0;
  }

  // Test if argv[1] is actual file
  struct stat buffer;   
  if (stat(argv[1],&buffer) != 0) {
    if(rank==0)cerr << "Cannot stat " << argv[1] <<  endl;

    MPI_Finalize();
    return 0;
  }

  // Test
  image_print_line(argv[1],rank);

  MPI_Finalize();

  return 0;
}  

#include "img_processing.h"

using namespace cv;

// 'ptr' is faster but 'at' is safer : https://docs.opencv.org/2.4/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html?highlight=accessing%20element
// Input:
//  - char* filename - full path to image
void image_print_line(const char* filename, int id)
{
  Mat image = imread(filename,IMREAD_GRAYSCALE); // Read the file

  // Check for invalid input
  assert(!image.empty());
  assert(image.rows>=id+1);

  cout << id << ": "; 
  for(int x = 0; x < image.cols; x++){
    // OpenCV -> row-major order: row/column -> (y/x)
    cout << (int)image.at<uchar>(id+1,x) << " ";
  }
  cout << endl;
}

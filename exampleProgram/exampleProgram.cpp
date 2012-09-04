/*
 ============================================================================
 Name        : exampleProgram.c
 Author      : Nicola Vigano
 Version     :
 Copyright   : 
 Description : Uses shared library to print greeting
               To run the resulting executable the LD_LIBRARY_PATH must be
               set to ${project_loc}/libEDF/.libs
               Alternatively, libtool creates a wrapper shell script in the
               build directory of this program which can be used to run it.
               Here the script will be called exampleProgram.
 ============================================================================
 */

#include "libEDF.h"

#include <cstdio>
#include <cstdlib>

using namespace std;

int
main(int argc, char** argv)
{
  char * filename = NULL;
  if (argc < 2) {
    fprintf(stderr, "No file specified\n");
    return EXIT_FAILURE;
  } else {
    filename = argv[1];
  }
  double yeah = 0;
  for (size_t count = 0; count < 200; count++) {
    EDF_File edfFile;
    if (!edfFile.load_file(filename)) {
      return EXIT_FAILURE;
    }
    yeah += edfFile.getData().getPixel<double>(40);
  }
  printf("Val: %lf\n", yeah);
  return EXIT_SUCCESS;
}

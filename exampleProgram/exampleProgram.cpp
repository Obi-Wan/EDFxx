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

int main(void) {
  EDF_File bene;
  EDF_parse_file(bene,
      "/media/windows/D/Documenti/Progetti/sandbox-fluo-tomography/single_channel01055.edf");
  return 0;
}

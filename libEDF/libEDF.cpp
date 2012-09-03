#include "libEDF.h"
#include <cstdio>
#include <stdexcept>
#include "edf_header_parser.h"

using namespace std;

extern FILE * yyin;

bool
EDF_parse_file(EDF_File & fileHeader, const char * fileName)
{
  yyin = fopen(fileName, "r");
  if (yyin) {
    try {
      int32_t res = yyparse(&fileHeader);
      if (res) {
        fprintf(stderr, "An error may have occurred, code: %3d\n", res);
        throw runtime_error("Error parsing\n");
      }
    } catch (const exception & e) {
      fclose(yyin);
      return false;
    }
  }
  printf("Num bytes read: %lu, in file: %s\n", fileHeader.getFields().headerLength, fileName);
  fclose(yyin);
  return true;
}

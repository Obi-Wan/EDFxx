#include "libEDF.h"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include "edf_header_parser.h"

using namespace std;

extern FILE * yyin;

const size_t
EDF_Data::getPixelSize() const
{
  switch (this->dataType) {
    case EDF_INT_08_S:
    case EDF_INT_08_U:
      return 1;
    case EDF_INT_16_S:
    case EDF_INT_16_U:
      return 2;
    case EDF_INT_32_S:
    case EDF_INT_32_U:
    case EDF_FLOAT_32:
      return 4;
    case EDF_INT_64_S:
    case EDF_INT_64_U:
    case EDF_FLOAT_64:
      return 8;
    default:
      throw runtime_error("Invalid datatype");
  }
}

EDF_Data::~EDF_Data()
{
  if (this->data) { free(this->data); this->data = 0; }
}

bool
EDF_File::parse_file(const char * fileName)
{
  yyin = fopen(fileName, "r");
  if (yyin) {
    yyrestart(yyin);
    try {
      int32_t res = yyparse(this);
      if (res) {
        fprintf(stderr, "An error may have occurred, code: %3d\n", res);
        throw runtime_error("Error parsing\n");
      }
    } catch (const exception & e) {
      fclose(yyin);
      return false;
    }
  }
  printf( "Header size: %lu, Image size: (%lu, %lu, %lu) -> %lu (bytes per pixel %lu)\n",
          this->getFields().headerLength, this->getData().dimensions[0],
          this->getData().dimensions[1], this->getData().dimensions[2],
          this->getData().totPixels, this->getData().getPixelSize());

  fclose(yyin);
  return true;
}

bool
EDF_File::load_file(const char * fileName)
{
  yyin = fopen(fileName, "r");
  if (yyin) {
    yyrestart(yyin);
    try {
      int32_t res = yyparse(this);
      if (res) {
        fprintf(stderr, "An error may have occurred, code: %3d\n", res);
        throw runtime_error("Error parsing\n");
      }
    } catch (const exception & e) {
      fclose(yyin);
      return false;
    }
  }
  fseek(yyin, this->getFields().headerLength, SEEK_SET);

  const size_t pixelSize = this->getData().getPixelSize();
//  printf( "Header size: %lu, Image size: (%lu, %lu, %lu) -> %lu (bytes per pixel %lu)\n",
//          this->getFields().headerLength, this->getData().dimensions[0],
//          this->getData().dimensions[1], this->getData().dimensions[2],
//          this->getData().totPixels, pixelSize);

  this->getData().data = malloc(this->getData().totPixels * pixelSize);
  if (!this->getData().data) {
    fclose(yyin);
    fprintf(stderr, "An error occurred in memory allocation\n");
    return false;
  }

  // Simple loading
  try {
    int8_t * bufferPos = (int8_t *)this->getData().data;
    const size_t block = 256;
    for(size_t count = 0; count < this->getData().totPixels; count += block) {
      size_t readBytes = fread(bufferPos, pixelSize, block, yyin);
      bufferPos += (pixelSize*block);
      if (readBytes < block && ferror(yyin)) {
        throw runtime_error("Error reading\n");
      }
    }
  } catch (const exception & e) {
    fclose(yyin);
    free(this->getData().data);
    this->getData().data = NULL;
    fprintf(stderr, "An error occurred in reading: %s\n", e.what());
    return false;
  }

  fclose(yyin);
  return true;
}



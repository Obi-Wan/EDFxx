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

EDF_Data::EDF_Data()
  : data(NULL), totPixels(0), dataType(EDF_NO_TYPE), allocator(malloc)
  , deallocator(free)
{ }

EDF_Data::~EDF_Data()
{
  this->dealloc();
}

template<typename Type>
inline void
EDF_Data::_transpose(Type * basePointer, const size_t & dimX, const size_t & dimY)
{
  for(size_t countY = 0; countY < dimY; countY++) {
    Type * __restrict const upperBase = basePointer + countY * dimX;
    for(size_t countX = countY+1; countX < dimX; countX++) {
      Type * __restrict const lower = basePointer + countX * dimX + countY;
      const Type temp = *lower;
      *lower = upperBase[countX];
      upperBase[countX] = temp;
    }
  }
}

void
EDF_Data::transpose()
{
  const size_t pixelSize = this->getPixelSize();
  const size_t blockSize = dimensions[0] * dimensions[1];
  const size_t blockBSize = blockSize * pixelSize;
  size_t numBlocks = dimensions[2];
  for(size_t count = 3; count < dimensions.size(); count++) {
    numBlocks *= dimensions[count];
  }
  for(size_t block = 0; block < numBlocks; block++) {
    const void * const basePointer = ((const int8_t *)data) + block * blockBSize;
    switch (this->dataType) {
      case EDF_INT_08_S: {
        this->_transpose((int8_t *)basePointer, dimensions[0], dimensions[1]);
        break;
      }
      case EDF_INT_08_U: {
        this->_transpose((uint8_t *)basePointer, dimensions[0], dimensions[1]);
        break;
      }
      case EDF_INT_16_S: {
        this->_transpose((int16_t *)basePointer, dimensions[0], dimensions[1]);
        break;
      }
      case EDF_INT_16_U: {
        this->_transpose((uint16_t *)basePointer, dimensions[0], dimensions[1]);
        break;
      }
      case EDF_INT_32_S: {
        this->_transpose((int32_t *)basePointer, dimensions[0], dimensions[1]);
        break;
      }
      case EDF_INT_32_U: {
        this->_transpose((uint32_t *)basePointer, dimensions[0], dimensions[1]);
        break;
      }
      case EDF_FLOAT_32: {
        this->_transpose((float *)basePointer, dimensions[0], dimensions[1]);
        break;
      }
      case EDF_INT_64_S: {
        this->_transpose((int64_t *)basePointer, dimensions[0], dimensions[1]);
        break;
      }
      case EDF_INT_64_U: {
        this->_transpose((uint64_t *)basePointer, dimensions[0], dimensions[1]);
        break;
      }
      case EDF_FLOAT_64: {
        this->_transpose((double *)basePointer, dimensions[0], dimensions[1]);
        break;
      }
      default:
        throw runtime_error("Invalid datatype");
    }
  }
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
EDF_File::load_file(const char * fileName, const bool transpose)
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
  // The parser moves a little bit too further
  fseek(yyin, this->getFields().headerLength, SEEK_SET);

  const size_t pixelSize = this->getData().getPixelSize();
//  printf( "Header size: %lu, Image size: (%lu, %lu, %lu) -> %lu (bytes per pixel %lu)\n",
//          this->getFields().headerLength, this->getData().dimensions[0],
//          this->getData().dimensions[1], this->getData().dimensions[2],
//          this->getData().totPixels, pixelSize);

  if (!this->getData().alloc()) {
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
    this->getData().dealloc();
    fprintf(stderr, "An error occurred in reading: %s\n", e.what());
    return false;
  }

  if (transpose) {
    this->getData().transpose();
  }

  fclose(yyin);
  return true;
}



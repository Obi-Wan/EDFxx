#include "libEDF.h"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include "edf_header_parser.h"
#include <malloc.h>

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
  : data(NULL), totPixels(0), dataType(EDF_NO_TYPE), allocator(&alignedAllocator)
  , deallocator(free), allocatedSize(0)
{ }

EDF_Data::~EDF_Data()
{
  this->dealloc();
}

inline void *
EDF_Data::alignedAllocator(size_t _size)
{
  void * const newMem = memalign(32, _size);
//  printf("in vecs %p %lu %d\n", &newMem, ((size_t)(&newMem)) % 32, sizeof(void *));
  return newMem;
}

inline void
EDF_Data::copyBufer(const int8_t * const in, int8_t * const out, const size_t & _size)
{
  typedef int8_t v16sb __attribute__ ((vector_size (16))) __attribute__ ((aligned(16)));
  const size_t roundedSize = ROUND_DOWN(_size, 16);
  for(size_t count = 0; count < roundedSize; count += 16)
  {
    *((v16sb *)&out[count]) = *((const v16sb *)&in[count]);
  }
  for(size_t count = roundedSize; count < _size; count++)
  {
    out[count] = in[count];
  }
}

template<typename Type>
inline void
EDF_Data::_transpose(const Type * const inPointer, Type * const outPointer,
    const size_t & dimX, const size_t & dimY)
{
#if defined(SLOW_TRANSPOSE)
  for(size_t countY = 0; countY < dimY; countY++) {
    const Type * __restrict const inBase = inPointer + countY * dimX;
    for(size_t countX = 0; countX < dimX; countX++) {
      outPointer[dimY * countX + countY] = inBase[countX];
    }
  }
#elif defined(NO_FAST_TRANSPOSE)
  const size_t blockSize = 4;
  const size_t roundedDimY = ROUND_DOWN(dimY, blockSize);
  const size_t roundedDimX = ROUND_DOWN(dimX, blockSize);
  for(size_t countY = 0; countY < roundedDimY; countY += blockSize) {
    const Type * __restrict const inBase = inPointer + countY * dimX;
    for(size_t countX = 0; countX < roundedDimX; countX += blockSize) {
      const Type * __restrict const inPtr = inBase + countX;
      const Type inArray[blockSize*blockSize] = {
          inPtr[ 0*dimX + 0 ], inPtr[ 0*dimX + 1 ], inPtr[ 0*dimX + 2 ], inPtr[ 0*dimX + 3 ],
          inPtr[ 1*dimX + 0 ], inPtr[ 1*dimX + 1 ], inPtr[ 1*dimX + 2 ], inPtr[ 1*dimX + 3 ],
          inPtr[ 2*dimX + 0 ], inPtr[ 2*dimX + 1 ], inPtr[ 2*dimX + 2 ], inPtr[ 2*dimX + 3 ],
          inPtr[ 3*dimX + 0 ], inPtr[ 3*dimX + 1 ], inPtr[ 3*dimX + 2 ], inPtr[ 3*dimX + 3 ] };

      Type * __restrict const outPtr = outPointer + dimY * countX + countY;
      outPtr[ 0*dimY + 0 ] = inArray[ 0*blockSize + 0 ];
      outPtr[ 0*dimY + 1 ] = inArray[ 1*blockSize + 0 ];
      outPtr[ 0*dimY + 2 ] = inArray[ 2*blockSize + 0 ];
      outPtr[ 0*dimY + 3 ] = inArray[ 3*blockSize + 0 ];
      outPtr[ 1*dimY + 0 ] = inArray[ 0*blockSize + 1 ];
      outPtr[ 1*dimY + 1 ] = inArray[ 1*blockSize + 1 ];
      outPtr[ 1*dimY + 2 ] = inArray[ 2*blockSize + 1 ];
      outPtr[ 1*dimY + 3 ] = inArray[ 3*blockSize + 1 ];
      outPtr[ 2*dimY + 0 ] = inArray[ 0*blockSize + 2 ];
      outPtr[ 2*dimY + 1 ] = inArray[ 1*blockSize + 2 ];
      outPtr[ 2*dimY + 2 ] = inArray[ 2*blockSize + 2 ];
      outPtr[ 2*dimY + 3 ] = inArray[ 3*blockSize + 2 ];
      outPtr[ 3*dimY + 0 ] = inArray[ 0*blockSize + 3 ];
      outPtr[ 3*dimY + 1 ] = inArray[ 1*blockSize + 3 ];
      outPtr[ 3*dimY + 2 ] = inArray[ 2*blockSize + 3 ];
      outPtr[ 3*dimY + 3 ] = inArray[ 3*blockSize + 3 ];
    }
    for(size_t countX = roundedDimX; countX < dimX; countX++) {
      outPointer[dimY * countX + countY] = inBase[countX];
    }
  }
  for(size_t countY = roundedDimY; countY < dimY; countY++) {
    const Type * __restrict const inBase = inPointer + countY * dimX;
    for(size_t countX = 0; countX < dimX; countX++) {
      outPointer[dimY * countX + countY] = inBase[countX];
    }
  }
#else
  typedef Type v8dt __attribute__ ((vector_size (sizeof(Type)*8))) __attribute__((aligned(32)));
  const size_t blockSize = 8;
  const size_t roundedDimY = ROUND_DOWN(dimY, blockSize);
  const size_t roundedDimX = ROUND_DOWN(dimX, blockSize);
  for(size_t countY = 0; countY < roundedDimY; countY += blockSize) {
    const Type * __restrict const inBase = inPointer + countY * dimX;
    for(size_t countX = 0; countX < roundedDimX; countX += blockSize) {
      const Type * __restrict const inPtr = inBase + countX;
      const v8dt inV0 = *((v8dt *)&inPtr[0*dimX]);
      const v8dt inV1 = *((v8dt *)&inPtr[1*dimX]);
      const v8dt inV2 = *((v8dt *)&inPtr[2*dimX]);
      const v8dt inV3 = *((v8dt *)&inPtr[3*dimX]);
      const v8dt inV4 = *((v8dt *)&inPtr[4*dimX]);
      const v8dt inV5 = *((v8dt *)&inPtr[5*dimX]);
      const v8dt inV6 = *((v8dt *)&inPtr[6*dimX]);
      const v8dt inV7 = *((v8dt *)&inPtr[7*dimX]);

      const Type * const finV0 = (const Type *) &inV0;
      const Type * const finV1 = (const Type *) &inV1;
      const Type * const finV2 = (const Type *) &inV2;
      const Type * const finV3 = (const Type *) &inV3;
      const Type * const finV4 = (const Type *) &inV4;
      const Type * const finV5 = (const Type *) &inV5;
      const Type * const finV6 = (const Type *) &inV6;
      const Type * const finV7 = (const Type *) &inV7;

      const v8dt outV0 = { finV0[0], finV1[0], finV2[0], finV3[0], finV4[0], finV5[0], finV6[0], finV7[0] };
      const v8dt outV1 = { finV0[1], finV1[1], finV2[1], finV3[1], finV4[1], finV5[1], finV6[1], finV7[1] };
      const v8dt outV2 = { finV0[2], finV1[2], finV2[2], finV3[2], finV4[2], finV5[2], finV6[2], finV7[2] };
      const v8dt outV3 = { finV0[3], finV1[3], finV2[3], finV3[3], finV4[3], finV5[3], finV6[3], finV7[3] };
      const v8dt outV4 = { finV0[4], finV1[4], finV2[4], finV3[4], finV4[4], finV5[4], finV6[4], finV7[4] };
      const v8dt outV5 = { finV0[5], finV1[5], finV2[5], finV3[5], finV4[5], finV5[5], finV6[5], finV7[5] };
      const v8dt outV6 = { finV0[6], finV1[6], finV2[6], finV3[6], finV4[6], finV5[6], finV6[6], finV7[6] };
      const v8dt outV7 = { finV0[7], finV1[7], finV2[7], finV3[7], finV4[7], finV5[7], finV6[7], finV7[7] };

      Type * __restrict const outPtr = outPointer + dimY * countX + countY;
      *((v8dt *)&outPtr[0*dimY]) = outV0;
      *((v8dt *)&outPtr[1*dimY]) = outV1;
      *((v8dt *)&outPtr[2*dimY]) = outV2;
      *((v8dt *)&outPtr[3*dimY]) = outV3;
      *((v8dt *)&outPtr[4*dimY]) = outV4;
      *((v8dt *)&outPtr[5*dimY]) = outV5;
      *((v8dt *)&outPtr[6*dimY]) = outV6;
      *((v8dt *)&outPtr[7*dimY]) = outV7;
    }
    for(size_t countX = roundedDimX; countX < dimX; countX++) {
      outPointer[dimY * countX + countY] = inBase[countX];
    }
  }
  for(size_t countY = roundedDimY; countY < dimY; countY++) {
    const Type * __restrict const inBase = inPointer + countY * dimX;
    for(size_t countX = 0; countX < dimX; countX++) {
      outPointer[dimY * countX + countY] = inBase[countX];
    }
  }
#endif
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
  void * const newData = this->allocator(pixelSize * this->totPixels);

  for(size_t block = 0; block < numBlocks; block++) {
    const void * const inPointer = ((const int8_t *)data) + block * blockBSize;
    void * const outPointer = ((int8_t *)newData) + block * blockBSize;
    switch (this->dataType) {
      case EDF_INT_08_S: {
        this->_transpose((const int8_t *)inPointer, (int8_t *)outPointer, dimensions[0], dimensions[1]);
        break;
      }
      case EDF_INT_08_U: {
        this->_transpose((const uint8_t *)inPointer, (uint8_t *)outPointer, dimensions[0], dimensions[1]);
        break;
      }
      case EDF_INT_16_S: {
        this->_transpose((const int16_t *)inPointer, (int16_t *)outPointer, dimensions[0], dimensions[1]);
        break;
      }
      case EDF_INT_16_U: {
        this->_transpose((const uint16_t *)inPointer, (uint16_t *)outPointer, dimensions[0], dimensions[1]);
        break;
      }
      case EDF_INT_32_S: {
        this->_transpose((const int32_t *)inPointer, (int32_t *)outPointer, dimensions[0], dimensions[1]);
        break;
      }
      case EDF_INT_32_U: {
        this->_transpose((const uint32_t *)inPointer, (uint32_t *)outPointer, dimensions[0], dimensions[1]);
        break;
      }
      case EDF_FLOAT_32: {
        this->_transpose((const float *)inPointer, (float *)outPointer, dimensions[0], dimensions[1]);
        break;
      }
      case EDF_INT_64_S: {
        this->_transpose((const int64_t *)inPointer, (int64_t *)outPointer, dimensions[0], dimensions[1]);
        break;
      }
      case EDF_INT_64_U: {
        this->_transpose((const uint64_t *)inPointer, (uint64_t *)outPointer, dimensions[0], dimensions[1]);
        break;
      }
      case EDF_FLOAT_64: {
        this->_transpose((const double *)inPointer, (double *)outPointer, dimensions[0], dimensions[1]);
        break;
      }
      default:
        throw runtime_error("Invalid datatype");
    }
  }

  this->deallocator(this->data);
  this->data = newData;
}

inline void
EDF_File::_parse_header(FILE * fid)
{
  if (fid) {
    yyrestart(fid);
    int32_t res = yyparse(this);
    if (res) {
      fprintf(stderr, "An error may have occurred, code: %3d\n", res);
      throw runtime_error("Error parsing\n");
    }
  }
}

inline void
EDF_File::_load_data(FILE * fid)
{
  const size_t pixelSize = this->getData().getPixelSize();
  // The parser moves a little bit too further ( YY_READ_BUF_SIZE )
  fseek(fid, this->getFields().headerLength, SEEK_SET);

  if (!this->getData().realloc()) {
    throw runtime_error("An error occurred in memory allocation\n");
  }

  int8_t * bufferPos = (int8_t *)this->getData().data;
  const size_t block = 256;

  // Simple loading
  for(size_t count = 0; count < this->getData().totPixels; count += block)
  {
    size_t readBytes = fread(bufferPos, pixelSize, block, fid);
    bufferPos += (pixelSize*block);

    if (readBytes < block && ferror(yyin)) {
      throw runtime_error("Error reading\n");
    }
  }
}

bool
EDF_File::parse_file(const char * fileName)
{
  yyin = fopen(fileName, "r");
  try {
    _parse_header(yyin);
  } catch (const exception & e) {
    fclose(yyin);
    return false;
  }
  printf( "Header size: %lu, Image size: (%lu, %lu, %lu) -> %lu (bytes per pixel %lu)\n",
          this->getFields().headerLength, this->getData().dimensions[0],
          this->getData().dimensions[1], this->getData().dimensions[2],
          this->getData().totPixels, this->getData().getPixelSize());

  fclose(yyin);
  return true;
}


bool
EDF_File::load_data(const char * fileName, const bool transpose)
{
  FILE * fid = fopen(fileName, "r");

  try {
    _load_data(fid);
  } catch (const exception & e) {
    fclose(fid);
    this->getData().dealloc();
    fprintf(stderr, "An error occurred in reading: %s\n", e.what());
    return false;
  }

  if (transpose) {
    this->getData().transpose();
  }

  fclose(fid);
  return true;
}

bool
EDF_File::load_file(const char * fileName, const bool transpose)
{
  yyin = fopen(fileName, "r");
  try {
    _parse_header(yyin);
  } catch (const exception & e) {
    fclose(yyin);
    return false;
  }

  try {
    _load_data(yyin);
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



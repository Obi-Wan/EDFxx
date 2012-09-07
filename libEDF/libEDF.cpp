#include "libEDF.h"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include "edf_header_parser.h"
#include <malloc.h>

#include <xmmintrin.h>

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

#if defined(__SSE2__) && defined(NO_INTRIN)
template<>
inline void
EDF_Data::_transpose(const float * const inPointer, float * const outPointer,
    const size_t & dimX, const size_t & dimY)
{
  typedef float v4sf __attribute__ ((vector_size (16))) __attribute__((aligned(16)));
  const size_t blockSize = 8;
  const size_t roundedDimY = ROUND_DOWN(dimY, blockSize);
  const size_t roundedDimX = ROUND_DOWN(dimX, blockSize);
  for(size_t countY = 0; countY < roundedDimY; countY += blockSize) {
    const float * __restrict const inBase = inPointer + countY * dimX;
    for(size_t countX = 0; countX < roundedDimX; countX += 2 * blockSize) {
      const float * __restrict const inPtr = inBase + countX;
      const v4sf inV00 = *((v4sf *)&inPtr[0*dimX + 0]);
      const v4sf inV10 = *((v4sf *)&inPtr[0*dimX + 4]);
      const v4sf inV20 = *((v4sf *)&inPtr[0*dimX + 8]);
      const v4sf inV30 = *((v4sf *)&inPtr[0*dimX +12]);

      const v4sf inV01 = *((v4sf *)&inPtr[1*dimX + 0]);
      const v4sf inV11 = *((v4sf *)&inPtr[1*dimX + 4]);
      const v4sf inV21 = *((v4sf *)&inPtr[1*dimX + 8]);
      const v4sf inV31 = *((v4sf *)&inPtr[1*dimX +12]);

      const v4sf inV02 = *((v4sf *)&inPtr[2*dimX + 0]);
      const v4sf inV12 = *((v4sf *)&inPtr[2*dimX + 4]);
      const v4sf inV22 = *((v4sf *)&inPtr[2*dimX + 8]);
      const v4sf inV32 = *((v4sf *)&inPtr[2*dimX +12]);

      const v4sf inV03 = *((v4sf *)&inPtr[3*dimX + 0]);
      const v4sf inV13 = *((v4sf *)&inPtr[3*dimX + 4]);
      const v4sf inV23 = *((v4sf *)&inPtr[3*dimX + 8]);
      const v4sf inV33 = *((v4sf *)&inPtr[3*dimX +12]);

      const v4sf inV04 = *((v4sf *)&inPtr[4*dimX + 0]);
      const v4sf inV14 = *((v4sf *)&inPtr[4*dimX + 4]);
      const v4sf inV24 = *((v4sf *)&inPtr[4*dimX + 8]);
      const v4sf inV34 = *((v4sf *)&inPtr[4*dimX +12]);

      const v4sf inV05 = *((v4sf *)&inPtr[5*dimX + 0]);
      const v4sf inV15 = *((v4sf *)&inPtr[5*dimX + 4]);
      const v4sf inV25 = *((v4sf *)&inPtr[5*dimX + 8]);
      const v4sf inV35 = *((v4sf *)&inPtr[5*dimX +12]);

      const v4sf inV06 = *((v4sf *)&inPtr[6*dimX + 0]);
      const v4sf inV16 = *((v4sf *)&inPtr[6*dimX + 4]);
      const v4sf inV26 = *((v4sf *)&inPtr[6*dimX + 8]);
      const v4sf inV36 = *((v4sf *)&inPtr[6*dimX +12]);

      const v4sf inV07 = *((v4sf *)&inPtr[7*dimX + 0]);
      const v4sf inV17 = *((v4sf *)&inPtr[7*dimX + 4]);
      const v4sf inV27 = *((v4sf *)&inPtr[7*dimX + 8]);
      const v4sf inV37 = *((v4sf *)&inPtr[7*dimX +12]);

      const float * const finV00 = (const float *) &inV00;
      const float * const finV01 = (const float *) &inV01;
      const float * const finV02 = (const float *) &inV02;
      const float * const finV03 = (const float *) &inV03;
      const float * const finV04 = (const float *) &inV04;
      const float * const finV05 = (const float *) &inV05;
      const float * const finV06 = (const float *) &inV06;
      const float * const finV07 = (const float *) &inV07;
      const float * const finV10 = (const float *) &inV10;
      const float * const finV11 = (const float *) &inV11;
      const float * const finV12 = (const float *) &inV12;
      const float * const finV13 = (const float *) &inV13;
      const float * const finV14 = (const float *) &inV14;
      const float * const finV15 = (const float *) &inV15;
      const float * const finV16 = (const float *) &inV16;
      const float * const finV17 = (const float *) &inV17;
      const float * const finV20 = (const float *) &inV20;
      const float * const finV21 = (const float *) &inV21;
      const float * const finV22 = (const float *) &inV22;
      const float * const finV23 = (const float *) &inV23;
      const float * const finV24 = (const float *) &inV24;
      const float * const finV25 = (const float *) &inV25;
      const float * const finV26 = (const float *) &inV26;
      const float * const finV27 = (const float *) &inV27;
      const float * const finV30 = (const float *) &inV30;
      const float * const finV31 = (const float *) &inV31;
      const float * const finV32 = (const float *) &inV32;
      const float * const finV33 = (const float *) &inV33;
      const float * const finV34 = (const float *) &inV34;
      const float * const finV35 = (const float *) &inV35;
      const float * const finV36 = (const float *) &inV36;
      const float * const finV37 = (const float *) &inV37;

      // First Block
      const v4sf outV00 = { finV00[0], finV01[0], finV02[0], finV03[0] };
      const v4sf outV01 = { finV00[1], finV01[1], finV02[1], finV03[1] };
      const v4sf outV02 = { finV00[2], finV01[2], finV02[2], finV03[2] };
      const v4sf outV03 = { finV00[3], finV01[3], finV02[3], finV03[3] };

      const v4sf outV04 = { finV10[0], finV11[0], finV12[0], finV13[0] };
      const v4sf outV05 = { finV10[1], finV11[1], finV12[1], finV13[1] };
      const v4sf outV06 = { finV10[2], finV11[2], finV12[2], finV13[2] };
      const v4sf outV07 = { finV10[3], finV11[3], finV12[3], finV13[3] };

      const v4sf outV10 = { finV04[0], finV05[0], finV06[0], finV07[0] };
      const v4sf outV11 = { finV04[1], finV05[1], finV06[1], finV07[1] };
      const v4sf outV12 = { finV04[2], finV05[2], finV06[2], finV07[2] };
      const v4sf outV13 = { finV04[3], finV05[3], finV06[3], finV07[3] };

      const v4sf outV14 = { finV14[0], finV15[0], finV16[0], finV17[0] };
      const v4sf outV15 = { finV14[1], finV15[1], finV16[1], finV17[1] };
      const v4sf outV16 = { finV14[2], finV15[2], finV16[2], finV17[2] };
      const v4sf outV17 = { finV14[3], finV15[3], finV16[3], finV17[3] };

      // Second Block
      const v4sf outV20 = { finV20[0], finV21[0], finV22[0], finV23[0] };
      const v4sf outV21 = { finV20[1], finV21[1], finV22[1], finV23[1] };
      const v4sf outV22 = { finV20[2], finV21[2], finV22[2], finV23[2] };
      const v4sf outV23 = { finV20[3], finV21[3], finV22[3], finV23[3] };

      const v4sf outV24 = { finV30[0], finV31[0], finV32[0], finV33[0] };
      const v4sf outV25 = { finV30[1], finV31[1], finV32[1], finV33[1] };
      const v4sf outV26 = { finV30[2], finV31[2], finV32[2], finV33[2] };
      const v4sf outV27 = { finV30[3], finV31[3], finV32[3], finV33[3] };

      const v4sf outV30 = { finV24[0], finV25[0], finV26[0], finV27[0] };
      const v4sf outV31 = { finV24[1], finV25[1], finV26[1], finV27[1] };
      const v4sf outV32 = { finV24[2], finV25[2], finV26[2], finV27[2] };
      const v4sf outV33 = { finV24[3], finV25[3], finV26[3], finV27[3] };

      const v4sf outV34 = { finV34[0], finV35[0], finV36[0], finV37[0] };
      const v4sf outV35 = { finV34[1], finV35[1], finV36[1], finV37[1] };
      const v4sf outV36 = { finV34[2], finV35[2], finV36[2], finV37[2] };
      const v4sf outV37 = { finV34[3], finV35[3], finV36[3], finV37[3] };

      float * __restrict const outPtr1 = outPointer + dimY * countX + countY;
      *((v4sf *)&outPtr1[0*dimY + 0]) = outV00;
      *((v4sf *)&outPtr1[0*dimY + 4]) = outV10;
      *((v4sf *)&outPtr1[1*dimY + 0]) = outV01;
      *((v4sf *)&outPtr1[1*dimY + 4]) = outV11;
      *((v4sf *)&outPtr1[2*dimY + 0]) = outV02;
      *((v4sf *)&outPtr1[2*dimY + 4]) = outV12;
      *((v4sf *)&outPtr1[3*dimY + 0]) = outV03;
      *((v4sf *)&outPtr1[3*dimY + 4]) = outV13;
      *((v4sf *)&outPtr1[4*dimY + 0]) = outV04;
      *((v4sf *)&outPtr1[4*dimY + 4]) = outV14;
      *((v4sf *)&outPtr1[5*dimY + 0]) = outV05;
      *((v4sf *)&outPtr1[5*dimY + 4]) = outV15;
      *((v4sf *)&outPtr1[6*dimY + 0]) = outV06;
      *((v4sf *)&outPtr1[6*dimY + 4]) = outV16;
      *((v4sf *)&outPtr1[7*dimY + 0]) = outV07;
      *((v4sf *)&outPtr1[7*dimY + 4]) = outV17;

      float * __restrict const outPtr2 = outPointer + dimY * (countX + blockSize) + countY;
      *((v4sf *)&outPtr2[0*dimY + 0]) = outV20;
      *((v4sf *)&outPtr2[0*dimY + 4]) = outV30;
      *((v4sf *)&outPtr2[1*dimY + 0]) = outV21;
      *((v4sf *)&outPtr2[1*dimY + 4]) = outV31;
      *((v4sf *)&outPtr2[2*dimY + 0]) = outV22;
      *((v4sf *)&outPtr2[2*dimY + 4]) = outV32;
      *((v4sf *)&outPtr2[3*dimY + 0]) = outV23;
      *((v4sf *)&outPtr2[3*dimY + 4]) = outV33;
      *((v4sf *)&outPtr2[4*dimY + 0]) = outV24;
      *((v4sf *)&outPtr2[4*dimY + 4]) = outV34;
      *((v4sf *)&outPtr2[5*dimY + 0]) = outV25;
      *((v4sf *)&outPtr2[5*dimY + 4]) = outV35;
      *((v4sf *)&outPtr2[6*dimY + 0]) = outV26;
      *((v4sf *)&outPtr2[6*dimY + 4]) = outV36;
      *((v4sf *)&outPtr2[7*dimY + 0]) = outV27;
      *((v4sf *)&outPtr2[7*dimY + 4]) = outV37;
    }
    for(size_t countX = roundedDimX; countX < dimX; countX++) {
      outPointer[dimY * countX + countY] = inBase[countX];
    }
  }
  for(size_t countY = roundedDimY; countY < dimY; countY++) {
    const float * __restrict const inBase = inPointer + countY * dimX;
    for(size_t countX = 0; countX < dimX; countX++) {
      outPointer[dimY * countX + countY] = inBase[countX];
    }
  }
}
#elif defined(__SSE2__)
template<>
inline void
EDF_Data::_transpose(const float * const inPointer, float * const outPointer,
    const size_t & dimX, const size_t & dimY)
{
  typedef float v4sf __attribute__ ((vector_size (16))) __attribute__((aligned(16)));
  const size_t blockSize = 8;
  const size_t roundedDimY = ROUND_DOWN(dimY, blockSize);
  const size_t roundedDimX = ROUND_DOWN(dimX, 2*blockSize);
  for(size_t countY = 0; countY < roundedDimY; countY += blockSize) {
    const float * __restrict const inBase = inPointer + countY * dimX;
    for(size_t countX = 0; countX < roundedDimX; countX += 2*blockSize) {
      const float * __restrict const inPtr = inBase + countX;
      const v4sf inV00 = *((v4sf *)&inPtr[0*dimX + 0]);
      const v4sf inV10 = *((v4sf *)&inPtr[0*dimX + 4]);
      const v4sf inV20 = *((v4sf *)&inPtr[0*dimX + 8]);
      const v4sf inV30 = *((v4sf *)&inPtr[0*dimX +12]);

      const v4sf inV01 = *((v4sf *)&inPtr[1*dimX + 0]);
      const v4sf inV11 = *((v4sf *)&inPtr[1*dimX + 4]);
      const v4sf inV21 = *((v4sf *)&inPtr[1*dimX + 8]);
      const v4sf inV31 = *((v4sf *)&inPtr[1*dimX +12]);

      const v4sf inV02 = *((v4sf *)&inPtr[2*dimX + 0]);
      const v4sf inV12 = *((v4sf *)&inPtr[2*dimX + 4]);
      const v4sf inV22 = *((v4sf *)&inPtr[2*dimX + 8]);
      const v4sf inV32 = *((v4sf *)&inPtr[2*dimX +12]);

      const v4sf inV03 = *((v4sf *)&inPtr[3*dimX + 0]);
      const v4sf inV13 = *((v4sf *)&inPtr[3*dimX + 4]);
      const v4sf inV23 = *((v4sf *)&inPtr[3*dimX + 8]);
      const v4sf inV33 = *((v4sf *)&inPtr[3*dimX +12]);

      const v4sf inV04 = *((v4sf *)&inPtr[4*dimX + 0]);
      const v4sf inV14 = *((v4sf *)&inPtr[4*dimX + 4]);
      const v4sf inV24 = *((v4sf *)&inPtr[4*dimX + 8]);
      const v4sf inV34 = *((v4sf *)&inPtr[4*dimX +12]);

      const v4sf inV05 = *((v4sf *)&inPtr[5*dimX + 0]);
      const v4sf inV15 = *((v4sf *)&inPtr[5*dimX + 4]);
      const v4sf inV25 = *((v4sf *)&inPtr[5*dimX + 8]);
      const v4sf inV35 = *((v4sf *)&inPtr[5*dimX +12]);

      const v4sf inV06 = *((v4sf *)&inPtr[6*dimX + 0]);
      const v4sf inV16 = *((v4sf *)&inPtr[6*dimX + 4]);
      const v4sf inV26 = *((v4sf *)&inPtr[6*dimX + 8]);
      const v4sf inV36 = *((v4sf *)&inPtr[6*dimX +12]);

      const v4sf inV07 = *((v4sf *)&inPtr[7*dimX + 0]);
      const v4sf inV17 = *((v4sf *)&inPtr[7*dimX + 4]);
      const v4sf inV27 = *((v4sf *)&inPtr[7*dimX + 8]);
      const v4sf inV37 = *((v4sf *)&inPtr[7*dimX +12]);

      // First Block
      const v4sf temp00 = _mm_unpacklo_ps(inV00, inV01);
      const v4sf temp01 = _mm_unpackhi_ps(inV00, inV01);
      const v4sf temp02 = _mm_unpacklo_ps(inV02, inV03);
      const v4sf temp03 = _mm_unpackhi_ps(inV02, inV03);

      const v4sf temp10 = _mm_unpacklo_ps(inV04, inV05);
      const v4sf temp11 = _mm_unpackhi_ps(inV04, inV05);
      const v4sf temp12 = _mm_unpacklo_ps(inV06, inV07);
      const v4sf temp13 = _mm_unpackhi_ps(inV06, inV07);

      const v4sf temp04 = _mm_unpacklo_ps(inV10, inV11);
      const v4sf temp05 = _mm_unpackhi_ps(inV10, inV11);
      const v4sf temp06 = _mm_unpacklo_ps(inV12, inV13);
      const v4sf temp07 = _mm_unpackhi_ps(inV12, inV13);

      const v4sf temp14 = _mm_unpacklo_ps(inV14, inV15);
      const v4sf temp15 = _mm_unpackhi_ps(inV14, inV15);
      const v4sf temp16 = _mm_unpacklo_ps(inV16, inV17);
      const v4sf temp17 = _mm_unpackhi_ps(inV16, inV17);

      // Second Block
      const v4sf temp20 = _mm_unpacklo_ps(inV20, inV21);
      const v4sf temp21 = _mm_unpackhi_ps(inV20, inV21);
      const v4sf temp22 = _mm_unpacklo_ps(inV22, inV23);
      const v4sf temp23 = _mm_unpackhi_ps(inV22, inV23);

      const v4sf temp30 = _mm_unpacklo_ps(inV24, inV25);
      const v4sf temp31 = _mm_unpackhi_ps(inV24, inV25);
      const v4sf temp32 = _mm_unpacklo_ps(inV26, inV27);
      const v4sf temp33 = _mm_unpackhi_ps(inV26, inV27);

      const v4sf temp24 = _mm_unpacklo_ps(inV30, inV31);
      const v4sf temp25 = _mm_unpackhi_ps(inV30, inV31);
      const v4sf temp26 = _mm_unpacklo_ps(inV32, inV33);
      const v4sf temp27 = _mm_unpackhi_ps(inV32, inV33);

      const v4sf temp34 = _mm_unpacklo_ps(inV34, inV35);
      const v4sf temp35 = _mm_unpackhi_ps(inV34, inV35);
      const v4sf temp36 = _mm_unpacklo_ps(inV36, inV37);
      const v4sf temp37 = _mm_unpackhi_ps(inV36, inV37);

      float * __restrict const outPtr1 = outPointer + dimY * countX + countY;
      *((v4sf *)&outPtr1[0*dimY + 0]) = _mm_movelh_ps(temp00, temp02);
      *((v4sf *)&outPtr1[0*dimY + 4]) = _mm_movelh_ps(temp10, temp12);
      *((v4sf *)&outPtr1[1*dimY + 0]) = _mm_movehl_ps(temp00, temp02);
      *((v4sf *)&outPtr1[1*dimY + 4]) = _mm_movehl_ps(temp10, temp12);
      *((v4sf *)&outPtr1[2*dimY + 0]) = _mm_movelh_ps(temp01, temp03);
      *((v4sf *)&outPtr1[2*dimY + 4]) = _mm_movelh_ps(temp11, temp13);
      *((v4sf *)&outPtr1[3*dimY + 0]) = _mm_movehl_ps(temp01, temp03);
      *((v4sf *)&outPtr1[3*dimY + 4]) = _mm_movehl_ps(temp11, temp13);
      *((v4sf *)&outPtr1[4*dimY + 0]) = _mm_movelh_ps(temp04, temp06);
      *((v4sf *)&outPtr1[4*dimY + 4]) = _mm_movelh_ps(temp14, temp16);
      *((v4sf *)&outPtr1[5*dimY + 0]) = _mm_movehl_ps(temp04, temp06);
      *((v4sf *)&outPtr1[5*dimY + 4]) = _mm_movehl_ps(temp14, temp16);
      *((v4sf *)&outPtr1[6*dimY + 0]) = _mm_movelh_ps(temp05, temp07);
      *((v4sf *)&outPtr1[6*dimY + 4]) = _mm_movelh_ps(temp15, temp17);
      *((v4sf *)&outPtr1[7*dimY + 0]) = _mm_movehl_ps(temp05, temp07);
      *((v4sf *)&outPtr1[7*dimY + 4]) = _mm_movehl_ps(temp15, temp17);

      float * __restrict const outPtr2 = outPointer + dimY * (countX + blockSize) + countY;
      *((v4sf *)&outPtr2[0*dimY + 0]) = _mm_movelh_ps(temp20, temp22);
      *((v4sf *)&outPtr2[0*dimY + 4]) = _mm_movelh_ps(temp30, temp32);
      *((v4sf *)&outPtr2[1*dimY + 0]) = _mm_movehl_ps(temp20, temp22);
      *((v4sf *)&outPtr2[1*dimY + 4]) = _mm_movehl_ps(temp30, temp32);
      *((v4sf *)&outPtr2[2*dimY + 0]) = _mm_movelh_ps(temp21, temp23);
      *((v4sf *)&outPtr2[2*dimY + 4]) = _mm_movelh_ps(temp31, temp33);
      *((v4sf *)&outPtr2[3*dimY + 0]) = _mm_movehl_ps(temp21, temp23);
      *((v4sf *)&outPtr2[3*dimY + 4]) = _mm_movehl_ps(temp31, temp33);
      *((v4sf *)&outPtr2[4*dimY + 0]) = _mm_movelh_ps(temp24, temp26);
      *((v4sf *)&outPtr2[4*dimY + 4]) = _mm_movelh_ps(temp34, temp36);
      *((v4sf *)&outPtr2[5*dimY + 0]) = _mm_movehl_ps(temp24, temp26);
      *((v4sf *)&outPtr2[5*dimY + 4]) = _mm_movehl_ps(temp34, temp36);
      *((v4sf *)&outPtr2[6*dimY + 0]) = _mm_movelh_ps(temp25, temp27);
      *((v4sf *)&outPtr2[6*dimY + 4]) = _mm_movelh_ps(temp35, temp37);
      *((v4sf *)&outPtr2[7*dimY + 0]) = _mm_movehl_ps(temp25, temp27);
      *((v4sf *)&outPtr2[7*dimY + 4]) = _mm_movehl_ps(temp35, temp37);
    }
    for(size_t countX = roundedDimX; countX < dimX; countX++) {
      outPointer[dimY * countX + countY] = inBase[countX];
    }
  }
  for(size_t countY = roundedDimY; countY < dimY; countY++) {
    const float * __restrict const inBase = inPointer + countY * dimX;
    for(size_t countX = 0; countX < dimX; countX++) {
      outPointer[dimY * countX + countY] = inBase[countX];
    }
  }
}
#endif

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



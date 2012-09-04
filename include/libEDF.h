#ifndef libEDF_H
#define libEDF_H

/* ********* Types Definitions for Programmer's Use ****************************
 */
/* Let's take care of MS VisualStudio before 2010 that don't ship with stdint */
#if defined(_MSC_VER) && _MSC_VER < 1600
  typedef __int8 int8_t;
  typedef unsigned __int8 uint8_t;
  typedef __int32 int32_t;
  typedef unsigned __int32 uint32_t;
  typedef __int64 int64_t;
  typedef unsigned __int64 uint64_t;
#else
# include <stdint.h>
#endif

#define ROUND_DOWN(x, s) ((x) & ~((s)-1))

#include <vector>
#include <string>

using namespace std;

enum EDF_DataType {
  EDF_NO_TYPE = 0,
  EDF_INT_08_S,
  EDF_INT_08_U,
  EDF_INT_16_S,
  EDF_INT_16_U,
  EDF_INT_32_S,
  EDF_INT_32_U,
  EDF_INT_64_S,
  EDF_INT_64_U,
  EDF_FLOAT_32,
  EDF_FLOAT_64,
};

template<typename Type>
struct EDF_Field {
  string name;
  Type content;

  EDF_Field(const string & _name, const Type & _content)
    : name(_name), content(_content)
  { }
  EDF_Field(const EDF_Field<Type> & old)
    : name(old.name), content(old.content)
  { }
};
//typedef EDF_Field<int32_t> EDF_IntField;
//typedef EDF_Field<string> EDF_StringField;
//typedef EDF_Field<double> EDF_FloatField;

template<typename Type>
class EDF_TypeFields : public vector<EDF_Field<Type> > {
//  typename std::vector<EDF_Field<Type> >::value_type value_type;
public:
  void push_back(const string & name, const Type & content) {
    const EDF_Field<Type> field(name, content);
    vector<EDF_Field<Type> >::push_back(field);
  }
};

struct EDF_Fields {
  EDF_TypeFields<int32_t> intFields;
  EDF_TypeFields<double> floatFields;
  EDF_TypeFields<string> stringFields;

  size_t headerLength;
};

struct EDF_Data {
  void * data;
  size_t totPixels;
  vector<size_t> dimensions;
  EDF_DataType dataType;

private:
  void *(* allocator)(size_t);
  void (* deallocator)(void *);

public:
  EDF_Data();
  ~EDF_Data();

  void setAllocator(void *(* _alloc)(size_t)) { this->allocator = _alloc; }
  void setDeallocator(void (* _dealloc)(void *)) { this->deallocator = _dealloc; }

  bool alloc() { data = allocator(totPixels * getPixelSize()); return data; }
  void dealloc() { if (data) { deallocator(data); data = NULL; } }

  void transpose();

  const size_t getPixelSize() const;

  template<typename Type>
  const Type getPixel(size_t pos) const
  {
    switch (this->dataType) {
      case EDF_INT_08_S: {
        const int8_t * pixels = (const int8_t *) this->data;
        return pixels[pos];
      }
      case EDF_INT_08_U: {
        const uint8_t * pixels = (const uint8_t *) this->data;
        return pixels[pos];
      }
      case EDF_INT_16_S: {
        const int16_t * pixels = (const int16_t *) this->data;
        return pixels[pos];
      }
      case EDF_INT_16_U: {
        const uint16_t * pixels = (const uint16_t *) this->data;
        return pixels[pos];
      }
      case EDF_INT_32_S: {
        const int32_t * pixels = (const int32_t *) this->data;
        return pixels[pos];
      }
      case EDF_INT_32_U: {
        const uint32_t * pixels = (const uint32_t *) this->data;
        return pixels[pos];
      }
      case EDF_FLOAT_32: {
        const float * pixels = (const float *) this->data;
        return pixels[pos];
      }
      case EDF_INT_64_S: {
        const int64_t * pixels = (const int64_t *) this->data;
        return pixels[pos];
      }
      case EDF_INT_64_U: {
        const uint64_t * pixels = (const uint64_t *) this->data;
        return pixels[pos];
      }
      case EDF_FLOAT_64: {
        const double * pixels = (const double *) this->data;
        return pixels[pos];
      }
      default:
        return 0;
    }
  }

private:
  template<typename Type>
  void _transpose(Type * basePointer, const size_t & dimX, const size_t & dimY);
};

class EDF_File {
protected:
  EDF_Fields fields;
  EDF_Data data;
public:
  const EDF_Fields & getFields() const { return fields; }
  const EDF_Data & getData() const { return data; }

  EDF_Fields & getFields() { return fields; }
  EDF_Data & getData() { return data; }

  bool parse_file(const char *);
  bool load_file(const char *, const bool transpose = false);
};

#endif

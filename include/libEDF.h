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

#include <vector>
#include <string>

using namespace std;

enum EDF_DataType {
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
  size_t numDimensions;
  size_t * dimensions;
  EDF_DataType dataType;
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
};

extern bool EDF_parse_file(EDF_File &, const char *);

#endif

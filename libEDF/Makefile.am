# Trigger header generation
AM_YFLAGS = -d

#######################################
# The list of libraries we are building seperated by spaces.
# The 'lib_' indicates that these build products will be installed
# in the $(libdir) directory. For example /usr/lib
lib_LTLIBRARIES = libEDF.la

#######################################
# Build information for each library

# Sources for libEDF
libEDF_la_SOURCES = edf_header_parser.ypp \
		  							edf_header_lexer.lpp \
		  							libEDF.cpp

# Linker options libTestProgram
libEDF_la_LDFLAGS = 

# Compiler options. Here we are adding the include directory
# to be searched for headers included in the source code.
libEDF_la_CPPFLAGS = -I$(top_srcdir)/include


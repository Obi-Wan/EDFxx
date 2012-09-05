dnl Process this file with autoconf to produce a configure script.

AC_PREREQ(2.59)
AC_INIT([EDFxx], [0.1], [ben.vighy@gmail.com])

AC_PROG_CXX
AM_PROG_LEX
AC_PROG_YACC

AC_CANONICAL_SYSTEM

dnl Initialize automake
AM_INIT_AUTOMAKE([-Wall dist-bzip2])

dnl this allows us specify individual liking flags for each target
AM_PROG_CC_C_O 

dnl Initialize Libtool
LT_INIT

dnl Check if Libtool is present
dnl Libtool is used for building share libraries 
AC_PROG_LIBTOOL

AC_CONFIG_FILES(Makefile
                exampleProgram/Makefile
                libEDF/Makefile
                include/Makefile)

CFLAGS="${CFLAGS=}"
CXXFLAGS="${CXXFLAGS=}"
AC_SUBST([CFLAGS],["${CFLAGS} -Wall"])
AC_SUBST([CXXFLAGS],["${CXXFLAGS} -Wall"])

AC_OUTPUT

dnl
dnl Output some configuration info for the user
dnl
echo ""
echo "  Configuration summary:"
echo ""
echo "        prefix:          $prefix"
echo "        exec_prefix:     $exec_prefix"
echo "        libdir:          $libdir"
echo "        includedir:      $includedir"

dnl
dnl Output the features included
dnl
echo ""
echo "        C Compiler:      $CC"
echo "        C++ Compiler:    $CXX"
echo "        CFLAGS:          $CFLAGS"
echo "        CXXFLAGS:        $CXXFLAGS"

echo ""
echo ""


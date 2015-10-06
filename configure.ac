# Ensure that a recent enough version of Autoconf is being used. 
# If the version of Autoconf being used to create configure is earlier than
# version, print an error message to the standard error output and exit with
# failure (exit status is 63).
 AC_PREREQ([2.69])

# Initialize AC: ACINIT( package_name, version, bug_report_address)
AC_INIT([Ursus], [0.0.1], [jwkim.unist@gmail.com])
AC_CONFIG_HEADERS([config.h:config-h.in])
AC_CONFIG_AUX_DIR([config])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([1.0 foreign subdir-objects])

# Enable static library
AC_PROG_LIBTOOL

AC_LANG([C++])
AC_PROG_CXX([g++])

######################################################################
# GCC COLOR
######################################################################

# FIXME
#AX_CHECK_COMPILE_FLAG([-fdiagnostics-color=auto], [CXXFLAGS="$CXXFLAGS -fdiagnostics-color=auto"])

######################################################################
# VALGRIND 
######################################################################

# TODO check available cuda device.
AX_VALGRIND_CHECK

######################################################################
# PARALLEL MAKE 
######################################################################

NPROCS=1
if [[ "$build_os" == "linux-gnu" ]]
then
  NPROCS=$(grep -c ^processor /proc/cpuinfo)
elif [[ "$build_os" == "darwin" ]]
then
  NPROCS=$(sysctl -n hw.physicalcpu)
fi
  
#NPROCS=`expr $NPROCS / 2`
MAKEFLAGS="-j$NPROCS"

AC_CONFIG_FILES([Makefile
                 src/Makefile
                 tests/Makefile ])

AC_OUTPUT

echo \
"---------------------------------------------------------------------

${PACKAGE_NAME} 

Version        : ${PACKAGE_VERSION}
OS             : ${build_os}
Prefix         : ${prefix}
Compiler       : ${CC}
CPP Flags      : ${CPPFLAGS} ${DEBUG_CPPFLAGS} 
C++ Flags      : ${CXXFLAGS} ${DEBUG_CXXFLAGS}
Make Flags     : ${MAKEFLAGS}
Libraries      : ${LIBS} $TBB_LIBS

---------------------------------------------------------------------"

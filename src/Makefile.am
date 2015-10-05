## Makefile.am -- Process this file with automake to produce Makefile.in

######################################################################
# COMMON
######################################################################

## Common flags
include $(top_srcdir)/common.am

######################################################################
# SUBDIRS
######################################################################

include $(srcdir)/mapper/Makefile.am

bin_PROGRAMS = ursus

lib_LTLIBRARIES = libursus.la 

######################################################################
# LIBRARY
######################################################################

libursus_la_SOURCES = \
		      $(common_FILES) \
		      $(mapper_FILES) 
										
######################################################################
# URSUS
######################################################################

ursus_SOURCES = main/main.cpp
ursus_LDFLAGS =
ursus_CPPFLAGS = -I. -I$(top_srcdir)/src -I.. 

ursus_LDADD = libursus.la

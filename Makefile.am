# Makefile.am -- Process this file with automake to produce Makefile.in
AUTOMAKE_OPTIONS = foreign
ACLOCAL_AMFLAGS	 = -I m4

# subdirectories of the project 
SUBDIRS = \
          src \
          tests 

clean-local:
	find . -type f -name "*.gcno" -delete; find . -type f -name "*.gcda" -delete

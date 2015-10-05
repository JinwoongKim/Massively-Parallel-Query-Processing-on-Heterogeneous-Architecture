You can clone this template project to jump start your project with a automake/autoconf
 based build system.
It has the basic ingredients, so all you need is to replace the libraries and main file
 with your own code
while modifying the Makefile.am's and configure.ac's with the names of your files.

In order to make everything work, once you clone this directory, go the root of project
, and type:
> aclocal

> autoconf

> automake --add-missing

This should create all "configure" script along with all the Makefile's. Now, you can t
ype
> ./configure

> make

And your project should compile. Now you are all set with a lean template to build upon
.

CC=g++
CFLAGS=-Wall -std=c++11 -fPIC
INCLUDES=-I../eigen
SOURCES=random_matrix_factory.cc random_orthonormal_basis.cc random_svd.cc

all: $(SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) $(SOURCES)

random_svd.o: random_svd.cc random.

random_orthonormal_basis.o: random_orthonormal_basis.cc random_orthonormal_basis.h random_matrix_factory.o
	$(CC) $(CFLAGS) random_matrix_factory.o random_orthonormal_basis.cc

random_matrix_factory.o: random_matrix_factory.cc random_matrix_factory.h
	$(CC) $(CFLAGS) random_matrix_factory.cc -o random_matrix_factory.o

clean:
	rm *o

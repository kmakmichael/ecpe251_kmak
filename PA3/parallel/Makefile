# The variable CC specifies which compiler will be used.
# (because different unix systems may use different compilers)
CC=mpicxx -fopenmp

# The variable CFLAGS specifies compiler options
#   -c :    Only compile (don't link)
#   -Wall:  Enable all warnings about lazy / dangerous C programming 
#  You can add additional options on this same line..
#  WARNING: NEVER REMOVE THE -c FLAG, it is essential to proper operation
CFLAGS=-c -Wall -O3

# All of the .h header files to use as dependencies
HEADERS=image_template.h sort.h

# All of the object files to produce as intermediary work
OBJECTS=main.o

# The final program to build
EXECUTABLE=canny

# --------------------------------------------

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(EXECUTABLE)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf *.o $(EXECUTABLE)

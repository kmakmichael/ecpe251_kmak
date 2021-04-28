#From Cody Balos
CC=gcc
CCFLAGS=-Wall -std=c99
LDFLAGS=-lm
SOURCES=$(wildcard *.c)
OBJECTS=$(SOURCES:.c=.o)
TARGET=snn_morris

a2cloud: release

all: debug

debug: CCFLAGS += -DDEBUG -g
debug: $(TARGET)

release: CCFLAGS += -O3
release: $(TARGET)

benchmode: CCFLAGS += -O3 -DBENCHMARKMODE
benchmode: $(TARGET)

$(TARGET): $(OBJECTS) $(CXXOBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c %.h
	$(CC) $(CCFLAGS) -c $<

%.o: %.c
	$(CC) $(CCFLAGS) -c $<

clean:
	rm -f *.pgm *.o $(TARGET) *.e* *.o*

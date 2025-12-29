CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm

# Targets
all: example

# Build the example program
example: example.o llm.o
	$(CC) $(CFLAGS) -o example example.o llm.o $(LDFLAGS)

# Object files
example.o: example.c llm.h
	$(CC) $(CFLAGS) -c example.c

llm.o: llm.c llm.h
	$(CC) $(CFLAGS) -c llm.c

# Clean build artifacts
clean:
	rm -f *.o example

# Run the example
run: example
	./example

.PHONY: all clean run

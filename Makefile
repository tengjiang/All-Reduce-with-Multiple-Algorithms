CC = mpicc -lm
OBJ = *.o
EXE = allreduce_tree
FLAGS = -O3 -march=native -g -Wall
all:${EXE}

allreduce_tree: allreduce_tree.c
	$(CC) -o $@ $^ $(FLAGS) 

clean:
	rm -f $(OBJ) $(EXE)

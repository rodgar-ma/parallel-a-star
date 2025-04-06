CC = mpicc
DEF = -DTIME -DDEBUG
CFLAGS = -O3 -std=gnu99
LD_FLAGS = -fopenmp -lm

OMPI_DIR = $(MPI_HOME)
OMPI_INC = -I$(OMPI_DIR)/include

N =  100
NP = 1
H1 = "marte"
H2 = "marte"
FILE = 
EXEC_DIR = exec

$(FILE): $(FILE).c
	$(CC) $(CFLAGS) $(OMPI_INC) $(FILE).c -o $(EXEC_DIR)/$(FILE) $(DEF)

test_mpi:
	mpirun -np $(NP) -host $(H1) ./$(EXEC_DIR)/$(FILE) $(N) : -np $(NP) -host $(H2) ./$(EXEC_DIR)/$(FILE) $(N)

clean:
	rm -f $(EXEC_DIR)/*

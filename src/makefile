windows: main.c astar.c astar.h spastar.c spastar.h hdastar.c hdastar.h MapUtils.c MapUtils.h heap.c heap.h buffer.c buffer.h
	gcc -Wall -g -fopenmp main.c astar.c spastar.c hdastar.c MapUtils.c heap.c buffer.c -lm -o main.exe

linux: main.c astar.c astar.h spastar.c spastar.h hdastar.c hdastar.h MapUtils.c MapUtils.h heap.c heap.h buffer.c buffer.h
	gcc -Wall -g -fopenmp main.c astar.c spastar.c hdastar.c MapUtils.c heap.c buffer.c -lm -o main

clean:
	rm main main.exe

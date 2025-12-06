all:
	nvcc vecadd_cpu.cpp -o vecadd_cpu

run: all
	./vecadd_cpu

clean:
	rm -f vecadd_cpu


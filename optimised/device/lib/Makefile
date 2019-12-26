library.aoclib: library.aoco
	fpga_libtool --target aoc --create library.aoclib library.aoco

library.aoco: library.cpp
	fpga_crossgen --target aoc -I /home/cc/intelFPGA_pro/19.3/hls/include library.cpp 

clean:
	rm *.aoco *.i *.aoclib



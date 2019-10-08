CPP = g++
SRC = src/sim/sim_driver.cc \
			src/runtime.cc \
			src/vta_api.cc
CPPFLAGS = -std=c++11 -g
OBJ = $(patsubst %.cc, %.o, $(SRC))
OBJ2 = $(notdir $(OBJ))
LINK_FLAGS = -shared -fPIC

all: vta test_add test_mm

vta: $(OBJ) 
	@echo $(OBJ)
	@echo $(OBJ2)
	$(CPP) $(LINK_FLAGS) -o libVTA.so $(OBJ2) 

%.o: %.cc
	$(CPP) $(CPPFLAGS) -c $< -o $@
	mv $@ ./

test_add:
	$(CPP) $(CPPFLAGS) -L./ tests/test_add.cpp -o tests/test_add -lVTA 

test_mm:
	$(CPP) $(CPPFLAGS) -L./ tests/test_mm.cpp -o tests/test_mm -lVTA 

test_mm2:
	$(CPP) $(CPPFLAGS) -L./ tests/test_mm2.cpp -o tests/test_mm2 -lVTA 

.PHONY: all clean test_add test_mm
clean:
	rm -rf *.o *.so tests/test_add tests/test_mm tests/test_mm2 *.dSYM

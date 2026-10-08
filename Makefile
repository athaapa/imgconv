CXX      = clang++ 
CXXFLAGS = -O3 -g -std=c++20 -ffast-math
TARGET   = imgconv

ifeq ($(shell uname -s),Darwin)
LDLIBS += -framework Accelerate
endif

all:
	$(CXX) $(CXXFLAGS) src/main.cpp -o $(TARGET) $(LDLIBS)

clean:
	rm -f $(TARGET)

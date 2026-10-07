CXX      = clang++ 
CXXFLAGS = -O3 -g -std=c++20 -ffast-math
TARGET   = imgconv

all:
	$(CXX) $(CXXFLAGS) src/main.cpp -o $(TARGET)

clean:
	rm -f $(TARGET)

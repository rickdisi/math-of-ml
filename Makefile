CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

# Compile and run a single implementation file.
# Usage: make run FILE=impl/linear_algebra/matrix_multiply.cpp
run:
	$(CXX) $(CXXFLAGS) $(FILE) -o /tmp/mlrun && /tmp/mlrun

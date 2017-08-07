CXX = g++

CXXFLAGS = -g -std=c++11 -I . -lpthread -msse4.1 -lopenblas -D USE_BLAS

OBJ = kws.o net.o fft.o

BIN = tools/net-quantization

TEST = test/fbank-test test/kws-test

all: $(TEST) $(BIN) $(OBJ)

test/%: test/%.cc $(OBJ)
	$(CXX) $< $(OBJ) $(CXXFLAGS) -o $@

tools/%: tools/%.cc $(OBJ)
	$(CXX) $< $(OBJ) $(CXXFLAGS) -o $@

net.o: net.h utils.h
test/fbank-test: fbank.h

.PHONY: clean

clean:
	rm -rf $(OBJ); rm -rf $(TEST)


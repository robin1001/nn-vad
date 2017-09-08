CXX = g++

CXXFLAGS = -g -std=c++11 -I . -lpthread -msse4.1 -lopenblas -D USE_BLAS

OBJ = vad.o net.o feature-pipeline.o fft.o

BIN = tools/net-quantization tools/apply-vad

TEST = test/fbank-test

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


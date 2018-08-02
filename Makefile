CXX = g++

CXXFLAGS = -g -std=c++11 -MMD -I . -lpthread -msse4.1 -lopenblas -D USE_BLAS

OBJ = vad.o net.o feature-pipeline.o fft.o

BIN = tools/net-quantization tools/apply-vad

TEST = test/fbank-test

all: $(TEST) $(BIN) $(OBJ)

test/%: test/%.cc $(OBJ)
	$(CXX) $< $(OBJ) $(CXXFLAGS) -o $@

tools/%: tools/%.cc $(OBJ)
	$(CXX) $< $(OBJ) $(CXXFLAGS) -o $@

test: $(TEST)
	@for x in $(TEST); do \
		printf "Running $$x ..."; \
		./$$x;  \
		if [ $$? -ne 0 ]; then \
			echo "... Fail $$x"; \
		else \
			echo "... Success $$x"; \
		fi \
	done

check:
	for file in *.h *.cc test/*.cc tools/*.cc; do \
		echo $$file; \
        cpplint --filter=-build/header_guard,-readability/check,-build/include_subdir $$file; \
	done

net.o: net.h utils.h
test/fbank-test: fbank.h

.PHONY: clean

clean:
	rm -rf $(OBJ); rm -rf $(TEST); rm -f *.d; rm -f */*.d;

-include *.d
-include */*.d


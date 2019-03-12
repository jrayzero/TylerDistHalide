include ../support/Makefile.inc

W ?=
H ?=

# If HL_TARGET isn't set, use host
HL_TARGET ?= host

all: wavelet

distributed_wavelet: wavelet_distributed.cpp
	$(CXX) $(CXXFLAGS) wavelet_distributed.cpp -g $(LIB_HALIDE) -o distributed_wavelet -lpthread -ldl -lz $(LDFLAGS)

run_distributed_wavelet: distributed_wavelet
	$(MPI_RUN) ./distributed_wavelet $(W) $(H)

clean:
	@rm -rf distributed_wavelet 

# By default, %.generator is produced by building %_generator.cpp
%.generator: %_generator.cpp $(GENERATOR_DEPS)
	@echo Building Generator $(filter %_generator.cpp,$^)
	@mkdir -p $(BUILD_DIR)
	@$(CXX) $(CXXFLAGS) -fno-rtti $(filter-out %.h,$^) -lz -ldl -lpthread -o $@

# By default, %.o/.h are produced by executing %.generator
%.o $(BUILD_DIR)/%.h: $(BUILD_DIR)/%.generator
	@echo Running Generator $<
	@mkdir -p $(BUILD_DIR)
	@$< -g $(notdir $*) -o $(BUILD_DIR) target=$(HL_TARGET)

HL_MODULES = \
	$(BUILD_DIR)/daubechies_x.o \
	$(BUILD_DIR)/haar_x.o \
	$(BUILD_DIR)/inverse_daubechies_x.o \
	$(BUILD_DIR)/inverse_haar_x.o

$(BUILD_DIR)/wavelet.o: wavelet.cpp $(HL_MODULES)
	@$(CXX) $(CXXFLAGS) $(LIBPNG_CXX_FLAGS) -I$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/wavelet: $(BUILD_DIR)/wavelet.o
	@$(CXX) $(CXXFLAGS) $^ $(HL_MODULES) $(PNGFLAGS) -ldl -lpthread -o $@

test: $(BUILD_DIR)/wavelet
	@echo Testing wavelet...
	@$< ../images/gray.png $(BUILD_DIR)

# Don't auto-delete the generators.
.SECONDARY:

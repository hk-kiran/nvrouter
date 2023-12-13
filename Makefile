# Compiler and flags
NVCC := nvcc
CFLAGS := -std=c++11

# Directories
SRC_DIR := src
BIN_DIR := bin

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.cu)

# Target executable
TARGET := $(BIN_DIR)/nvrouter

run: all
	$(BIN_DIR)/nvrouter
	
# Default target
all: $(TARGET)

# Rule to compile and link the target executable
$(TARGET): $(SRCS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CFLAGS) $(SRC_DIR)/main.cu -o $@

# Clean build artifacts
clean:
	rm -rf $(BIN_DIR)


# Compiler and flags
NVCC := nvcc
CFLAGS := -std=c++11

# Directories
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.cu)
OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRCS))

# Target executable
TARGET := $(BIN_DIR)/nvrouter

# Default target
all: $(TARGET)

# Rule to compile object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

# Rule to link the target executable
$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CFLAGS) $^ -o $@

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

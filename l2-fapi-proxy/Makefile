CC = gcc

CFLAGS = -g -O3 -Wall -Werror
CXX = g++ -O3 -std=c++11
CXXFLAGS = -g -O3 -Wall -Wextra -Werror

# CXX = g++  -std=c++11
# CFLAGS = -g -Wall #-Werror
# CXXFLAGS = -g -Wall -Wextra #-Werror


BUILD_DIR = ./build
TARGET_EXEC = $(BUILD_DIR)/proxy

top_srcdir = .

LIB_DIR = .
SRC_DIR = ./src
INC_DIRS = \
  $(SRC_DIR) \
  $(LIB_DIR)/open-nFAPI/nfapi/public_inc \
  $(LIB_DIR)/open-nFAPI/pnf_sim/inc \
  $(LIB_DIR)/open-nFAPI/nfapi/inc \
  $(LIB_DIR)/open-nFAPI/fapi/inc \
  $(LIB_DIR)/open-nFAPI/pnf/public_inc \
  $(LIB_DIR)/open-nFAPI/pnf/inc \
  $(LIB_DIR)/open-nFAPI/common/public_inc \
  $(LIB_DIR)/open-nFAPI/common/inc \
  $(LIB_DIR)/common \
  $(LIB_DIR)/common/utils \
  $(LIB_DIR)/common/utils/LOG \

LIB_SOURCES = \
  $(LIB_DIR)/open-nFAPI/nfapi/src/nfapi_p4.c \
  $(LIB_DIR)/open-nFAPI/nfapi/src/nfapi_p5.c \
  $(LIB_DIR)/open-nFAPI/nfapi/src/nfapi_p7.c \
  $(LIB_DIR)/open-nFAPI/fapi/src/nr_fapi.c \
  $(LIB_DIR)/open-nFAPI/fapi/src/nr_fapi_p5.c \
  $(LIB_DIR)/open-nFAPI/fapi/src/nr_fapi_p7.c \
  $(LIB_DIR)/open-nFAPI/fapi/src/nr_fapi_p7_utils.c \
  $(LIB_DIR)/open-nFAPI/pnf/src/pnf_interface.c \
  $(LIB_DIR)/open-nFAPI/pnf/src/pnf_p7_interface.c \
  $(LIB_DIR)/open-nFAPI/pnf/src/pnf.c \
  $(LIB_DIR)/open-nFAPI/pnf/src/pnf_p7.c \
  $(LIB_DIR)/open-nFAPI/common/src/debug.c \
  $(LIB_DIR)/open-nFAPI/common/src/nfapi.c \
  
SRCS = \
	$(SRC_DIR)/proxy.cc \
	$(SRC_DIR)/nr_proxy.cc \
	$(SRC_DIR)/nfapiutils.c \
	$(SRC_DIR)/nfapi_pnf.c \
	$(SRC_DIR)/queue.c \
	$(LIB_SOURCES)

OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS = $(OBJS:.o=.d)

INC_FLAGS := $(addprefix -I,$(INC_DIRS))

LDFLAGS = -pthread -lpthread -lz -lsctp -rdynamic 

MKDIR_P ?= mkdir -p

$(TARGET_EXEC): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

# c source
$(BUILD_DIR)/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(INC_FLAGS) $(CFLAGS) -c $< -o $@

# c++ source
$(BUILD_DIR)/%.cc.o: %.cc
	$(MKDIR_P) $(dir $@)
	$(CXX) $(INC_FLAGS) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)

# Run static analysis checks on Python scripts
.PHONY: mypy
mypy:
	python3 -m mypy proxy_testscript.py

PROTO_DIR = protos
OUTPUT_DIR = generated
PROTO_FILES = $(wildcard $(PROTO_DIR)/*.proto)
PB2_FILES = $(PROTO_FILES:$(PROTO_DIR)/%.proto=$(OUTPUT_DIR)/%_pb2.py)

.PHONY: all clean proto-check proto-build

all: proto-build

# Check protobuf installation
proto-check:
	@echo "Checking protobuf installation..."
	@protoc --version || (echo "protoc not found" && exit 1)
	@python -c "import google.protobuf; print('Python protobuf:', google.protobuf.__version__)" || (echo "Python protobuf not found" && exit 1)

# Build all proto files
proto-build: proto-check $(OUTPUT_DIR)
	protoc --proto_path=$(PROTO_DIR) \
	       --python_out=$(OUTPUT_DIR) \
	       --pyi_out=$(OUTPUT_DIR) \
	       $(PROTO_FILES)
	touch $(OUTPUT_DIR)/__init__.py

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Clean generated files
clean:
	rm -rf $(OUTPUT_DIR)

# Force rebuild
rebuild: clean proto-build

# Test imports
test:
	python -c "import sys; sys.path.append('$(OUTPUT_DIR)'); \
	          import $(shell basename $(firstword $(PB2_FILES)) _pb2.py)_pb2; \
	          print('Import successful')"


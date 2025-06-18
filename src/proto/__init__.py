"""Proto package for FastEmbed gRPC service."""

import os
import warnings

# Check if proto files exist
proto_files = ['embed_pb2.py', 'embed_pb2_grpc.py']
proto_dir = os.path.dirname(__file__)

for proto_file in proto_files:
    if not os.path.exists(os.path.join(proto_dir, proto_file)):
        warnings.warn("Proto files not found. Please ensure protobuf files are generated before importing.")
        break

# Only import if files exist
if all(os.path.exists(os.path.join(proto_dir, f)) for f in proto_files):
    from .embed_pb2 import *
    from .embed_pb2_grpc import *

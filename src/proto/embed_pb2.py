# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: embed.proto
# Protobuf Python Version: 6.31.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    6,
    31,
    0,
    '',
    'embed.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0b\x65mbed.proto\x12\x05proto\"5\n\x10\x45mbeddingRequest\x12\r\n\x05texts\x18\x01 \x03(\t\x12\x12\n\nmodel_name\x18\x02 \x01(\t\"9\n\x11\x45mbeddingResponse\x12$\n\nembeddings\x18\x01 \x03(\x0b\x32\x10.proto.Embedding\".\n\tEmbedding\x12\x0e\n\x06vector\x18\x01 \x03(\x02\x12\x11\n\tdimension\x18\x02 \x01(\x05\"\x94\x01\n\x10LoadModelRequest\x12\x12\n\nmodel_name\x18\x01 \x01(\t\x12\x15\n\x08use_cuda\x18\x02 \x01(\x08H\x00\x88\x01\x01\x12\x17\n\nmax_length\x18\x03 \x01(\x05H\x01\x88\x01\x01\x12\x14\n\x07threads\x18\x04 \x01(\x05H\x02\x88\x01\x01\x42\x0b\n\t_use_cudaB\r\n\x0b_max_lengthB\n\n\x08_threads\"]\n\x11LoadModelResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t\x12\x17\n\nmodel_info\x18\x03 \x01(\tH\x00\x88\x01\x01\x42\r\n\x0b_model_info\"\xd7\x01\n\x0c\x46ileMetadata\x12\x10\n\x08\x66ilename\x18\x01 \x01(\t\x12\x14\n\x0c\x63ontent_type\x18\x02 \x01(\t\x12\x11\n\tfile_size\x18\x03 \x01(\x03\x12@\n\x0f\x63ustom_metadata\x18\x04 \x03(\x0b\x32\'.proto.FileMetadata.CustomMetadataEntry\x12\x13\n\x0b\x64ocument_id\x18\x05 \x01(\t\x1a\x35\n\x13\x43ustomMetadataEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"p\n\x11ProcessingOptions\x12\x12\n\nchunk_size\x18\x01 \x01(\x05\x12\x15\n\rchunk_overlap\x18\x02 \x01(\x05\x12\x17\n\x0fstore_in_qdrant\x18\x03 \x01(\x08\x12\x17\n\x0f\x63ollection_name\x18\x04 \x01(\t\"\x9e\x01\n\x11\x46ileStreamRequest\x12\'\n\x08metadata\x18\x01 \x01(\x0b\x32\x13.proto.FileMetadataH\x00\x12\x14\n\nchunk_data\x18\x02 \x01(\x0cH\x00\x12+\n\x07options\x18\x03 \x01(\x0b\x32\x18.proto.ProcessingOptionsH\x00\x12\x12\n\nmodel_name\x18\x04 \x01(\tB\t\n\x07\x63ontent\"\xc0\x01\n\x12\x46ileStreamResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t\x12\x18\n\x10\x63hunks_processed\x18\x03 \x01(\x05\x12\x1a\n\x12\x65mbeddings_created\x18\x04 \x01(\x05\x12\x15\n\rpoints_stored\x18\x05 \x01(\x05\x12\x11\n\tchunk_ids\x18\x06 \x03(\t\x12\x18\n\x0b\x64ocument_id\x18\x07 \x01(\tH\x00\x88\x01\x01\x42\x0e\n\x0c_document_id\"\x0f\n\rStatusRequest\"\x80\x02\n\x0eStatusResponse\x12\x16\n\x0eserver_version\x18\x01 \x01(\t\x12\x15\n\rcurrent_model\x18\x02 \x01(\t\x12\x16\n\x0e\x63uda_available\x18\x03 \x01(\x08\x12\x18\n\x10qdrant_connected\x18\x04 \x01(\x08\x12?\n\rconfiguration\x18\x05 \x03(\x0b\x32(.proto.StatusResponse.ConfigurationEntry\x12\x16\n\x0euptime_seconds\x18\x06 \x01(\x03\x1a\x34\n\x12\x43onfigurationEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\x13\n\x11ListModelsRequest\"\x9a\x01\n\tModelInfo\x12\x12\n\nmodel_name\x18\x01 \x01(\t\x12\x12\n\ndimensions\x18\x02 \x01(\x05\x12\x12\n\nmax_length\x18\x03 \x01(\x05\x12\x1a\n\x12\x64\x65\x66\x61ult_chunk_size\x18\x04 \x01(\x05\x12\x0f\n\x07size_gb\x18\x05 \x01(\x02\x12\x0f\n\x07license\x18\x06 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x07 \x01(\t\"6\n\x12ListModelsResponse\x12 \n\x06models\x18\x01 \x03(\x0b\x32\x10.proto.ModelInfo\"I\n\x18UpdateModelConfigRequest\x12\x12\n\nmodel_name\x18\x01 \x01(\t\x12\x19\n\x11qdrant_collection\x18\x02 \x01(\t\"\x99\x01\n\x19UpdateModelConfigResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t\x12\x14\n\x0c\x61\x63tive_model\x18\x03 \x01(\t\x12\x12\n\nchunk_size\x18\x04 \x01(\x05\x12\x15\n\rchunk_overlap\x18\x05 \x01(\x05\x12\x19\n\x11qdrant_collection\x18\x06 \x01(\t2\xc3\x03\n\x10\x45mbeddingService\x12\x44\n\rGetEmbeddings\x12\x17.proto.EmbeddingRequest\x1a\x18.proto.EmbeddingResponse\"\x00\x12@\n\tLoadModel\x12\x17.proto.LoadModelRequest\x1a\x18.proto.LoadModelResponse\"\x00\x12L\n\x11ProcessFileStream\x12\x18.proto.FileStreamRequest\x1a\x19.proto.FileStreamResponse\"\x00(\x01\x12:\n\tGetStatus\x12\x14.proto.StatusRequest\x1a\x15.proto.StatusResponse\"\x00\x12\x43\n\nListModels\x12\x18.proto.ListModelsRequest\x1a\x19.proto.ListModelsResponse\"\x00\x12X\n\x11UpdateModelConfig\x12\x1f.proto.UpdateModelConfigRequest\x1a .proto.UpdateModelConfigResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'embed_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_FILEMETADATA_CUSTOMMETADATAENTRY']._loaded_options = None
  _globals['_FILEMETADATA_CUSTOMMETADATAENTRY']._serialized_options = b'8\001'
  _globals['_STATUSRESPONSE_CONFIGURATIONENTRY']._loaded_options = None
  _globals['_STATUSRESPONSE_CONFIGURATIONENTRY']._serialized_options = b'8\001'
  _globals['_EMBEDDINGREQUEST']._serialized_start=22
  _globals['_EMBEDDINGREQUEST']._serialized_end=75
  _globals['_EMBEDDINGRESPONSE']._serialized_start=77
  _globals['_EMBEDDINGRESPONSE']._serialized_end=134
  _globals['_EMBEDDING']._serialized_start=136
  _globals['_EMBEDDING']._serialized_end=182
  _globals['_LOADMODELREQUEST']._serialized_start=185
  _globals['_LOADMODELREQUEST']._serialized_end=333
  _globals['_LOADMODELRESPONSE']._serialized_start=335
  _globals['_LOADMODELRESPONSE']._serialized_end=428
  _globals['_FILEMETADATA']._serialized_start=431
  _globals['_FILEMETADATA']._serialized_end=646
  _globals['_FILEMETADATA_CUSTOMMETADATAENTRY']._serialized_start=593
  _globals['_FILEMETADATA_CUSTOMMETADATAENTRY']._serialized_end=646
  _globals['_PROCESSINGOPTIONS']._serialized_start=648
  _globals['_PROCESSINGOPTIONS']._serialized_end=760
  _globals['_FILESTREAMREQUEST']._serialized_start=763
  _globals['_FILESTREAMREQUEST']._serialized_end=921
  _globals['_FILESTREAMRESPONSE']._serialized_start=924
  _globals['_FILESTREAMRESPONSE']._serialized_end=1116
  _globals['_STATUSREQUEST']._serialized_start=1118
  _globals['_STATUSREQUEST']._serialized_end=1133
  _globals['_STATUSRESPONSE']._serialized_start=1136
  _globals['_STATUSRESPONSE']._serialized_end=1392
  _globals['_STATUSRESPONSE_CONFIGURATIONENTRY']._serialized_start=1340
  _globals['_STATUSRESPONSE_CONFIGURATIONENTRY']._serialized_end=1392
  _globals['_LISTMODELSREQUEST']._serialized_start=1394
  _globals['_LISTMODELSREQUEST']._serialized_end=1413
  _globals['_MODELINFO']._serialized_start=1416
  _globals['_MODELINFO']._serialized_end=1570
  _globals['_LISTMODELSRESPONSE']._serialized_start=1572
  _globals['_LISTMODELSRESPONSE']._serialized_end=1626
  _globals['_UPDATEMODELCONFIGREQUEST']._serialized_start=1628
  _globals['_UPDATEMODELCONFIGREQUEST']._serialized_end=1701
  _globals['_UPDATEMODELCONFIGRESPONSE']._serialized_start=1704
  _globals['_UPDATEMODELCONFIGRESPONSE']._serialized_end=1857
  _globals['_EMBEDDINGSERVICE']._serialized_start=1860
  _globals['_EMBEDDINGSERVICE']._serialized_end=2311
# @@protoc_insertion_point(module_scope)

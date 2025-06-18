# FastEmbed Service Sessionization: Architecture Overview

Based on your current implementation, here's my analysis and recommendations for making FastEmbed session-aware to serve multiple clients with different configurations.

## Current State Analysis

From your existing setup, the FastEmbed service is currently:
- Single-model focused (default: BAAI/bge-large-en-v1.5)
- Single Qdrant collection target
- Statically configured via environment variables
- Optimized for one configuration per service instance

## Sessionization Strategy Options

### Option 1: **Multi-Tenant Single Instance** (Recommended)
```
┌─────────────────────────────────────────┐
│            FastEmbed Service            │
├─────────────────────────────────────────┤
│  Session Manager                        │
│  ├── Session ID → Model Config          │
│  ├── Session ID → Qdrant Config         │
│  └── Session ID → Client State          │
├─────────────────────────────────────────┤
│  Model Pool                             │
│  ├── BAAI/bge-large-en-v1.5 (1024d)    │
│  ├── BAAI/bge-base-en-v1.5 (768d)      │
│  └── sentence-transformers/...         │
├─────────────────────────────────────────┤
│  Qdrant Connection Pool                 │
│  ├── Collection A (Client 1)           │
│  ├── Collection B (Client 2)           │
│  └── Collection C (Client 3)           │
└─────────────────────────────────────────┘
```

### Option 2: **Service-Per-Configuration** (Current Pattern)
Keep your current approach but orchestrate multiple instances with different configs.

## Recommended Architecture

### **Multi-Tenant Approach with Session Management**

#### Session Configuration Structure
```python
@dataclass
class SessionConfig:
    session_id: str
    model_name: str
    qdrant_config: QdrantConfig
    chunk_size: int
    chunk_overlap: int
    metadata_schema: Optional[Dict]
```

#### Key Components to Add

1. **Session Manager**
   - Create/destroy sessions
   - Track active sessions and their configs
   - Session timeout/cleanup

2. **Model Pool Manager**
   - Lazy-load models on first use
   - Share models across sessions when possible
   - Memory management and model eviction

3. **Enhanced gRPC Interface**
   ```protobuf
   service FastEmbedService {
     rpc CreateSession(CreateSessionRequest) returns (SessionResponse);
     rpc ProcessFileStream(stream ProcessFileStreamRequest) returns (ProcessFileStreamResponse);
     rpc GetEmbeddings(GetEmbeddingsRequest) returns (GetEmbeddingsResponse);
     rpc DestroySession(DestroySessionRequest) returns (SessionResponse);
   }
   ```

## Implementation Strategy

### Phase 1: **Direct Session Enhancement**
```python
# Add session_id to all endpoints - no backward compatibility needed
class EnhancedFastEmbedService:
    def __init__(self):
        self.sessions: Dict[str, SessionConfig] = {}
        self.model_pool: Dict[str, FastEmbed] = {}
        self.qdrant_pool: Dict[str, QdrantEmbeddingStore] = {}
    
    async def ProcessFileStream(self, request_iterator, context):
        # Extract session_id from metadata or first message
        session_config = self.sessions.get(session_id)
        model = self._get_or_load_model(session_config.model_name)
        qdrant = self._get_or_create_qdrant_client(session_config.qdrant_config)
        # ... existing logic with session-specific configs
```

### Phase 2: **Resource Optimization**
- Implement model sharing across compatible sessions
- Add connection pooling for Qdrant
- Memory-efficient model management

### Phase 3: **Advanced Features**
- Session persistence across service restarts
- Resource quotas per session
- Metrics and monitoring per session

## Performance Considerations

### **Model Loading Bottlenecks**
- **Problem**: Each model loads ~500MB-2GB into memory
- **Solution**: 
  - Share models across sessions with same model_name
  - Implement LRU eviction for unused models
  - Pre-warm popular models

### **ONNX Runtime Instances**
- **Current**: One ONNXRuntime per model instance
- **Optimized**: Share runtime, separate inference sessions
- **Memory Impact**: Significant reduction when serving multiple sessions

### **Qdrant Connections**
- **Pattern**: Connection per collection, pooled per host
- **Optimization**: Connection multiplexing for same Qdrant instance

## Implementation Approach

### **Direct Enhancement Strategy**
Since this is pre-production, we can implement the full session-aware architecture directly:

1. **Update Protobuf Definitions**
   - Add session management RPCs
   - Add session_id to all existing message types

2. **Implement Session Management**
   - SessionManager class with create/destroy/lookup
   - SessionConfig dataclass with all per-session settings
   - Model and connection pooling

3. **Update gRPC Server**
   - All endpoints require session_id
   - Session validation on each call
   - Resource cleanup on session destroy

4. **Update FastAPI Wrapper**
   - Session creation/management endpoints
   - Session-aware file upload and embedding endpoints

5. **Update Clients**
   - Add session creation to workflow
   - Pass session_id in all subsequent calls

## Resource Requirements

### **Memory Scaling**
- **Current**: ~4GB for single model
- **Multi-model**: 4GB × number of unique models loaded
- **Mitigation**: Model eviction policy, memory limits per session

### **CPU Considerations**
- ONNX Runtime handles concurrent inference well
- Bottleneck likely in text chunking and Qdrant I/O
- Consider async processing queues for heavy loads

## Enhanced Protobuf Schema

```protobuf
// New session management messages
message CreateSessionRequest {
  string session_id = 1;
  string model_name = 2;
  QdrantConfig qdrant_config = 3;
  int32 chunk_size = 4;
  int32 chunk_overlap = 5;
  map<string, string> metadata_schema = 6;
}

message SessionResponse {
  string session_id = 1;
  bool success = 2;
  string message = 3;
}

// Updated existing messages to include session_id
message ProcessFileStreamRequest {
  string session_id = 1;  // NEW
  oneof data {
    FileMetadata metadata = 2;
    bytes chunk = 3;
  }
}

message GetEmbeddingsRequest {
  string session_id = 1;  // NEW
  repeated string texts = 2;
}
```

## Implementation Benefits

✅ **Advantages of Direct Enhancement:**
- Clean architecture from the start
- No legacy code to maintain
- Optimal resource utilization
- Better testing and validation
- Cleaner codebase

✅ **Resource Efficiency:**
- Shared models across compatible sessions
- Connection pooling for Qdrant instances
- Memory-efficient model management
- Better horizontal scaling characteristics

## Recommendation

Implement the full session-aware architecture directly since we're in pre-production. This provides:

1. **Better Resource Utilization** - Share models and connections
2. **Cleaner Architecture** - No backward compatibility constraints
3. **Optimal Performance** - Designed for multi-tenancy from the ground up
4. **Future-Proof Design** - Scales well as usage grows

The session-aware approach aligns well with your current gRPC + FastAPI architecture and provides a robust foundation for multiple clients with different configurations.
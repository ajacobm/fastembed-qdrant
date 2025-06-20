# FastEmbed gRPC Server Initialization Deadlock Analysis

## Problem Summary
The gRPC server initialization is hanging during the model loading phase due to a **deadlock** in the async lock mechanism.

## Root Cause Analysis

### Deadlock Location
The issue occurs in `grpc_server.py` where there's a nested lock acquisition:

1. **Outer Lock**: `initialize()` method acquires `self.model_lock`
2. **Inner Lock**: `load_model()` method also tries to acquire the same `self.model_lock`

### Code Flow Leading to Deadlock

```python
async def initialize(self):
    """Initialize the service including Qdrant connection and model loading, then set ready."""
    async with self.model_lock:  # <-- FIRST LOCK ACQUISITION
        with operation_context("service_initialization") as op:
            # ... Qdrant initialization ...
            
            # Load default model
            if self.config.default_model:
                with operation_context("default_model_loading", model=self.config.default_model) as model_op:
                    success = await self.load_model(  # <-- CALLS load_model()
                        model_name=self.config.default_model,
                        # ... other params
                    )

async def load_model(self, model_name: str, use_cuda: bool = True, max_length: int = 512, threads: int = 8) -> bool:
    """Load or switch the FastEmbed model."""
    async with self.model_lock:  # <-- SECOND LOCK ACQUISITION (DEADLOCK!)
        # ... model loading logic
```

### Why This Happens
- The `initialize()` method holds the lock to prevent concurrent initialization
- It then calls `load_model()` which tries to acquire the same lock
- Since asyncio locks are not reentrant by default, this creates a deadlock
- The thread waits forever for the lock that it already holds

## Evidence from Logs
In the server logs, we can see:
1. Qdrant initialization completes successfully
2. The log shows starting "default_model_loading" operation with `"model": null` 
3. The process hangs indefinitely - no further logs appear
4. When Docker container is stopped, model loading suddenly begins but then terminates

## Solutions

### Option 1: Remove Nested Lock (Recommended)
Create an internal `_load_model_unlocked()` method for internal use:

```python
async def initialize(self):
    """Initialize the service including Qdrant connection and model loading, then set ready."""
    async with self.model_lock:
        # ... initialization logic ...
        if self.config.default_model:
            success = await self._load_model_unlocked(  # No lock here
                model_name=self.config.default_model,
                # ... params
            )

async def _load_model_unlocked(self, model_name: str, use_cuda: bool = True, max_length: int = 512, threads: int = 8) -> bool:
    """Internal model loading without lock acquisition."""
    # Model loading logic without async with self.model_lock

async def load_model(self, model_name: str, use_cuda: bool = True, max_length: int = 512, threads: int = 8) -> bool:
    """Public model loading with lock."""
    async with self.model_lock:
        return await self._load_model_unlocked(model_name, use_cuda, max_length, threads)
```

### Option 2: Use Reentrant Lock
Replace `asyncio.Lock()` with a reentrant lock implementation, though this is more complex.

### Option 3: Separate Initialization Lock
Use separate locks for initialization vs model loading operations.

## Additional Issues Found

### 1. Protobuf Import Issue (Fixed)
The generated `embed_pb2_grpc.py` had incorrect relative imports. Fixed by changing:
```python
import embed_pb2 as embed__pb2  # WRONG
```
to:
```python
from . import embed_pb2 as embed__pb2  # CORRECT
```

### 2. Environment Variable Loading
The server may not be reliably loading environment variables from the `.env` file in all contexts.

## Impact
- Server appears to start but is not functional
- HTTP health checks will fail
- Model loading never completes
- All embedding requests will timeout
- Container startup scripts detect this as a failed initialization

## Next Steps
1. Implement Option 1 (separate unlocked method)
2. Test the fix with both direct Python execution and Docker
3. Verify startup script health checks work correctly
4. Confirm model loading operations complete successfully

## Files to Modify
- `src/grpc_server.py` - Fix the deadlock
- Possibly update startup scripts to handle this edge case better
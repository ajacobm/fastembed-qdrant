# Model Cache Directory

This directory is used to cache FastEmbed model files to avoid re-downloading them on each startup.

## Structure

The cache will contain subdirectories for different models:
- Each model gets its own subdirectory
- Model files are downloaded on first use
- Subsequent uses load from cache for faster startup

## Size Considerations

Model files can be quite large:
- Small models: ~67MB (BAAI/bge-small-en-v1.5)
- Base models: ~210MB (BAAI/bge-base-en-v1.5) 
- Large models: ~1.2GB (BAAI/bge-large-en-v1.5)

## Docker Volume Mounting

When using Docker, mount this directory as a volume to persist models between container restarts:

```yaml
volumes:
  - ./cache:/app/.cache
```

## Cleanup

To free up space, you can safely delete model subdirectories. They will be re-downloaded when needed.
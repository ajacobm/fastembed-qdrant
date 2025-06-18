#!/bin/bash

# Script to source .env file and export variables to current shell
# Usage: source ./load_env.sh

ENV_FILE=".env"

if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: $ENV_FILE not found in current directory"
    return 1
fi

echo "🔧 Loading environment variables from $ENV_FILE..."

# Read .env file and export variables
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # Extract variable name and value
    if [[ "$line" =~ ^[[:space:]]*([^=]+)=(.*)$ ]]; then
        var_name="${BASH_REMATCH[1]}"
        var_value="${BASH_REMATCH[2]}"
        
        # Remove leading/trailing whitespace from name and value
        var_name=$(echo "$var_name" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        var_value=$(echo "$var_value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        
        # Export the variable
        export "$var_name"="$var_value"
        
        # Mask sensitive values in output
        if [[ "$var_name" =~ .*API_KEY.* || "$var_name" =~ .*TOKEN.* || "$var_name" =~ .*SECRET.* ]]; then
            echo "  ✅ $var_name=***MASKED***"
        else
            echo "  ✅ $var_name=$var_value"
        fi
    fi
done < "$ENV_FILE"

echo "🎉 Environment variables loaded successfully!"
echo ""
echo "Key variables:"
echo "  QDRANT_HOST: ${QDRANT_HOST:-not set}"
echo "  QDRANT_PORT: ${QDRANT_PORT:-not set}"
echo "  QDRANT_COLLECTION: ${QDRANT_COLLECTION:-not set}"
echo "  GRPC_PORT: ${GRPC_PORT:-not set}"
echo "  HTTP_PORT: ${HTTP_PORT:-not set}"
echo "  DEFAULT_MODEL: ${DEFAULT_MODEL:-not set}"
echo ""
echo "🚀 You can now run Python scripts that will use these environment variables!"
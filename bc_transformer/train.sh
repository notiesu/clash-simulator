#!/bin/sh

# upload s3 info & then download info

export $(grep -v '^#' .env | xargs)

TRAIN_DIR_PATH=$1
RUN_NAME=$2

# upload train package to runpod
aws s3 cp "$TRAIN_DIR_PATH" "$RUNPOD_NETWORK_BUCKET/$RUN_NAME" \
    --region us-east-2 \
    --recursive || {
    echo "Error downloading output for $RUN_NAME"
    exit 1
}

RESPONSE_AND_CODE=$(curl -sS -w "\nHTTP_CODE:%{http_code}\n" \
  -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "train_dir": "'"$TRAIN_DIR_PATH"'",
      "run_name": "'"$RUN_NAME"'"
    }
  }'
)

echo "$RESPONSE_AND_CODE"
RESPONSE=$(echo "$RESPONSE_AND_CODE" | sed '/HTTP_CODE:/d')
HTTP_CODE=$(echo "$RESPONSE_AND_CODE" | awk -F: '/HTTP_CODE:/ {print $2}')

RUN_ID=$(echo "$RESPONSE" | grep -o '"id":"[^"]*' | cut -d'"' -f4)
STATUS=$(echo "$RESPONSE" | grep -o '"status":"[^"]*' | cut -d'"' -f4)

if [ -z "$RUN_ID" ] || [ "$STATUS" = "null" ]; then
    echo "Error: Failed to start RunPod job"
    exit 1
fi

while [ "$STATUS" != "COMPLETED" ]; do
    sleep 30
    RESPONSE=$(curl -s -X GET "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/status/$RUN_ID" \
        -H "Authorization: Bearer $RUNPOD_API_KEY")
    STATUS=$(echo "$RESPONSE" | grep -o '"status":"[^"]*' | cut -d'"' -f4)
    echo "Current job status: $STATUS"
    if [ "$STATUS" = "FAILED" ]; then
        echo "Error: RunPod job failed"
        exit 1
    fi
done

# download model training outputs
aws s3 sync $RUNPOD_NETWORK_BUCKET/cr-checkpts/$RUN_NAME/output \
            outputs/$RUN_NAME/output \
            --size-only \
            --region us-east-2 \
            --recursive || {
    echo "Error downloading output for $RUN_NAME"
    exit 1
}

echo "Training completed successfully. Outputs downloaded to outputs/ppo_reduced_learning2/output"

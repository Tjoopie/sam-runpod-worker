# SAM RunPod Worker

Segment Anything Model (SAM) serverless worker for RunPod.

## Features

- Click-to-segment: Click on a point to segment the object at that location
- Returns GeoJSON polygon coordinates
- Supports geographic coordinate transformation

## API

### Input

```json
{
  "input": {
    "image_base64": "base64_encoded_image",
    "click_x": 320,
    "click_y": 240,
    "bounds": {
      "west": 28.0,
      "east": 28.1,
      "north": -25.7,
      "south": -25.8
    }
  }
}
```

### Output

```json
{
  "polygon": {
    "type": "Polygon",
    "coordinates": [[[lng, lat], ...]]
  },
  "confidence": 0.95,
  "processing_time_ms": 150
}
```

## Deployment

1. Fork this repo to your GitHub
2. Connect GitHub to RunPod
3. Create new serverless endpoint from this repo
4. Select GPU (RTX 3090 or better recommended)


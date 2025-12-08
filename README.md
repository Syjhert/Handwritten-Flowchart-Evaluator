Flowchart Evaluator – Detection → Connections → Text Extraction → Evaluation

End-to-end pipeline for evaluating handwritten flowchart submissions against assignment instructions.

## Pipeline Overview
1. **Detection**: Faster R-CNN detects flowchart elements (shapes, arrows)
2. **Connection Inference**: Arrow-based + geometric fallback algorithms
3. **Text Extraction**: Gemini 2.5 Flash extracts handwritten text with full context awareness
4. **Evaluation**: LLM scores flowcharts (0-10) against assignment instructions

## Usage

### Folder Structure
```
test_cases/
  images/
    1/              # Activity 1 submissions
      student1.jpg
      student2.png
    2/              # Activity 2 submissions
      ...
  instructions/
    1.md            # Instructions for Activity 1
    2.md            # Instructions for Activity 2
model/
  model_best.pth    # Trained Faster R-CNN checkpoint
```

### Run
```pwsh
uv run flowchart_pipeline.py <activity_number>
```

**Example:**
```pwsh
uv run flowchart_pipeline.py 1    # Evaluate all Activity 1 submissions
```

### Environment Setup
Set `GOOGLE_API_KEY` environment variable for Gemini 2.5 Flash:
```pwsh
$env:GOOGLE_API_KEY = "your-api-key"
```

## Outputs

Results saved to `debug/<activity_number>/<image_basename>/`:

- `bounding_boxes.jpg`: Visualization of detected boxes
- `nodes.initial.json`: Initial detected nodes (scan nodes filtered out)
- `nodes.connected.json`: Nodes with inferred `connecting_to` relationships
- `cropped/<id>.png`: Per-node cropped images (shape-aware padding)
- `ocr_extracted.json`: Nodes with extracted text
- `ocr_extracted.jpg`: Labeled visualization (boxes + node IDs + text)
- `connections.jpg`: Connection arrows visualization
- `nodes.simplified.json`: Simplified nodes (id, label, text, connecting_to)
- `evaluation.json`: Score (0-10) and detailed analysis

## Dependencies

**Required:**
- `torch`, `torchvision` – Detection model
- `Pillow`, `opencv-python`, `numpy` – Image processing
- `google-generativeai` – Gemini 2.5 Flash for text extraction & evaluation
- `python-dotenv` – Environment variable loading

**Optional:**
- None (pytesseract removed; pipeline now requires Gemini)

Install with:
```pwsh
uv sync
```

## Features

### Context-Aware Text Extraction
- Single LLM call extracts text from all nodes using full flowchart image
- Assignment instructions provided as context for better handwriting interpretation
- Handles common OCR errors (division '/' vs '1', multiplication '*' vs '+')
- Cross-validates variable names across connected nodes

### Connection Inference
- Arrow detection with directional analysis (up/down/left/right)
- Geometric fallback using vertical proximity + horizontal overlap
- Supports multiple connections per node (branches)

### Shape-Aware Cropping
- Type-specific padding to exclude borders while preserving text
- Rectangle: 6-8% padding
- Rhombus: 18% horizontal, 12% vertical (accounts for slanted edges)
- Circle/Ellipse: 12% uniform
- Parallelogram: 14-10% padding

### Evaluation Scoring
- Expert flowchart analyzer prompt
- Checks: completeness, logical order, correct flow, syntax accuracy
- Returns 0-10 score with detailed analysis

## Cost Efficiency

**Per flowchart image:**
- Text extraction: 1 call (~4-6K tokens)
- Evaluation: 1 call (~2-3K tokens)
- **Total: ~2 calls ≈ $0.001-0.002** (Gemini 2.5 Flash pricing)

## Notes

- Detection threshold: 0.65 confidence
- "scan" nodes automatically filtered out
- Old per-crop OCR methods preserved in `old_utils.py` for reference
- Requires Google API key; no offline fallback mode

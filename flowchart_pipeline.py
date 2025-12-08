import os
import json
import sys
import argparse
from typing import List, Dict, Tuple
from dotenv import load_dotenv
load_dotenv()

import torch
from torchvision.transforms import functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import numpy as np
import cv2
import warnings

# Optional: Google Gemini client
try:
    import google.generativeai as genai  # type: ignore
    HAS_GEMINI = True
except Exception:
    print("[warn] google-generativeai not installed; Gemini integration disabled.")
    HAS_GEMINI = False

def get_model(num_classes: int) -> torch.nn.Module:
    # Load a pre-trained model for classification and return only the features
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # Replace the classifier with a new one, that has num_classes which is user-defined
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_model_and_map(ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu")
    class_map = state.get("class_map", {}) if isinstance(state, dict) else {}
    id_to_name = {v: k for k, v in class_map.items()} if class_map else {}
    num_classes = (len(class_map) + 1) if class_map else 2
    model = get_model(num_classes)
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) else state)
    return model, id_to_name


# --- Detection and debug drawing ---
def detect_boxes(model, device, image_path: str, score_thresh: float = 0.65) -> List[Tuple[np.ndarray, int, float]]:
    pil_img = Image.open(image_path).convert("RGB")
    tensor_img = F.to_tensor(pil_img).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model([tensor_img])
    out = outputs[0]
    boxes = out["boxes"].cpu().numpy()
    labels = out["labels"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    dets = [
        (boxes[i], int(labels[i]), float(scores[i]))
        for i in range(len(scores))
        if scores[i] >= score_thresh
    ]
    return dets


def save_bboxes_debug_image(image_path: str, dets: List[Tuple[np.ndarray, int, float]], id_to_name: Dict[int, str], out_path: str):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    for box, label, score in dets:
        x1, y1, x2, y2 = [int(v) for v in box]
        name = id_to_name.get(int(label), str(int(label)))
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(bgr, f"{name}:{score:.2f}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, bgr)


# --- Nodes JSON helpers ---
def dets_to_nodes_json(dets: List[Tuple[np.ndarray, int, float]], id_to_name: Dict[int, str]) -> List[Dict]:
    nodes = []
    for idx, (box, label, score) in enumerate(dets):
        x1, y1, x2, y2 = [int(v) for v in box]
        nodes.append({
            "id": idx,
            "label": id_to_name.get(int(label), str(int(label))),
            "score": score,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "type": id_to_name.get(int(label), "unknown")
        })
    return nodes


# --- Gemini integration ---
def call_gemini_add_connections(base_image_path: str, nodes: List[Dict]) -> List[Dict]:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not HAS_GEMINI or not api_key:
        # If Gemini not available, return nodes unchanged with empty connections
        for n in nodes:
            n["connecting_to"] = []
        return nodes
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = (
        "You are given a base flowchart image and a JSON array of nodes (elements). "
        "Each node has an id and bbox coordinates (x1,y1,x2,y2). "
        "Infer connections between nodes based on the image's arrows/lines and return the SAME array of nodes, "
        "adding a field 'connecting_to' which is a list of target node ids. Do NOT include arrow nodes in the result; "
        "only connect actual element nodes. Preserve original fields and ids. Return pure JSON only."
    )

    # Prepare inputs
    img = Image.open(base_image_path).convert("RGB")
    content = [
        prompt,
        {"mime_type": "application/json", "data": json.dumps(nodes)},
        img,
    ]

    try:
        resp = model.generate_content(content)
        text = resp.text or ""
        # Attempt to parse JSON from response
        parsed = None
        try:
            parsed = json.loads(text)
        except Exception:
            # Try to extract JSON block
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(text[start:end + 1])
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # Fallback: no changes
    for n in nodes:
        n["connecting_to"] = []
    return nodes


# --- Text extraction using full image context ---


def extract_all_text_from_image(nodes: List[Dict], base_image_path: str, instructions: str = "") -> List[Dict]:
    """Use LLM to extract text from all nodes in a single call using the full flowchart image.
    This provides better context awareness than processing individual crops.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not HAS_GEMINI or not api_key:
        print("[warn] Gemini not available; text extraction skipped.")
        return nodes

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Prepare nodes with bbox info for extraction
        extraction_nodes = [{
            "id": n["id"],
            "label": n.get("label", ""),
            "bbox": n["bbox"],
            "connecting_to": n.get("connecting_to", [])
        } for n in nodes]
        
        img = Image.open(base_image_path).convert("RGB")
        
        # Build prompt with optional instructions context
        instructions_context = ""
        if instructions:
            instructions_context = (
                "\n\nCONTEXT - ASSIGNMENT INSTRUCTIONS:\n"
                "The student was asked to create a flowchart for the following task:\n"
                f"{instructions}\n\n"
                "Use these instructions as context to understand what the student is trying to accomplish. "
                "This helps interpret ambiguous handwriting and mathematical symbols.\n"
            )
        
        prompt = (
            "You are extracting HANDWRITTEN text from a flowchart image. "
            "Given the flowchart image and a JSON array of nodes with bounding boxes (x1,y1,x2,y2), "
            "extract the text content from each node.\n\n"
            "The handwriting may vary in style, neatness, slant, and legibility - interpret it as accurately as possible.\n\n"
            "Common handwriting patterns to watch for:\n"
            "- Division symbol '/' may look like '1' or 'l'\n"
            "- Multiplication '*' may look like '+' or 'x'\n"
            "- 'n' may look like 'h'\n"
            "- 'l' (lowercase L) may look like '1' or 'I'\n"
            "- 'o' may look like '0'\n\n"
            "Expected flowchart syntax patterns:\n"
            "- Variable declarations: 'Declare x, y = 5, z' or 'declare var1, var2'\n"
            "- Input statements: 'Input x, y, z' or 'input name, age'\n"
            "- Print statements: 'Print x + \" text \" + y' or 'print result'\n"
            "- Arithmetic: x + 1, count++, sum -= n, x / 6, x * 2\n"
            "- Comparisons: x < 10, age >= 18, count == 0\n"
            "- Branch labels: TRUE, FALSE (case-insensitive)\n"
            f"{instructions_context}\n"
            "For each node in the JSON:\n"
            "1. Look at the bbox coordinates to find the flowchart element\n"
            "2. Extract ONLY the text inside that shape (ignore shape borders)\n"
            "3. Use surrounding context and connections to validate variable consistency\n"
            "4. Return text exactly as written, typically 1-7 words\n\n"
            "Return the SAME JSON array with a 'text' field added to each node. "
            "Keep id, label, bbox, and connecting_to unchanged.\n"
            "Return ONLY valid JSON, no other text.\n\n"
            f"NODES TO EXTRACT TEXT FROM:\n{json.dumps(extraction_nodes, indent=2)}"
        )
        
        content = [prompt, img]
        resp = model.generate_content(content)
        text = (resp.text or "").strip()
        
        # Try to parse extracted nodes
        try:
            extracted = json.loads(text)
            if isinstance(extracted, list) and len(extracted) == len(nodes):
                # Update original nodes with extracted text
                for orig, extr in zip(nodes, extracted):
                    if isinstance(extr, dict) and "text" in extr:
                        orig["text"] = extr["text"]
                        print(f"[ocr] node {orig['id']} text: '{extr['text']}'")
                return nodes
        except Exception:
            # Try to extract JSON block
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                try:
                    extracted = json.loads(text[start:end + 1])
                    if isinstance(extracted, list) and len(extracted) == len(nodes):
                        for orig, extr in zip(nodes, extracted):
                            if isinstance(extr, dict) and "text" in extr:
                                orig["text"] = extr["text"]
                                print(f"[ocr] node {orig['id']} text: '{extr['text']}'")
                        return nodes
                except Exception:
                    pass
        
        print(f"[warn] Could not parse extraction response, text fields not populated")
        for n in nodes:
            n["text"] = ""
        return nodes
        
    except Exception as e:
        print(f"[warn] Text extraction failed: {e}, text fields not populated")
        for n in nodes:
            n["text"] = ""
        return nodes





def draw_labeled_image(bgr: np.ndarray, nodes: List[Dict]) -> np.ndarray:
    canvas = bgr.copy()
    for n in nodes:
        bb = n["bbox"]
        x1, y1, x2, y2 = bb["x1"], bb["y1"], bb["x2"], bb["y2"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = n.get("text", n.get("label", ""))
        cv2.putText(canvas, str(n["id"]), (x1, max(0, y1 - 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        if label:
            cv2.putText(canvas, label, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return canvas


def draw_connections_image(bgr: np.ndarray, nodes: List[Dict]) -> np.ndarray:
    """Draw lines/arrows between node bboxes according to 'connecting_to'."""
    canvas = bgr.copy()
    # Choose colors
    box_color = (0, 255, 0)
    conn_color = (255, 0, 0)
    text_color = (0, 0, 255)

    # Draw boxes and ids first
    for n in nodes:
        bb = n["bbox"]
        x1, y1, x2, y2 = bb["x1"], bb["y1"], bb["x2"], bb["y2"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(canvas, str(n["id"]), (x1, max(0, y1 - 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # Map ids to center points
    centers = {n["id"]: ((n["bbox"]["x1"] + n["bbox"]["x2"]) // 2,
                          (n["bbox"]["y1"] + n["bbox"]["y2"]) // 2) for n in nodes}

    # Draw connections
    for n in nodes:
        src_id = n["id"]
        src_pt = centers.get(src_id)
        for tgt_id in n.get("connecting_to", []) or []:
            tgt_pt = centers.get(tgt_id)
            if src_pt and tgt_pt:
                cv2.arrowedLine(canvas, src_pt, tgt_pt, conn_color, 2, tipLength=0.03)
    return canvas


def create_simplified_nodes_json(nodes: List[Dict]) -> List[Dict]:
    """Extract only label, text, and connecting_to fields for evaluation."""
    simplified = []
    for n in nodes:
        simplified.append({
            "id": n["id"],
            "label": n.get("label", ""),
            "text": n.get("text", ""),
            "connecting_to": n.get("connecting_to", [])
        })
    return simplified


def load_instructions(instructions_dir: str, activity_number: str) -> str:
    """Load instructions from .md file matching the activity number."""
    instr_path = os.path.join(instructions_dir, f"{activity_number}.md")
    if os.path.isfile(instr_path):
        with open(instr_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    print(f"[warn] Instructions not found at {instr_path}")
    return ""


def evaluate_flowchart_with_llm(nodes_json: List[Dict], instructions: str) -> Dict:
    """Use Gemini to evaluate the flowchart against instructions.
    Returns a dict with 'score' (0-10) and 'analysis' fields.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not HAS_GEMINI or not api_key:
        print("[warn] Gemini not available; evaluation skipped.")
        return {"score": None, "analysis": "Gemini not available"}

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (
            "You are an expert flowchart analyzer. You have been given:\n"
            "1. A JSON representation of a flowchart with nodes (label, text, connecting_to)\n"
            "2. Instructions describing what the flowchart should accomplish\n\n"
            "Your task: Analyze how correctly the flowchart implements the given instructions.\n"
            "Consider:\n"
            "- Does the flowchart have all required steps?\n"
            "- Are the steps in the correct logical order?\n"
            "- Are the connections/flow correct?\n"
            "- Does the text in each node match the instruction requirements?\n"
            "- Are there unnecessary or missing branches?\n\n"
            "Return your evaluation as JSON with exactly these fields:\n"
            "{\n"
            '  "score": <integer 0-10>,\n'
            '  "analysis": "<detailed explanation of why you gave this score>"\n'
            "}\n\n"
            "FLOWCHART NODES:\n"
            f"{json.dumps(nodes_json, indent=2)}\n\n"
            "INSTRUCTIONS:\n"
            f"{instructions}\n\n"
            "Return ONLY the JSON, no other text."
        )
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        # Try to parse JSON
        try:
            result = json.loads(text)
            if isinstance(result, dict) and "score" in result and "analysis" in result:
                return result
        except Exception:
            # Try to extract JSON block
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    result = json.loads(text[start:end + 1])
                    if isinstance(result, dict) and "score" in result and "analysis" in result:
                        return result
                except Exception:
                    pass
        print(f"[warn] Could not parse evaluation response: {text[:100]}")
        return {"score": None, "analysis": f"Parse error: {text[:200]}"}
    except Exception as e:
        print(f"[error] Evaluation failed: {e}")
        return {"score": None, "analysis": str(e)}


def process_image(image_path: str, checkpoint_path: str, debug_root: str, instructions_dir: str = None, activity_number: str = None) -> None:
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(debug_root, base)
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model, id_to_name = load_model_and_map(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 1) Detect boxes and save debug drawing
    dets = detect_boxes(model, device, image_path, score_thresh=0.65)
    save_bboxes_debug_image(
        image_path, dets, id_to_name, os.path.join(out_dir, 'bounding_boxes.jpg')
    )

    # 2) Convert to nodes.json
    nodes = dets_to_nodes_json(dets, id_to_name)
    # Remove nodes labeled "scan"
    nodes = [n for n in nodes if str(n.get('label', '')).lower() != 'scan']
    nodes_path = os.path.join(out_dir, 'nodes.initial.json')
    with open(nodes_path, 'w', encoding='utf-8') as f:
        json.dump(nodes, f, indent=2)

    # 3) Infer connections using detected arrow nodes (skip Gemini call)
    # Initialize empty connections
    for n in nodes:
        n['connecting_to'] = []
    # Infer connections using detected arrow nodes; then remove arrows.
    nodes_connected = infer_connections_with_arrows(nodes)
    nodes_conn_path = os.path.join(out_dir, 'nodes.connected.json')
    with open(nodes_conn_path, 'w', encoding='utf-8') as f:
        json.dump(nodes_connected, f, indent=2)

    # 4) Crop nodes
    bgr = cv2.imread(image_path)
    crops_dir = os.path.join(out_dir, 'cropped')
    os.makedirs(crops_dir, exist_ok=True)
    for n in nodes_connected:
        bb = n['bbox']
        # Adjust crop inward to focus text and reduce shape borders
        x1c, y1c, x2c, y2c = adjust_bbox_for_text(bb, n.get('type', ''))
        # Clamp to image bounds
        x1c, y1c = max(0, x1c), max(0, y1c)
        x2c, y2c = min(bgr.shape[1], x2c), min(bgr.shape[0], y2c)
        if x2c <= x1c or y2c <= y1c:
            # Fallback to original bbox if adjustment collapses
            x1c, y1c, x2c, y2c = bb['x1'], bb['y1'], bb['x2'], bb['y2']
            x1c, y1c = max(0, x1c), max(0, y1c)
            x2c, y2c = min(bgr.shape[1], x2c), min(bgr.shape[0], y2c)
        crop = bgr[y1c:y2c, x1c:x2c]
        cv2.imwrite(os.path.join(crops_dir, f"{n['id']}.png"), crop)

    # 5) Extract text from all nodes using full image (single LLM call with context awareness)
    print("[info] Extracting text from all nodes using full flowchart image...")
    # Load instructions for extraction context if available
    extraction_instructions = ""
    if instructions_dir and activity_number:
        extraction_instructions = load_instructions(instructions_dir, activity_number)
    nodes_connected = extract_all_text_from_image(nodes_connected, image_path, extraction_instructions)

    # # OLD METHOD: OCR each crop and compile text (concurrent processing)
    # print("[info] Starting concurrent OCR extraction for all nodes...")
    # max_workers = min(4, len(nodes_connected))  # Use up to 4 threads, or fewer if fewer nodes
    # ocr_results = {}
    # 
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     futures = {executor.submit(_ocr_node_concurrent, n, crops_dir): n['id'] for n in nodes_connected}
    #     for future in as_completed(futures):
    #         node_id, text = future.result()
    #         ocr_results[node_id] = text
    #         print(f"[ocr] node {node_id} text: '{text}'")
    # 
    # # Assign extracted texts back to nodes
    # for n in nodes_connected:
    #     n['text'] = ocr_results.get(n['id'], "")

    # # 5b) Validate and correct OCR using context from the full flowchart
    # # COMMENTED OUT: Not needed since extraction already uses context awareness
    # print("[info] Validating OCR results and correcting common errors...")
    # # Load instructions for validation context if available
    # validation_instructions = ""
    # if instructions_dir and activity_number:
    #     validation_instructions = load_instructions(instructions_dir, activity_number)
    # nodes_connected = validate_and_correct_ocr(nodes_connected, image_path, validation_instructions)

    ocr_json_path = os.path.join(out_dir, 'ocr_extracted.json')
    with open(ocr_json_path, 'w', encoding='utf-8') as f:
        json.dump(nodes_connected, f, indent=2)

    # 6) Save labeled image
    labeled = draw_labeled_image(bgr, nodes_connected)
    cv2.imwrite(os.path.join(out_dir, 'ocr_extracted.jpg'), labeled)
    # Save connections visualization
    conn_vis = draw_connections_image(bgr, nodes_connected)
    cv2.imwrite(os.path.join(out_dir, 'connections.jpg'), conn_vis)
    # Print summary
    print("[summary] Extracted texts:")
    for n in nodes_connected:
        print(f"  id={n['id']}, label={n.get('label','')}, text='{n.get('text','')}'")

    # 7) Create simplified nodes JSON for evaluation
    simplified_nodes = create_simplified_nodes_json(nodes_connected)
    simplified_path = os.path.join(out_dir, 'nodes.simplified.json')
    with open(simplified_path, 'w', encoding='utf-8') as f:
        json.dump(simplified_nodes, f, indent=2)
    print(f"[info] Simplified nodes saved to {simplified_path}")

    # 8) Load instructions and evaluate flowchart
    if instructions_dir and activity_number:
        instructions = load_instructions(instructions_dir, activity_number)
        if instructions:
            print(f"[info] Evaluating flowchart against instructions for activity {activity_number}...")
            evaluation = evaluate_flowchart_with_llm(simplified_nodes, instructions)
            eval_path = os.path.join(out_dir, 'evaluation.json')
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation, f, indent=2)
            print(f"[evaluation] Score: {evaluation.get('score', 'N/A')}/10")
            print(f"[analysis] {evaluation.get('analysis', 'No analysis available')}")
            print(f"[info] Evaluation saved to {eval_path}")
        else:
            print(f"[warn] No instructions found for activity {activity_number}")
    else:
        print(f"[info] Skipping evaluation (no instructions directory or activity number provided)")

def adjust_bbox_for_text(bb: Dict[str,int], shape: str) -> Tuple[int,int,int,int]:
    """Shrink the bbox inward to exclude shape borders while keeping text.
    - rectangle/process/parallelogram: small uniform inward padding (keep more area)
    - diamond/rhombus: stronger horizontal inward padding, moderate vertical
    - ellipse/circle/start_end: moderate uniform inward padding
    """
    x1, y1, x2, y2 = bb['x1'], bb['y1'], bb['x2'], bb['y2']
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    s = (shape or '').lower()
    if s in ('rectangle', 'process', 'box'):
        pad_x = int(0.06 * w)
        pad_y = int(0.08 * h)
    elif s in ('diamond', 'rhombus'):
        # Crop more on left/right to avoid slanted edges
        pad_x = int(0.18 * w)
        pad_y = int(0.12 * h)
    elif s in ('ellipse', 'circle', 'start_end'):
        pad_x = int(0.12 * w)
        pad_y = int(0.12 * h)
    elif s in ('parallelogram',):
        pad_x = int(0.14 * w)
        pad_y = int(0.10 * h)
    else:
        # Unknown: conservative
        pad_x = int(0.10 * w)
        pad_y = int(0.10 * h)
    nx1 = x1 + pad_x
    ny1 = y1 + pad_y
    nx2 = x2 - pad_x
    ny2 = y2 - pad_y
    # Ensure minimal size remains
    min_w = max(10, int(0.25 * w))
    min_h = max(8, int(0.25 * h))
    if (nx2 - nx1) < min_w:
        # reduce horizontal padding proportionally
        reduce = (min_w - (nx2 - nx1)) // 2 + 1
        nx1 = max(x1, nx1 - reduce)
        nx2 = min(x2, nx2 + reduce)
    if (ny2 - ny1) < min_h:
        reduce = (min_h - (ny2 - ny1)) // 2 + 1
        ny1 = max(y1, ny1 - reduce)
        ny2 = min(y2, ny2 + reduce)
    return nx1, ny1, nx2, ny2


def infer_connections_geometry(nodes: List[Dict]) -> List[Dict]:
    """Add multiple 'connecting_to' based on vertical layout and horizontal overlap.
    Strategy:
    - Assume arrows removed already.
    - Sort by top y (bbox.y1).
    - For each node, consider ALL nodes below with horizontal overlap >= 0.3.
    - Select targets with smallest vertical gaps; include all within 1.5x of the minimal dy (to allow branches).
    - Append to existing 'connecting_to' without duplicating.
    """
    # Sort by top y
    nodes_sorted = sorted(nodes, key=lambda n: n['bbox']['y1'])

    def horiz_overlap(a: Dict, b: Dict) -> float:
        ax1, ax2 = a['bbox']['x1'], a['bbox']['x2']
        bx1, bx2 = b['bbox']['x1'], b['bbox']['x2']
        inter = max(0, min(ax2, bx2) - max(ax1, bx1))
        union = max(ax2, bx2) - min(ax1, bx1)
        return inter / union if union > 0 else 0.0

    for i, n in enumerate(nodes_sorted):
        n.setdefault('connecting_to', [])
        candidates: List[Tuple[float, Dict]] = []
        for j in range(i + 1, len(nodes_sorted)):
            m = nodes_sorted[j]
            if m['bbox']['y1'] <= n['bbox']['y2']:
                continue
            if horiz_overlap(n, m) < 0.30:
                continue
            dy = m['bbox']['y1'] - n['bbox']['y2']
            candidates.append((dy, m))
        if not candidates:
            continue
        candidates.sort(key=lambda t: t[0])
        min_dy = candidates[0][0]
        threshold = 1.5 * min_dy
        for dy, m in candidates:
            if dy <= threshold:
                tgt_id = m['id']
                if tgt_id not in n['connecting_to'] and tgt_id != n['id']:
                    n['connecting_to'].append(tgt_id)
            else:
                break
    return nodes_sorted


def infer_connections_with_arrows(nodes: List[Dict]) -> List[Dict]:
    """Infer multiple connections using explicit arrow detections when available.
    Approach:
    - Identify arrow nodes (type startswith 'arrow'). For each arrow, determine direction
      from its type (e.g., 'arrow_line_down', 'arrow_line_up', 'arrow_line_left', 'arrow_line_right').
    - Compute arrow tail and head points:
        down: tail at top-center, head at bottom-center
        up: tail at bottom-center, head at top-center
        left: tail at right-center, head at left-center
        right: tail at left-center, head at right-center
    - Find source node whose bbox contains or is nearest to tail; find target node for head.
      Use padding and nearest distance if not contained.
    - Add edge src -> tgt. Allow multiple edges per src.
    - Remove arrow nodes and, for remaining nodes without any outgoing edge, apply geometric fallback.
    """
    def is_arrow(t: str) -> bool:
        return t.lower().startswith('arrow') if isinstance(t, str) else False

    def center(bb: Dict[str, int]) -> Tuple[int, int]:
        return ((bb['x1'] + bb['x2']) // 2, (bb['y1'] + bb['y2']) // 2)

    def pts_for_arrow(bb: Dict[str, int], direction: str) -> Tuple[Tuple[int,int], Tuple[int,int]]:
        cx, cy = center(bb)
        if direction.endswith('down'):
            tail = (cx, bb['y1'])
            head = (cx, bb['y2'])
        elif direction.endswith('up'):
            tail = (cx, bb['y2'])
            head = (cx, bb['y1'])
        elif direction.endswith('left'):
            tail = (bb['x2'], cy)
            head = (bb['x1'], cy)
        elif direction.endswith('right'):
            tail = (bb['x1'], cy)
            head = (bb['x2'], cy)
        else:
            tail = (cx, bb['y1'])
            head = (cx, bb['y2'])
        return tail, head

    def point_in_bbox(pt: Tuple[int,int], bb: Dict[str,int], pad: int = 10) -> bool:
        x, y = pt
        return (bb['x1'] - pad <= x <= bb['x2'] + pad) and (bb['y1'] - pad <= y <= bb['y2'] + pad)

    def dist_to_bbox(pt: Tuple[int,int], bb: Dict[str,int]) -> float:
        px, py = pt
        x1, y1, x2, y2 = bb['x1'], bb['y1'], bb['x2'], bb['y2']
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5

    arrows = [n for n in nodes if is_arrow(n.get('type', ''))]
    elems = [n for n in nodes if not is_arrow(n.get('type', ''))]

    # Build id map for elements
    id_to_elem = {n['id']: n for n in elems}
    # Initialize connections list
    for n in elems:
        n.setdefault('connecting_to', [])

    # Infer edges from arrows
    for a in arrows:
        bb = a['bbox']
        direction = str(a.get('label', a.get('type', ''))).lower()
        tail, head = pts_for_arrow(bb, direction)
        # Find source: containing tail, else nearest by center distance
        src_candidates = []
        for n in elems:
            if point_in_bbox(tail, n['bbox'], pad=12):
                src_candidates.append((0.0, n))
            else:
                src_candidates.append((dist_to_bbox(tail, n['bbox']), n))
        src_candidates.sort(key=lambda t: t[0])
        src = src_candidates[0][1] if src_candidates else None

        # Find target: containing head, else nearest
        tgt_candidates = []
        for n in elems:
            if point_in_bbox(head, n['bbox'], pad=12):
                tgt_candidates.append((0.0, n))
            else:
                tgt_candidates.append((dist_to_bbox(head, n['bbox']), n))
        tgt_candidates.sort(key=lambda t: t[0])
        tgt = tgt_candidates[0][1] if tgt_candidates else None

        if src and tgt and src['id'] != tgt['id']:
            # Add connection if not already present
            ct = src['connecting_to']
            if tgt['id'] not in ct:
                ct.append(tgt['id'])

    # Fallback geometry adds multiple targets for nodes with no outgoing edges
    no_edge_nodes = [n for n in elems if not n.get('connecting_to')]
    if no_edge_nodes:
        elems = infer_connections_geometry(elems)
    return elems


def main():
    parser = argparse.ArgumentParser(description='Flowchart evaluation pipeline')
    parser.add_argument('activity', type=str, help='Activity number to evaluate (e.g., "1", "2", "3")')
    args = parser.parse_args()

    root = os.getcwd()
    activity_number = args.activity
    activity_images_dir = os.path.join(root, 'test_cases', 'images', activity_number)
    instructions_dir = os.path.join(root, 'test_cases', 'instructions')
    ckpt = os.path.join(root, 'model', 'model_best.pth')
    debug_root = os.path.join(root, 'debug', activity_number)

    if not os.path.isfile(ckpt):
        print(f"[error] Checkpoint not found at {ckpt}")
        sys.exit(1)
    if not os.path.isdir(activity_images_dir):
        print(f"[error] Activity images folder not found at {activity_images_dir}")
        sys.exit(1)

    images = [
        f for f in os.listdir(activity_images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    if not images:
        print(f"[warn] No images found in {activity_images_dir}")
        return

    print(f"[info] Processing activity {activity_number} with {len(images)} image(s)")
    for fname in images:
        img_path = os.path.join(activity_images_dir, fname)
        print(f"[info] Processing {img_path}")
        try:
            process_image(img_path, ckpt, debug_root, instructions_dir, activity_number)
        except Exception as e:
            print(f"[error] Failed on {fname}: {e}")


if __name__ == '__main__':
    main()

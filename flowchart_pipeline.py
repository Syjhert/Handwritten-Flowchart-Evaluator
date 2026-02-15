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


def is_arrow_detection(label: int, id_to_name: Dict[int, str]) -> bool:
    """Check if a detection is an arrow based on its label name."""
    name = id_to_name.get(label, str(label))
    return name.lower().startswith('arrow') if isinstance(name, str) else False


def is_box_contained(box1: np.ndarray, box2: np.ndarray, threshold: float = 0.80) -> bool:
    """Check if at least threshold (default 80%) of box1's area is contained within box2."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return False  # No intersection
    
    intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    
    if box1_area == 0:
        return False
    
    # Check if intersection is at least threshold of box1's area
    return (intersection_area / box1_area) >= threshold


def filter_nested_non_arrows(dets: List[Tuple[np.ndarray, int, float]], 
                             id_to_name: Dict[int, str]) -> List[Tuple[np.ndarray, int, float]]:
    """Remove non-arrow detections that are at least 80% contained within another non-arrow detection.
    Keeps the detection with higher confidence when one is nested inside another.
    """
    if len(dets) <= 1:
        return dets
    
    # Separate arrow and non-arrow detections
    arrow_dets = []
    non_arrow_dets = []
    
    for det in dets:
        box, label, score = det
        if is_arrow_detection(label, id_to_name):
            arrow_dets.append(det)
        else:
            non_arrow_dets.append(det)
    
    # Filter nested non-arrow detections
    to_remove = set()
    
    for i, (box1, label1, score1) in enumerate(non_arrow_dets):
        if i in to_remove:
            continue
        
        for j, (box2, label2, score2) in enumerate(non_arrow_dets):
            if i == j or j in to_remove:
                continue
            
            # Check if box1 is contained in box2
            if is_box_contained(box1, box2):
                # Remove the one with lower confidence
                if score1 < score2:
                    to_remove.add(i)
                else:
                    to_remove.add(j)
                break
            
            # Check if box2 is contained in box1
            elif is_box_contained(box2, box1):
                # Remove the one with lower confidence
                if score1 < score2:
                    to_remove.add(i)
                    break
                else:
                    to_remove.add(j)
    
    # Build filtered list
    filtered_non_arrow = [det for idx, det in enumerate(non_arrow_dets) if idx not in to_remove]
    
    # Combine filtered non-arrows with all arrows
    return arrow_dets + filtered_non_arrow


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

    # 1) Detect boxes
    dets = detect_boxes(model, device, image_path, score_thresh=0.65)
    
    # Filter nested non-arrow detections (remove lower confidence ones)
    dets = filter_nested_non_arrows(dets, id_to_name)

    # 2) Convert to nodes.json
    nodes = dets_to_nodes_json(dets, id_to_name)
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

    # 4) Extract text from all nodes using full image (single LLM call with context awareness)
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

    # 6) Print summary
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
    - Only decision/diamond nodes can have multiple outgoing connections; others get only the closest target.
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
    
    def is_decision_node(node: Dict) -> bool:
        node_type = (node.get('type') or node.get('label') or '').lower()
        return 'diamond' in node_type or 'rhombus' in node_type or 'decision' in node_type

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
        
        # Decision nodes can have multiple connections; others get only the closest
        if is_decision_node(n):
            min_dy = candidates[0][0]
            threshold = 1.5 * min_dy
            for dy, m in candidates:
                if dy <= threshold:
                    tgt_id = m['id']
                    if tgt_id not in n['connecting_to'] and tgt_id != n['id']:
                        n['connecting_to'].append(tgt_id)
                else:
                    break
        else:
            # Non-decision nodes: only connect to the closest target
            m = candidates[0][1]
            tgt_id = m['id']
            if tgt_id not in n['connecting_to'] and tgt_id != n['id']:
                n['connecting_to'].append(tgt_id)
    return nodes_sorted


def infer_connections_with_arrows(nodes: List[Dict]) -> List[Dict]:
    """Infer multiple connections using explicit arrow detections when available.
    Improved approach:
    - Identify arrow nodes and determine direction
    - Find source and target nodes considering flow direction
    - Check for intermediate nodes between source and target
    - Connect through intermediate nodes to avoid skipping components
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

    def is_in_direction(src_bb: Dict[str, int], tgt_bb: Dict[str, int], direction: str) -> bool:
        """Check if target is in the correct direction relative to source."""
        src_cx = (src_bb['x1'] + src_bb['x2']) / 2
        src_cy = (src_bb['y1'] + src_bb['y2']) / 2
        tgt_cx = (tgt_bb['x1'] + tgt_bb['x2']) / 2
        tgt_cy = (tgt_bb['y1'] + tgt_bb['y2']) / 2
        
        if direction.endswith('down'):
            return tgt_cy > src_cy  # Target below source
        elif direction.endswith('up'):
            return tgt_cy < src_cy  # Target above source
        elif direction.endswith('right'):
            return tgt_cx > src_cx  # Target right of source
        elif direction.endswith('left'):
            return tgt_cx < src_cx  # Target left of source
        return True  # Default: allow any direction

    def directional_distance(src_bb: Dict[str, int], tgt_bb: Dict[str, int], direction: str) -> float:
        """Calculate distance considering flow direction (penalize wrong direction)."""
        src_cx = (src_bb['x1'] + src_bb['x2']) / 2
        src_cy = (src_bb['y1'] + src_bb['y2']) / 2
        tgt_cx = (tgt_bb['x1'] + tgt_bb['x2']) / 2
        tgt_cy = (tgt_bb['y1'] + tgt_bb['y2']) / 2
        
        dx = tgt_cx - src_cx
        dy = tgt_cy - src_cy
        
        if direction.endswith('down'):
            # Prefer nodes below, penalize if above
            if dy < 0:
                return float('inf')  # Wrong direction
            return abs(dx) + dy  # Horizontal distance + vertical distance
        elif direction.endswith('up'):
            if dy > 0:
                return float('inf')
            return abs(dx) + abs(dy)
        elif direction.endswith('right'):
            if dx < 0:
                return float('inf')
            return abs(dy) + dx
        elif direction.endswith('left'):
            if dx > 0:
                return float('inf')
            return abs(dy) + abs(dx)
        
        # Default: Euclidean distance
        return ((dx ** 2) + (dy ** 2)) ** 0.5

    def find_intermediate_nodes(src: Dict, tgt: Dict, direction: str, all_elems: List[Dict]) -> List[Dict]:
        """Find nodes that are between source and target in the flow direction."""
        intermediate = []
        src_bb = src['bbox']
        tgt_bb = tgt['bbox']
        
        for node in all_elems:
            if node['id'] == src['id'] or node['id'] == tgt['id']:
                continue
            node_bb = node['bbox']
            
            # Check if node is between source and target
            if direction.endswith('down'):
                # Node should be below source and above target
                src_bottom = src_bb['y2']
                tgt_top = tgt_bb['y1']
                node_top = node_bb['y1']
                node_bottom = node_bb['y2']
                if src_bottom < node_top < tgt_top or src_bottom < node_bottom < tgt_top:
                    # Check horizontal overlap
                    src_cx = (src_bb['x1'] + src_bb['x2']) / 2
                    tgt_cx = (tgt_bb['x1'] + tgt_bb['x2']) / 2
                    node_cx = (node_bb['x1'] + node_bb['x2']) / 2
                    if min(src_cx, tgt_cx) - 50 <= node_cx <= max(src_cx, tgt_cx) + 50:
                        intermediate.append(node)
            elif direction.endswith('up'):
                src_top = src_bb['y1']
                tgt_bottom = tgt_bb['y2']
                node_top = node_bb['y1']
                node_bottom = node_bb['y2']
                if tgt_bottom < node_top < src_top or tgt_bottom < node_bottom < src_top:
                    src_cx = (src_bb['x1'] + src_bb['x2']) / 2
                    tgt_cx = (tgt_bb['x1'] + tgt_bb['x2']) / 2
                    node_cx = (node_bb['x1'] + node_bb['x2']) / 2
                    if min(src_cx, tgt_cx) - 50 <= node_cx <= max(src_cx, tgt_cx) + 50:
                        intermediate.append(node)
            elif direction.endswith('right'):
                src_right = src_bb['x2']
                tgt_left = tgt_bb['x1']
                node_left = node_bb['x1']
                node_right = node_bb['x2']
                if src_right < node_left < tgt_left or src_right < node_right < tgt_left:
                    src_cy = (src_bb['y1'] + src_bb['y2']) / 2
                    tgt_cy = (tgt_bb['y1'] + tgt_bb['y2']) / 2
                    node_cy = (node_bb['y1'] + node_bb['y2']) / 2
                    if min(src_cy, tgt_cy) - 50 <= node_cy <= max(src_cy, tgt_cy) + 50:
                        intermediate.append(node)
            elif direction.endswith('left'):
                src_left = src_bb['x1']
                tgt_right = tgt_bb['x2']
                node_left = node_bb['x1']
                node_right = node_bb['x2']
                if tgt_right < node_left < src_left or tgt_right < node_right < src_left:
                    src_cy = (src_bb['y1'] + src_bb['y2']) / 2
                    tgt_cy = (tgt_bb['y1'] + tgt_bb['y2']) / 2
                    node_cy = (node_bb['y1'] + node_bb['y2']) / 2
                    if min(src_cy, tgt_cy) - 50 <= node_cy <= max(src_cy, tgt_cy) + 50:
                        intermediate.append(node)
        
        # Sort intermediate nodes by distance from source
        intermediate.sort(key=lambda n: directional_distance(src_bb, n['bbox'], direction))
        return intermediate

    def is_decision_node(node: Dict) -> bool:
        """Check if a node is a decision/diamond node."""
        node_type = (node.get('type') or node.get('label') or '').lower()
        return 'diamond' in node_type or 'rhombus' in node_type or 'decision' in node_type

    def euclidean_distance(bb1: Dict[str, int], bb2: Dict[str, int]) -> float:
        """Calculate Euclidean distance between two bounding boxes."""
        cx1 = (bb1['x1'] + bb1['x2']) / 2.0
        cy1 = (bb1['y1'] + bb1['y2']) / 2.0
        cx2 = (bb2['x1'] + bb2['x2']) / 2.0
        cy2 = (bb2['y1'] + bb2['y2']) / 2.0
        return ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5

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

        if not src:
            continue

        # Find target: containing head, else nearest in correct direction
        tgt_candidates = []
        for n in elems:
            if n['id'] == src['id']:
                continue
            if point_in_bbox(head, n['bbox'], pad=12):
                # Check if in correct direction
                if is_in_direction(src['bbox'], n['bbox'], direction):
                    tgt_candidates.append((0.0, n))
            else:
                # Use directional distance (penalizes wrong direction)
                dir_dist = directional_distance(src['bbox'], n['bbox'], direction)
                if dir_dist != float('inf'):
                    tgt_candidates.append((dir_dist, n))
        
        if not tgt_candidates:
            continue
            
        tgt_candidates.sort(key=lambda t: t[0])
        tgt = tgt_candidates[0][1]

        if src['id'] == tgt['id']:
            continue

        # Check for intermediate nodes
        intermediate = find_intermediate_nodes(src, tgt, direction, elems)
        
        if intermediate:
            # Connect through intermediate nodes
            current = src
            for inter_node in intermediate:
                ct = current['connecting_to']
                if inter_node['id'] not in ct:
                    ct.append(inter_node['id'])
                current = inter_node
            # Finally connect last intermediate to target
            ct = current['connecting_to']
            if tgt['id'] not in ct:
                ct.append(tgt['id'])
        else:
            # Direct connection
            ct = src['connecting_to']
            if tgt['id'] not in ct:
                ct.append(tgt['id'])

    # Post-process: For non-decision nodes, keep only the closest connection
    for node in elems:
        connections = node.get('connecting_to', [])
        if len(connections) > 1 and not is_decision_node(node):
            # Find the closest target
            node_bb = node['bbox']
            target_distances = []
            for tgt_id in connections:
                if tgt_id in id_to_elem:
                    tgt_node = id_to_elem[tgt_id]
                    dist = euclidean_distance(node_bb, tgt_node['bbox'])
                    target_distances.append((dist, tgt_id))
            
            # Sort by distance and keep only the closest
            target_distances.sort(key=lambda t: t[0])
            closest_id = target_distances[0][1]
            node['connecting_to'] = [closest_id]

    # Fallback geometry adds multiple targets for nodes with no outgoing edges
    no_edge_nodes = [n for n in elems if not n.get('connecting_to')]
    if no_edge_nodes:
        elems = infer_connections_geometry(elems)
        # Apply the same filtering to geometry-inferred connections
        for node in elems:
            connections = node.get('connecting_to', [])
            if len(connections) > 1 and not is_decision_node(node):
                node_bb = node['bbox']
                target_distances = []
                for tgt_id in connections:
                    if tgt_id in id_to_elem:
                        tgt_node = id_to_elem[tgt_id]
                        dist = euclidean_distance(node_bb, tgt_node['bbox'])
                        target_distances.append((dist, tgt_id))
                if target_distances:
                    target_distances.sort(key=lambda t: t[0])
                    closest_id = target_distances[0][1]
                    node['connecting_to'] = [closest_id]
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

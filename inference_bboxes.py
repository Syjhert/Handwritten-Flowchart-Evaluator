"""
Inference script for bounding box detection on all test case images.
Saves annotated images to output/<activity_number>/<image_name>/
"""

import os
import sys
import json
import torch
from torchvision.transforms import functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import numpy as np
import cv2
from typing import List, Tuple, Dict


def get_model(num_classes: int) -> torch.nn.Module:
    """Load a pre-trained Faster R-CNN model with custom classifier."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_model_and_map(ckpt_path: str):
    """Load model checkpoint and class mappings."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    class_map = state.get("class_map", {}) if isinstance(state, dict) else {}
    id_to_name = {v: k for k, v in class_map.items()} if class_map else {}
    num_classes = (len(class_map) + 1) if class_map else 2
    model = get_model(num_classes)
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) else state)
    return model, id_to_name


def detect_boxes(model, device, image_path: str, score_thresh: float = 0.65) -> List[Tuple[np.ndarray, int, float]]:
    """Run inference on an image and return detections above threshold."""
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


def infer_connections_geometry(nodes: List[Dict]) -> List[Dict]:
    """Add multiple 'connecting_to' based on vertical layout and horizontal overlap.
    Only decision/diamond nodes can have multiple outgoing connections.
    """
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


def save_annotated_image(image_path: str, dets: List[Tuple[np.ndarray, int, float]], 
                         id_to_name: Dict[int, str], out_path: str):
    """Draw bounding boxes with labels and connections on image and save."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Build nodes structure for connection inference
    nodes = []
    for idx, (box, label, score) in enumerate(dets):
        x1, y1, x2, y2 = [int(v) for v in box]
        name = id_to_name.get(int(label), str(int(label)))
        nodes.append({
            'id': idx,
            'type': name,
            'label': name,
            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
            'score': score
        })
    
    # Infer connections
    nodes_with_connections = infer_connections_with_arrows(nodes)
    id_to_node = {n['id']: n for n in nodes_with_connections}
    
    # Draw connections first (so they appear under the boxes)
    connection_count = 0
    for node in nodes_with_connections:
        bbox = node['bbox']
        src_cx = (bbox['x1'] + bbox['x2']) // 2
        src_cy = (bbox['y1'] + bbox['y2']) // 2
        
        for tgt_id in node.get('connecting_to', []):
            if tgt_id in id_to_node:
                tgt_bbox = id_to_node[tgt_id]['bbox']
                tgt_cx = (tgt_bbox['x1'] + tgt_bbox['x2']) // 2
                tgt_cy = (tgt_bbox['y1'] + tgt_bbox['y2']) // 2
                
                # Draw arrow from source to target
                cv2.arrowedLine(bgr, (src_cx, src_cy), (tgt_cx, tgt_cy), 
                               (255, 0, 0), 2, tipLength=0.03)
                connection_count += 1
    
    # Draw bounding boxes
    for box, label, score in dets:
        x1, y1, x2, y2 = [int(v) for v in box]
        name = id_to_name.get(int(label), str(int(label)))
        
        # Draw bounding box
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label with background
        label_text = f"{name}:{score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(bgr, (x1, y1 - text_height - baseline - 5), 
                     (x1 + text_width, y1), (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(bgr, label_text, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
    
    # Save annotated image
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, bgr)
    
    print(f"[saved] {out_path} ({len(dets)} detections, {connection_count} connections)")


def process_activity_images(activity_dir: str, activity_num: str, model, device, 
                            id_to_name: Dict[int, str], output_root: str, score_thresh: float = 0.60):
    """Process all images in an activity directory."""
    if not os.path.isdir(activity_dir):
        print(f"[skip] Activity directory not found: {activity_dir}")
        return
    
    images = [f for f in os.listdir(activity_dir) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        print(f"[skip] No images found in {activity_dir}")
        return
    
    print(f"\n[info] Processing activity {activity_num}: {len(images)} image(s)")
    
    for img_name in images:
        img_path = os.path.join(activity_dir, img_name)
        img_basename = os.path.splitext(img_name)[0]
        
        # Create output path: output/<activity_num>/<img_basename>.jpg
        out_dir = os.path.join(output_root, activity_num)
        out_path = os.path.join(out_dir, f'{img_basename}.jpg')
        
        try:
            # Run detection
            dets = detect_boxes(model, device, img_path, score_thresh=score_thresh)
            
            # Filter nested non-arrow detections (remove lower confidence ones)
            dets = filter_nested_non_arrows(dets, id_to_name)
            
            # Save annotated image with connections
            save_annotated_image(img_path, dets, id_to_name, out_path)
            
        except Exception as e:
            print(f"[error] Failed on {img_name}: {e}")


def main():
    # Paths
    root = os.getcwd()
    test_cases_dir = os.path.join(root, 'test_cases', 'images')
    checkpoint_path = os.path.join(root, 'model', 'model_best.pth')
    output_root = os.path.join(root, 'output')
    
    # Check checkpoint exists
    if not os.path.isfile(checkpoint_path):
        print(f"[error] Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    # Check test_cases directory exists
    if not os.path.isdir(test_cases_dir):
        print(f"[error] Test cases directory not found at {test_cases_dir}")
        sys.exit(1)
    
    # Load model
    print("[info] Loading model...")
    model, id_to_name = load_model_and_map(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"[info] Model loaded on {device}")
    print(f"[info] Classes: {id_to_name}")
    
    # Find all activity folders
    activity_folders = [f for f in os.listdir(test_cases_dir) 
                       if os.path.isdir(os.path.join(test_cases_dir, f))]
    
    if not activity_folders:
        print(f"[error] No activity folders found in {test_cases_dir}")
        sys.exit(1)
    
    print(f"[info] Found {len(activity_folders)} activity folder(s): {sorted(activity_folders)}")
    
    # Process each activity folder
    for activity_num in sorted(activity_folders):
        activity_dir = os.path.join(test_cases_dir, activity_num)
        process_activity_images(activity_dir, activity_num, model, device, 
                               id_to_name, output_root)
    
    print(f"\n[done] All annotations and connections saved to {output_root}/")


if __name__ == '__main__':
    main()

import os
import json
import tempfile
from typing import Dict
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import cv2

from flowchart_pipeline import (
    load_model_and_map,
    detect_boxes,
    filter_nested_non_arrows,
    dets_to_nodes_json,
    infer_connections_with_arrows,
    extract_all_text_from_image,
    create_simplified_nodes_json,
    evaluate_flowchart_with_llm,
    draw_labeled_image,
    draw_connections_image
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global model cache
_model_cache = {
    'model': None,
    'id_to_name': None,
    'device': None,
    'checkpoint_path': None
}


def load_model_once(checkpoint_path: str):
    """Load model once and cache it for subsequent requests."""
    if (_model_cache['model'] is None or 
        _model_cache['checkpoint_path'] != checkpoint_path):
        print(f"[info] Loading model from {checkpoint_path}...")
        model, id_to_name = load_model_and_map(checkpoint_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        _model_cache['model'] = model
        _model_cache['id_to_name'] = id_to_name
        _model_cache['device'] = device
        _model_cache['checkpoint_path'] = checkpoint_path
        print(f"[info] Model loaded and cached (device: {device})")
    
    return _model_cache['model'], _model_cache['id_to_name'], _model_cache['device']


def process_flowchart_image(image_path: str, instructions_context: str = "") -> Dict:
    """
    Process a flowchart image and return results as a dictionary.
    
    Args:
        image_path: Path to the flowchart image
        instructions_context: Optional instructions context string for evaluation
        
    Returns:
        Dictionary containing nodes, evaluation, and metadata
    """
    # Get checkpoint path from environment or use default
    root = os.getcwd()
    checkpoint_path = os.path.join(root, 'model', 'model_best.pth')
    
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Load model (cached)
    model, id_to_name, device = load_model_once(checkpoint_path)
    
    # 1) Detect boxes
    dets = detect_boxes(model, device, image_path, score_thresh=0.65)
    
    # Filter nested non-arrow detections
    dets = filter_nested_non_arrows(dets, id_to_name)
    
    # 2) Convert to nodes JSON
    nodes = dets_to_nodes_json(dets, id_to_name)
    
    # 3) Infer connections using detected arrow nodes
    for n in nodes:
        n['connecting_to'] = []
    nodes_connected = infer_connections_with_arrows(nodes)
    
    # 4) Extract text from all nodes using full image with instructions context
    print("[info] Extracting text from all nodes using full flowchart image...")
    nodes_connected = extract_all_text_from_image(
        nodes_connected, 
        image_path, 
        instructions_context
    )
    
    # 5) Create simplified nodes JSON for evaluation
    simplified_nodes = create_simplified_nodes_json(nodes_connected)
    
    # 6) Evaluate flowchart if instructions provided
    evaluation = None
    if instructions_context and instructions_context.strip():
        print("[info] Evaluating flowchart against instructions...")
        evaluation = evaluate_flowchart_with_llm(simplified_nodes, instructions_context)
    
    # 7) Generate visualization images (base64 encoded for API response)
    bgr = cv2.imread(image_path)
    labeled_image = draw_labeled_image(bgr, nodes_connected)
    connections_image = draw_connections_image(bgr, nodes_connected)
    
    # Convert images to base64 for JSON response
    import base64
    from io import BytesIO
    
    def image_to_base64(img_array):
        """Convert OpenCV image array to base64 string."""
        _, buffer = cv2.imencode('.jpg', img_array)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"
    
    result = {
        'success': True,
        'nodes': simplified_nodes,
        'nodes_detailed': nodes_connected,
        'evaluation': evaluation,
        'visualizations': {
            'labeled': image_to_base64(labeled_image),
            'connections': image_to_base64(connections_image)
        },
        'metadata': {
            'num_nodes': len(nodes_connected),
            'num_arrows_detected': len([n for n in nodes if id_to_name.get(n[1], '').lower().startswith('arrow')])
        }
    }
    
    return result


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'flowchart-evaluator'})


@app.route('/evaluate', methods=['POST'])
def evaluate_flowchart():
    """
    Endpoint to evaluate a handwritten flowchart image.
    
    Expected request:
    - Content-Type: multipart/form-data
    - Fields:
        - 'image': image file (jpg, jpeg, png)
        - 'instructions_context': (optional) string with instructions context
    
    Returns:
        JSON response with nodes, evaluation, and visualizations
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided. Use form field "image".'
            }), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image file selected.'
            }), 400
        
        # Get instructions context (optional)
        instructions_context = request.form.get('instructions_context', '').strip()
        
        # Validate image file
        filename = secure_filename(image_file.filename)
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return jsonify({
                'success': False,
                'error': 'Invalid image format. Supported: jpg, jpeg, png'
            }), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
            image_file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Validate it's a valid image
            img = Image.open(tmp_path)
            img.verify()
            img.close()
            
            # Process the flowchart
            result = process_flowchart_image(tmp_path, instructions_context)
            return jsonify(result)
            
        except Exception as e:
            print(f"[error] Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'Processing failed: {str(e)}'
            }), 500
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        print(f"[error] Request handling failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Request handling failed: {str(e)}'
        }), 500


@app.route('/evaluate/json', methods=['POST'])
def evaluate_flowchart_json():
    """
    Alternative endpoint that accepts JSON with base64-encoded image.
    
    Expected request:
    - Content-Type: application/json
    - Body:
        {
            "image_base64": "data:image/jpeg;base64,..." or just base64 string,
            "instructions_context": "optional instructions string"
        }
    
    Returns:
        JSON response with nodes, evaluation, and visualizations
    """
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        
        if not data or 'image_base64' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "image_base64" field in JSON body'
            }), 400
        
        image_base64 = data['image_base64']
        instructions_context = data.get('instructions_context', '').strip()
        
        # Extract base64 data (handle data URI format)
        if ',' in image_base64:
            image_base64 = image_base64.split(',', 1)[1]
        
        # Decode and save temporarily
        import base64
        from io import BytesIO
        
        try:
            image_data = base64.b64decode(image_base64)
            img = Image.open(BytesIO(image_data))
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                img.save(tmp_file.name, format='PNG')
                tmp_path = tmp_file.name
            
            try:
                # Process the flowchart
                result = process_flowchart_image(tmp_path, instructions_context)
                return jsonify(result)
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid base64 image data: {str(e)}'
            }), 400
    
    except Exception as e:
        print(f"[error] Request handling failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Request handling failed: {str(e)}'
        }), 500


if __name__ == '__main__':
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get configuration from environment
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"[info] Starting Flask app on {host}:{port}")
    print(f"[info] Debug mode: {debug}")
    app.run(host=host, port=port, debug=debug)


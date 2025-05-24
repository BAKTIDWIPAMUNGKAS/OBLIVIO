from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
from PIL import Image, ExifTags
import io
import base64
import cv2
from scipy.fftpack import dct, idct
import os
import json
import re
from datetime import datetime
import hashlib
import tempfile
import logging
from werkzeug.datastructures import FileStorage

# Import functions from video_audio.py
from metadata import (
    extract_video_metadata,
    extract_audio_metadata,
    sanitize_video_metadata,
    sanitize_audio_metadata,
    obscure_file_metadata,
    check_ffmpeg_environment
)

# Import functions from audio_video.py
from stegano import (
    safe_embed_audio,
    safe_extract_audio,
    safe_embed_video,
    safe_extract_video,
    validate_video_file,
    check_ffmpeg_installed as check_ffmpeg_environment
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Quantization table for DCT (standard JPEG quantization table)
quant = np.array([[16,11,10,16,24,40,51,61],
                  [12,12,14,19,26,58,60,55],
                  [14,13,16,24,40,57,69,56],
                  [14,17,22,29,51,87,80,62],
                  [18,22,37,56,68,109,103,77],
                  [24,35,55,64,81,104,113,92],
                  [49,64,78,87,103,121,120,101],
                  [72,92,95,98,112,100,103,99]])

@app.route('/')
def home():
    return render_template('index.html')

def chunks(l, n):
    """Helper function to break the image into chunks"""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def add_padding(img, row, col):
    """Add padding to make image dimensions divisible by 8"""
    return cv2.resize(img, (col + (8 - col % 8) if col % 8 != 0 else col, 
                           row + (8 - row % 8) if row % 8 != 0 else row))

def encrypt_message(message, password):
    """Encrypt message using password"""
    if not password:
        return message
    
    # Create a simple key from password using SHA-256
    key = hashlib.sha256(password.encode()).digest()
    
    # XOR encryption
    encrypted = bytearray()
    for i, char in enumerate(message.encode()):
        encrypted.append(char ^ key[i % len(key)])
    
    return base64.b64encode(encrypted).decode()

def decrypt_message(encrypted_message, password):
    """Decrypt message using password"""
    if not password:
        return encrypted_message
    
    try:
        # Decode base64
        encrypted = base64.b64decode(encrypted_message)
        
        # Create a simple key from password using SHA-256
        key = hashlib.sha256(password.encode()).digest()
        
        # XOR decryption
        decrypted = bytearray()
        for i, char in enumerate(encrypted):
            decrypted.append(char ^ key[i % len(key)])
        
        return decrypted.decode()
    except:
        return "Decryption failed. Incorrect password."

def embed_message_dct(img_array, message, password=""):
    """Embed message using DCT method with optional encryption"""
    # Encrypt message if password is provided
    if password:
        message = encrypt_message(message, password)
    
    # Prepare message with marker for size extraction later
    message_with_marker = str(len(message)) + '*' + message
    binary_message = []
    for char in message_with_marker:
        binary_message.append(format(ord(char), '08b'))
    
    # Flatten binary message into bits
    bits = []
    for byte in binary_message:
        for bit in byte:
            bits.append(int(bit))
    
    # Get image dimensions
    row, col = img_array.shape[:2]
    
    # Check if message fits in the image
    # Each 8x8 block can store multiple bits (using mid-frequency coefficients)
    max_capacity = (row // 8) * (col // 8) * 3  # Each block can store ~3 bits
    if len(bits) > max_capacity:
        raise ValueError(f"Message too large to embed. Max capacity: ~{max_capacity//8} characters")
    
    # Make dimensions divisible by 8
    if row % 8 != 0 or col % 8 != 0:
        img_array = add_padding(img_array, row, col)
        row, col = img_array.shape[:2]
    
    # Split image into RGB channels
    b_img, g_img, r_img = cv2.split(img_array)
    
    # Convert channel to float for DCT
    b_img = np.float32(b_img)
    
    # Break into 8x8 blocks
    img_blocks = []
    block_positions = []  # Store positions for reconstruction
    for i in range(0, row, 8):
        for j in range(0, col, 8):
            if i + 8 <= row and j + 8 <= col:
                img_blocks.append(np.float32(b_img[i:i+8, j:j+8] - 128))
                block_positions.append((i, j))
    
    # Apply DCT to each block
    dct_blocks = []
    for block in img_blocks:
        dct_blocks.append(cv2.dct(block))
    
    # Quantize DCT coefficients
    quantized_dct = []
    for block in dct_blocks:
        quantized_dct.append(np.round(block / quant))
    
    # Embed message bits into mid-frequency coefficients
    # These positions offer good balance of imperceptibility and robustness
    # Using zigzag pattern for mid-frequencies
    embedding_positions = [(1,2), (2,1), (2,2), (3,0), (0,3)]
    bit_index = 0
    
    for block_idx, quantized_block in enumerate(quantized_dct):
        modified_block = quantized_block.copy()
        
        for pos in embedding_positions:
            if bit_index < len(bits):
                # Get coefficient at position
                i, j = pos
                coef = modified_block[i, j]
                
                # Embed bit by modifying LSB (even/odd)
                if (coef % 2 == 0 and bits[bit_index] == 1) or (coef % 2 == 1 and bits[bit_index] == 0):
                    if coef > 0:
                        modified_block[i, j] = coef + 1
                    else:
                        modified_block[i, j] = coef - 1
                
                bit_index += 1
                
                # Stop if all bits embedded
                if bit_index >= len(bits):
                    break
        
        quantized_dct[block_idx] = modified_block
        
        # Stop if all bits embedded
        if bit_index >= len(bits):
            break
    
    # Dequantize DCT coefficients
    dequantized_blocks = []
    for block in quantized_dct:
        dequantized_blocks.append(block * quant)
    
    # Apply inverse DCT
    idct_blocks = []
    for block in dequantized_blocks:
        idct_blocks.append(cv2.idct(block) + 128)
    
    # Rebuild blue channel
    stego_b = np.zeros_like(b_img)
    for idx, (i, j) in enumerate(block_positions):
        if idx < len(idct_blocks):
            stego_b[i:i+8, j:j+8] = idct_blocks[idx]
    
    # Ensure values are within 0-255 range
    stego_b = np.uint8(np.clip(stego_b, 0, 255))
    
    # Merge channels back into image
    stego_img = cv2.merge([stego_b, g_img, r_img])
    
    return stego_img

def extract_message_dct(img_array, password=""):
    """Extract message using DCT method with optional decryption"""
    # Get image dimensions
    row, col = img_array.shape[:2]
    
    # Split image into RGB channels
    b_img, g_img, r_img = cv2.split(img_array)
    
    # Convert blue channel to float for DCT
    b_img = np.float32(b_img)
    
    # Break into 8x8 blocks
    img_blocks = []
    for i in range(0, row, 8):
        for j in range(0, col, 8):
            if i + 8 <= row and j + 8 <= col:
                img_blocks.append(np.float32(b_img[i:i+8, j:j+8] - 128))
    
    # Apply DCT to each block
    dct_blocks = []
    for block in img_blocks:
        dct_blocks.append(cv2.dct(block))
    
    # Quantize DCT coefficients
    quantized_dct = []
    for block in dct_blocks:
        quantized_dct.append(np.round(block / quant))
    
    # Extract bits from mid-frequency coefficients (same positions as embed)
    embedding_positions = [(1,2), (2,1), (2,2), (3,0), (0,3)]
    extracted_bits = []
    
    for block in quantized_dct:
        for pos in embedding_positions:
            i, j = pos
            coef = block[i, j]
            # Extract LSB
            bit = int(coef % 2)
            extracted_bits.append(bit)
    
    # Convert bits to bytes
    extracted_bytes = []
    for i in range(0, len(extracted_bits), 8):
        if i + 8 <= len(extracted_bits):
            byte = ''.join(map(str, extracted_bits[i:i+8]))
            extracted_bytes.append(int(byte, 2))
    
    # Convert bytes to string
    extracted_message = ''.join(chr(byte) for byte in extracted_bytes if 32 <= byte <= 126 or byte in [9, 10, 13])
    
    # Find message size marker
    if '*' in extracted_message:
        try:
            size_str, message = extracted_message.split('*', 1)
            message_size = int(size_str)
            
            # Truncate to actual message size
            if len(message) >= message_size:
                extracted_message = message[:message_size]
                
                # Decrypt if password provided
                if password:
                    extracted_message = decrypt_message(extracted_message, password)
                
                return extracted_message
        except:
            pass
    
    # If marker not found or parsing failed, return raw message
    return extracted_message

def extract_image_metadata(img):
    """Extract metadata from an image and return it as a dictionary"""
    metadata = {}
    
    try:
        # Extract Exif data
        exif_data = {}
        if hasattr(img, '_getexif') and img._getexif() is not None:
            for tag, value in img._getexif().items():
                if tag in ExifTags.TAGS:
                    tag_name = ExifTags.TAGS[tag]
                    # Format datetime values properly
                    if tag_name == 'DateTimeOriginal' or tag_name == 'DateTime':
                        try:
                            # Convert to a standard format
                            date_obj = datetime.strptime(str(value), '%Y:%m:%d %H:%M:%S')
                            value = date_obj.strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            pass
                    # Skip binary data or convert to string representation
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8')
                        except:
                            value = str(len(value)) + " bytes of binary data"
                    exif_data[tag_name] = value
        
        metadata['exif'] = exif_data
        
        # Image dimensions
        metadata['dimensions'] = {
            'width': img.width,
            'height': img.height
        }
        
        # Image format
        metadata['format'] = img.format
        
        # Image mode (RGB, RGBA, etc.)
        metadata['mode'] = img.mode
        
        # Try to extract XMP metadata if present
        if hasattr(img, 'info') and 'xmp' in img.info:
            try:
                metadata['xmp'] = img.info['xmp'].decode('utf-8')
            except:
                metadata['xmp'] = "XMP data present but could not be decoded"
        
    except Exception as e:
        metadata['error'] = str(e)
    
    return metadata

def sanitize_metadata(metadata, options):
    """Sanitize metadata based on user preferences"""
    sanitized = metadata.copy()
    
    # Handle exif data
    if 'exif' in sanitized:
        exif = sanitized['exif']
        
        # Remove GPS data if requested
        if options.get('remove_gps', False):
            for key in list(exif.keys()):
                if 'GPS' in key:
                    del exif[key]
        
        # Remove timestamp data if requested
        if options.get('remove_timestamps', False):
            for key in list(exif.keys()):
                if any(time_related in key for time_related in ['Date', 'Time', 'Created', 'Modified']):
                    del exif[key]
        
        # Remove device info if requested
        if options.get('remove_device_info', False):
            for key in list(exif.keys()):
                if any(info in key for info in ['Make', 'Model', 'Software', 'Device', 'Camera', 'Serial']):
                    del exif[key]
        
        # Remove all metadata if requested
        if options.get('remove_all', False):
            sanitized['exif'] = {}
    
    # Handle XMP data similarly if present
    if 'xmp' in sanitized and options.get('remove_all', False):
        sanitized['xmp'] = None
    
    return sanitized

@app.route('/api/embed', methods=['POST'])
def api_embed():
    try:
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image file provided'})
        
        image_file = request.files['image']
        message = request.form.get('message', '')
        password = request.form.get('password', '')
        
        if not message:
            return jsonify({'status': 'error', 'message': 'No message provided'})
        
        # Read image
        img = Image.open(image_file)
        img_array = np.array(img)
        
        # Embed message with optional encryption
        stego_array = embed_message_dct(img_array, message, password)
        stego_img = Image.fromarray(stego_array)
        
        # Convert to base64
        img_buffer = io.BytesIO()
        stego_img.save(img_buffer, format="PNG")
        img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'image': f"data:image/png;base64,{img_str}"
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/extract', methods=['POST'])
def api_extract():
    try:
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image file provided'})
        
        image_file = request.files['image']
        password = request.form.get('password', '')
        
        # Read image
        img = Image.open(image_file)
        img_array = np.array(img)
        
        # Extract message with optional decryption
        message = extract_message_dct(img_array, password)
        
        return jsonify({
            'status': 'success',
            'message': message
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/embed_media', methods=['POST'])
def api_embed_media():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'})
        
        file = request.files['file']
        message = request.form.get('message', '')
        password = request.form.get('password', '')
        
        if not message:
            return jsonify({'status': 'error', 'message': 'No message provided'})
        
        filename = file.filename.lower()
        file_ext = os.path.splitext(filename)[1].lower() if '.' in filename else ''
        
        # Handle audio files
        if file_ext in ['.mp3', '.wav', '.ogg']:
            result = safe_embed_audio(file, message, password)
            if result.get('status') == 'success':
                return jsonify({
                    'status': 'success',
                    'file': result['file'],
                    'file_type': 'audio'
                })
            return jsonify(result)
        
        # Handle video files
        elif file_ext in ['.mp4', '.mov', '.avi']:
            result = safe_embed_video(file, message, password)
            if result.get('status') == 'success':
                return jsonify({
                    'status': 'success',
                    'file': result['file'],
                    'file_type': 'video'
                })
            return jsonify(result)
        
        else:
            return jsonify({'status': 'error', 'message': 'Unsupported file format'})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/extract_media', methods=['POST'])
def extract_media():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
        
        file = request.files['file']
        password = request.form.get('password', '')
        
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file'}), 400
        
        # Check file type and route to appropriate handler
        if file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            result = safe_extract_video(file, password)
        elif file.filename.lower().endswith(('.wav', '.mp3')):
            result = safe_extract_audio(file, password)
        else:
            return jsonify({'status': 'error', 'message': 'Unsupported file type'}), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/metadata/extract', methods=['POST'])
def api_metadata_extract():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'})
        
        file = request.files['file']
        
        # Get file type from extension
        filename = file.filename.lower()
        file_ext = os.path.splitext(filename)[1] if '.' in filename else None
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
            # Handle image files
            img = Image.open(file)
            metadata = extract_image_metadata(img)
        elif file_ext in ['.mp3', '.wav', '.ogg', '.oga']:
            # Use the imported function for audio metadata extraction
            metadata = extract_audio_metadata(file)
        elif file_ext in ['.mp4', '.mov', '.avi', '.m4v']:
            # Use the imported function for video metadata extraction
            metadata = extract_video_metadata(file)
        else:
            return jsonify({'status': 'error', 'message': 'Unsupported file format'})
        
        return jsonify({
            'status': 'success',
            'metadata': metadata
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/metadata/sanitize', methods=['POST'])
def api_metadata_sanitize():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'})
        
        file = request.files['file']
        options_str = request.form.get('options', '{}')
        
        try:
            options = json.loads(options_str)
        except json.JSONDecodeError:
            options = {}
        
        filename = file.filename.lower()
        file_ext = os.path.splitext(filename)[1] if '.' in filename else None
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
            # Handle image files
            img = Image.open(file)
            metadata = extract_image_metadata(img)
            
            # Sanitize metadata
            sanitized_metadata = sanitize_metadata(metadata, options)
            
            # Create a new image without the unwanted metadata
            img_array = np.array(img)
            new_img = Image.fromarray(img_array)
            
            # Convert to base64
            img_buffer = io.BytesIO()
            new_img.save(img_buffer, format="PNG")
            img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            return jsonify({
                'status': 'success',
                'metadata': sanitized_metadata,
                'image': f"data:image/png;base64,{img_str}"
            })
        elif file_ext in ['.mp3', '.wav', '.ogg', '.oga']:
            # Use the imported function for audio metadata sanitization
            try:
                result = sanitize_audio_metadata(file, options)
                return jsonify({
                    'status': 'success',
                    'original_metadata': result['original_metadata'],
                    'sanitized_metadata': result['sanitized_metadata'],
                    'file': result['file'],
                    'file_type': 'audio'
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f"Error sanitizing audio: {str(e)}"
                })
        elif file_ext in ['.mp4', '.m4v', '.mov', '.avi']:
            # Use the imported function for video metadata sanitization
            try:
                result = sanitize_video_metadata(file, options)
                return jsonify({
                    'status': 'success',
                    'original_metadata': result['original_metadata'],
                    'sanitized_metadata': result['sanitized_metadata'],
                    'file': result['file'],
                    'file_type': 'video'
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f"Error sanitizing video: {str(e)}"
                })
        else:
            return jsonify({'status': 'error', 'message': 'Unsupported file format for sanitization'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/metadata/obscure', methods=['POST'])
def api_metadata_obscure():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'})
        
        file = request.files['file']
        options = json.loads(request.form.get('options', '{}'))
        password = request.form.get('password', '')
        
        # Get file type
        filename = file.filename.lower()
        file_ext = os.path.splitext(filename)[1] if '.' in filename else None
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
            # Handle image files
            img = Image.open(file)
            metadata = extract_image_metadata(img)
            
            # Convert metadata to JSON string
            metadata_json = json.dumps(metadata)
            
            # Embed the metadata into the image itself using DCT
            img_array = np.array(img)
            
            # Create a new clean image without original metadata
            clean_img = Image.fromarray(img_array)
            
            # Embed metadata with optional encryption
            stego_array = embed_message_dct(img_array, metadata_json, password)
            stego_img = Image.fromarray(stego_array)
            
            # Convert to base64
            img_buffer = io.BytesIO()
            stego_img.save(img_buffer, format="PNG")
            img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            return jsonify({
                'status': 'success',
                'image': f"data:image/png;base64,{img_str}"
            })
        else:
            # Use the imported function for audio/video metadata obscuring
            metadata = None
            
            # Extract metadata first based on file type
            if file_ext in ['.mp3', '.wav', '.ogg', '.oga']:
                metadata = extract_audio_metadata(file)
            elif file_ext in ['.mp4', '.m4v', '.mov', '.avi']:
                metadata = extract_video_metadata(file)
            else:
                return jsonify({'status': 'error', 'message': 'Unsupported file format for metadata obscuring'})
            
            # Convert metadata to JSON string
            metadata_json = json.dumps(metadata)
            
            # Use obscure_file_metadata from video_audio.py
            result = obscure_file_metadata(file, metadata_json, password)
            
            return jsonify({
                'status': 'success',
                'file': result,
                'file_type': 'audio' if file_ext in ['.mp3', '.wav', '.ogg', '.oga'] else 'video'
            })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/sanitize_video_metadata', methods=['POST'])
def handle_sanitize_video_metadata():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
        
    file = request.files['file']
    options = request.form.get('options', '{}')
    
    try:
        options = json.loads(options)
    except json.JSONDecodeError:
        options = {}
        
    try:
        result = sanitize_video_metadata(file, options)
        return jsonify({
            'status': 'success',
            'original_metadata': result['original_metadata'],
            'sanitized_metadata': result['sanitized_metadata'],
            'file': result['file']
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/obscure_video_metadata', methods=['POST'])
def handle_obscure_video_metadata():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
        
    file = request.files['file']
    options = request.form.get('options', '{}')
    
    try:
        options = json.loads(options)
    except json.JSONDecodeError:
        options = {}
        
    try:
        metadata_json = request.form.get('metadata', '{}')
        password = request.form.get('password', '')
        
        result = obscure_file_metadata(file, metadata_json, password)
        return jsonify({
            'status': 'success',
            'file': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

from stegano import safe_embed_audio, safe_extract_audio, safe_embed_video, safe_extract_video

@app.route('/api/embed_audio', methods=['POST'])
def api_embed_audio():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No audio file provided'})
    
    audio_file = request.files['file']
    message = request.form.get('message', '')
    password = request.form.get('password', '')
    
    result = safe_embed_audio(audio_file, message, password)
    if result.get('status') == 'success':
        # `file` from safe_embed_audio is expected to be base64 encoded string
        return jsonify({'status': 'success', 'file': result['file'], 'file_type': 'audio'})
    else:
        return jsonify(result)

@app.route('/api/extract_audio', methods=['POST'])
def api_extract_audio():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No audio file provided'})
    
    audio_file = request.files['file']
    password = request.form.get('password', '')
    
    result = safe_extract_audio(audio_file, password)
    return jsonify(result)

# API endpoints untuk video (menggunakan LSB melalui audio_video.py)
@app.route('/api/embed_video', methods=['POST'])
def api_embed_video():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file provided'})
    
    video_file = request.files['file']
    message = request.form.get('message', '')
    password = request.form.get('password', '')
    
    # Validasi ukuran file
    video_file.seek(0, 2)  # Pindah ke akhir file
    file_size = video_file.tell()
    video_file.seek(0)  # Kembali ke awal
    
    if file_size > 50 * 1024 * 1024:  # Batas 50MB
        return jsonify({'status': 'error', 'message': 'File too large (max 50MB)'})
    
    if len(message) > 1000:  # Batas pesan
        return jsonify({'status': 'error', 'message': 'Message too long (max 1000 chars)'})
    
    result = safe_embed_video(video_file, message, password)
    return jsonify(result)

@app.route('/api/extract_media', methods=['POST'])
def api_extract_media():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'})
        
        file = request.files['file']
        password = request.form.get('password', '')
        
        filename = file.filename.lower()
        file_ext = os.path.splitext(filename)[1].lower() if '.' in filename else ''
        
        # Handle audio files
        if file_ext in ['.mp3', '.wav', '.ogg']:
            result = safe_extract_audio(file, password)
            return jsonify(result)
        
        # Handle video files
        elif file_ext in ['.mp4', '.mov', '.avi']:
            result = safe_extract_video(file, password)
            return jsonify(result)
        
        else:
            return jsonify({'status': 'error', 'message': 'Unsupported file format'})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/check_ffmpeg', methods=['GET'])
def api_check_ffmpeg():
    result = check_ffmpeg_environment()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
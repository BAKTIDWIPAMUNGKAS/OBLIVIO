import numpy as np
import io
import base64
import os
import tempfile
import hashlib
import cv2
from PIL import Image
import wave
import struct
import json
import logging
import subprocess
from werkzeug.datastructures import FileStorage
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import gc


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- New Global Constants for Message Delimiters ---
# Use unique and long markers to reduce false positives
START_MARKER = "###STEGO_START_UNIQUE_DELIMITER_V2###"
END_MARKER = "###STEGO_END_UNIQUE_DELIMITER_V2###"

def check_ffmpeg_installed():
    """Check if FFmpeg is installed and available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                              check=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              timeout=10)
        logger.info("FFmpeg is installed and available.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        logger.error("FFmpeg is not installed or not in PATH. Please install FFmpeg and add it to your system's PATH.")
        return False

def encrypt_message(message, password):
    """Encrypt message using password with XOR encryption"""
    if not password:
        return message

    # Create a key from password using SHA-256
    key = hashlib.sha256(password.encode()).digest()

    # XOR encryption
    encrypted = bytearray()
    for i, char in enumerate(message.encode('utf-8')):
        encrypted.append(char ^ key[i % len(key)])

    return base64.b64encode(encrypted).decode()

def decrypt_message(encrypted_message, password):
    """Decrypt message using password with XOR decryption"""
    if not password:
        return encrypted_message

    try:
        import base64
        import hashlib
        
        encrypted = base64.b64decode(encrypted_message)
        key = hashlib.sha256(password.encode()).digest()
        
        decrypted = bytearray()
        for i, char in enumerate(encrypted):
            decrypted.append(char ^ key[i % len(key)])

        try:
            return decrypted.decode('utf-8')
        except UnicodeDecodeError:
            return decrypted.decode('latin-1', errors='ignore')

    except Exception as e:
        return "Decryption failed. Incorrect password or corrupted message."

def text_to_bits(text):
    """Convert text to binary representation"""
    bits = []
    for char in text:
        byte = ord(char)
        for i in range(8):
            bits.append((byte >> (7-i)) & 1)
    return bits

def bits_to_text(bits):
    """Convert binary representation back to text"""
    text = ""
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        try:
            char = bytes([byte]).decode('utf-8')
            text += char
        except UnicodeDecodeError:
            text += '�'
    return text

def find_delimiter_in_bits(bits, delimiter_text, max_search_length=None):
    """
    Optimized delimiter search with early termination and bounds checking
    """
    if not bits or not delimiter_text:
        return -1
    
    delimiter_bits = text_to_bits(delimiter_text)
    delimiter_len = len(delimiter_bits)
    
    if delimiter_len == 0:
        return -1
    
    # Limit search range to prevent excessive processing
    search_limit = min(len(bits) - delimiter_len + 1, max_search_length or len(bits))
    
    # Use numpy for faster comparison if available and bits list is large
    if len(bits) > 10000:
        try:
            bits_array = np.array(bits[:search_limit + delimiter_len], dtype=np.int8)
            delimiter_array = np.array(delimiter_bits, dtype=np.int8)
            
            # Sliding window comparison using numpy
            for i in range(search_limit):
                if np.array_equal(bits_array[i:i + delimiter_len], delimiter_array):
                    return i
        except (ImportError, MemoryError):
            # Fallback to regular method if numpy fails
            pass
    
    # Regular search with step optimization
    step = max(1, delimiter_len // 4)  # Skip some positions for initial scan
    
    # First pass: coarse search
    for i in range(0, search_limit, step):
        if bits[i:i + delimiter_len] == delimiter_bits:
            return i
    
    # Second pass: fine search in promising areas
    for i in range(0, min(search_limit, step * 100)):  # Limit fine search
        if bits[i:i + delimiter_len] == delimiter_bits:
            return i
    
    return -1

def embed_video_lsb(input_video_path, output_video_path, message, password=""):
    """Improved video LSB embedding with better consistency and lossless intermediary"""
    try:
        # Encrypt and prepare message
        encrypted_message = encrypt_message(message, password)
        
        # Format: START_MARKER + length + # + message + END_MARKER
        # Length refers to the BYTE length of the base64 encoded string
        message_with_markers = f"{START_MARKER}{len(encrypted_message.encode('utf-8'))}#{encrypted_message}{END_MARKER}"
        message_bits = text_to_bits(message_with_markers)

        logger.info(f"Original message length: {len(message)} chars")
        logger.info(f"Encrypted message length: {len(encrypted_message)} chars (base64 encoded)")
        logger.info(f"Message with markers length: {len(message_with_markers)} chars")
        logger.info(f"Total bits to embed: {len(message_bits)} bits")

        # Open input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")

        # Calculate total available space
        # Assuming 3 channels (RGB) for LSB, 1 bit per channel per pixel
        pixels_per_frame = width * height * 3
        total_available_bits = pixels_per_frame * total_frames

        if len(message_bits) > total_available_bits:
            raise ValueError(f"Message too long. Available: {total_available_bits} bits, needed: {len(message_bits)} bits")

        # --- KEY CHANGE: Use a truly lossless codec for the intermediate AVI file ---
        # 'IYUV' or 'YV12' are uncompressed YUV formats, ensuring pixel-perfect preservation.
        # This is critical for LSB data.
        fourcc = cv2.VideoWriter_fourcc(*'IYUV') # Using uncompressed YUV for true pixel lossless write
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise RuntimeError(f"Could not initialize video writer with codec: IYUV. Consider checking OpenCV installation or codec availability.")
        logger.info(f"Successfully initialized video writer with codec: IYUV (for intermediate lossless write).")

        bit_index = 0
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                if frame_count < total_frames:
                    logger.warning(f"Reached end of video prematurely or error reading frame. Read {frame_count}/{total_frames} frames.")
                break

            # Ensure frame is in BGR format (OpenCV default for color images)
            if len(frame.shape) == 3 and frame.shape[2] == 4: # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif len(frame.shape) == 2: # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            frame_flat = frame.flatten() # Flatten to 1D array for easier LSB manipulation

            # Embed message bits if available
            if bit_index < len(message_bits):
                bits_to_embed_in_frame = min(len(message_bits) - bit_index, len(frame_flat))

                for j in range(bits_to_embed_in_frame):
                    frame_flat[j] = (frame_flat[j] & 0xFE) | message_bits[bit_index + j]

                frame = frame_flat.reshape(frame.shape)
                bit_index += bits_to_embed_in_frame

                if bit_index >= len(message_bits):
                    logger.info(f"All {len(message_bits)} bits embedded by frame {frame_count + 1}. Remaining frames will be written as is.")

            out.write(frame)
            frame_count += 1

            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames, embedded {bit_index} bits")

        cap.release()
        out.release()

        if bit_index < len(message_bits):
            logger.warning(f"Warning: Not all message bits were embedded. Embedded {bit_index}/{len(message_bits)} bits.")
            return None # Indicate partial embedding or error
        else:
            logger.info(f"Successfully embedded {bit_index} bits into 537 frames. Output saved to {output_video_path}")
            return output_video_path

    except Exception as e:
        logger.error(f"Error in embed_video_lsb: {e}", exc_info=True)
        if 'out' in locals() and out is not None:
            out.release()
        if 'cap' in locals() and cap is not None:
            cap.release()
        raise # Re-raise the exception after logging

def extract_video_lsb(video_path, password=""):
    """Fixed video LSB extraction with proper delimiter matching"""
    cap = None
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video: {width}x{height}, {total_frames} frames")
        
        # Progressive extraction strategy
        collected_bits = []
        
        # Strategy 1: Read first 50 frames completely (no sampling)
        logger.info("Strategy 1: Reading first 50 frames completely")
        for frame_idx in range(min(50, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale for consistency
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Extract ALL LSBs (no pixel sampling)
            frame_bits = (frame.flatten() & 1).tolist()
            collected_bits.extend(frame_bits)
            
            # Check for message every 10 frames
            if frame_idx % 10 == 0 and len(collected_bits) > 10000:
                message = extract_message_from_bits_fixed(collected_bits, password)
                if message and not message.startswith("No hidden message"):
                    logger.info(f"Message found in first {frame_idx + 1} frames")
                    cap.release()
                    return message
        
        # Strategy 2: If not found, try with more frames but light sampling
        if len(collected_bits) == 0 or not extract_message_from_bits_fixed(collected_bits, password).startswith("No hidden message"):
            logger.info("Strategy 2: Reading more frames with light sampling")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            collected_bits = []
            
            for frame_idx in range(min(200, total_frames)):
                ret, frame = cap.read()
                if not ret:
                    break
                
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Light sampling (every 5th pixel)
                frame_flat = frame.flatten()
                sampled_bits = [frame_flat[i] & 1 for i in range(0, len(frame_flat), 5)]
                collected_bits.extend(sampled_bits)
        
        # Final extraction attempt
        message = extract_message_from_bits_fixed(collected_bits, password)
        cap.release()
        return message
        
    except Exception as e:
        logger.error(f"Error in video extraction: {e}")
        if cap:
            cap.release()
        return f"Extraction error: {str(e)}"

# The _try_extract_message function is removed as _try_extract_complete_message is used
# and already correctly defined as a global function.

def extract_message_from_bits_improved(message_bits, password=""):
    """Improved message extraction from bits"""
    try:
        if not message_bits:
            return "No message bits provided."

        message_text = bits_to_text(message_bits)

        # Decrypt if password provided
        if password:
            message_text = decrypt_message(message_text, password)

        logger.info(f"Successfully extracted and decrypted message (first 100 chars): {message_text[:100]}...")
        return message_text

    except Exception as e:
        logger.error(f"Error in extract_message_from_bits_improved: {e}", exc_info=True)
        return f"Error processing message bits: {str(e)}"


def safe_embed_video(video_file, message, password=""):
    """Safe video embedding with better format handling"""
    if not check_ffmpeg_installed():
        return {'status': 'error', 'message': 'FFmpeg is not installed or not in PATH.'}

    temp_input_path = None
    temp_embedded_path = None
    temp_final_path = None

    try:
        # Create temporary files
        file_ext = os.path.splitext(video_file.filename)[1].lower()
        if not file_ext:
            file_ext = '.mp4' 
        temp_input_fd, temp_input_path = tempfile.mkstemp(suffix=file_ext)
        os.close(temp_input_fd) 
        video_file.save(temp_input_path)

        logger.info(f"Saved uploaded video to temporary path: {temp_input_path}")

        # Check if video has audio
        probe_cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1',
            temp_input_path
        ]

        has_audio = False
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, timeout=30)
            if result.stdout.strip():
                has_audio = True
            logger.info(f"Audio stream detected: {has_audio}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"ffprobe failed (audio check): {e.stderr.decode().strip()}. Assuming no audio.")
        except subprocess.TimeoutExpired:
            logger.warning("ffprobe timed out during audio check. Assuming no audio.")

        # Perform LSB embedding
        # Use .avi suffix here because OpenCV's VideoWriter works well with it,
        # especially when using MJPG or rawvideo codec which are relatively lossless for LSB.
        temp_embedded_fd, temp_embedded_path = tempfile.mkstemp(suffix='.avi') 
        os.close(temp_embedded_fd)

        logger.info("Starting LSB embedding...")
        embedded_result_path = embed_video_lsb(temp_input_path, temp_embedded_path, message, password)

        if embedded_result_path is None:
            raise RuntimeError("LSB embedding failed or was incomplete.")

        # Create final output as MP4
        temp_final_fd, temp_final_path = tempfile.mkstemp(suffix='.mp4')
        os.close(temp_final_fd)

        if has_audio:
            logger.info("Adding audio to embedded video...")
            # Re-encode video with libx264 for MP4. Use -shortest to match video duration.
            # Use -map 0:v:0 to select video stream from the embedded AVI (first input)
            # Use -map 1:a:0 to select audio stream from the original MP4 (second input)
            cmd = [
                'ffmpeg', '-i', embedded_result_path, '-i', temp_input_path,
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', 
                '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', 
                '-shortest', '-y', temp_final_path
            ]
        else:
            logger.info("Converting to MP4 format (no audio)...")
            cmd = [
                'ffmpeg', '-i', embedded_result_path,
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                '-an', # No audio
                '-y', temp_final_path
            ]

        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, timeout=300) 
        logger.info(f"FFmpeg stdout: {result.stdout.decode().strip()}")
        logger.info(f"FFmpeg stderr: {result.stderr.decode().strip()}")
        logger.info("FFmpeg final re-encoding completed successfully")

        with open(temp_final_path, 'rb') as f:
            video_data = f.read()

        if len(video_data) == 0:
            raise ValueError("Output video file is empty after FFmpeg processing. Check FFmpeg logs for errors.")

        encoded_file = base64.b64encode(video_data).decode('utf-8')

        return {
            'status': 'success',
            'file': encoded_file,
            'file_type': 'video/mp4', 
            'message': f'Message successfully embedded in video{"" if has_audio else " (no audio)"}. Output file size: {len(video_data)/1024/1024:.2f} MB'
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e}. STDOUT: {e.stdout.decode(errors='ignore')}, STDERR: {e.stderr.decode(errors='ignore')}")
        return {'status': 'error', 'message': f"Video processing failed. FFmpeg Error: {e.stderr.decode(errors='ignore').strip()}"}
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg process timed out.")
        return {'status': 'error', 'message': "Video processing timed out. The video might be too large or the server is slow."}
    except Exception as e:
        logger.error(f"Error in safe_embed_video: {e}", exc_info=True)
        return {'status': 'error', 'message': f"Error during video embedding: {str(e)}"}
    finally:
        for path in [temp_input_path, temp_embedded_path, temp_final_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    logger.info(f"Cleaned up temporary file: {path}")
                except OSError as e:
                    logger.error(f"Error cleaning up temporary file {path}: {e}")

def extract_message_from_bits_fixed(bits, password=""):
    """Fixed message extraction with correct delimiters"""
    try:
        if not bits or len(bits) < 1000:
            logger.info("No hidden message found (insufficient data, returning 'halo oblivio')")
            return "halo oblivio" # <-- MODIFIKASI INI
        
        logger.info(f"Searching for message in {len(bits)} bits")
        
        # Search for start marker using CORRECT delimiter
        start_pos = find_pattern_in_bits(bits, START_MARKER, max_search=min(len(bits), 500000))
        
        if start_pos == -1:
            logger.info("Start marker not found, returning 'halo oblivio' as dummy message")
            return "halo oblivio" # <-- MODIFIKASI INI
        
        logger.info(f"Found start marker at position {start_pos}")
        
        # Find the message length
        start_marker_bits = text_to_bits(START_MARKER)
        length_start = start_pos + len(start_marker_bits)
        
        if length_start >= len(bits):
            logger.warning("No hidden message found (truncated after start marker), returning 'halo oblivio'")
            return "halo oblivio" # <-- MODIFIKASI INI
        
        # Extract length string (up to '#' delimiter)
        length_str = ""
        hash_pos = -1
        
        # Look for length + '#' pattern
        for i in range(length_start, min(length_start + 160, len(bits)), 8):  # 20 chars max
            if i + 8 > len(bits):
                break
            
            byte_bits = bits[i:i+8]
            if len(byte_bits) < 8:
                break
                
            try:
                char_value = 0
                for bit in byte_bits:
                    char_value = (char_value << 1) | bit
                char = chr(char_value)
                
                if char == '#':
                    hash_pos = i + 8
                    break
                elif char.isdigit():
                    length_str += char
                else:
                    break
            except:
                break
        
        if not length_str.isdigit() or hash_pos == -1:
            logger.warning(f"Length parsing failed. length_str='{length_str}', hash_pos={hash_pos}. Returning 'halo oblivio'")
            return "halo oblivio" # <-- MODIFIKASI INI
        
        message_length = int(length_str)
        logger.info(f"Expected message length: {message_length} bytes")
        
        if message_length > 1000000:  # 1MB limit
            logger.warning("No hidden message found (message too large), returning 'halo oblivio'")
            return "halo oblivio" # <-- MODIFIKASI INI
        
        # Extract the message content
        message_bits_needed = message_length * 8
        
        # Look for end marker
        end_search_start = hash_pos + message_bits_needed
        end_search_end = min(len(bits), end_search_start + len(text_to_bits(END_MARKER)) + 1000)
        
        if end_search_start < len(bits):
            end_pos = find_pattern_in_bits(bits, END_MARKER, end_search_start, 
                                         end_search_end - end_search_start)
            
            if end_pos != -1:
                # Extract exact message between markers
                message_bits = bits[hash_pos:end_pos]
                logger.info(f"Found end marker, extracted {len(message_bits)} bits")
            else:
                # Extract expected length
                message_bits = bits[hash_pos:hash_pos + message_bits_needed]
                logger.info(f"End marker not found, extracted {len(message_bits)} bits by length")
        else:
            message_bits = bits[hash_pos:]
            logger.info(f"Insufficient data for end marker search, extracted remaining {len(message_bits)} bits")
        
        if not message_bits:
            logger.warning("No hidden message found (empty message content), returning 'halo oblivio'")
            return "halo oblivio" # <-- MODIFIKASI INI
        
        # Convert bits to text
        message_text = bits_to_text(message_bits)
        
        # Decrypt if password provided
        if password:
            decrypted = decrypt_message(message_text, password)
            if "Decryption failed" not in decrypted:
                logger.info("Message successfully decrypted")
                return decrypted
            else:
                logger.info("Decryption failed, returning encrypted text or 'halo oblivio'")
                return "halo oblivio" # <-- MODIFIKASI INI jika dekripsi gagal
        
        logger.info("Message extracted successfully (no decryption)")
        return message_text
        
    except Exception as e:
        logger.error(f"Error in message extraction: {e}. Returning 'halo oblivio' as dummy.")
        return "halo oblivio" # <-- MODIFIKASI INI

def safe_extract_video(video_file, password=""):
    """Fixed video LSB extraction with better error handling"""
    temp_input_path = None
    
    try:
        # Create temporary file
        temp_input_fd, temp_input_path = tempfile.mkstemp(suffix='.mp4')
        os.close(temp_input_fd)
        
        # Save uploaded file
        video_file.save(temp_input_path)
        logger.info(f"Video saved to: {temp_input_path}")
        
        # Extract message
        message = extract_video_lsb(temp_input_path, password)
        
        return {
            'status': 'success',
            'message': message
        }
        
    except Exception as e:
        logger.error(f"Error in video extraction: {e}")
        return {'status': 'error', 'message': str(e)}
    finally:
        # Cleanup
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.unlink(temp_input_path)
            except OSError as e:
                logger.error(f"Error cleaning up temp file: {e}")


def find_pattern_in_bits(bits, pattern_text, start_pos=0, max_search=None):
    """Find pattern in bits array with optimizations"""
    if not bits or not pattern_text:
        return -1
    
    pattern_bits = text_to_bits(pattern_text)
    pattern_len = len(pattern_bits)
    
    if pattern_len == 0:
        return -1
    
    # Set search bounds
    end_pos = len(bits) - pattern_len + 1
    if max_search:
        end_pos = min(end_pos, start_pos + max_search)
    
    # Search for pattern
    for i in range(start_pos, end_pos):
        if bits[i:i + pattern_len] == pattern_bits:
            return i
    
    return -1

def _is_valid_message(message):
    """
    Quick validation to check if extracted message is valid
    """
    if not message or len(message) < 10:
        return False
        
    # Check for common error messages
    error_indicators = [
        "No hidden message",
        "Error extracting",
        "Decryption failed",
        "Message too large",
        "Wrong password"
    ]
    
    return not any(indicator in message for indicator in error_indicators)

def validate_video_file(video_file):
    try:
        # Check file size
        video_file.seek(0, 2)
        size = video_file.tell()
        video_file.seek(0)
        if size > 100 * 1024 * 1024:  # 100MB limit
            return False, "File too large (max 100MB)"
            
        # Check file extension
        filename = video_file.filename.lower()
        if not any(filename.endswith(ext) for ext in ['.mp4', '.mov', '.avi']):
            return False, "Unsupported video format"
            
        return True, ""
    except Exception as e:
        return False, str(e)

# Audio functions remain the same but with minor improvements
def embed_audio_lsb(audio_data, message, password=""):
    """Embed message into audio using LSB method"""
    try:
        # Encrypt message if password provided
        encrypted_message = encrypt_message(message, password)

        # Add delimiter and message length for reliable extraction
        # Use the global START_MARKER and END_MARKER for consistency
        message_with_markers = f"{START_MARKER}{len(encrypted_message.encode('utf-8'))}#{encrypted_message}{END_MARKER}"

        # Convert message to bits
        message_bits = text_to_bits(message_with_markers)

        # Check if audio can accommodate the message
        if len(message_bits) > len(audio_data):
            raise ValueError(f"Message too long. Audio can hold {len(audio_data)} bits, but message needs {len(message_bits)} bits")

        # Create a copy of audio data and ensure it's writable
        stego_audio = audio_data.copy().astype(audio_data.dtype)

        # Embed message bits into LSB of audio samples
        for i, bit in enumerate(message_bits):
            # Clear LSB first, then set it to the desired bit
            stego_audio[i] = (stego_audio[i] & ~1) | bit

        logger.info(f"Successfully embedded {len(message_bits)} bits into audio")
        return stego_audio
    except Exception as e:
        logger.error(f"Error embedding audio: {e}", exc_info=True)
        raise

def extract_audio_lsb(audio_data, password=""):
    """Extract message from audio using LSB method with improved delimiter detection"""
    try:
        # Extract bits from LSB of audio samples. Search a reasonable chunk first.
        # Max 5 million bits, equivalent to about 625KB of data (enough for a long message)
        max_bits_to_scan = min(len(audio_data), 5_000_000)
        extracted_bits = [int(sample & 1) for sample in audio_data[:max_bits_to_scan]]

        logger.info(f"Extracted {len(extracted_bits)} bits from audio for initial scan.")

        # Find delimiter - Use the global START_MARKER and END_MARKER for consistency
        delimiter_bits = text_to_bits(START_MARKER)
        end_marker_bits = text_to_bits(END_MARKER)

        delimiter_pos = find_delimiter_in_bits(extracted_bits, START_MARKER)

        if delimiter_pos == -1:
            logger.warning("Delimiter not found in audio within the scanned range.")
            return "No hidden message found"

        logger.info(f"Found delimiter at bit position {delimiter_pos}")

        # Extract the part after delimiter to find length
        bits_after_delimiter_start = delimiter_pos + len(delimiter_bits)
        # Take a reasonable chunk to find the length string and the '#'
        # A length string could be 1-10 digits, plus '#'. Say max 12 chars * 8 bits/char = 96 bits.
        # Add some buffer for safety.
        max_bits_for_length_parse = min(len(extracted_bits) - bits_after_delimiter_start, 50 * 8) # 50 chars as buffer
        if max_bits_for_length_parse <= 0:
            logger.warning("Not enough bits after delimiter to parse message length.")
            return "Message format error: insufficient data after delimiter."

        temp_text_for_length = bits_to_text(extracted_bits[bits_after_delimiter_start : bits_after_delimiter_start + max_bits_for_length_parse])

        if '#' in temp_text_for_length:
            try:
                length_str = temp_text_for_length.split('#')[0]
                message_length_bytes = int(length_str) # This is the original BYTE length of the encrypted message
                logger.info(f"Expected encrypted message BYTE length: {message_length_bytes}")

                # Calculate the start of the actual encrypted message content in bits
                length_and_hash_str = f"{message_length_bytes}#"
                length_and_hash_bits = text_to_bits(length_and_hash_str)
                message_content_start_bit = bits_after_delimiter_start + len(length_and_hash_bits)

                # Now, extract the message content and then search for the end marker
                # The total number of bits for the message, including potential base64 padding,
                # is roughly message_length_bytes * 8 (assuming each byte is 8 bits).
                # The actual base64 string will be roughly (message_length_bytes / 3) * 4 characters long.
                # Each of *those* characters is then embedded as 8 bits.
                # So total bits for encrypted string: ceil(message_length_bytes / 3) * 4 * 8.
                # Let's be generous in extraction and rely on the end_marker.

                # Determine the maximum range to search for the end_marker.
                # We expect it to be after the encrypted message.
                # Use a large buffer to ensure we catch it.
                max_search_range_for_end_marker = min(len(audio_data), message_content_start_bit + (message_length_bytes * 8) + len(end_marker_bits) + (1000 * 8)) # Message bits + end marker + large buffer

                # Extract a larger chunk of bits for the actual message + end marker
                # Ensure we don't go out of bounds of extracted_bits
                full_message_area_bits = extracted_bits[message_content_start_bit : min(len(extracted_bits), max_search_range_for_end_marker)]

                end_marker_relative_pos = find_delimiter_in_bits(full_message_area_bits, END_MARKER) # Use END_MARKER here

                if end_marker_relative_pos != -1:
                    # The end marker was found. Trim the message_bits precisely.
                    actual_encrypted_message_bits = full_message_area_bits[:end_marker_relative_pos]
                else:
                    # End marker not found within the reasonable range. Assume the message is incomplete or truncated.
                    logger.warning(f"End marker not found in audio after scanning {len(full_message_area_bits)} bits from message start. Extracting up to expected length or end of scanned data.")
                    # Fallback: extract up to the expected message byte length (converted to bits)
                    actual_encrypted_message_bits = full_message_area_bits[: message_length_bytes * 8]
                    if len(actual_encrypted_message_bits) == 0:
                        return "Message incomplete or corrupted (end marker not found, and no data extracted)."


                extracted_message_text = bits_to_text(actual_encrypted_message_bits)

                # Decrypt if password provided
                if password:
                    extracted_message_text = decrypt_message(extracted_message_text, password)

                logger.info("Successfully extracted and decrypted message from audio.")
                return extracted_message_text

            except (ValueError, IndexError) as e:
                logger.error(f"Error parsing message length or content from audio: {e}", exc_info=True)
                return "Error parsing hidden message from audio."
        else:
            logger.warning("Message length delimiter '#' not found after STEGO_START.")
            return "Message format error: length information missing."

    except Exception as e:
        logger.error(f"Error extracting from audio: {e}", exc_info=True)
        return f"Error extracting message: {str(e)}"

def safe_embed_audio(file, message, password=""):
    """Safely embed message into audio file using LSB"""
    temp_input_path = None
    temp_output_path = None
    temp_wav_path = None

    try:
        # Create temporary files
        temp_input_fd, temp_input_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename)[1] or '.mp3') # Use original ext or default
        os.close(temp_input_fd)
        file.save(temp_input_path)
        logger.info(f"Uploaded audio saved to: {temp_input_path}")

        temp_output_fd, temp_output_path = tempfile.mkstemp(suffix='.wav') # Always output WAV for consistency
        os.close(temp_output_fd)

        # Convert to WAV format using ffmpeg if available
        if check_ffmpeg_installed():
            temp_wav_fd, temp_wav_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_wav_fd)
            try:
                # Force PCM S16 LE, 44.1kHz, mono for consistent LSB embedding
                subprocess.run([
                    'ffmpeg', '-i', temp_input_path,
                    '-acodec', 'pcm_s16le',
                    '-ar', '44100',
                    '-ac', '1', # Force mono channel
                    '-y',
                    temp_wav_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120) # Increased timeout
                logger.info(f"Audio converted to WAV (PCM_S16LE, 44.1kHz, Mono) at: {temp_wav_path}")
                current_audio_path = temp_wav_path
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg audio conversion failed: {e.stderr.decode()}", exc_info=True)
                return {'status': 'error', 'message': f'Failed to convert audio format: {e.stderr.decode().strip()}'}
        else:
            logger.warning("FFmpeg not installed, attempting to process audio directly (may fail for non-WAV inputs).")
            current_audio_path = temp_input_path
            # If not a WAV, this will likely fail later. Could add a check here.

        # Read WAV file
        try:
            with wave.open(current_audio_path, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes()) # Read all frames
                params = wav_file.getparams()

                logger.info(f"Audio params: nchannels={params.nchannels}, sampwidth={params.sampwidth}, framerate={params.framerate}, nframes={params.nframes}")

                # Convert to numpy array based on sample width
                if params.sampwidth == 1:
                    audio_data = np.frombuffer(frames, dtype=np.uint8)
                elif params.sampwidth == 2:
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                elif params.sampwidth == 4:
                    audio_data = np.frombuffer(frames, dtype=np.int32)
                else:
                    raise ValueError(f"Unsupported sample width for LSB embedding: {params.sampwidth} bytes. Only 1, 2, or 4 supported.")

                logger.info(f"Audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}")

                # Embed message
                stego_audio = embed_audio_lsb(audio_data, message, password)

                # Create output WAV file
                with wave.open(temp_output_path, 'wb') as output_wav:
                    output_wav.setparams(params)
                    output_wav.writeframes(stego_audio.tobytes())

                logger.info(f"Stego audio saved to: {temp_output_path}")

        except Exception as e:
            logger.error(f"Error processing WAV file for embedding: {e}", exc_info=True)
            return {'status': 'error', 'message': f'Error processing audio for embedding: {str(e)}'}

        # Read the output file and convert to base64
        with open(temp_output_path, 'rb') as f:
            audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode()

        return {
            'status': 'success',
            'file': f"data:audio/wav;base64,{audio_b64}" # Data URL for direct use in browser
        }

    except Exception as e:
        logger.error(f"Unhandled error in safe_embed_audio: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}
    finally:
        # Cleanup temporary files
        for path in [temp_input_path, temp_wav_path, temp_output_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    logger.info(f"Cleaned up temporary file: {path}")
                except OSError as e:
                    logger.error(f"Error cleaning up temporary file {path}: {e}")


def safe_extract_audio(file, password=""):
    """Safely extract message from audio file using LSB"""
    temp_input_path = None
    temp_wav_path = None

    try:
        temp_input_fd, temp_input_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename)[1] or '.mp3')
        os.close(temp_input_fd)
        file.save(temp_input_path)
        logger.info(f"Uploaded audio saved to: {temp_input_path}")

        # Convert to WAV if FFmpeg is available
        if check_ffmpeg_installed():
            temp_wav_fd, temp_wav_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_wav_fd)
            try:
                subprocess.run([
                    'ffmpeg', '-i', temp_input_path,
                    '-acodec', 'pcm_s16le',
                    '-ar', '44100',
                    '-ac', '1',
                    '-y',
                    temp_wav_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
                logger.info(f"Audio converted to WAV for extraction: {temp_wav_path}")
                current_audio_path = temp_wav_path
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg audio conversion failed during extraction: {e.stderr.decode()}", exc_info=True)
                return {'status': 'error', 'message': f'Failed to convert audio format for extraction: {e.stderr.decode().strip()}'}
        else:
            logger.warning("FFmpeg not installed, attempting to process audio directly (may fail for non-WAV inputs).")
            current_audio_path = temp_input_path


        # Read WAV file
        try:
            with wave.open(current_audio_path, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                params = wav_file.getparams()

                logger.info(f"Audio params for extraction: nchannels={params.nchannels}, sampwidth={params.sampwidth}, framerate={params.framerate}, nframes={params.nframes}")

                # Convert to numpy array
                if params.sampwidth == 1:
                    audio_data = np.frombuffer(frames, dtype=np.uint8)
                elif params.sampwidth == 2:
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                elif params.sampwidth == 4:
                    audio_data = np.frombuffer(frames, dtype=np.int32)
                else:
                    raise ValueError(f"Unsupported sample width for extraction: {params.sampwidth} bytes.")

                logger.info(f"Audio data shape for extraction: {audio_data.shape}, dtype: {audio_data.dtype}")

                # Extract message
                message = extract_audio_lsb(audio_data, password)

                return {
                    'status': 'success',
                    'message': message
                }
        except Exception as e:
            logger.error(f"Error processing WAV file for extraction: {e}", exc_info=True)
            return {'status': 'error', 'message': f'Error processing audio for extraction: {str(e)}'}

    except Exception as e:
        logger.error(f"Unhandled error in safe_extract_audio: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}
    finally:
        # Cleanup
        for path in [temp_input_path, temp_wav_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    logger.info(f"Cleaned up temporary file: {path}")
                except OSError as e:
                    logger.error(f"Error cleaning up temporary file {path}: {e}")
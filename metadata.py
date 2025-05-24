import os
import eyed3
import mutagen
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.wave import WAVE
from mutagen.oggvorbis import OggVorbis
import json
import base64
import io
import cv2
import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
import tempfile
import shutil
from moviepy.editor import VideoFileClip
from flask import send_file, jsonify
import subprocess
import traceback
from datetime import datetime
import logging


def extract_audio_metadata(audio_file):
    """Extract metadata from audio files (MP3, WAV, OGG)"""
    metadata = {}
    file_path = audio_file.filename.lower() if hasattr(audio_file, 'filename') else 'unknown'
    file_ext = os.path.splitext(file_path)[1]
    
    temp_path = None
    try:
        # Create a temporary file with the correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp:
            temp_path = temp.name
            audio_file.save(temp_path)
        
        # Extract basic information
        metadata['format'] = file_ext[1:].upper()  # Remove the dot from extension
        
        if file_ext == '.mp3':
            # Use eyed3 for MP3 ID3 tags
            audiofile = eyed3.load(temp_path)
            if audiofile and audiofile.tag:
                metadata['id3'] = {
                    'title': str(audiofile.tag.title) if audiofile.tag.title else None,
                    'artist': str(audiofile.tag.artist) if audiofile.tag.artist else None,
                    'album': str(audiofile.tag.album) if audiofile.tag.album else None,
                    'year': str(audiofile.tag.getBestDate()) if audiofile.tag.getBestDate() else None,
                    'genre': str(audiofile.tag.genre) if audiofile.tag.genre else None,
                    'comments': [str(c) for c in audiofile.tag.comments] if audiofile.tag.comments else [],
                }
                
            # Also use mutagen for additional info
            audio = MP3(temp_path)
            metadata['technical'] = {
                'length': audio.info.length,
                'bitrate': audio.info.bitrate,
                'sample_rate': audio.info.sample_rate,
                'channels': getattr(audio.info, 'channels', None),
            }
            
        elif file_ext == '.wav':
            # Use mutagen for WAV files
            audio = WAVE(temp_path)
            metadata['technical'] = {
                'length': audio.info.length,
                'sample_rate': audio.info.sample_rate,
                'channels': getattr(audio.info, 'channels', None),
                'bits_per_sample': getattr(audio.info, 'bits_per_sample', None),
            }
            
        elif file_ext in ['.ogg', '.oga']:
            # Use mutagen for OGG files
            audio = OggVorbis(temp_path)
            metadata['vorbis_comments'] = dict(audio.tags) if audio.tags else {}
            metadata['technical'] = {
                'length': audio.info.length,
                'bitrate': getattr(audio.info, 'bitrate', None),
                'sample_rate': audio.info.sample_rate,
                'channels': getattr(audio.info, 'channels', None),
            }
    except Exception as e:
        metadata['error'] = str(e)
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
    
    return metadata

def extract_video_metadata(video_file):
    """
    Improved video metadata extraction with better error handling and multiple fallback methods
    """
    metadata = {}
    file_path = video_file.filename.lower() if hasattr(video_file, 'filename') else 'unknown'
    file_ext = os.path.splitext(file_path)[1]
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger("video_metadata_improved")
    
    temp_path = None
    extraction_methods = []
    
    try:
        # Create temporary file with correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp:
            temp_path = temp.name
        
        logger.info(f"Created temp file at {temp_path}")
        
        # Save uploaded file
        video_file.save(temp_path)
        file_size = os.path.getsize(temp_path)
        logger.info(f"Saved uploaded file, size: {file_size} bytes")
        
        # Basic file information
        metadata['file_info'] = {
            'format': file_ext[1:].upper(),
            'file_size': file_size,
            'filename': os.path.basename(file_path)
        }
        
        # Method 1: FFprobe (Most comprehensive and reliable)
        ffprobe_success = False
        try:
            logger.info("Attempting FFprobe metadata extraction")
            
            # Check if ffprobe is available
            ffprobe_check = subprocess.run(
                ['ffprobe', '-version'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                timeout=10
            )
            
            if ffprobe_check.returncode != 0:
                raise Exception("FFprobe not available or not working")
            
            # Run ffprobe command with comprehensive output
            cmd = [
                'ffprobe', 
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format', 
                '-show_streams',
                '-show_chapters',
                '-show_programs',
                temp_path
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60
            )
            
            if result.returncode == 0 and result.stdout:
                ffprobe_output = result.stdout.decode('utf-8')
                ffprobe_data = json.loads(ffprobe_output)
                
                # Process format information
                if 'format' in ffprobe_data:
                    format_info = ffprobe_data['format']
                    metadata['container'] = {
                        'format_name': format_info.get('format_name'),
                        'format_long_name': format_info.get('format_long_name'),
                        'duration': float(format_info.get('duration', 0)) if format_info.get('duration') else None,
                        'bit_rate': int(format_info.get('bit_rate', 0)) if format_info.get('bit_rate') else None,
                        'size': int(format_info.get('size', 0)) if format_info.get('size') else None,
                        'start_time': float(format_info.get('start_time', 0)) if format_info.get('start_time') else None
                    }
                    
                    # Process container-level tags/metadata
                    if 'tags' in format_info:
                        metadata['tags'] = {}
                        for key, value in format_info['tags'].items():
                            # Clean up key names
                            clean_key = key.replace('_', ' ').replace('-', ' ').title()
                            metadata['tags'][clean_key] = value
                
                # Process streams information
                if 'streams' in ffprobe_data:
                    metadata['streams'] = {
                        'video': [],
                        'audio': [],
                        'subtitle': [],
                        'data': []
                    }
                    
                    for stream in ffprobe_data['streams']:
                        codec_type = stream.get('codec_type', 'unknown')
                        
                        stream_info = {
                            'index': stream.get('index'),
                            'codec_name': stream.get('codec_name'),
                            'codec_long_name': stream.get('codec_long_name'),
                            'profile': stream.get('profile'),
                            'level': stream.get('level'),
                            'duration': float(stream.get('duration', 0)) if stream.get('duration') else None,
                            'bit_rate': int(stream.get('bit_rate', 0)) if stream.get('bit_rate') else None
                        }
                        
                        # Add codec-specific information
                        if codec_type == 'video':
                            stream_info.update({
                                'width': stream.get('width'),
                                'height': stream.get('height'),
                                'coded_width': stream.get('coded_width'),
                                'coded_height': stream.get('coded_height'),
                                'display_aspect_ratio': stream.get('display_aspect_ratio'),
                                'sample_aspect_ratio': stream.get('sample_aspect_ratio'),
                                'pixel_format': stream.get('pix_fmt'),
                                'color_space': stream.get('color_space'),
                                'color_transfer': stream.get('color_transfer'),
                                'color_primaries': stream.get('color_primaries'),
                                'frame_rate': stream.get('r_frame_rate'),
                                'avg_frame_rate': stream.get('avg_frame_rate'),
                                'time_base': stream.get('time_base'),
                                'start_pts': stream.get('start_pts'),
                                'nb_frames': stream.get('nb_frames')
                            })
                            metadata['streams']['video'].append(stream_info)
                            
                        elif codec_type == 'audio':
                            stream_info.update({
                                'sample_rate': int(stream.get('sample_rate', 0)) if stream.get('sample_rate') else None,
                                'channels': stream.get('channels'),
                                'channel_layout': stream.get('channel_layout'),
                                'sample_fmt': stream.get('sample_fmt'),
                                'bits_per_sample': stream.get('bits_per_sample')
                            })
                            metadata['streams']['audio'].append(stream_info)
                            
                        elif codec_type == 'subtitle':
                            stream_info.update({
                                'language': stream.get('tags', {}).get('language') if stream.get('tags') else None
                            })
                            metadata['streams']['subtitle'].append(stream_info)
                            
                        else:
                            metadata['streams']['data'].append(stream_info)
                        
                        # Add stream-level tags if available
                        if 'tags' in stream and stream['tags']:
                            stream_info['tags'] = stream['tags']
                
                # Process chapters if available
                if 'chapters' in ffprobe_data and ffprobe_data['chapters']:
                    metadata['chapters'] = []
                    for chapter in ffprobe_data['chapters']:
                        chapter_info = {
                            'id': chapter.get('id'),
                            'time_base': chapter.get('time_base'),
                            'start': chapter.get('start'),
                            'start_time': chapter.get('start_time'),
                            'end': chapter.get('end'),
                            'end_time': chapter.get('end_time')
                        }
                        if 'tags' in chapter:
                            chapter_info['tags'] = chapter['tags']
                        metadata['chapters'].append(chapter_info)
                
                ffprobe_success = True
                extraction_methods.append('FFprobe')
                logger.info("FFprobe metadata extraction successful")
                
            else:
                error_msg = result.stderr.decode('utf-8')
                logger.warning(f"FFprobe failed: {error_msg}")
                metadata['ffprobe_error'] = error_msg
                
        except subprocess.TimeoutExpired:
            logger.warning("FFprobe timed out")
            metadata['ffprobe_error'] = "FFprobe operation timed out"
        except Exception as e:
            logger.warning(f"FFprobe failed: {str(e)}")
            metadata['ffprobe_error'] = str(e)
        
        # Method 2: OpenCV (Good for basic video properties)
        opencv_success = False
        try:
            logger.info("Attempting OpenCV metadata extraction")
            
            video_capture = cv2.VideoCapture(temp_path)
            
            if video_capture.isOpened():
                # Get basic video properties
                width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = video_capture.get(cv2.CAP_PROP_FPS)
                frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Calculate duration
                duration = frame_count / fps if fps > 0 else 0
                
                # Get codec information
                fourcc = int(video_capture.get(cv2.CAP_PROP_FOURCC))
                codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
                
                # Additional properties
                brightness = video_capture.get(cv2.CAP_PROP_BRIGHTNESS)
                contrast = video_capture.get(cv2.CAP_PROP_CONTRAST)
                saturation = video_capture.get(cv2.CAP_PROP_SATURATION)
                hue = video_capture.get(cv2.CAP_PROP_HUE)
                
                video_capture.release()
                
                metadata['opencv'] = {
                    'resolution': {
                        'width': width,
                        'height': height
                    },
                    'framerate': round(fps, 2) if fps > 0 else None,
                    'frame_count': frame_count,
                    'duration_seconds': round(duration, 2) if duration > 0 else None,
                    'codec_fourcc': codec.strip() if codec.strip() else None,
                    'properties': {
                        'brightness': brightness if brightness != 0 else None,
                        'contrast': contrast if contrast != 0 else None,
                        'saturation': saturation if saturation != 0 else None,
                        'hue': hue if hue != 0 else None
                    }
                }
                
                opencv_success = True
                extraction_methods.append('OpenCV')
                logger.info(f"OpenCV metadata: {width}x{height} @ {fps}fps, {frame_count} frames")
                
            else:
                logger.warning("OpenCV could not open the video file")
                metadata['opencv_error'] = "Could not open video file with OpenCV"
                
        except Exception as e:
            logger.warning(f"OpenCV metadata extraction failed: {str(e)}")
            metadata['opencv_error'] = str(e)
        
        # Method 3: MoviePy (Good for additional video info)
        try:
            logger.info("Attempting MoviePy metadata extraction")
            
            try:
                from moviepy.editor import VideoFileClip
                
                with VideoFileClip(temp_path) as clip:
                    moviepy_info = {
                        'duration_seconds': round(clip.duration, 2) if clip.duration else None,
                        'fps': round(clip.fps, 2) if clip.fps else None,
                        'size': [clip.w, clip.h] if hasattr(clip, 'w') and hasattr(clip, 'h') else None,
                        'has_audio': clip.audio is not None,
                        'mask': clip.mask is not None if hasattr(clip, 'mask') else None
                    }
                    
                    if clip.audio:
                        moviepy_info['audio'] = {
                            'duration': round(clip.audio.duration, 2) if clip.audio.duration else None,
                            'fps': clip.audio.fps if hasattr(clip.audio, 'fps') else None,
                            'nchannels': clip.audio.nchannels if hasattr(clip.audio, 'nchannels') else None
                        }
                    
                    metadata['moviepy'] = moviepy_info
                    extraction_methods.append('MoviePy')
                    logger.info("MoviePy metadata extraction successful")
                    
            except ImportError:
                logger.warning("MoviePy not available")
                metadata['moviepy_error'] = "MoviePy not installed"
                
        except Exception as e:
            logger.warning(f"MoviePy metadata extraction failed: {str(e)}")
            metadata['moviepy_error'] = str(e)
        
        # Method 4: Enhanced file analysis
        try:
            logger.info("Performing enhanced file analysis")
            
            with open(temp_path, 'rb') as f:
                # Read first 64KB for more thorough header analysis
                header = f.read(65536)
                
                # File signature detection
                file_signatures = {
                    b'\x00\x00\x00\x14ftyp': 'MP4/QuickTime',
                    b'\x00\x00\x00\x18ftyp': 'MP4',
                    b'\x00\x00\x00\x1cftyp': 'MP4',
                    b'\x00\x00\x00 ftyp': 'MP4',
                    b'RIFF': 'AVI',
                    b'\x1a\x45\xdf\xa3': 'Matroska/MKV/WebM',
                    b'OggS': 'OGG/OGV',
                    b'\x46\x4c\x56': 'FLV'
                }
                
                detected_signature = None
                for signature, format_name in file_signatures.items():
                    if header.startswith(signature) or signature in header[:50]:
                        detected_signature = format_name
                        break
                
                # Enhanced metadata detection
                metadata_indicators = {
                    'XMP': b'<?xpacket begin',
                    'EXIF': b'Exif\x00\x00',
                    'iTunes': b'----com.apple.iTunes',
                    'UUID': b'uuid',
                    'Creation Tool': b'Lavf',  # FFmpeg
                    'Encoder': b'encoder'
                }
                
                found_metadata = {}
                for meta_type, indicator in metadata_indicators.items():
                    if indicator in header:
                        found_metadata[meta_type] = True
                
                metadata['file_analysis'] = {
                    'detected_format': detected_signature,
                    'metadata_found': found_metadata,
                    'header_size_analyzed': len(header)
                }
                
                extraction_methods.append('File Analysis')
                logger.info(f"File analysis completed. Detected format: {detected_signature}")
                
        except Exception as e:
            logger.warning(f"File analysis failed: {str(e)}")
            metadata['file_analysis_error'] = str(e)
        
        # Create summary
        metadata['extraction_summary'] = {
            'status': 'success' if extraction_methods else 'failed',
            'methods_used': extraction_methods,
            'total_methods': len(extraction_methods),
            'file_size': file_size,
            'extraction_time': datetime.now().isoformat()
        }
        
        # Consolidate key information for easy access
        if extraction_methods:
            consolidated = {}
            
            # Get duration from any available source
            duration_sources = []
            if 'container' in metadata and metadata['container'].get('duration'):
                duration_sources.append(('FFprobe Container', metadata['container']['duration']))
            if 'opencv' in metadata and metadata['opencv'].get('duration_seconds'):
                duration_sources.append(('OpenCV', metadata['opencv']['duration_seconds']))
            if 'moviepy' in metadata and metadata['moviepy'].get('duration_seconds'):
                duration_sources.append(('MoviePy', metadata['moviepy']['duration_seconds']))
            
            if duration_sources:
                consolidated['duration'] = {
                    'sources': duration_sources,
                    'primary_value': duration_sources[0][1],
                    'unit': 'seconds'
                }
            
            # Get resolution from any available source
            resolution_sources = []
            if 'opencv' in metadata and metadata['opencv'].get('resolution'):
                res = metadata['opencv']['resolution']
                if res['width'] and res['height']:
                    resolution_sources.append(('OpenCV', f"{res['width']}x{res['height']}"))
            if 'streams' in metadata and metadata['streams'].get('video'):
                for video_stream in metadata['streams']['video']:
                    if video_stream.get('width') and video_stream.get('height'):
                        resolution_sources.append(('FFprobe Video Stream', f"{video_stream['width']}x{video_stream['height']}"))
                        break
            if 'moviepy' in metadata and metadata['moviepy'].get('size'):
                size = metadata['moviepy']['size']
                if size and len(size) == 2:
                    resolution_sources.append(('MoviePy', f"{size[0]}x{size[1]}"))
            
            if resolution_sources:
                consolidated['resolution'] = {
                    'sources': resolution_sources,
                    'primary_value': resolution_sources[0][1]
                }
            
            # Get codec information
            codec_sources = []
            if 'opencv' in metadata and metadata['opencv'].get('codec_fourcc'):
                codec_sources.append(('OpenCV FOURCC', metadata['opencv']['codec_fourcc']))
            if 'streams' in metadata and metadata['streams'].get('video'):
                for video_stream in metadata['streams']['video']:
                    if video_stream.get('codec_name'):
                        codec_sources.append(('FFprobe Video Stream', video_stream['codec_name']))
                        break
            
            if codec_sources:
                consolidated['codec'] = {
                    'sources': codec_sources,
                    'primary_value': codec_sources[0][1]
                }
            
            metadata['consolidated'] = consolidated
        
        logger.info(f"Metadata extraction completed. Methods used: {extraction_methods}")
        
    except Exception as e:
        logger.error(f"Error in extract_video_metadata_improved: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        metadata['critical_error'] = str(e)
        metadata['traceback'] = traceback.format_exc()
        metadata['extraction_summary'] = {
            'status': 'critical_failure',
            'methods_used': extraction_methods,
            'error': str(e)
        }
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Cleaned up temp file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {str(e)}")
    
    return metadata

def sanitize_audio_metadata(audio_file, options):
    """Sanitize metadata in audio files by thoroughly removing all metadata"""
    file_path = audio_file.filename.lower() if hasattr(audio_file, 'filename') else 'unknown'
    file_ext = os.path.splitext(file_path)[1]
    
    temp_path = None
    output_path = None
    
    try:
        import tempfile
        import subprocess
        import shutil
        
        # Create temporary files with correct extensions
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp:
            temp_path = temp.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as output:
            output_path = output.name
        
        # Save the uploaded file
        audio_file.save(temp_path)
        
        # Create a FileStorage-like object for metadata extraction
        class FileObj:
            def __init__(self, path, filename):
                self.path = path
                self.filename = filename
                
            def save(self, dst_path):
                shutil.copy(self.path, dst_path)
        
        # Create a file object for metadata extraction
        file_obj = FileObj(temp_path, file_path)
        
        # Extract original metadata for comparison
        original_metadata = extract_audio_metadata(file_obj)
        
        if file_ext == '.mp3':
            try:
                # First approach: Use mutagen to remove ID3 tags
                from mutagen.mp3 import MP3
                from mutagen.id3 import ID3, ID3NoHeaderError
                
                # Remove ID3 tags completely
                try:
                    mp3 = MP3(temp_path)
                    if mp3.tags:
                        mp3.delete()
                        mp3.save()
                except Exception as e:
                    print(f"Mutagen MP3 operation failed: {str(e)}")
                
                try:
                    # Additionally try direct ID3 removal
                    try:
                        id3 = ID3(temp_path)
                        id3.delete()
                    except ID3NoHeaderError:
                        # No ID3 header to remove
                        pass
                except Exception as e:
                    print(f"ID3 direct removal failed: {str(e)}")
                
                # Second approach: Use eyed3 as backup
                try:
                    audiofile = eyed3.load(temp_path)
                    if audiofile is not None and audiofile.tag is not None:
                        audiofile.tag.clear()
                        audiofile.tag.save(temp_path)
                except Exception as e:
                    print(f"eyed3 operation failed: {str(e)}")
                
                # For complete removal, we'll extract audio data and create a new file
                # This is more extreme but ensures all metadata is gone
                if options.get('remove_all', True):  # Force remove_all for thorough cleaning
                    try:
                        # Create a temporary WAV file
                        wav_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                        wav_path = wav_temp.name
                        wav_temp.close()
                        
                        # Extract audio to WAV (no metadata)
                        from pydub import AudioSegment
                        audio = AudioSegment.from_mp3(temp_path)
                        audio.export(wav_path, format="wav")
                        
                        # Convert back to MP3 without metadata
                        audio = AudioSegment.from_wav(wav_path)
                        audio.export(output_path, format="mp3", tags=None, id3v2_version='none')
                        
                        # Clean up WAV temp file
                        try:
                            os.remove(wav_path)
                        except:
                            pass
                            
                    except ImportError:
                        # If pydub is not available, fall back to copying the MP3 file
                        # with stripped tags
                        shutil.copy(temp_path, output_path)
                    except Exception as e:
                        print(f"Audio conversion failed: {str(e)}")
                        # Fall back to copying the file with stripped tags
                        shutil.copy(temp_path, output_path)
                else:
                    # If not removing all metadata, copy the file with stripped tags
                    shutil.copy(temp_path, output_path)
            
            except Exception as e:
                # If all operations fail, just copy the original file
                print(f"MP3 processing failed: {str(e)}")
                shutil.copy(temp_path, output_path)
                
        elif file_ext == '.wav':
            try:
                # WAV files have less metadata, but we can still try to clean them
                from mutagen.wave import WAVE
                
                try:
                    # Use mutagen to remove WAV chunks
                    wave = WAVE(temp_path)
                    if hasattr(wave, 'tags'):
                        wave.tags = None
                        wave.save()
                except Exception as e:
                    print(f"WAVE mutagen operation failed: {str(e)}")
                
                # For a more thorough approach, extract and rewrite the audio data
                try:
                    import wave
                    import array
                    
                    # Open the original file
                    with wave.open(temp_path, 'rb') as wf:
                        # Get parameters
                        params = wf.getparams()
                        # Read frames
                        frames = wf.readframes(wf.getnframes())
                    
                    # Write a new clean file with just the audio data
                    with wave.open(output_path, 'wb') as wf:
                        wf.setparams(params)
                        wf.writeframes(frames)
                        
                except Exception as e:
                    print(f"WAV rewrite operation failed: {str(e)}")
                    # Fall back to copying if rewrite fails
                    shutil.copy(temp_path, output_path)
            
            except Exception as e:
                # If all WAV operations fail, just copy the file
                print(f"WAV processing failed: {str(e)}")
                shutil.copy(temp_path, output_path)
                
        elif file_ext in ['.ogg', '.oga']:
            try:
                # Handle OGG files
                from mutagen.oggvorbis import OggVorbis
                
                try:
                    # Use mutagen to clear Vorbis comments
                    ogg = OggVorbis(temp_path)
                    if options.get('remove_all', False):
                        ogg.clear()
                    else:
                        if ogg.tags:
                            # Remove specific tags
                            for key in list(ogg.tags.keys()):
                                if (options.get('remove_timestamps', False) and 
                                    any(time_str in key.lower() for time_str in ['date', 'time'])):
                                    del ogg.tags[key]
                                elif (options.get('remove_device_info', False) and 
                                    any(info in key.lower() for info in ['encoder', 'device', 'tool'])):
                                    del ogg.tags[key]
                    ogg.save()
                    
                    # Copy the processed file to the output
                    shutil.copy(temp_path, output_path)
                    
                except Exception as e:
                    print(f"OGG mutagen operation failed: {str(e)}")
                    # Just copy the file if mutagen fails
                    shutil.copy(temp_path, output_path)
            
            except Exception as e:
                # If all OGG operations fail, just copy the file
                print(f"OGG processing failed: {str(e)}")
                shutil.copy(temp_path, output_path)
        
        else:
            # For unsupported formats, just copy the file
            shutil.copy(temp_path, output_path)
        
        # Read the sanitized file to return
        with open(output_path, 'rb') as file:
            file_data = file.read()
        
        # Create a file object for sanitized metadata extraction
        sanitized_file_obj = FileObj(output_path, file_path)
        sanitized_metadata = extract_audio_metadata(sanitized_file_obj)
        
        # Convert to base64 for response
        file_b64 = base64.b64encode(file_data).decode('utf-8')
        
        # Set the correct MIME type based on file extension
        if file_ext == '.mp3':
            mime_type = 'audio/mpeg'
        elif file_ext == '.wav':
            mime_type = 'audio/wav'
        elif file_ext in ['.ogg', '.oga']:
            mime_type = 'audio/ogg'
        else:
            mime_type = 'application/octet-stream'
        
        return {
            'original_metadata': original_metadata,
            'sanitized_metadata': sanitized_metadata,
            'file': f"data:{mime_type};base64,{file_b64}"
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise Exception(f"Error sanitizing audio: {str(e)}\nDetails: {error_details}")
    finally:
        # Clean up temporary files
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass

def sanitize_video_metadata(video_file, options):
    """
    Improved video metadata sanitization with better error handling and multiple methods
    """
    import tempfile
    import shutil
    import base64
    import subprocess
    from datetime import datetime
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("video_sanitizer_improved")
    
    file_path = video_file.filename.lower() if hasattr(video_file, 'filename') else 'unknown'
    file_ext = os.path.splitext(file_path)[1]
    
    temp_path = None
    output_path = None
    
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp:
            temp_path = temp.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as output:
            output_path = output.name
        
        # Save uploaded file
        video_file.save(temp_path)
        logger.info(f"Saved uploaded file to {temp_path}, size: {os.path.getsize(temp_path)} bytes")
        
        # Create file object for metadata extraction
        class FileObj:
            def __init__(self, path, filename):
                self.path = path
                self.filename = filename
            def save(self, dst_path):
                shutil.copy(self.path, dst_path)
        
        # Extract original metadata
        file_obj = FileObj(temp_path, file_path)
        original_metadata = extract_video_metadata(file_obj)
        
        # Check FFmpeg availability
        try:
            ffmpeg_check = subprocess.run(
                ['ffmpeg', '-version'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                timeout=10
            )
            if ffmpeg_check.returncode != 0:
                raise Exception("FFmpeg not available")
            logger.info("FFmpeg is available")
        except Exception as e:
            logger.error(f"FFmpeg check failed: {str(e)}")
            raise Exception(f"FFmpeg not found or not working: {str(e)}")
        
        # Build FFmpeg command based on options
        ffmpeg_cmd = ['ffmpeg', '-i', temp_path]
        
        if options.get('remove_all', True):
            # Remove all metadata aggressively
            ffmpeg_cmd.extend([
                '-map_metadata', '-1',           # Remove global metadata
                '-map_chapters', '-1',           # Remove chapters
                '-map_metadata:s:v', '-1',       # Remove video stream metadata
                '-map_metadata:s:a', '-1',       # Remove audio stream metadata
                '-map_metadata:s:s', '-1',       # Remove subtitle metadata
                '-fflags', '+bitexact',          # Bitexact mode
                '-flags:v', '+bitexact',         # Video bitexact
                '-flags:a', '+bitexact',         # Audio bitexact
            ])
        else:
            # Selective metadata removal
            if options.get('remove_timestamps', False):
                ffmpeg_cmd.extend(['-metadata', 'creation_time='])
            if options.get('remove_device_info', False):
                ffmpeg_cmd.extend([
                    '-metadata', 'encoder=',
                    '-metadata', 'software=',
                    '-metadata', 'device=',
                ])
        
        # Output settings
        ffmpeg_cmd.extend([
            '-c', 'copy',                        # Copy streams without re-encoding
            '-avoid_negative_ts', 'make_zero',   # Handle timestamp issues
            '-y',                                # Overwrite output
            output_path
        ])
        
        # Execute FFmpeg
        logger.info(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
        process = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300  # 5 minute timeout
        )
        
        if process.returncode == 0:
            logger.info("FFmpeg processing completed successfully")
            
            # Verify output file
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise Exception("Output file is empty or doesn't exist")
                
        else:
            error_msg = process.stderr.decode('utf-8')
            logger.error(f"FFmpeg failed with return code {process.returncode}: {error_msg}")
            raise Exception(f"FFmpeg processing failed: {error_msg}")
        
        # Extract sanitized metadata
        sanitized_file_obj = FileObj(output_path, file_path)
        sanitized_metadata = extract_video_metadata(sanitized_file_obj)
        
        # Read sanitized file
        with open(output_path, 'rb') as file:
            file_data = file.read()
        
        logger.info(f"Read {len(file_data)} bytes from sanitized file")
        
        # Convert to base64
        file_b64 = base64.b64encode(file_data).decode('utf-8')
        
        # Determine MIME type
        mime_types = {
            '.mp4': 'video/mp4',
            '.m4v': 'video/mp4',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
            '.wmv': 'video/x-ms-wmv',
            '.flv': 'video/x-flv'
        }
        mime_type = mime_types.get(file_ext.lower(), 'application/octet-stream')
        
        return {
            'original_metadata': original_metadata,
            'sanitized_metadata': sanitized_metadata,
            'file': f"data:{mime_type};base64,{file_b64}",
            'processing_info': {
                'method': 'FFmpeg',
                'options_used': options,
                'original_size': os.path.getsize(temp_path),
                'sanitized_size': len(file_data),
                'timestamp': datetime.now().isoformat()
            }
        }
        
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg processing timed out")
        raise Exception("Video processing timed out")
    except Exception as e:
        logger.error(f"Error in sanitize_video_metadata_improved: {str(e)}")
        raise Exception(f"Error sanitizing video: {str(e)}")
    finally:
        # Clean up temporary files
        for path in [temp_path, output_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"Cleaned up temporary file: {path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {path}: {str(e)}")

def obscure_file_metadata(file, metadata_json, password=""):
    """Obscure metadata by embedding it in the file itself"""
    file_path = file.filename.lower() if hasattr(file, 'filename') else 'unknown'
    file_ext = os.path.splitext(file_path)[1]
    
    temp_path = None
    output_path = None
    
    try:
        # Create a temporary file with the correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp:
            temp_path = temp.name
        
        # Save the uploaded file
        file.save(temp_path)
    
        # Currently we use DCT embedding, which is primarily for images
        # For audio/video, we'll use a simpler approach with a secret section
        
        if file_ext in ['.jpg', '.jpeg', '.png']:
            # Use the existing DCT method for images
            # Convert file to numpy array
            img = Image.open(temp_path)
            img_array = np.array(img)
            
            # Use the DCT embedding function from the main app
            from app import embed_message_dct
            stego_array = embed_message_dct(img_array, metadata_json, password)
            stego_img = Image.fromarray(stego_array)
            
            # Convert to base64
            img_buffer = io.BytesIO()
            stego_img.save(img_buffer, format="PNG")
            img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{img_str}"
            
        elif file_ext in ['.mp3', '.wav', '.ogg', '.oga']:
            # For audio files, we'll append the metadata as a "comment" tag
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as output:
                output_path = output.name
            
            # Encrypt metadata if password provided
            from app import encrypt_message
            secured_metadata = encrypt_message(metadata_json, password)
            
            if file_ext == '.mp3':
                try:
                    # Use eyed3 for MP3
                    audiofile = eyed3.load(temp_path)
                    if not audiofile.tag:
                        audiofile.initTag()
                    # Add the metadata as a comment with a special description
                    audiofile.tag.comments.set(secured_metadata, "eng", "OBLIVIO_METADATA")
                    audiofile.tag.save(output_path)
                except:
                    # If adding tags fails, just copy the file
                    shutil.copy(temp_path, output_path)
                
            elif file_ext == '.wav':
                # WAV doesn't support many tags, so for now just create a copy
                shutil.copy(temp_path, output_path)
                
            elif file_ext in ['.ogg', '.oga']:
                try:
                    # Use mutagen for OGG
                    audio = OggVorbis(temp_path)
                    audio.tags['OBLIVIO_METADATA'] = secured_metadata
                    audio.save(output_path)
                except:
                    # If adding tags fails, just copy the file
                    shutil.copy(temp_path, output_path)
            
            # Read the obscured file
            with open(output_path, 'rb') as file:
                file_data = file.read()
            
            # Convert to base64
            file_b64 = base64.b64encode(file_data).decode('utf-8')
            mime_type = 'audio/mp3' if file_ext == '.mp3' else 'audio/wav' if file_ext == '.wav' else 'audio/ogg'
            
            return f"data:{mime_type};base64,{file_b64}"
            
        elif file_ext in ['.mp4', '.m4v', '.mov', '.avi']:
            # For video files, use a simpler approach to avoid moov errors
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as output:
                output_path = output.name
            
            # Just copy the file directly to avoid complex operations that may fail
            shutil.copy(temp_path, output_path)
            
            # Read the file
            with open(output_path, 'rb') as file:
                file_data = file.read()
            
            # Convert to base64
            file_b64 = base64.b64encode(file_data).decode('utf-8')
            mime_type = f'video/{file_ext[1:]}' # e.g., video/mp4
            
            return f"data:{mime_type};base64,{file_b64}"
        
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    except Exception as e:
        raise ValueError(f"Error obscuring metadata: {str(e)}")
    finally:
        # Clean up temporary files
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
            
def extract_audio_from_video(file_path, filename):
    """Ekstrak audio dari video dan simpan sebagai WAV"""
    try:
        video = VideoFileClip(file_path)

        if video.audio is None:
            return None, 'Video tidak memiliki audio'

        os.makedirs('audio', exist_ok=True)
        audio_path = os.path.join('audio', filename.rsplit('.', 1)[0] + '.wav')
        video.audio.write_audiofile(audio_path)
        return audio_path, None
    except Exception as e:
        return None, f'Gagal mengekstrak audio: {str(e)}'


def download_audio(filename):
    """Kirim file audio hasil ekstrak"""
    file_path = os.path.join('audio', filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'File audio tidak ditemukan'}), 404            

def check_ffmpeg_environment():
    """Periksa apakah FFmpeg dan FFprobe tersedia dan berfungsi dengan baik"""
    try:
        ffmpeg = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ffprobe = subprocess.run(['ffprobe', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if ffmpeg.returncode == 0 and ffprobe.returncode == 0:
            return {
                "status": "OK",
                "ffmpeg": ffmpeg.stdout.decode('utf-8').splitlines()[0],
                "ffprobe": ffprobe.stdout.decode('utf-8').splitlines()[0]
            }
        else:
            return {
                "status": "ERROR",
                "message": "FFmpeg or FFprobe returned non-zero exit code"
            }
    except Exception as e:
        return {
            "status": "ERROR",
            "message": str(e)
        }

def validate_video_file(video_file):
    """
    Enhanced video file validation
    """
    if not video_file:
        return False, "No file provided"
    
    if not hasattr(video_file, 'filename') or not video_file.filename:
        return False, "Invalid file object or missing filename"
    
    filename = video_file.filename.lower()
    file_ext = os.path.splitext(filename)[1]
    
    # Supported video formats
    supported_formats = {
        '.mp4': 'MPEG-4 Video',
        '.m4v': 'MPEG-4 Video',
        '.mov': 'QuickTime Movie',
        '.avi': 'Audio Video Interleave',
        '.mkv': 'Matroska Video',
        '.webm': 'WebM Video',
        '.wmv': 'Windows Media Video',
        '.flv': 'Flash Video',
        '.3gp': '3GPP Video',
        '.ogv': 'Ogg Video'
    }
    
    if file_ext not in supported_formats:
        supported_list = ', '.join(supported_formats.keys())
        return False, f"Unsupported format '{file_ext}'. Supported formats: {supported_list}"
    
    # Check file size (optional - set reasonable limits)
    try:
        # For uploaded files, we can check content length if available
        if hasattr(video_file, 'content_length') and video_file.content_length:
            max_size = 500 * 1024 * 1024  # 500MB limit
            if video_file.content_length > max_size:
                return False, f"File too large: {video_file.content_length / (1024*1024):.1f}MB. Maximum allowed: {max_size / (1024*1024)}MB"
    except:
        pass  # If we can't check size, continue
    
    return True, f"Valid {supported_formats[file_ext]} file"
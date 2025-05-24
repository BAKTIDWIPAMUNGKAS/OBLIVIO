document.addEventListener('DOMContentLoaded', function() {
    // Tab switching functionality
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const tabId = this.getAttribute('data-tab');
            
            // Remove active class from all tabs and contents
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to current tab and content
            this.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        });
    });
    
    // File upload previews
    setupFileUpload('embed-image', showImagePreview);
    setupFileUpload('extract-image', showImagePreview);
    setupFileUpload('metadata-file', showFilePreview);
    
    // Button event listeners
    document.getElementById('embed-button').addEventListener('click', embedMessage);
    document.getElementById('extract-button').addEventListener('click', extractMessage);
    document.getElementById('process-metadata-button').addEventListener('click', processMetadata);
    
    // Clear buttons
    document.getElementById('clear-embed').addEventListener('click', () => clearSection('embed'));
    document.getElementById('clear-extract').addEventListener('click', () => clearSection('extract'));
    document.getElementById('clear-metadata').addEventListener('click', () => clearSection('metadata'));
    
    // Copy buttons
    document.getElementById('copy-message').addEventListener('click', copyToClipboard);
    document.getElementById('copy-image-btn').addEventListener('click', copyImageToClipboard);
    
    // Auto-select "remove all" when "obscure" is selected
    document.getElementById('obscure').addEventListener('change', function() {
        if (this.checked) {
            document.getElementById('remove-all').checked = true;
        }
    });
    
    // Prevent "remove all" from being unchecked when "obscure" is selected
    document.getElementById('remove-all').addEventListener('change', function() {
        if (!this.checked && document.getElementById('obscure').checked) {
            this.checked = true;
            showToast('Semua metadata harus dihapus ketika menggunakan metode "Aburkan"');
        }
    });
});

// Helper function to set up file uploads
function setupFileUpload(inputId, previewCallback) {
    const fileInput = document.getElementById(inputId);
    const dropArea = fileInput.closest('.file-upload');
    
    // Handle file selection through input
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            previewCallback(file, this.closest('.stego-form').querySelector('.image-preview img'));
        }
    });
    
    // Handle drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.add('highlight');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.remove('highlight');
        }, false);
    });
    
    dropArea.addEventListener('drop', function(e) {
        const file = e.dataTransfer.files[0];
        if (file) {
            fileInput.files = e.dataTransfer.files;
            previewCallback(file, this.closest('.stego-form').querySelector('.image-preview img'));
        }
    }, false);
}

// Function to show image preview
function showImagePreview(file, previewElement) {
    const previewContainer = previewElement.parentElement;
    
    // Clear any existing media elements
    const existingMedia = previewContainer.querySelector('audio, video');
    if (existingMedia) {
        previewContainer.removeChild(existingMedia);
    }

    if (file.type.match('image.*')) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewElement.src = e.target.result;
            previewElement.style.display = 'block';
        };
        reader.readAsDataURL(file);
    } else if (file.type.match('audio.*')) {
        previewElement.style.display = 'none';
        const audio = document.createElement('audio');
        audio.controls = true;
        audio.preload = 'metadata';
        audio.style.width = '100%';
        audio.style.marginTop = '10px';
        
        const reader = new FileReader();
        reader.onload = function(e) {
            audio.src = e.target.result;
        };
        reader.readAsDataURL(file);
        
        previewContainer.appendChild(audio);
    } else if (file.type.match('video.*')) {
        previewElement.style.display = 'none';
        const video = document.createElement('video');
        video.controls = true;
        video.preload = 'metadata';
        video.muted = false; // Helps with autoplay policies
        video.style.width = '100%';
        video.style.maxHeight = '300px';
        video.style.marginTop = '10px';
        
        const reader = new FileReader();
        reader.onload = function(e) {
            video.src = e.target.result;
        };
        reader.readAsDataURL(file);
        
        previewContainer.appendChild(video);
    } else {
        showToast('Format file tidak didukung', 'error');
    }
}

// Function to show file preview for various file types
function showFilePreview(file, previewElement) {
    if (file.type.match('image.*')) {
        // For images, show the actual image
        const reader = new FileReader();
        reader.onload = function(e) {
            previewElement.src = e.target.result;
        };
        reader.readAsDataURL(file);
    } else if (file.type.match('audio.*')) {
        // For audio, show an audio icon
        previewElement.src = '/static/images/audio-icon.png';
        
        // Create an audio preview if it doesn't exist
        let audioPreview = previewElement.parentElement.querySelector('audio');
        if (!audioPreview) {
            audioPreview = document.createElement('audio');
            audioPreview.controls = true;
            audioPreview.style.width = '100%';
            audioPreview.style.marginTop = '10px';
            previewElement.parentElement.appendChild(audioPreview);
        }
        
        // Set the audio source
        const reader = new FileReader();
        reader.onload = function(e) {
            audioPreview.src = e.target.result;
        };
        reader.readAsDataURL(file);
    } else if (file.type.match('video.*')) {
        // For video, show a video preview
        previewElement.src = '/static/images/video-icon.png';
        
        // Create a video preview if it doesn't exist
        let videoPreview = previewElement.parentElement.querySelector('video');
        if (!videoPreview) {
            videoPreview = document.createElement('video');
            videoPreview.controls = true;
            videoPreview.style.width = '100%';
            videoPreview.style.marginTop = '10px';
            previewElement.parentElement.appendChild(videoPreview);
        }
        
        // Set the video source
        const reader = new FileReader();
        reader.onload = function(e) {
            videoPreview.src = e.target.result;
        };
        reader.readAsDataURL(file);
    } else {
        showToast('Format file tidak didukung', 'error');
    }
}

// Enhanced progress simulation for video processing
function simulateVideoProgress(progressBar) {
    let width = 0;
    const interval = setInterval(function() {
        if (width >= 85) { // Stop at 85% to show it's still processing
            clearInterval(interval);
        } else {
            // Slower progress for video to reflect longer processing time
            width += Math.random() * 3;
            if (width > 85) width = 85;
            progressBar.style.width = width + '%';
        }
    }, 500); // Slower updates
}

// Function to embed a message in an image
function embedMessage() {
    const fileInput = document.getElementById('embed-image');
    const message = document.getElementById('secret-message').value;
    const password = document.getElementById('password').value;
    const file = fileInput.files[0];
    
    if (!file) {
        showToast('Pilih file terlebih dahulu', 'error');
        return;
    }
    
    if (!message) {
        showToast('Masukkan pesan yang ingin disembunyikan', 'error');
        return;
    }
    
    // Check file size for videos (warn if too large)
    if (file.type.startsWith('video/') && file.size > 100 * 1024 * 1024) { // 100MB
        if (!confirm('File video berukuran besar. Proses mungkin memakan waktu lama. Lanjutkan?')) {
            return;
        }
    }
    
    // Determine endpoint based on file type
    const endpoint = file.type.startsWith('image/') ? '/api/embed' : '/api/embed_media';
    
    // Show progress indicator
    const progressContainer = document.getElementById('embed-progress');
    const embedButton = document.getElementById('embed-button');
    
    progressContainer.style.display = 'block';
    embedButton.disabled = true;
    embedButton.textContent = 'Memproses...';
    
    // Enhanced progress simulation for video files
    if (file.type.startsWith('video/')) {
        simulateVideoProgress(progressContainer.querySelector('.progress'));
    } else {
        simulateProgress(progressContainer.querySelector('.progress'));
    }
    
    // Create form data
    const formData = new FormData();
    formData.append(file.type.startsWith('image/') ? 'image' : 'file', file);
    formData.append('message', message);
    formData.append('password', password);
    
    // Send request with extended timeout for video files
    const timeoutDuration = file.type.startsWith('video/') ? 300000 : 60000; // 5 min for video, 1 min for others
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutDuration);
    
    fetch(endpoint, {
        method: 'POST',
        body: formData,
        signal: controller.signal
    })
    .then(response => {
        clearTimeout(timeoutId);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        progressContainer.style.display = 'none';
        embedButton.disabled = false;
        embedButton.textContent = 'Sembunyikan Pesan';
        
        if (data.status === 'success') {
            // Show result
            const resultContainer = document.getElementById('embed-result');
            resultContainer.style.display = 'block';
            
            // Update preview based on file type
            const previewImg = resultContainer.querySelector('.image-preview img');
            const previewContainer = previewImg.parentElement;
            
            // Clear any existing media elements
            const existingMedia = previewContainer.querySelector('audio, video');
            if (existingMedia) {
                previewContainer.removeChild(existingMedia);
            }
            
            const downloadLink = document.getElementById('download-image');
            
            if (file.type.startsWith('image/')) {
                previewImg.src = data.image;
                previewImg.style.display = 'block';
                
                // Set download link
                downloadLink.href = data.image;
                downloadLink.download = 'stego-image.png';
            } else if (file.type.startsWith('audio/')) {
                previewImg.style.display = 'none';
                
                // Create audio element with better compatibility
                const audio = document.createElement('audio');
                audio.controls = true;
                audio.preload = 'metadata';
                audio.style.width = '100%';
                
                // Handle base64 data URLs properly
                if (data.file.startsWith('data:')) {
                    audio.src = data.file;
                } else {
                    audio.src = 'data:audio/wav;base64,' + data.file;
                }
                
                previewContainer.appendChild(audio);
                
                // Set download link
                downloadLink.href = audio.src;
                downloadLink.download = 'stego-audio.wav';
            } else if (file.type.startsWith('video/')) {
                previewImg.style.display = 'none';
                
                // Create video element with enhanced compatibility
                const video = document.createElement('video');
                video.controls = true;
                video.preload = 'metadata';
                video.muted = true; // Helps with autoplay policies
                video.style.width = '100%';
                video.style.maxHeight = '400px';
                
                // Add error handling for video
                video.onerror = function(e) {
                    console.error('Video error:', e);
                    showToast('Video tidak dapat diputar. Coba download dan putar dengan player eksternal.', 'warning');
                };
                
                video.onloadstart = function() {
                    console.log('Video loading started');
                };
                
                video.oncanplay = function() {
                    console.log('Video can play');
                    showToast('Video berhasil dimuat dan siap diputar');
                };
                
                // Handle base64 data URLs properly for video
                if (data.file.startsWith('data:')) {
                    video.src = data.file;
                } else {
                    // Create blob URL for better video handling
                    try {
                        const byteCharacters = atob(data.file);
                        const byteNumbers = new Array(byteCharacters.length);
                        for (let i = 0; i < byteCharacters.length; i++) {
                            byteNumbers[i] = byteCharacters.charCodeAt(i);
                        }
                        const byteArray = new Uint8Array(byteNumbers);
                        const blob = new Blob([byteArray], { type: 'video/mp4' });
                        video.src = URL.createObjectURL(blob);
                        
                        // Clean up blob URL when component is removed
                        video.addEventListener('error', () => {
                            URL.revokeObjectURL(video.src);
                        });
                    } catch (e) {
                        console.error('Error creating blob URL:', e);
                        video.src = 'data:video/mp4;base64,' + data.file;
                    }
                }
                
                previewContainer.appendChild(video);
                
                // Set download link with proper MIME type
                const originalExt = file.name.split('.').pop().toLowerCase();
                const downloadExt = originalExt === 'mp4' ? 'mp4' : 'mp4'; // Force MP4 for compatibility
                
                if (data.file.startsWith('data:')) {
                    downloadLink.href = data.file;
                } else {
                    downloadLink.href = 'data:video/mp4;base64,' + data.file;
                }
                downloadLink.download = `stego-video.${downloadExt}`;
            }
            
            showToast('Pesan berhasil disembunyikan!');
        } else {
            showToast('Error: ' + data.message, 'error');
        }
    })
    .catch(error => {
        clearTimeout(timeoutId);
        progressContainer.style.display = 'none';
        embedButton.disabled = false;
        embedButton.textContent = 'Sembunyikan Pesan';
        
        if (error.name === 'AbortError') {
            showToast('Proses dibatalkan karena timeout. Coba dengan file yang lebih kecil.', 'error');
        } else {
            console.error('Embed error:', error);
            showToast('Error: ' + error.message, 'error');
        }
    });
}


// Function to extract a message from an image
function extractMessage() {
    const fileInput = document.getElementById('extract-image');
    const password = document.getElementById('extract-password').value;
    const file = fileInput.files[0];
    
    if (!file) {
        showToast('Pilih file terlebih dahulu', 'error');
        return;
    }
    
    // Check file size for videos
    if (file.type.startsWith('video/') && file.size > 100 * 1024 * 1024) {
        if (!confirm('File video berukuran besar. Proses ekstraksi mungkin memakan waktu lama. Lanjutkan?')) {
            return;
        }
    }
    
    // Determine endpoint based on file type
    const endpoint = file.type.startsWith('image/') ? '/api/extract' : '/api/extract_media';
    
    // Show progress indicator
    const progressContainer = document.getElementById('extract-progress');
    const extractButton = document.getElementById('extract-button');
    
    progressContainer.style.display = 'block';
    extractButton.disabled = true;
    extractButton.textContent = 'Mengekstrak...';
    
    // Enhanced progress for video files
    if (file.type.startsWith('video/')) {
        simulateVideoProgress(progressContainer.querySelector('.progress'));
    } else {
        simulateProgress(progressContainer.querySelector('.progress'));
    }
    
    // Create form data
    const formData = new FormData();
    formData.append(file.type.startsWith('image/') ? 'image' : 'file', file);
    formData.append('password', password);
    
    // Extended timeout for video files
    const timeoutDuration = file.type.startsWith('video/') ? 300000 : 60000;
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutDuration);
    
    // Send request to the server
    fetch(endpoint, {
        method: 'POST',
        body: formData,
        signal: controller.signal
    })
    .then(response => {
        clearTimeout(timeoutId);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        progressContainer.style.display = 'none';
        extractButton.disabled = false;
        extractButton.textContent = 'Ekstrak Pesan';
        
        if (data.status === 'success') {
            // Show result
            const resultContainer = document.getElementById('extract-result');
            resultContainer.style.display = 'block';
            document.getElementById('extracted-message').value = data.message;
            
            showToast('Pesan berhasil diekstrak!');
        } else {
            showToast('Error: ' + data.message, 'error');
        }
    })
    .catch(error => {
        clearTimeout(timeoutId);
        progressContainer.style.display = 'none';
        extractButton.disabled = false;
        extractButton.textContent = 'Ekstrak Pesan';
        
        if (error.name === 'AbortError') {
            showToast('Proses dibatalkan karena timeout. Coba dengan file yang lebih kecil.', 'error');
        } else {
            console.error('Extract error:', error);
            showToast('Error: ' + error.message, 'error');
        }
    });
}

// Function to process metadata
function processMetadata() {
    const fileInput = document.getElementById('metadata-file');
    const removeGPS = document.getElementById('remove-gps').checked;
    const removeTimestamps = document.getElementById('remove-timestamps').checked;
    const removeDeviceInfo = document.getElementById('remove-device-info').checked;
    const removeAll = document.getElementById('remove-all').checked;
    const method = document.querySelector('input[name="method"]:checked').value;
    
    if (!fileInput.files[0]) {
        showToast('Pilih file terlebih dahulu', 'error');
        return;
    }
    
    const progressContainer = document.getElementById('metadata-progress');
    progressContainer.style.display = 'block';
    simulateProgress(progressContainer.querySelector('.progress'));
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('options', JSON.stringify({
        remove_gps: removeGPS,
        remove_timestamps: removeTimestamps,
        remove_device_info: removeDeviceInfo,
        remove_all: removeAll
    }));
    
    const endpoint = method === 'sanitize' ? '/api/metadata/sanitize' : '/api/metadata/obscure';
    
    fetch(endpoint, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        progressContainer.style.display = 'none';
        
        if (data.status === 'success') {
            const resultContainer = document.getElementById('metadata-result');
            resultContainer.style.display = 'block';
            
            // Show metadata
            document.getElementById('original-metadata').textContent = 
                JSON.stringify(data.original_metadata, null, 2);
            
            if (method === 'sanitize') {
                document.getElementById('processed-metadata').textContent = 
                    JSON.stringify(data.sanitized_metadata, null, 2);
            } else {
                document.getElementById('processed-metadata').textContent = 
                    "Metadata telah diamankan di dalam file (terenkripsi).";
            }
            
            // Update preview and download link
            const previewImg = resultContainer.querySelector('.image-preview img');
            const downloadLink = document.getElementById('download-processed-file');
            const file = fileInput.files[0];
            
            // Clear any existing media elements
            const previewContainer = previewImg.parentElement;
            const existingMedia = previewContainer.querySelector('audio, video');
            if (existingMedia) {
                previewContainer.removeChild(existingMedia);
            }
            
            if (data.file) {
                // For audio/video files
                if (file.type.startsWith('audio/')) {
                    // Create audio element
                    const audio = document.createElement('audio');
                    audio.controls = true;
                    audio.src = data.file;
                    previewContainer.appendChild(audio);
                    previewImg.style.display = 'none';
                } else if (file.type.startsWith('video/')) {
                    // Create video element
                    const video = document.createElement('video');
                    video.controls = true;
                    video.src = data.file;
                    previewContainer.appendChild(video);
                    previewImg.style.display = 'none';
                }
                
                // Set download link
                downloadLink.href = data.file;
                downloadLink.download = 'processed-' + file.name;
            } else if (data.image) {
                // For images
                previewImg.src = data.image;
                previewImg.style.display = 'block';
                downloadLink.href = data.image;
                downloadLink.download = 'processed-image.png';
            }
            
            showToast('Metadata berhasil diproses!');
        } else {
            showToast('Error: ' + data.message, 'error');
        }
    })
    .catch(error => {
        progressContainer.style.display = 'none';
        showToast('Error: ' + error.message, 'error');
    });
}

// Determine the correct endpoint based on file type and method
function getMetadataEndpoint(fileType, method) {
    const base = method === 'sanitize' ? 'sanitize' : 'obscure';
    
    if (fileType.startsWith('video/')) {
        return `/api/${base}_video_metadata`;
    } else if (fileType.startsWith('audio/')) {
        return `/api/${base}_audio_metadata`;
    } else if (fileType.startsWith('image/')) {
        return `/api/${base}_image_metadata`;
    }
    throw new Error('Format file tidak didukung');
}



// Helper function to clear a section
function clearSection(sectionId) {
    if (sectionId === 'embed') {
        document.getElementById('embed-image').value = '';
        document.getElementById('secret-message').value = '';
        document.getElementById('password').value = '';
        document.getElementById('embed-result').style.display = 'none';
        document.querySelector('#embed .image-preview img').src = '/static/images/placeholder.png';
    } else if (sectionId === 'extract') {
        document.getElementById('extract-image').value = '';
        document.getElementById('extract-password').value = '';
        document.getElementById('extract-result').style.display = 'none';
        document.querySelector('#extract .image-preview img').src = '/static/images/placeholder.png';
    } else if (sectionId === 'metadata') {
        document.getElementById('metadata-file').value = '';
        document.getElementById('metadata-result').style.display = 'none';
        document.querySelector('#metadata .image-preview img').src = '/static/images/placeholder.png';
        
        // Remove any audio or video elements
        const previewContainer = document.querySelector('#metadata .image-preview');
        const audioElement = previewContainer.querySelector('audio');
        const videoElement = previewContainer.querySelector('video');
        
        if (audioElement) previewContainer.removeChild(audioElement);
        if (videoElement) previewContainer.removeChild(videoElement);
    }
}

// Function to copy text to clipboard
function copyToClipboard() {
    const text = document.getElementById('extracted-message');
    text.select();
    document.execCommand('copy');
    showToast('Pesan disalin ke clipboard');
}

// Function to copy image to clipboard
function copyImageToClipboard() {
    const img = document.querySelector('#embed-result img');
    
    // Create a canvas element
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    
    // Draw the image onto the canvas
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    
    // Convert the canvas to a blob
    canvas.toBlob(function(blob) {
        // Create a ClipboardItem
        const item = new ClipboardItem({ 'image/png': blob });
        
        // Copy to clipboard
        navigator.clipboard.write([item]).then(
            function() {
                showToast('Gambar disalin ke clipboard');
            },
            function(err) {
                showToast('Error: ' + err, 'error');
            }
        );
    });
}

// Show toast notification
function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toast-message');
    
    toastMessage.textContent = message;
    toast.className = 'toast';
    
    if (type === 'error') {
        toast.classList.add('error');
        toast.querySelector('i').className = 'fas fa-exclamation-circle';
    } else {
        toast.classList.add('success');
        toast.querySelector('i').className = 'fas fa-check-circle';
    }
    
    toast.classList.add('show');
    
    setTimeout(function() {
        toast.classList.remove('show');
    }, 3000);
}

// Simulate progress bar
function simulateProgress(progressBar) {
    let width = 0;
    const interval = setInterval(function() {
        if (width >= 90) {
            clearInterval(interval);
        } else {
            width += Math.random() * 10;
            if (width > 90) width = 90;
            progressBar.style.width = width + '%';
        }
    }, 200);
}

// Fungsi untuk menangani proses sanitasi metadata
function sanitizeMetadata() {
    const fileInput = document.getElementById('sanitize-file');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Pilih file terlebih dahulu');
        return;
    }
    
    // Mendapatkan pilihan sanitasi
    const removeGps = document.getElementById('remove-gps').checked;
    const removeTimestamps = document.getElementById('remove-timestamps').checked;
    const removeDeviceInfo = document.getElementById('remove-device-info').checked;
    const removeAll = document.getElementById('remove-all').checked;
    
    const options = {
        remove_gps: removeGps,
        remove_timestamps: removeTimestamps,
        remove_device_info: removeDeviceInfo,
        remove_all: removeAll
    };
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('options', JSON.stringify(options));
    
    // Tampilkan indikator loading
    document.getElementById('sanitize-loading').style.display = 'block';
    
    fetch('/api/metadata/sanitize', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Sembunyikan indikator loading
        document.getElementById('sanitize-loading').style.display = 'none';
        
        if (data.status === 'error') {
            alert('Error: ' + data.message);
            return;
        }
        
        // Tampilkan metadata asli dan yang sudah disanitasi
        const originalMetadataEl = document.getElementById('original-metadata');
        const sanitizedMetadataEl = document.getElementById('sanitized-metadata');
        
        originalMetadataEl.textContent = JSON.stringify(data.original_metadata, null, 2);
        sanitizedMetadataEl.textContent = JSON.stringify(data.sanitized_metadata, null, 2);
        
        // Tampilkan preview dan tombol download berdasarkan jenis file
        const previewContainer = document.getElementById('sanitized-preview');
        const downloadContainer = document.getElementById('download-container');
        
        if (data.image) {
            // Untuk file gambar
            previewContainer.innerHTML = `<img src="${data.image}" alt="Sanitized Image" class="img-fluid">`;
            downloadContainer.innerHTML = `
                <a href="${data.image}" download="sanitized-image.png" class="btn btn-success">
                    <i class="fas fa-download"></i> Download Sanitized Image
                </a>
            `;
        } else if (data.file) {
            // Untuk file audio/video
            const fileType = file.type;
            
            if (fileType.startsWith('audio/')) {
                // Preview audio
                previewContainer.innerHTML = `
                    <audio controls class="w-100">
                        <source src="${data.file}" type="${fileType}">
                        Your browser does not support the audio element.
                    </audio>
                `;
            } else if (fileType.startsWith('video/')) {
                // Preview video
                previewContainer.innerHTML = `
                    <video controls class="w-100">
                        <source src="${data.file}" type="${fileType}">
                        Your browser does not support the video element.
                    </video>
                `;
            } else {
                previewContainer.innerHTML = `<p>Preview tidak tersedia untuk jenis file ini</p>`;
            }
            
            // Tombol download
            let fileExt = file.name.split('.').pop();
            downloadContainer.innerHTML = `
                <a href="${data.file}" download="sanitized-file.${fileExt}" class="btn btn-success">
                    <i class="fas fa-download"></i> Download Sanitized File
                </a>
            `;
        }
        
        // Tampilkan hasil
        document.getElementById('sanitize-result').style.display = 'block';
        
        // Debug output ke konsol
        console.log("Sanitize result:", data);
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('sanitize-loading').style.display = 'none';
        alert('Error: ' + error.message);
    });
}


// Video metadata management functionality
// Modified Video metadata handling code

document.addEventListener('DOMContentLoaded', function() {
    // ... (keep existing tab switching and other functionality)

    // Modified video metadata processing
    async function processMetadata() {
        const fileInput = document.getElementById('metadata-file');
        const removeGPS = document.getElementById('remove-gps').checked;
        const removeTimestamps = document.getElementById('remove-timestamps').checked;
        const removeDeviceInfo = document.getElementById('remove-device-info').checked;
        const removeAll = document.getElementById('remove-all').checked;
        const method = document.querySelector('input[name="method"]:checked').value;
        
        if (!fileInput.files[0]) {
            showToast('Pilih file terlebih dahulu', 'error');
            return;
        }
        
        const file = fileInput.files[0];
        const progressContainer = document.getElementById('metadata-progress');
        const resultContainer = document.getElementById('metadata-result');
        const previewImg = resultContainer.querySelector('.image-preview img');
        const downloadLink = document.getElementById('download-processed-file');
        
        // Show progress indicator
        progressContainer.style.display = 'block';
        resultContainer.style.display = 'none';
        
        try {
            // Prepare form data
            const formData = new FormData();
            formData.append('file', file);
            formData.append('options', JSON.stringify({
                remove_gps: removeGPS,
                remove_timestamps: removeTimestamps,
                remove_device_info: removeDeviceInfo,
                remove_all: removeAll
            }));
            
            // Determine endpoint based on file type and method
            let endpoint;
            if (file.type.startsWith('video/')) {
                endpoint = method === 'sanitize' ? '/api/sanitize_video_metadata' : '/api/obscure_video_metadata';
            } else if (file.type.startsWith('audio/')) {
                endpoint = method === 'sanitize' ? '/api/sanitize_audio_metadata' : '/api/obscure_audio_metadata';
            } else if (file.type.startsWith('image/')) {
                endpoint = method === 'sanitize' ? '/api/sanitize_image_metadata' : '/api/obscure_image_metadata';
            } else {
                showToast('Format file tidak didukung', 'error');
                return;
            }
            
            // Send request
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.status === 'error') {
                throw new Error(data.message);
            }
            
            // Hide progress, show results
            progressContainer.style.display = 'none';
            resultContainer.style.display = 'block';
            
            // Update metadata display
            document.getElementById('original-metadata').textContent = 
                JSON.stringify(data.original_metadata, null, 2);
            
            if (method === 'sanitize') {
                document.getElementById('processed-metadata').textContent = 
                    JSON.stringify(data.sanitized_metadata, null, 2);
            } else {
                document.getElementById('processed-metadata').textContent = 
                    "Metadata telah diamankan di dalam file (terenkripsi).";
            }
            
            // Clear any existing media elements
            const previewContainer = previewImg.parentElement;
            const existingMedia = previewContainer.querySelector('audio, video');
            if (existingMedia) {
                previewContainer.removeChild(existingMedia);
            }
            
            // Handle different file types
            if (file.type.startsWith('video/')) {
                // Create video element
                const video = document.createElement('video');
                video.controls = true;
                video.style.width = '100%';
                video.style.marginTop = '10px';
                
                // Set video source
                if (data.file) {
                    video.src = data.file;
                    previewContainer.appendChild(video);
                    previewImg.style.display = 'none';
                    
                    // Set download link
                    downloadLink.href = data.file;
                    downloadLink.download = 'processed-' + file.name;
                } else {
                    throw new Error('No processed video data returned');
                }
            } else if (file.type.startsWith('audio/')) {
                // Create audio element
                const audio = document.createElement('audio');
                audio.controls = true;
                audio.style.width = '100%';
                audio.style.marginTop = '10px';
                
                // Set audio source
                if (data.file) {
                    audio.src = data.file;
                    previewContainer.appendChild(audio);
                    previewImg.style.display = 'none';
                    
                    // Set download link
                    downloadLink.href = data.file;
                    downloadLink.download = 'processed-' + file.name;
                } else {
                    throw new Error('No processed audio data returned');
                }
            } else if (file.type.startsWith('image/') && data.image) {
                // For images
                previewImg.src = data.image;
                previewImg.style.display = 'block';
                downloadLink.href = data.image;
                downloadLink.download = 'processed-image.png';
            }
            
            showToast('Metadata berhasil diproses!');
        } catch (error) {
            progressContainer.style.display = 'none';
            console.error('Error:', error);
            showToast(`Gagal memproses metadata: ${error.message}`, 'error');
        }
    }

    // Update the event listener for the process button
    document.getElementById('process-metadata-button').addEventListener('click', processMetadata);

    // ... (keep rest of the existing code)
});

// ... (keep other existing functions like showToast, simulateProgress, etc.)

// Debug function for testing endpoints
function testEndpoints() {
    const endpoints = [
        '/api/metadata/process',
        '/api/metadata/sanitize',
        '/api/metadata/obscure',
        '/api/video/sanitize',
        '/api/video/obscure',
        '/api/audio/sanitize',
        '/api/audio/obscure'
    ];
    
    endpoints.forEach(endpoint => {
        fetch(endpoint, { method: 'GET' })
            .then(response => {
                console.log(`${endpoint}: ${response.status}`);
            })
            .catch(error => {
                console.log(`${endpoint}: Error - ${error.message}`);
            });
    });
}
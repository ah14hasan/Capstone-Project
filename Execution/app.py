import os  
import subprocess  
import ssl  
import cv2  
import logging  
from flask import Flask, render_template, request, jsonify  
import whisper  
from googletrans import Translator  

# Set up logging  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  

# Create an unverified SSL context for development/testing  
ssl._create_default_https_context = ssl._create_unverified_context  

app = Flask(__name__)  
app.config['UPLOAD_FOLDER'] = 'uploads'  
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'mov', 'avi', 'mkv', 'mpg'}  

# Ensure upload folder exists  
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  

def allowed_file(filename):  
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']  

def has_visible_speaker(video_path):  
    """Check if the video has at least one frame with a detectable face"""  
    try:  
        # Load the face cascade classifier  
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  
        
        # Open the video  
        cap = cv2.VideoCapture(video_path)  
        if not cap.isOpened():  
            return False  
            
        # Count frames with faces  
        faces_found = 0  
        frame_count = 0  
        
        while cap.isOpened() and frame_count < 30:  # Check up to 30 frames  
            ret, frame = cap.read()  
            if not ret:  
                break  
                
            frame_count += 1  
            
            # Skip every other frame to speed up processing  
            if frame_count % 2 != 0:  
                continue  
                
            # Convert to grayscale for face detection  
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
            
            # Detect faces  
            faces = face_cascade.detectMultiScale(  
                gray,   
                scaleFactor=1.1,   
                minNeighbors=5,  
                minSize=(30, 30)  
            )  
            
            if len(faces) > 0:  
                faces_found += 1  
                if faces_found >= 1:  # Only need to find 1 face  
                    cap.release()  
                    return True  
        
        cap.release()  
        return False  # No faces found  
        
    except Exception as e:  
        logger.error(f"Error in face detection: {str(e)}")  
        return False  # Return False on error to trigger the error message  

@app.route('/')  
def index():  
    return render_template('index.html')  

@app.route('/upload', methods=['POST'])  
def upload_video():  
    if 'video' not in request.files:  
        return jsonify(success=False, error="No video file part")  
    file = request.files['video']  
    if file.filename == '':  
        return jsonify(success=False, error="No selected file")  
    if file and allowed_file(file.filename):  
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)  
        file.save(video_path)  

        # Check if video has a visible speaker  
        if not has_visible_speaker(video_path):  
            # Clean up the file  
            if os.path.exists(video_path):  
                os.remove(video_path)  
            return jsonify(success=False, error="no_speaker")  

        # Extract audio from the video using ffmpeg  
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], "extracted_audio.wav")  
        try:  
            command = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", audio_path]  
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  
        except Exception as e:  
            return jsonify(success=False, error=f"Error processing video: {str(e)}")  

        # Load Whisper model and transcribe audio  
        try:  
            model = whisper.load_model("base")  
            result = model.transcribe(audio_path)  
            transcript = result.get('text', 'No transcription found.')  
        except Exception as e:  
            return jsonify(success=False, error=f"Error transcribing audio: {str(e)}")  
        finally:  
            # Clean up the files if desired  
            if os.path.exists(video_path):  
                os.remove(video_path)  
            if os.path.exists(audio_path):  
                os.remove(audio_path)  

        # Retrieve desired language from form, default to English ('en')  
        desired_lang = request.form.get("language", "en")  
        
        # Translate transcript if desired language is not English  
        if desired_lang != "en":  
            try:  
                # Using a try/except block to handle potential translation errors  
                translator = Translator()  
                translated_text = translator.translate(transcript, dest=desired_lang).text  
                transcript = translated_text  
            except Exception as e:  
                # If translation fails, return the original transcript with a note  
                return jsonify(  
                    success=True,   
                    transcript=f"{transcript}\n\n(Translation failed: {str(e)})"  
                )  
        
        return jsonify(success=True, transcript=transcript)  
    else:  
        return jsonify(success=False, error="Invalid file type.")  

# In your app.py  
if __name__ == "__main__":  
    app.run(host='0.0.0.0', port=8080, debug=True)  # Using port 8080 instead  
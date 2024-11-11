import sounddevice as sd
import numpy as np
import wave
import logging
from datetime import datetime
import os
from pretrained_detector import PretrainedGunshotDetector
import math
import sys

# Configuration
SAMPLE_RATE = 44100
DURATION = 5
CHANNELS = 1
SAVE_PATH = "recordings/"

# VU meter configuration
METER_WIDTH = 40  # Width of the console meter
DB_MIN = -60  # Minimum dB to show
DB_MAX = 0    # Maximum dB to show

# ASCII characters for the meter (Windows-compatible)
METER_CHARS = {
    'full': '#',
    'empty': '-',
    'peak': '|'
}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioRecorder:
    def __init__(self):
        self.stream = None
        self.audio_buffer = []
        self.is_recording = False
        self.current_db = DB_MIN
        
        # Check available devices and select default
        try:
            devices = sd.query_devices()
            input_device = sd.default.device[0]
            device_info = sd.query_devices(input_device, 'input')
            logger.info(f"Using input device: {device_info['name']}")
            
            if device_info['default_samplerate'] != SAMPLE_RATE:
                logger.warning(f"Adjusting sample rate from {SAMPLE_RATE} to {device_info['default_samplerate']}")
                global LOCAL_SAMPLE_RATE
                LOCAL_SAMPLE_RATE = int(device_info['default_samplerate'])
                
        except Exception as e:
            logger.error(f"Error querying audio devices: {str(e)}")
            raise RuntimeError("Failed to initialize audio device")

    def calculate_db(self, audio_data):
        """Calculate decibel level from audio data."""
        if len(audio_data) == 0:
            return DB_MIN
        
        # Calculate RMS value
        rms = np.sqrt(np.mean(np.square(audio_data)))
        
        # Convert to decibels
        if rms > 0:
            db = 20 * math.log10(rms)
        else:
            db = DB_MIN
            
        # Clamp the value
        return max(min(db, DB_MAX), DB_MIN)

    def draw_vu_meter(self, db):
        """Draw a VU meter in the console using ASCII characters."""
        # Normalize db value to 0-1 range
        normalized = (db - DB_MIN) / (DB_MAX - DB_MIN)
        meter_level = int(normalized * METER_WIDTH)
        
        # Create meter segments using ASCII characters
        meter = (METER_CHARS['full'] * meter_level + 
                METER_CHARS['empty'] * (METER_WIDTH - meter_level))
        
        # Create the scale with ASCII characters
        scale = f"{db:>6.1f} dB {METER_CHARS['peak']}{meter}{METER_CHARS['peak']} {DB_MAX}"
        
        # Use Windows-compatible color codes
        if sys.platform == 'win32':
            # On Windows, we'll just print without colors
            print(f"{scale}", end='\r')
        else:
            # On Unix-like systems, we can use colors
            if db > -10:  # High level - red
                print(f"\033[91m{scale}\033[0m", end='\r')
            elif db > -20:  # Medium level - yellow
                print(f"\033[93m{scale}\033[0m", end='\r')
            else:  # Low level - green
                print(f"\033[92m{scale}\033[0m", end='\r')
        
        sys.stdout.flush()  # Ensure the output is displayed immediately
    
    def _initialize_stream(self):
        """Initialize the audio stream with error handling."""
        try:
            if self.stream is not None and self.stream.active:
                self.stream.stop()
                self.stream.close()
            
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                callback=self._audio_callback,
                blocksize=int(SAMPLE_RATE * 0.1),  # 100ms blocks
                device=sd.default.device[0]
            )
            return True
        except Exception as e:
            logger.error(f"Error initializing audio stream: {str(e)}")
            return False
        
    def _audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Status: {status}")
        if self.is_recording:
            self.audio_buffer.extend(indata.copy())
            # Calculate and display current decibel level
            self.current_db = self.calculate_db(indata)
            self.draw_vu_meter(self.current_db)
            
    def record(self, duration):
        """Record audio with improved error handling."""
        self.audio_buffer = []
        self.is_recording = False
        
        # Initialize stream
        if not self._initialize_stream():
            raise RuntimeError("Failed to initialize audio stream")
        
        try:
            self.is_recording = True
            with self.stream:
                logger.debug("Recording started...")
                print("\nRecording... (Press Ctrl+C to stop)")
                print("VU Meter:")
                sd.sleep(int(duration * 1000))
        except Exception as e:
            logger.error(f"Error during recording: {str(e)}")
            raise
        finally:
            self.is_recording = False
            if self.stream is not None and self.stream.active:
                self.stream.stop()
                self.stream.close()
            print()  # New line after VU meter
        
        return np.array(self.audio_buffer)
        
    def save_audio(self, audio_data, filename):
        os.makedirs(SAVE_PATH, exist_ok=True)
        filepath = os.path.join(SAVE_PATH, filename)
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
        return filepath

def main():
    try:
        recorder = AudioRecorder()
        detector = PretrainedGunshotDetector()
        
        # Print available classes for debugging
        print("\nAvailable sound classes:")
        detector.print_all_classes()
        
        logger.info("Starting gunshot detection system using pre-trained model...")
        print("\nMonitoring audio levels...")
        
        while True:
            try:
                # Record audio
                logger.debug("Recording audio...")
                audio_data = recorder.record(DURATION)
                
                # Log audio statistics
                logger.debug(f"Recorded audio shape: {audio_data.shape}")
                logger.debug(f"Audio range: [{np.min(audio_data)}, {np.max(audio_data)}]")
                
                # Detect gunshots
                probability = detector.detect(audio_data, SAMPLE_RATE)
                print(f"\nGunshot probability: {probability:.3f}")
                
                # If probability exceeds threshold, save the audio
                # Lower threshold for testing
                if probability > 0.2:  # Even lower threshold for testing
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"gunshot_{timestamp}.wav"
                    filepath = recorder.save_audio(audio_data, filename)
                    logger.info(f"\nGunshot detected! (Probability: {probability:.3f}) Saved to {filepath}")
                    
            except Exception as e:
                logger.error(f"\nError in recording loop: {str(e)}")
                logger.info("Attempting to reinitialize audio stream...")
                sd.sleep(1000)  # Wait 1 second before retrying
                continue
                
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    except Exception as e:
        logger.error(f"\nFatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
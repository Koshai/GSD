# mic_test.py
import sounddevice as sd
import numpy as np
import wave
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MicrophoneTester:
    def __init__(self):
        # List available devices
        self.list_devices()
        
    def list_devices(self):
        """List all available audio devices."""
        devices = sd.query_devices()
        logger.info("Available audio devices:")
        for i, device in enumerate(devices):
            logger.info(f"Device {i}: {device['name']}")
            logger.info(f"  Inputs: {device['max_input_channels']}")
            logger.info(f"  Outputs: {device['max_output_channels']}")
            logger.info(f"  Default Sample Rate: {device['default_samplerate']}")
            
    def test_recording(self, device_id=None, duration=5, sample_rate=44100):
        """Test recording from specified device."""
        try:
            # Configure recording parameters
            channels = 1
            filename = f"test_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            
            logger.info(f"Starting {duration} second test recording...")
            logger.info(f"Using device: {sd.query_devices(device_id) if device_id is not None else 'default'}")
            
            # Record audio
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                device=device_id
            )
            sd.wait()
            
            # Save recording
            os.makedirs("test_recordings", exist_ok=True)
            filepath = os.path.join("test_recordings", filename)
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes((recording * 32767).astype(np.int16).tobytes())
            
            logger.info(f"Recording saved to: {filepath}")
            logger.info(f"Recording shape: {recording.shape}")
            logger.info(f"Max amplitude: {np.max(np.abs(recording))}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during recording: {str(e)}")
            return False

def main():
    tester = MicrophoneTester()
    
    # Let user choose device
    device_id = input("\nEnter device ID to test (press Enter for default): ").strip()
    device_id = None if device_id == "" else int(device_id)
    
    # Test recording
    success = tester.test_recording(device_id=device_id)
    if success:
        logger.info("Recording test completed successfully!")
    else:
        logger.error("Recording test failed!")

if __name__ == "__main__":
    main()
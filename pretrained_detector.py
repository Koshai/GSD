import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import logging
import csv
import io
import requests

logger = logging.getLogger(__name__)

class PretrainedGunshotDetector:
    def __init__(self):
        logger.info("Loading YAMNet model...")
        try:
            # Load the pre-trained YAMNet model
            self.model = hub.load('https://tfhub.dev/google/yamnet/1')
            logger.info("YAMNet model loaded successfully")
            
            # Load class names directly from GitHub
            class_map_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
            response = requests.get(class_map_url)
            class_map_csv = io.StringIO(response.text)
            
            self.class_names = []
            reader = csv.reader(class_map_csv)
            next(reader)  # Skip header
            for row in reader:
                self.class_names.append(row[2])
            
            # Gunshot-related class indices and names in YAMNet
            self.gunshot_classes = [
                417,  # Gunshot, gunfire
                418,  # Machine gun
                419,  # Fusillade, artillery fire
            ]
            
            gunshot_class_names = [self.class_names[i] for i in self.gunshot_classes]
            logger.info(f"Monitoring for classes: {gunshot_class_names}")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
        
    def process_audio(self, audio_data, sample_rate=44100):
        """Process audio data for model input."""
        try:
            # Ensure the input is a numpy array
            audio_data = np.array(audio_data)
            
            # Convert to float32 if not already
            audio_data = audio_data.astype(np.float32)
            
            # Reshape if needed
            if len(audio_data.shape) == 2:
                audio_data = audio_data.flatten()
            
            # Normalize audio to [-1, 1] range
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            logger.debug(f"Audio shape before resample: {audio_data.shape}, Sample rate: {sample_rate}")
            
            # Resample to 16kHz for YAMNet
            if sample_rate != 16000:
                audio_data = librosa.resample(
                    y=audio_data,
                    orig_sr=sample_rate,
                    target_sr=16000
                )
            
            logger.debug(f"Processed audio shape: {audio_data.shape}")
            logger.debug(f"Audio range: [{np.min(audio_data)}, {np.max(audio_data)}]")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            raise
        
    def detect(self, audio_data, sample_rate=44100):
        """Detect gunshots in audio data."""
        try:
            # Process audio
            processed_audio = self.process_audio(audio_data, sample_rate)
            
            # Run inference
            scores, embeddings, spectrogram = self.model(processed_audio)
            scores = scores.numpy()
            
            # Get probabilities for all classes
            all_probs = np.max(scores, axis=0)
            
            # Get maximum probability for gunshot-related classes
            gunshot_probs = scores[:, self.gunshot_classes]
            max_gunshot_prob = np.max(gunshot_probs)
            
            # Get the index of highest gunshot probability
            max_gunshot_class = self.gunshot_classes[np.argmax(np.max(gunshot_probs, axis=0))]
            
            # Find the top 5 detected classes for debugging
            top_5_indices = np.argsort(all_probs)[-5:][::-1]
            
            logger.info("\nTop 5 detected sounds:")
            for idx in top_5_indices:
                if idx < len(self.class_names):  # Ensure index is valid
                    logger.info(f"{self.class_names[idx]}: {all_probs[idx]:.3f}")
            
            if max_gunshot_prob > 0:
                logger.info(f"Detected gunshot type: {self.class_names[max_gunshot_class]}")
            logger.info(f"Gunshot probability: {max_gunshot_prob:.3f}")
            
            return max_gunshot_prob
            
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            return 0.0

    def print_all_classes(self):
        """Debug method to print all available sound classes."""
        for i, name in enumerate(self.class_names):
            print(f"{i}: {name}")
from faster_whisper import WhisperModel

class AudioAnalytics:
    def __init__(self, audio_path: str, model_size: str, device=None, compute_type=None) -> None:
        self.__transcription_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.__audio_path = audio_path
        self.language = None
        self.dead_air_duration = None
        self.sampling_rate = None
        self.duration_after_vad = None
        self.duration = None
        
    def get_language(self):
        pass
    
    def get_dead_air_duration(self):
        pass
    
    def get_sampling_rate(self):
        pass
    
    def get_duration_after_vad(self):
        pass
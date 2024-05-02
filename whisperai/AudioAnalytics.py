from faster_whisper import WhisperModel
from pydub.utils import mediainfo

class AudioAnalytics:
    def __init__(self, audio_path: str, model_size: str, device=None, compute_type=None, **kwargs) -> None:
        self.__transcription_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.__audio_path = audio_path
        
        vad_filter = kwargs.get('vad_filter', True)
        min_silence_duration_ms = kwargs.get('min_silence_duration_ms', 1000)
        temperature=kwargs.get('temperature', 0)
        
        if not kwargs.get('language', None) or not kwargs.get('duration_after_vad', None) or not kwargs.get('duration', None):
            segments, transcript_meta_info = self.__transcription_model(
                self.__audio_path,
                temperature=temperature,
                vad_filter=vad_filter,
                vad_parameters=dict(min_silence_duration_ms=min_silence_duration_ms)
            )
        
            self.language = transcript_meta_info.language
            self.duration_after_vad = transcript_meta_info.duration_after_vad   # In seconds
            self.duration = transcript_meta_info.duration   # In seconds
        else:
            self.language = kwargs['language']
            self.duration_after_vad = kwargs['duration_after_vad']
            self.duration = kwargs['duration']
            
        audio_metadata = mediainfo(self.__audio_path)
        self.sample_rate = audio_metadata['sample_rate']
        self.channels = audio_metadata['channels']
        self.bit_rate = audio_metadata['bit_rate']
        self.codec_name = audio_metadata['codec_name']
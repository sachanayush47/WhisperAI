from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from uuid import uuid4
import os
from pydub import AudioSegment
from pydub.utils import make_chunks
import torch

from whisperai.helpers.utils import get_env

class WhisperAI:
    def __init__(self, model_size: str, device=None, compute_type=None) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type   
        self.transcription_model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        self.diarization_model = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=get_env('HF_AUTH_TOKEN'))
        self.diarization_model.to(torch.device(self.device))

    def __transcriber(self, audio: str, **kwargs):
        vad_filter = kwargs.get('vad_filter', True)
        min_silence_duration_ms = kwargs.get('min_silence_duration_ms', 1000)
        
        options = dict(
            word_timestamps=kwargs.get('word_timestamps', True),
            language=kwargs.get('language', None),
            initial_prompt=kwargs.get('initial_prompt', None),
            task=kwargs.get('task', 'transcribe'),  # Accepted tasks: transcribe, translate
            temperature=kwargs.get('temperature', 0)
        )
        
        segments, transcript_meta_info = self.transcription_model.transcribe(
            audio,
            vad_filter=vad_filter,
            vad_parameters=dict(min_silence_duration_ms=min_silence_duration_ms),
            **options
        )
        
        return segments, transcript_meta_info


    def transcribe(self, audio: str, **kwargs):
        options = dict(
            word_timestamps=False,
            min_silence_duration_ms = kwargs.get('min_silence_duration_ms', 1000),
            vad_filter = kwargs.get('vad_filter', True),
            language=kwargs.get('language', None),
            initial_prompt=kwargs.get('initial_prompt', None),
            task=kwargs.get('task', 'transcribe'),  # Accepted tasks: transcribe, translate
            temperature=kwargs.get('temperature', 0)
        )
        
        segments, transcript_meta_info = self.__transcriber(audio, **options)
        
        transcript = ''
        for segment in segments:
            transcript += segment.text + ' '
        
        return transcript
    
        
    def diarize(self, audio: str, **kwargs):
        """
        Segments the audio into speaker turns.
        It uses pyannote's speaker diarization model.
        Currently, it only supports 2 speakers, it will be extended to support more speakers in the future updates.
        """
        
        OUTPUT_DIRECTORY = 'metadata'
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        
        dual_channel = kwargs.get('dual_channel', False)
        
        options = dict(
            word_timestamps=True,
            language = kwargs.get('language', None),
            initial_prompt = kwargs.get('initial_prompt', None),
            task = kwargs.get('task', 'transcribe'),
            temperature = kwargs.get('temperature', 0),
            vad_filter = kwargs.get('vad_filter', True),
            min_silence_duration_ms = kwargs.get('min_silence_duration_ms', 1000)
        )
        
        diarization = self.diarization_model(audio, min_speakers=1, max_speakers=2)
        diarization_list = list(diarization.itertracks(yield_label=True))

        output = {
            1: [],
            2: []
        }

        for i in diarization_list:
            speaker_label = i[2]
            start = round(i[0].start * 1000)
            end = round(i[0].end * 1000)
            if speaker_label == 'SPEAKER_00':
                output[1].append((start, end))
            else:
                output[2].append((start, end))
                
        file_extension = os.path.splitext(audio)[1].strip('.')
        audio_format = file_extension if file_extension else "mp3"
        mono_audio = AudioSegment.from_file(audio, format=audio_format)

        # Create empty audio segments for both channels
        left_channel = AudioSegment.silent(duration=len(mono_audio))
        right_channel = AudioSegment.silent(duration=len(mono_audio))

        # Function to insert segments into a channel
        def insert_segments(channel, intervals, audio):
            for start, end in intervals:
                segment = audio[start:end]
                channel = channel.overlay(segment, position=start)
            return channel

        # Insert audio segments into the appropriate channels
        left_channel = insert_segments(left_channel, output[1], mono_audio)
        right_channel = insert_segments(right_channel, output[2], mono_audio)

        # Export the left and right channels to separate files
        left_channel_path = os.path.join(OUTPUT_DIRECTORY, f'{uuid4()}_left.wav')
        right_channel_path = os.path.join(OUTPUT_DIRECTORY, f'{uuid4()}_right.wav')
        
        left_channel.export(left_channel_path, format='wav')
        right_channel.export(right_channel_path, format='wav')

        # Export the stereo audio file
        # stereo_audio = AudioSegment.from_mono_audiosegments(left_channel, right_channel)
        # stereo_audio.export(f"{base_file_name}_stereo.mp3", format="mp3")
        
        segments_left, transcript_meta_info_left = self.__transcriber(left_channel_path, **options)
        segments_right, transcript_meta_info_right = self.__transcriber(right_channel_path, **options)

        def get_words(segments, speaker):
            words = []
            for segment in segments:
                for word in segment.words:
                    words.append({
                        'start': word.start,
                        'end': word.end,
                        'word': word.word,
                        'speaker': speaker
                    })
            return words

        words_left = get_words(segments_left, 'A')
        words_right = get_words(segments_right, 'B')

        words_left.extend(words_right)
        sorted_words = sorted(words_left, key=lambda x: x['start'])

        merged_data = []
        for item in sorted_words:
            if merged_data and merged_data[-1]['speaker'] == item['speaker']:
                # Extend the previous entry
                merged_data[-1]['end'] = max(merged_data[-1]['end'], item['end'])
                merged_data[-1]['word'] += ' ' + item['word'].strip()
            else:
                # Add a new entry to the merged list
                merged_data.append(item.copy())

        return merged_data
      
      
    def stream(self, audio: str, **kwargs):
        """
        Streams the segment's text as it is transcribed.
        It does not wait for the entire transcription to finish.
        It does not support diarization. 
        """
        
        options = dict(
            word_timestamps=False,
            vad_filter = kwargs.get('vad_filter', True),
            min_silence_duration_ms = kwargs.get('min_silence_duration_ms', 1000),
            language=kwargs.get('language', None),
            initial_prompt=kwargs.get('initial_prompt', None),
            task=kwargs.get('task', 'transcribe'),  # Accepted tasks: transcribe, translate
            temperature=kwargs.get('temperature', 0)
        )
        
        segments, transcript_meta_info = self.__transcriber(audio, **options)
        
        for segment in segments:
            yield segment.text
        
class Transcript:
    def __init__(self, transcript, **kwargs):
        self.transcript = transcript
        # self.raw_transcript = transcript
        # self.profanity = self.get_profanity()
        # self.summary = self.get_summary()
        # self.language = self.get_language()
        self.duration = kwargs.get('duration')
        self.duration_after_vad = kwargs.get('duration_after_vad', None)

    def get_profanity(self):
        pass
    
    def get_sentiment(self):
        pass
    
    def get_summary(self):
        pass
    
    def get_category(self):
        pass
    
    
    
    
    
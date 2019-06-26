import pyttsx3


class speaker:
    def __init__(self, rate=None, volume=None, voice_id=None, dontspeak=False):
        self.engine = pyttsx3.init()
        if rate is not None:
            self.engine.setProperty('rate', rate)
        if volume is not None:
            self.engine.setProperty('volume', volume)
        if voice_id is not None:
            voices = self.engine.getProperty('voices')
            self.engine.setProperty('voice', voices[voice_id].id)
        self.dontspeak = dontspeak

    def say(self, text, print_text=False):
        if not self.dontspeak:
            self.engine.say(text)
            print_text = True
        if print_text:
            print(text)
        self.engine.runAndWait()

    def asyncsay(self, text, print_text=False):
        if not self.dontspeak:
            self.engine.say(text)
            print_text = True
        if print_text:
            print(text)

    def wait(self):
        self.engine.runAndWait()

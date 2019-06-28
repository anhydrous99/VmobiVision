"""
speech_tools.py
=================================
This module contains speech tools
"""
import pyttsx3


class Speaker:
    """This class wraps pyttsx3 to enable easier use of text-to-speech tools"""
    def __init__(self, rate=None, volume=None, voice_id=None, dontspeak=False):
        """
        Initializes Speaker Class

        :param rate: Set the speech rate here with an integer
        :param volume: Set the volume with a floating point number between 0 and 1
        :param voice_id: Set this to 0 for a male voice or set it to 1 for a female voice
        :param dontspeak: Set this to True if you want the object not to use text to speech
        but to print instead
        """
        self.engine = pyttsx3.init()
        if rate is not None:
            self.engine.setProperty('rate', rate)
        if volume is not None:
            self.engine.setProperty('volume', volume)
        if voice_id is not None:
            voices = self.engine.getProperty('voices')
            self.engine.setProperty('voice', voices[voice_id].id)
        self.dont_speak = dontspeak

    def say(self, text, print_text=False):
        """
        Use this function to use text to speech

        :param text: This is the text to say
        :param print_text: Set this to True if you want the function to also print the text on screen
        """
        if not self.dont_speak:
            self.engine.say(text)
        else:
            print_text = True
        if print_text:
            print(text)
        self.engine.runAndWait()

    def asyncsay(self, text, print_text=False):
        """
        Use this function to queue some text for text-to-speech

        :param text: This is the text to say
        :param print_text: Set this to True if you want the function to also print the text on screen
        """
        if not self.dont_speak:
            self.engine.say(text)
        else:
            print_text = True
        if print_text:
            print(text)

    def runAndWait(self):
        """
        When using the asyncsay function use this function to run queued text
        """
        self.engine.runAndWait()

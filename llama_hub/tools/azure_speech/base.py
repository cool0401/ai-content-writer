"""Azure Speech tool spec."""

from llama_index.tools.tool_spec.base import BaseToolSpec
from typing import Optional, List
import time

class AzureSpeechToolSpec(BaseToolSpec):
    """Azure Speech tool spec."""

    spec_functions = ["speech_to_text", "text_to_speech"]

    def __init__(
        self,
        region: str,
        speech_key: str,
        language: Optional[str] = "en-US"
    ) -> None:
        import azure.cognitiveservices.speech as speechsdk
        """Initialize with parameters."""
        self.config = speechsdk.SpeechConfig(subscription=speech_key, region=region)
        self.config.speech_recognition_language = language

    def text_to_speech(self, text: str) -> None:
        """
        This tool accepts a natural language string and will use Azure speech services to create an
        audio version of the text, and play it on the users computer.

        args:
            text (str): The text to play
        """
        import azure.cognitiveservices.speech as speechsdk
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.config)
        result = speech_synthesizer.speak_text(text)

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            stream = speechsdk.AudioDataStream(result)
            return 'Audio playback complete.'
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))

    def _transcribe(self, speech_recognizer) -> List[str]:
        done = False
        results = []

        def stop_cb(evt) -> None:
            """callback that stop continuous recognition"""
            speech_recognizer.stop_continuous_recognition_async()
            nonlocal done
            done = True

        speech_recognizer.recognized.connect(lambda evt, results=results: results.append(evt.result.text))
        speech_recognizer.session_stopped.connect(stop_cb)
        speech_recognizer.canceled.connect(stop_cb)

        # Start continuous speech recognition
        speech_recognizer.start_continuous_recognition_async()
        while not done:
            time.sleep(0.5)

        return results

    def speech_to_text(self, filename: str) -> List[str]:
        """
        This tool accepts a filename for a speech audio file and uses Azure to transcribe it into text

        args:
            filename (str): The name of the file to transcribe
        """
        import azure.cognitiveservices.speech as speechsdk
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.config,
            audio_config=speechsdk.audio.AudioConfig(filename=filename)
        )
        return self._transcribe(speech_recognizer)

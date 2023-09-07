import unittest
from app import transcription

class Test(unittest.TestCase):

    def test_transcription(self):
        file = "test.wav"
        transcription_text = transcription(file)
        self.assertNotEquals(transcription_text, "")

if __name__ == '__main__':
    unittest.main()

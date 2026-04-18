import threading
import queue
import pyttsx3
import pythoncom
import time

class LocalSpeaker:
    """
    Handles local text-to-speech using pyttsx3 in a dedicated worker thread.
    Optimized for real-time navigation: prioritize the LATEST message.
    """
    def __init__(self):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print("[Speaker] Shared zero-lag TTS initialised.")

    def _worker(self):
        """Dedicated thread for pyttsx3 processing."""
        pythoncom.CoInitialize()
        try:
            while True:
                # Get the next item
                text = self.queue.get()
                if text is None: break
                
                # ZERO-LAG Logic: If there are MORE items in the queue, skip this one.
                # We only care about the absolute latest instruction.
                if not self.queue.empty():
                    self.queue.task_done()
                    continue

                clean_text = text.replace("[FAST]", "").strip()
                
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 190) 
                    engine.setProperty('volume', 1.0)
                    
                    # Optional: Add a small delay if needed, but not for navigation
                    engine.say(clean_text)
                    engine.runAndWait()
                    engine.stop()
                    del engine
                except Exception as e:
                    print(f"[Speaker] Engine error: {e}")
                finally:
                    self.queue.task_done()
        except Exception as e:
            print(f"[Speaker] Thread critical error: {e}")
        finally:
            pythoncom.CoUninitialize()

    def speak(self, text, clear=True):
        """
        Add text to the speech queue.
        If clear=True, it effectively replaces any pending messages with this new one.
        """
        if not text: return
        
        if clear:
            # Drain the queue of any pending (stale) messages
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except queue.Empty:
                    break
                    
        self.queue.put(text)

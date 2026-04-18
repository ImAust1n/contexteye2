import time
import multiprocessing

def tts_worker(q):
    """
    Runs in a completely separate process to avoid any Windows COM thread blocks.
    """
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        
        while True:
            text = q.get()
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()
    except Exception as e:
        print(f"[TTS Worker Error] {e}")

class Narrator:
    def __init__(self):
        self.q = multiprocessing.Queue()
        self.cooldowns = {}
        self.cooldown_sec = 4.0 
        
        # Start background TTS process 
        self.process = multiprocessing.Process(target=tts_worker, args=(self.q,), daemon=True)
        self.process.start()

    def announce(self, detections):
        """
        Evaluates hazards and narrates them based on priority.
        HIGH: object in CENTER + NEAR
        MEDIUM: object in LEFT/RIGHT + NEAR
        LOW: FAR objects
        """
        high_hazards = []
        med_hazards = []
        
        for det in detections:
            if det["distance"] == "NEAR":
                if det["zone"] == "CENTER":
                    high_hazards.append(det)
                else:
                    med_hazards.append(det)
                    
        to_speak = None
        # Prioritize HIGH hazards
        if high_hazards:
            obj = high_hazards[0]
            if obj.get("motion") == "APPROACHING":
                to_speak = f"{obj['label']} approaching directly ahead."
            else:
                to_speak = f"{obj['label']} directly ahead, very close."
        elif med_hazards:
            obj = med_hazards[0]
            to_speak = f"{obj['label']} near on the {obj['zone'].lower()}."
            
        if to_speak:
            now = time.time()
            if to_speak not in self.cooldowns or (now - self.cooldowns[to_speak]) > self.cooldown_sec:
                self.cooldowns[to_speak] = now
                self.q.put(to_speak)
                print(f"[NARRATOR] {to_speak}")

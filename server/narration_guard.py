import re

def select_primary_hazard(hazards: list) -> dict:
    """
    Selects the single most critical hazard from the list.
    Rules:
    - Choose HIGH priority first.
    - If multiple, choose the closest or most central one.
    """
    if not hazards:
        return None
    
    # hazards are already sorted by priority from hazard.py
    # We just take the first one, but could add distance-based tie-breaking here
    return hazards[0]

def clean_text(text: str) -> str:
    """
    Strips noise, formatting, and repeated phrases.
    """
    # Remove bullet points and weird newlines
    text = text.replace("- ", "").replace("\n", " ").strip()
    
    # Remove technical tags like [FAST] or [HIGH] if the LLM leaked them
    text = re.sub(r"\[.*?\]", "", text)
    
    # Simple phrase-based deduplication
    words = text.split()
    seen = []
    for w in words:
        if not seen or w.lower() != seen[-1].lower():
            seen.append(w)
    clean = " ".join(seen)
    
    # Remove repeated neighboring sentences (e.g. "Path clear. Path clear.")
    sentences = [s.strip() for s in clean.split('.') if s.strip()]
    unique_sentences = []
    for s in sentences:
        if not unique_sentences or s.lower() != unique_sentences[-1].lower():
            unique_sentences.append(s)
            
    return ". ".join(unique_sentences) + "." if unique_sentences else ""

def resolve_contradictions(text: str) -> str:
    """
    Ensures safe guidance by removing 'Path clear' when hazards are present.
    """
    has_obstacle = any(kw in text.lower() for kw in ["wall", "obstacle", "person", "chair", "stairs", "door"])
    
    if has_obstacle:
        # If an obstacle is detected, "Path clear" is a dangerous contradiction.
        # Replace it with actionable guidance.
        if "path clear" in text.lower():
            text = re.sub(r"(?i)path clear\.?", "Adjust direction.", text)
            
    return text

def enforce_structure(text: str) -> str:
    """
    Normalizes output to '[Objective]. [Action].' format.
    """
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    if not sentences:
        return "Path clear."
    
    # If we only have one sentence, it might be just the object.
    # We try to ensure there's at least a simple instruction.
    if len(sentences) == 1:
        if "move" not in text.lower() and "stop" not in text.lower() and "turn" not in text.lower():
            # If no action mentioned, append a generic one based on context
            if "wall" in text.lower() or "stairs" in text.lower():
                return f"{sentences[0]}. Stop or turn."
            else:
                return f"{sentences[0]}. Use caution."
                
    return text

def limit_length(text: str, max_sentences=2, max_words_per_sentence=15) -> str:
    """
    Hard-caps length to ensure speed and clarity.
    """
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    final_sentences = []
    
    for s in sentences[:max_sentences]:
        words = s.split()
        if len(words) > max_words_per_sentence:
            s = " ".join(words[:max_words_per_sentence])
        final_sentences.append(s)
        
    return ". ".join(final_sentences) + "." if final_sentences else ""

def is_broken_output(text: str) -> bool:
    """
    Detects truncated or nonsensical output.
    """
    if not text:
        return True
    
    # Ends in the middle of a word or tag
    if re.search(r"[\w]$", text) and not text.endswith("."):
        # If it doesn't end with punctuation, it's likely truncated
        return True
        
    # Check for incomplete brackets or common LLM hallucinations
    if text.count("[") != text.count("]"):
        return True
        
    return False

def process_narration(raw_text: str, hazards: list) -> str:
    """
    Main entry point for the guard layer.
    """
    if is_broken_output(raw_text):
        return "" # Skip broken outputs
        
    text = clean_text(raw_text)
    text = resolve_contradictions(text)
    text = enforce_structure(text)
    text = limit_length(text)
    
    return text

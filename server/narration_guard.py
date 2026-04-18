import re

def select_primary_hazard(hazards: list) -> dict:
    """Selects the single most critical hazard."""
    if not hazards: return None
    return hazards[0]

def select_top_hazards(hazards: list, limit: int = 3) -> list:
    """
    Selects the top N most critical hazards.
    Hazards are already sorted by priority (HIGH > MEDIUM > LOW).
    """
    if not hazards: return []
    return hazards[:limit]

def clean_text(text: str) -> str:
    """
    Strips noise, formatting, filler words, and repeated phrases.
    """
    # Remove bullet points and weird newlines
    text = text.replace("- ", "").replace("\n", " ").strip()
    
    # Remove technical tags
    text = re.sub(r"\[.*?\]", "", text)
    
    # SYLLABLE STRIPPING: Remove words that add delay but no value
    fillers = ["gently", "smoothly", "immediately", "currently", "really", "quickly", "slowly", "very"]
    for word in fillers:
        text = re.sub(rf"(?i)\b{word}\b", "", text)
        
    text = re.sub(r"\s+", " ", text).strip()
    
    # Simple phrase-based deduplication
    words = text.split()
    seen = []
    for w in words:
        if not seen or w.lower() != seen[-1].lower():
            seen.append(w)
    clean = " ".join(seen)
    
    # Remove repeated neighboring sentences
    sentences = [s.strip() for s in clean.split('.') if s.strip()]
    unique_sentences = []
    for s in sentences:
        if not unique_sentences or s.lower() != unique_sentences[-1].lower():
            unique_sentences.append(s)
            
    return ". ".join(unique_sentences) + "." if unique_sentences else ""

def resolve_contradictions(text: str, hazards: list = None) -> str:
    """
    Ensures safe guidance by removing 'Path clear' when hazards are present,
    UNLESS it specifies a directional opening.
    Also replaces vague 'Adjust direction' with specific suggestions if available.
    """
    text_lower = text.lower()
    has_obstacle = any(kw in text_lower for kw in ["wall", "obstacle", "person", "chair", "stairs", "door"])
    
    # Grab the best suggestion from our hazard layer
    primary = select_primary_hazard(hazards)
    suggestion = primary.get("details", {}).get("suggestion") if primary else None

    if has_obstacle:
        # 1. Fix "Path clear" contradictions
        if "path clear" in text_lower:
            if not re.search(r"path clear (on|to|towards) (the )?(left|right)", text_lower):
                text = re.sub(r"(?i)path clear\.?", suggestion if suggestion else "Steer around.", text)
        
        # 2. Fix vague "Adjust direction"
        if "adjust direction" in text_lower and suggestion:
            text = text.replace("Adjust direction", suggestion)
            
    return text

def enforce_structure(text: str, hazards: list = None) -> str:
    """
    Normalizes output to '[Objective]. [Action].' format.
    Uses hazard data to ensure specific directional guidance is matched.
    """
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    if not sentences:
        return ""
    
    # If we only have one sentence, inject specific directional guidance.
    text_lower = text.lower()
    if not any(kw in text_lower for kw in ["move", "stop", "turn", "steer", "veer", "bypass", "path clear", "adjust"]):
        primary = select_primary_hazard(hazards)
        suggestion = primary.get("details", {}).get("suggestion") if primary else None
        
        if suggestion:
            return f"{sentences[0]}. {suggestion}."
        elif "wall" in text_lower or "stairs" in text_lower:
            return f"{sentences[0]}. Stop or turn."
        else:
            return f"{sentences[0]}. Use caution."
                
    return text

def limit_length(text: str, max_sentences=1, max_words_per_sentence=12) -> str:
    """
    Hard-caps length to telegraphic style for speed.
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
    text = resolve_contradictions(text, hazards)
    text = enforce_structure(text, hazards)
    text = limit_length(text)
    
    return text

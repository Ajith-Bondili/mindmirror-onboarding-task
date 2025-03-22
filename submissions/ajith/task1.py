import json
from transformers import pipeline
from preprocessing import preprocess_text, load_journal_entries


""""
I chose j-hartmann/emotion-english-distilroberta-base because...
	•	Efficiency and Speed:
The model is built on a distilled version of RoBERTa. This means it’s much lighter and faster than larger models, making it ideal for a journaling app where you want real-time processing without sacrificing too much accuracy.
	•	Training Data:
It was fine-tuned on the GoEmotions dataset. This dataset consists of human-annotated text examples, which provides a solid foundation for understanding and classifying emotional content from journal entries.
	•	Emotion Classes:
The GoEmotions dataset originally includes 27 detailed emotion labels along with a neutral category. In total, the model supports 28 emotion classes. This wide range of classes allows the model to capture the subtle and mixed emotions that people often express in their journals.
"""
emotion_pipeline = pipeline(
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base", 
    top_k=None
)

def detect_emotions(text_segments, threshold=0.2):
    emotion_scores = {}
    for segment in text_segments:
        results = emotion_pipeline(segment)

        for emotion_data in results[0]:
            emotion = emotion_data["label"]
            score = emotion_data["score"]

             # Keep the highest score for each emotion across all segments
            if emotion not in emotion_scores or score > emotion_scores[emotion]:
                emotion_scores[emotion] = score

    # Filter emotions based on threshold
    detected_emotions = [
        emotion for emotion, score in emotion_scores.items() 
        if score > threshold
    ]

    # If no emotions passed the threshold, include the highest scoring one
    if not detected_emotions and emotion_scores:
        top_emotion = None
        top_score = -1
        for emotion, score in emotion_scores.items():
            if score > top_score:
                top_score = score
                top_emotion = emotion
        detected_emotions = [top_emotion]
        
    return detected_emotions

def process_journal_entries():
     journal_entries = load_journal_entries()
     print(f"Processing {len(journal_entries)} journal entries...")

     results = []
    
     count = 0
     total_entries = len(journal_entries)
    
     for entry in journal_entries:
       content = entry.get("content", "")
       
       processed_segments = preprocess_text(content)
       
       emotions = detect_emotions(processed_segments)
       
       results.append({
          "journal_entry": content,
          "emotions": emotions
       })
    
       # Show progress for me 
       count += 1
       if count % 5 == 0 or count == total_entries:
          print(f"Processed {count}/{total_entries} entries")
    
     return results

def main():
    results = process_journal_entries()
    
    with open("emotion_detection_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("Results saved to emotion_detection_results.json")
    
if __name__ == "__main__":
    main()


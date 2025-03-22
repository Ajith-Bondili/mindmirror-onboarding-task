from transformers import pipeline
from preprocessing import preprocess_text, load_journal_entries


"""
I chose facebook/bart-large-cnn b ecause...

• Great at Summarizing:
BART is really strong at turning long texts into short, clear summaries. It’s been shown to work well for this task, so you can trust it to capture the key ideas.
• Solid Training Background:
It was fine-tuned on the CNN/DailyMail dataset, which features news articles and their highlights. Even though journal entries aren’t news, the model’s knack for picking out the main points makes it a good fit for our needs.
• Output Control:
Using BART lets you easily adjust settings like the maximum and minimum length of the summary. This means you can fine-tune the results so they’re neither too brief nor too wordy—just right for highlighting both the main ideas and any emotions.
• Practical and Reliable:
BART is one of the most popular models on Hugging Face. Its proven performance and efficiency make it perfect for a journaling app where users need quick, insightful summaries of their entries.
"""
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length=60, min_length=20):
    if len(text.split()) < 30:
        return "Text too short for meaningful summarization"
    
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']


def process_journal_entry(entry):
    content = entry.get("content", "")
    processed_segments = preprocess_text(content)

    if len(processed_segments) > 1:
        summaries = []
        for i, segment in enumerate(processed_segments):
            segment_summary = summarize_text(segment)
            summaries.append(segment_summary)
    # Join the summaries with a connecting phrase
        final_summary = " ... ".join(summaries)
    else:
        # For single segment entries
        final_summary = summarize_text(processed_segments[0])
    return {"entry": content,
            "summary": final_summary
            }

def main():
    journal_entries = load_journal_entries()
    print(f"Processing {len(journal_entries)} entries")

    results = []
    for i, entry in enumerate(journal_entries):

        if i % 5 == 0 or i == len(journal_entries):
            print(f"Summarizing entry {i}/{len(journal_entries)}")
        summary_result = process_journal_entry(entry)
        results.append(summary_result)

    with open("journal_summaries.txt", "w") as f:
        for i, result in enumerate(results, 1):
            f.write("### Entry::\n")
            f.write(f"{result['entry']}\n\n")
            f.write("### Generated Summary:\n")
            f.write(f"{result['summary']}\n\n")
    
    
    print("All summaries have been saved to journal_summaries.txt")


if __name__ == "__main__":
    main()

    
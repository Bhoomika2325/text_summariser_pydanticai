import asyncio
from pydantic import BaseModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.agent import Agent
from typing import List
import nltk
import re
import json

# Download punkt for sentence tokenization
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize

# 1. MODEL SETUP 
provider = OpenAIProvider(base_url="http://localhost:11434/v1")
model = OpenAIModel('llama3.2:1b', provider=provider)  
agent = Agent(model=model)

#  2. Pydantic Schema 
class SummaryOutput(BaseModel):
    heading: str
    main_point: str
    action_items: List[str]

#  3. Chunking Logic 
def chunk_text(text, max_words=200):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        words = word_tokenize(sentence)
        if current_length + len(words) <= max_words:
            current_chunk.append(sentence)
            current_length += len(words)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(words)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

#  4. Prompt Template 
def build_prompt(chunk_text):
    print(f"printing type of chunk_text: {type(chunk_text)}")
    return f"""You are an expert at summarizing technical content in strict JSON format.

INPUT TEXT:
{chunk_text}

YOUR TASK:
1. Create a SHORT heading (3-7 words)
2. Write a CONCISE main point (1 sentence)
3. List 3 ACTIONABLE items (bullet points)

REQUIRED JSON FORMAT:
```json
{{
    "heading": "string",
    "main_point": "string",
    "action_items": ["string", "string", "string"]
}}
EXAMPLE:
{{
    "heading": "Spreadsheet Formatting Basics",
    "main_point": "Proper formatting improves spreadsheet readability and functionality.",
    "action_items": [
        "Use clear column headers",
        "Apply consistent cell formatting",
        "Add borders to important sections"
    ]
}}
```"""

#  5. Summarize Each Chunk 
async def run_summary(chunk_text):
    print(f"printing type of chunk_text: {type(chunk_text)}")
    prompt = build_prompt(chunk_text)
    max_retries = 2
    for attempt in range(max_retries):
        try:
            result = await agent.run(prompt)
            if result and hasattr(result, 'output'):
                
                response_text = result.output
                
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start == -1 or json_end == 0:
                    raise ValueError("No JSON found in response")
                
                json_str = response_text[json_start:json_end]
                json_data = json.loads(json_str)
                
                # Validate we got all required fields
                if all(key in json_data for key in ['heading', 'main_point', 'action_items']):
                    return SummaryOutput(**json_data)
                
            elif attempt < max_retries - 1:
                continue
        except Exception as e:
            print(f" Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                continue
    return None

#  6. Summarize All Chunks 
async def summarize_chunks(chunks):
    all_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ({len(word_tokenize(chunk))} words) ---")
        try:
            summary = await run_summary(chunk)
            if summary:
                print(f"Summary: {summary.heading}")
                print(f"Main Point: {summary.main_point}")
                print("Action Items:")
                for j, item in enumerate(summary.action_items, 1):
                    print(f"  {j}. {item}")
                all_summaries.append(summary)
            else:
                print(" No valid summary - possible format issues")
        except Exception as e:
            print(f" Error: {str(e)}")
    return all_summaries
#  7. Final Merge 
def format_final_output(summaries):
    if not summaries:
        return "No valid summaries were generated."

    # Build detailed chunk summaries
    detailed_output = "=== DETAILED CHUNK SUMMARIES ===\n\n"
    for i, summary in enumerate(summaries, 1):
        detailed_output += f"CHUNK {i}: {summary.heading}\n"
        detailed_output += f"Main Point: {summary.main_point}\n"
        detailed_output += "Action Items:\n"
        for j, item in enumerate(summary.action_items, 1):
            detailed_output += f"  {j}. {item}\n"
        detailed_output += "\n"

    # Build consolidated action items (deduplicated)
    seen_actions = set()
    consolidated_output = "=== CONSOLIDATED ACTION ITEMS ===\n"
    action_counter = 1
    action_mapping = {}

    for summary in summaries:
        for item in summary.action_items:
            # Create a normalized version for comparison
            normalized = re.sub(r'[^\w\s]', '', item.lower()).strip()
            if normalized not in seen_actions:
                seen_actions.add(normalized)
                action_mapping[action_counter] = item
                action_counter += 1

    for num, action in sorted(action_mapping.items()):
        consolidated_output += f"{num}. {action}\n"

    return f"{detailed_output}\n{consolidated_output}"

# In your summarize_chunks function, modify the print statement:

#  8. Main Runner 
async def main():
    INPUT_FILE = "sample_input.txt"

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            full_text = f.read()
            print(f"Loaded input file with {len(word_tokenize(full_text))} words")
    except FileNotFoundError:
        print(f"File '{INPUT_FILE}' not found")
        return
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return

    chunks = chunk_text(full_text)
    print(f"\nSplit into {len(chunks)} chunks")
    
    summaries = await summarize_chunks(chunks)

    print("\n=== FINAL SUMMARY ===")
    print(format_final_output(summaries))

if __name__ == "__main__":
    asyncio.run(main())
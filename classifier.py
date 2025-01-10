import os
import json
import dotenv
import pandas as pd
from litellm import acompletion
from pydantic import BaseModel
from typing import Type, TypeVar
import asyncio
import nest_asyncio
import re

dotenv.load_dotenv()
nest_asyncio.apply()

# Type variable for generic JSON parsing
T = TypeVar('T', bound=BaseModel)


def parse_llm_json_response(content: str, model_class: Type[T]) -> T:
    """Parse JSON from LLM response, handling both direct JSON and markdown-fenced output."""
    try:
        # First try parsing as direct JSON
        return model_class.model_validate(json.loads(content))
    except json.JSONDecodeError:
        # If that fails, check for markdown code fence
        if '```json' in content:
            # Extract content between ```json and ```
            json_str = content.split('```json')[1].split('```')[0].strip()
        else:
            # Clean up the string by removing quotes and whitespace
            json_str = content.strip().strip('"\'')
        
        return model_class.model_validate(json.loads(json_str))

# Updated to accept a model_class argument
async def classify_text(prompt, model_class: Type[T]):
    response = await acompletion(
        model="deepseek/deepseek-chat", 
        messages=[
            {"role": "user", "content": prompt}
        ],
        format="json",
        api_key=os.getenv('DEEPSEEK_API_KEY')
    )
    return parse_llm_json_response(response['choices'][0]['message']['content'], model_class)

# Updated to accept a model_class argument
def process_csv(input_file, output_file, prompt_template, model_class: Type[T]):
    # Read the CSV file
    df = pd.read_csv(input_file)

    async def classify_all():
        tasks = []
        for _, row in df.iterrows():
            # Construct the format arguments programmatically
            placeholders = re.findall(r'\{(\w+)\}', prompt_template)
            format_args = {}
            for col in placeholders:
                if col not in row:
                    raise ValueError(f"Column '{col}' in prompt template not found in CSV file.")
                format_args[col] = row.get(col, '')
            current_prompt = prompt_template.format(**format_args)
            tasks.append(classify_text(current_prompt, model_class))
        return await asyncio.gather(*tasks)

    # Apply the classification and expand the result into two columns
    results = asyncio.run(classify_all())
    df[['reason', 'classification']] = [pd.Series(result.model_dump()) for result in results]

    # Write the updated DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_csv = "input.csv"
    output_csv = "output.csv"
    
    # Example of a prompt that uses the 'item' and 'category' columns
    prompt_template = """
    You are a helpful assistant that classifies items as "person", "place", or "thing".

    Return a JSON object with the following fields:
    - reason: a short explanation for the classification
    - classification: "person", "place", or "thing"

    Example output:
    {{
    "reason": "While Mickey Mouse is not technically a human being, he is a character or 'personality' created by Disney and most closely resembles the person category.",
    "classification": "person"
    }}

    Item to classify:
    {item}
    """

    # Define the response model
    class ClassificationResponse(BaseModel):
        reason: str
        classification: str
    
    process_csv(input_csv, output_csv, prompt_template, ClassificationResponse)

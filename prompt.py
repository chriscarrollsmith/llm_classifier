from pydantic import BaseModel

# Define the response model
class ClassificationResponse(BaseModel):
    reason: str
    classification: str

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
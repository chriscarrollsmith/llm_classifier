# LLM Classifier

This project classifies text data from a CSV file using a DeepSeek V3 in JSON mode (though LiteLLM makes it very easy to swap in a different LLM). DeepSeek V3 was chosen for its very low cost, very high throughput, and highly permissive rate limiting.

## Prerequisites

Before you begin, make sure you have the following installed:

1.  **uv**: This project uses `uv` for package management. Install it using the following command:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    See the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/) for more details.

2.  **Python**: Install Python using `uv`:
    ```bash
    uv python install
    ```

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/chriscarrollsmith/llm-classifier.git
    cd llm-classifier
    ```

2.  Install the project dependencies using `uv`:
    ```bash
    uv sync
    ```

3.  Create a `.env` file in the root directory and add your [DeepSeek API key](https://platform.deepseek.com/api_keys):
    ```
    DEEPSEEK_API_KEY=your_deepseek_api_key
    ```

## Usage

1.  Prepare your input CSV file named `input.csv` and define your `prompt_template` and `ClassificationResponse` model in `prompt.py`. Ensure that the column names in your CSV match the placeholders in `prompt_template`. For example, if your prompt uses `{item}`, your CSV should have a column named `item`.

    Note: The classifier supports Pydantic models with nested structures (which will be flattened in the output CSV) and Enum fields. If your output CSV contains rows that already have complete observations, these rows will be preserved unchanged when using the output as input for another run. This allows you to process missed items by feeding the output CSV back in as input.

2.  Run the `classifier.py` script:
    ```bash
    uv run python classifier.py
    ```

    This script will read `input.csv`, classify each item using the LLM, and output the results to `output.csv`.

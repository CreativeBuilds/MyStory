# Interactive Story Generator

This project is an interactive story generator that allows users to create and navigate through branching narratives. It uses AI-powered content generation to create dynamic story chapters based on user choices.

## Features

- Create new stories from initial ideas
- Generate AI-powered story content
- Navigate through branching narratives
- Save and load stories
- Interactive command-line interface

## Dependencies

- Python 3.7+
- OpenAI API (via OpenRouter)
- dotenv
- strictjson (from [TaskGen](https://github.com/simbianai/taskgen))

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install openai python-dotenv
   ```
3. Clone the TaskGen repository and install it:
   ```
   git clone https://github.com/simbianai/taskgen.git
   cd taskgen
   pip install -e .
   ```
4. Create a `.env` file in the project root with your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

## Usage

Run the script using Python:```
python main.py
```


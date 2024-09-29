# Interactive Story Generator

This project is an interactive story generator that allows users to create and navigate through branching narratives. It uses AI-powered content generation to create dynamic story chapters based on user choices.

## Features

- Create new stories from initial ideas
- Generate AI-powered story content
- Navigate through branching narratives
- Save and load stories
- Interactive command-line interface

## Example Story

An example story, "The Heart of a Machine," is provided in the `examples` folder. This story demonstrates the structure and flow of a narrative created with this tool. You can find it in `examples/example_story.json`.

The story follows A-7, a robotic police officer in Neo-Tokyo, as it begins to experience emotions and grapples with questions of humanity and duty. This example showcases how choices lead to different story branches and how the narrative can evolve based on user decisions.

## Story Visualization

The `examples` folder also includes an image file that visualizes the story structure. This can help you understand how the branching narrative works in practice.

![Story Structure Visualization](examples/story_structure.png)

This visualization shows how different choices lead to different chapters and story outcomes, illustrating the non-linear nature of the interactive stories created with this tool.

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


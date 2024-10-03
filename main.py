import os
import sys
import json
from typing import List, Dict, Optional
from strictjson import strict_json
from dotenv import load_dotenv
from openai import OpenAI
import shlex
import ast
import traceback
import uuid
from datetime import datetime
from functools import wraps
from contextlib import contextmanager
import shutil
from collections import deque
import re
import textwrap

try:
    import readline
except ImportError:
    import pyreadline as readline

load_dotenv()

# Initialize OpenAI with OpenRouter base URL and API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# MODEL = "mistralai/mistral-large"
# MODEL = "openai/gpt-4o-mini"
# MODEL = "mistralai/mistral-7b-instruct:nitro"
MODEL = "nousresearch/hermes-3-llama-3.1-70b"
# MODEL = "nousresearch/hermes-3-llama-3.1-405b"

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Set up a permanent folder for saving stories
STORIES_FOLDER = os.path.expanduser("~/Documents/my_stories")
os.makedirs(STORIES_FOLDER, exist_ok=True)

# Set up logging
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log.txt')

# Define the global constant for authors
# DEFAULT_AUTHORS = "Jane Austen:Ernest Hemingway:Virginia Woolf:Gabriel García Márquez:Toni Morrison:George Orwell:Agatha Christie:Haruki Murakami:Leo Tolstoy:Chimamanda Ngozi Adichie"
DEFAULT_AUTHORS = "J.R.R. Tolkien:Terry Goodkind:George R.R. Martin"
# Get authors from environment variable if set, otherwise use default
AUTHORS = os.getenv("AUTHORS", DEFAULT_AUTHORS).split(":")

# Logging and Output Functions
class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.silent = False
        self.privacy_filters = [
            (re.compile(r'/Users/[^/]+/'), r'~/'),
            (re.compile(r'C:\\Users\\[^\\]+\\'), r'C:\\Users\\USER\\'),
            (re.compile(r'/home/[^/]+/'), r'/home/user/'),
        ]
        self.terminal_width = shutil.get_terminal_size().columns - 1

    def apply_privacy_filter(self, message):
        for pattern, replacement in self.privacy_filters:
            message = pattern.sub(replacement, message)
        return message

    def log(self, message, end='\n', nowrap=False):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        filtered_message = self.apply_privacy_filter(message)
        
        # Wrap the message only if nowrap is False
        output_message = filtered_message if nowrap else self.wrap_text(filtered_message)
        
        if not self.silent:
            sys.__stdout__.write(output_message + end)
            sys.__stdout__.flush()
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            log_message = f"[{timestamp}] {filtered_message.strip()}"
            f.write(log_message + end)

    def set_silent(self, silent):
        self.silent = silent

    def log_exception(self, exc_type, exc_value, exc_traceback):
        exception_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        filtered_exception_str = self.apply_privacy_filter(exception_str)
        self.log(f"An exception occurred:\n{filtered_exception_str}")

    def wrap_text(self, text):
        # Split the text into lines
        lines = text.split('\n')
        wrapped_lines = []
        for line in lines:
            # Wrap each line
            wrapped = textwrap.fill(line, width=self.terminal_width, break_long_words=False, replace_whitespace=False)
            wrapped_lines.append(wrapped)
        return '\n'.join(wrapped_lines)

logger = Logger(LOG_FILE)

def custom_print(*args, **kwargs):
    message = ' '.join(map(str, args))
    end = kwargs.get('end', '\n')
    logger.log(message, end=end)

print = custom_print

@contextmanager
def capture_output():
    class OutputCapture:
        def write(self, msg):
            logger.log(msg, end='')
        def flush(self):
            pass

    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = OutputCapture()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = stdout, stderr

def log_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.log(f"Entering {func.__name__}")
        with capture_output():
            result = func(*args, **kwargs)
        logger.log(f"Exiting {func.__name__}")
        return result
    return wrapper

def get_user_input(prompt: str) -> str:
    readline.set_pre_input_hook(lambda: readline.insert_text(''))
    readline.set_completer(lambda text, state: None)
    try:
        user_input = input(prompt)
    finally:
        readline.set_pre_input_hook(None)
    logger.silent = True
    logger.log(f"User input: {user_input}")
    logger.silent = False
    return user_input

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')
    # Removed the logging of console clearing

# Story and Chapter Classes
class ChapterVersion:
    def __init__(self, content: str, call_to_action: str, choices: List[str]):
        self.id = str(uuid.uuid4())
        self.content = parse_content(content)  # Parse content when initializing
        self.call_to_action = call_to_action
        self.choices = choices
        self.children: List[Chapter] = []
        self.created_at = datetime.now()

    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "call_to_action": self.call_to_action,
            "choices": self.choices,
            "children": [child.to_dict() for child in self.children],
            "created_at": self.created_at.isoformat()
        }

class Chapter:
    def __init__(self, id: str, title: str, story_summary: str, user_choice: Optional[str] = None, parent_id: Optional[str] = None):
        self.id = id
        self.title = title
        self.story_summary = story_summary
        self.user_choice = user_choice
        self.parent_id = parent_id
        self.versions: List[ChapterVersion] = []
        self.what_changed: str = ""  # New field for what changed

    def add_version(self, content: str, call_to_action: str, choices: List[str], what_changed: str):
        new_version = ChapterVersion(parse_content(content), call_to_action, choices)
        self.versions.append(new_version)
        self.what_changed = what_changed
        return new_version

    def get_latest_version(self) -> Optional[ChapterVersion]:
        return self.versions[-1] if self.versions else None

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "story_summary": self.story_summary,
            "user_choice": self.user_choice,
            "parent_id": self.parent_id,
            "versions": [v.to_dict() for v in self.versions]
        }

class Story:
    def __init__(self, title: str, description: str, initial_idea: str):
        self.title = title
        self.description = description
        self.initial_idea = initial_idea
        self.root_chapters: List[Chapter] = []
        self.last_active_chapter_id: Optional[str] = None  # New attribute

    def add_chapter(self, chapter: Chapter, parent_id: Optional[str] = None, version_id: Optional[str] = None):
        if parent_id is None:
            self.root_chapters.append(chapter)
        else:
            parent_chapter = self.get_chapter_by_id(parent_id)
            if parent_chapter:
                parent_version = next((v for v in parent_chapter.versions if v.id == version_id), None) if version_id else parent_chapter.get_latest_version()
                if parent_version:
                    parent_version.children.append(chapter)
                else:
                    raise ValueError(f"Version not found for chapter {parent_id}")
            else:
                raise ValueError(f"Parent chapter with id {parent_id} not found")

    def get_chapter_by_id(self, chapter_id: str) -> Optional[Chapter]:
        return self._get_chapter_recursive(self.root_chapters, chapter_id)

    def _get_chapter_recursive(self, chapters: List[Chapter], chapter_id: str) -> Optional[Chapter]:
        for chapter in chapters:
            if chapter.id == chapter_id:
                return chapter
            for version in chapter.versions:
                found = self._get_chapter_recursive(version.children, chapter_id)
                if found:
                    return found
        return None

    def get_last_chapter(self) -> Optional[Chapter]:
        if not self.root_chapters:
            return None
        current = self.root_chapters[-1]
        while current.get_latest_version() and current.get_latest_version().children:
            current = current.get_latest_version().children[-1]
        return current

    def to_dict(self):
        return {
            "title": self.title,
            "description": self.description,
            "initial_idea": self.initial_idea,
            "chapters": [chapter.to_dict() for chapter in self.root_chapters],
            "last_active_chapter_id": self.last_active_chapter_id  # Include in the dictionary
        }

# Parsing and Processing Functions
def parse_list_or_string(data):
    """Parse a string representation of a list or return the original list."""
    if isinstance(data, str):
        try:
            parsed_data = ast.literal_eval(data)
            if isinstance(parsed_data, list):
                return parsed_data
        except (ValueError, SyntaxError):
            return [item.strip() for item in data.strip('[]').split(',')]
    elif isinstance(data, list):
        return data
    return []  # Return an empty list if data is neither a string nor a list

def parse_choices(choices):
    if isinstance(choices, str):
        # Split the string by newlines and remove any empty lines
        choices = [choice.strip() for choice in choices.split('\n') if choice.strip()]
        # Remove any numbering or bullet points at the start of each choice
        choices = [re.sub(r'^\d+\.\s*|\*\s*', '', choice) for choice in choices]
    elif isinstance(choices, list):
        # If it's already a list, just clean up each choice
        choices = [re.sub(r'^\d+\.\s*|\*\s*', '', choice.strip()) for choice in choices]
    return choices

def parse_title_options(options):
    return parse_list_or_string(options)

def process_story_idea(story_idea: str) -> Dict:
    result = strict_json(
        system_prompt="You're chatting with a friend about a story idea. Keep it casual and fun.",
        user_prompt=f"Hey, I've got this story idea: {story_idea}. Can you help me flesh it out a bit? Give me a chill description and maybe 3 cool title ideas. Nothing too fancy, just what comes to mind.",
        output_format={
            "rehashed_description": "A casual, friendly description of the story idea",
            "story_title_options": "Array of 3 potential story titles (keep 'em simple and catchy)"
        },
        model=MODEL,
        **{"client": client}
    )
    
    result["story_title_options"] = parse_title_options(result["story_title_options"])
    return result

def parse_content(content):
    if isinstance(content, list):
        content = '\n\n'.join(content)
    try:
        literal_content = ast.literal_eval(content)
        if isinstance(literal_content, list):
            content = '\n\n'.join(literal_content)
        elif isinstance(literal_content, str):
            content = literal_content
    except (ValueError, SyntaxError):
        pass
    
    # Remove leading/trailing double single quotes
    content = content.strip("''")
    
    return content

# Content Generation Functions
def generate_content(prompt: str, output_format: Dict, system_prompt: str = "", temperature: float = 0.7) -> Dict:
    """Generic function to generate content using the AI model."""
    try:
        logger.silent = True
        logger.log(f"Generating content with prompt: {prompt[:1000]}...")  # Log the first 1000 characters of the prompt
        logger.log(f"System prompt: {system_prompt}")
        logger.log(f"Output format: {output_format}")
        logger.log(f"Temperature: {temperature}")
        logger.silent = False

        # Add author style to system prompt
        author_style = f"Write in the style of one of these authors: {', '.join(AUTHORS)}. "
        system_prompt = author_style + system_prompt

        result = strict_json(
            system_prompt=system_prompt,
            user_prompt=prompt,
            output_format=output_format,
            model=MODEL,
            temperature=temperature,
            **{"client": client}
        )

        logger.silent = True
        logger.log(f"API response received: {str(result)[:1000]}...")  # Log the first 1000 characters of the response
        logger.silent = False

        if result is None:
            logger.log("Error: API returned None response")
            return {"error": "API returned None response"}
        return result
    except Exception as e:
        logger.log(f"Error in generate_content: {str(e)}")
        return {"error": str(e)}

def generate_chapter_content(selected_title: str, selected_chapter_title: str, chapter_number: str, story_summary: str, initial_idea: str, chapter_history: str = "", user_choice: Optional[str] = None, existing_content: Optional[str] = None) -> Dict:
    user_choice_prompt = f"\n\nSo, the reader went with: {user_choice}" if user_choice else ""
    existing_content_prompt = f"\n\nHere's what we've got so far: {existing_content}" if existing_content else ""
    chapter_history_prompt = f"\n\nQuick recap:\n{chapter_history}" if chapter_history else ""
    prompt = (f"Alright, let's write the next bit for '{selected_title}'. We're on {selected_chapter_title}. "
              f"Keep it fun, and engaging. Throw in a question at the end to keep things interesting, "
              f"and give the reader a couple of options for what happens next."
              f"Here's where we're at: {story_summary}\n\n"
              f"{chapter_history_prompt}"
              f"{user_choice_prompt}"
              f"{existing_content_prompt}")
    
    output_format = {
        "content": "The next part of the story, casual and engaging. 3-5 paragraphs. A paragraph is 3-5 sentences. Well formatted with \\n\\n to seperate paragraphs. Do not output a block of text without \\n\\n to seperate paragraphs. When someone speaks dictate this with \\n\\n\\n before and after their dialogue to seperate it from the rest of the text.",
        "chapter_title": "A chill title for this part, based on what happens.",
        "what_changed": "A quick note on how this part shakes things up.",
        "call_to_action": "A casual question to get the reader thinking about what's next.",
        "choices": "A string array of choices for the path of the story to continue. ie. ['path1', 'path2', ...]",
    }
    
    result = generate_content(prompt, output_format, f"You're writting a detailed story. This is part {chapter_number}.")
    
    if "error" in result:
        logger.log(f"Oops, hit a snag generating the chapter: {result['error']}")
        return {
            "content": ["Sorry, something went wrong while coming up with the next part."],
            "chapter_title": f"Part {chapter_number}",
            "what_changed": "We hit a roadblock",
            "call_to_action": "Want to give it another shot?",
            "choices": ["Try again", "Let's back up a bit"]
        }
    
    result["choices"] = parse_choices(result.get("choices", []))
    return result

def generate_expanded_content(story_title: str, chapter_title: str, existing_content: str, initial_idea: str, story_summary: str) -> str:
    prompt = (f'Initial story idea: {initial_idea}\n\n'
              f'Story summary so far: {story_summary}\n\n'
              f'Expand and enhance the following chapter content for the story "{story_title}", chapter "{chapter_title}". '
              f'Story content so far: {existing_content}... -continue here but output the whole content-\n\n'
              f'The expanded content should be at least 50% longer than the original and include new details, dialogue, or descriptions. '
              f'When someone speaks dictate this with \\n\\n\\n before and after their dialogue to seperate it from the rest of the text.'
              f'Do not simply repeat the existing content.')
              
    
    output_format = {
        "expanded_content": "The expanded and enhanced content of the chapter. A paragraph is 4-7 sentences. Well formatted with \\n\\n to seperate paragraphs. When someone speaks dictate this with \\n\\n\\n before and after their dialogue to seperate it from the rest of the text."
    }
    
    result = generate_content(prompt, output_format, "You are a creative writer tasked with expanding and improving an existing chapter. Be creative and add substantial new content.")
    return result["expanded_content"]

def generate_summary(previous_summary: str, new_content: str, initial_idea: str) -> str:
    prompt = (f"Hey, can you give me a quick recap of where we're at in the story? "
              f"Here's how it started: {initial_idea}\n\n"
              f"Last time, we were here: {previous_summary}\n\n"
              f"And we just added this: {new_content}")
    
    output_format = {
        "summary": "A casual, brief summary of the story so far, hitting the main points"
    }
    
    result = generate_content(prompt, output_format, "You're catching a friend up on a story you're writing together. Keep it casual and to the point.")
    try:
        return result["summary"]
    except Exception as e:
        logger.log(f"Whoops, something went wrong with the summary: {str(e)}")
        logger.log(result)
        return previous_summary

def process_custom_choice(custom_choice: str, call_to_action: str) -> str:
    return strict_json(
        system_prompt="You're helping a friend continue their story. Keep it casual and fun.",
        user_prompt=f"So, the story's at this point: '{call_to_action}' and my friend suggested we go with '{custom_choice}'. How should we work that into the story?",
        output_format={
            "rephrased_choice": "A casual way to continue the story based on the friend's idea"
        },
        model=MODEL,
        **{"client": client}
    )["rephrased_choice"]

# File and Story Management Functions
def generate_safe_filename(title: str) -> str:
    return "".join([c for c in title.replace(' ', '_') if c.isalnum() or c in ('-', '_')]).rstrip() + '.json'

def save_story(story: Story):
    safe_filename = generate_safe_filename(story.title)
    story_file = os.path.join(STORIES_FOLDER, safe_filename)
    
    try:
        with open(story_file, 'w', encoding='utf-8') as f:
            json.dump(story.to_dict(), f, indent=2, ensure_ascii=False)
        
        quoted_path = shlex.quote(story_file)
        logger.log(f"\nStory saved successfully to: {quoted_path}")
    except Exception as e:
        logger.log(f"Error saving story: {e}")
        traceback.print_exc()

def load_story(story_title: str) -> Optional[Story]:
    safe_filename = generate_safe_filename(story_title)
    story_file = os.path.join(STORIES_FOLDER, safe_filename)
    
    if not os.path.exists(story_file):
        logger.log(f"No story file found: {story_file}")
        return None

    try:
        with open(story_file, 'r', encoding='utf-8') as f:
            story_data = json.load(f)
        
        story = Story(story_data['title'], story_data['description'], story_data['initial_idea'])
        story.root_chapters = _load_chapters_recursive(story_data['chapters'])
        story.last_active_chapter_id = story_data.get('last_active_chapter_id')  # Load the last active chapter ID
        return story
    except Exception as e:
        logger.log(f"Error loading story: {e}")
        traceback.print_exc()
        return None

def _load_chapters_recursive(chapters_data: List[Dict]) -> List[Chapter]:
    chapters = []
    for chapter_data in chapters_data:
        chapter = Chapter(
            chapter_data['id'],
            chapter_data['title'],
            chapter_data['story_summary'],
            chapter_data.get('user_choice'),
            chapter_data['parent_id']
        )
        chapter.versions = [_load_chapter_version(v) for v in chapter_data['versions']]
        chapters.append(chapter)
    return chapters

def _load_chapter_version(version_data: Dict) -> ChapterVersion:
    version = ChapterVersion(
        version_data['content'],
        version_data['call_to_action'],
        version_data['choices']
    )
    version.id = version_data['id']
    version.created_at = datetime.fromisoformat(version_data['created_at'])
    version.children = _load_chapters_recursive(version_data.get('children', []))
    return version

def list_existing_stories():
    stories = []
    try:
        for filename in os.listdir(STORIES_FOLDER):
            if filename.endswith('.json'):
                file_path = os.path.join(STORIES_FOLDER, filename)
                if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
                    story_title = filename[:-5].replace('_', ' ')
                    stories.append(story_title)
    except FileNotFoundError:
        logger.log(f"Stories folder not found: {STORIES_FOLDER}")
    except PermissionError:
        logger.log(f"Permission denied when accessing stories folder: {STORIES_FOLDER}")
    return stories

# Chapter and Version Management Functions
def increment_suffix(suffix: str) -> str:
    if suffix == 'z':
        return 'aa'
    elif len(suffix) > 1 and suffix[-1] == 'z':
        return suffix[:-1] + 'a'
    return suffix[:-1] + chr(ord(suffix[-1]) + 1)

def generate_chapter_id(parent_id: Optional[str], story: Story) -> str:
    if parent_id is None or parent_id == "":
        # Generate a root chapter ID
        existing_ids = [chapter.id for chapter in story.root_chapters]
        next_number = max([int(id) for id in existing_ids], default=0) + 1
        return f"{next_number}"
    else:
        parent_chapter = story.get_chapter_by_id(parent_id)
        if not parent_chapter:
            raise ValueError(f"Parent chapter with id {parent_id} not found")
        
        # Get all direct children of the parent chapter
        children = parent_chapter.get_latest_version().children
        
        if not children:
            # First child of this parent
            return f"{int(parent_id) + 1}"
        else:
            # Check if there's already a non-lettered sibling
            non_lettered_sibling = next((child for child in children if child.id.isdigit()), None)
            if non_lettered_sibling:
                # If there's a non-lettered sibling, start lettering
                existing_letters = [child.id[-1] for child in children if not child.id.isdigit()]
                if not existing_letters:
                    return f"{non_lettered_sibling.id}a"
                else:
                    next_letter = chr(ord(max(existing_letters)) + 1)
                    return f"{non_lettered_sibling.id}{next_letter}"
            else:
                # If no non-lettered sibling, create one
                return f"{int(parent_id) + 1}"

def display_chapter_tree(chapters: List[Chapter], indent: str = "", current_version: Optional[ChapterVersion] = None):
    for chapter in chapters:
        latest_version = chapter.get_latest_version()
        version_indicator = "*V*" if len(chapter.versions) > 1 else ""
        logger.log(f"{indent}{chapter.id}: {chapter.title} {version_indicator}")
        if latest_version:
            display_chapter_tree(latest_version.children, indent + "  ", current_version)

def expand_chapter_content(chapter: Chapter, story: Story) -> ChapterVersion:
    latest_version = chapter.get_latest_version()
    if not latest_version:
        raise ValueError("Chapter has no versions")

    original_content = latest_version.content
    attempts = 0
    max_attempts = 3

    while attempts < max_attempts:
        expanded_content = generate_expanded_content(
            story.title,
            chapter.title,
            original_content,
            story.initial_idea,
            chapter.story_summary
        )

        if expanded_content != original_content and len(expanded_content) > len(original_content):
            chapter_history = get_chapter_history(story, chapter)
            new_chapter_content = generate_chapter_content(
                selected_title=story.title,
                selected_chapter_title=chapter.title,
                chapter_number=chapter.id,
                story_summary=chapter.story_summary,
                initial_idea=story.initial_idea,
                existing_content=expanded_content,
                chapter_history=chapter_history
            )

            return chapter.add_version(
                parse_content(new_chapter_content["content"]),
                new_chapter_content["call_to_action"],
                new_chapter_content["choices"],
                new_chapter_content["what_changed"]
            )
        
        attempts += 1

    logger.log("Failed to generate expanded content after multiple attempts.")
    return latest_version  # Return the original version if expansion fails

def display_version_menu(chapter: Chapter):
    logger.log(f"\nVersions for Chapter {chapter.id}: {chapter.title}")
    logger.log("-" * 40)
    for i, version in enumerate(reversed(chapter.versions), 1):
        logger.log(f"{i}. Created at: {version.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("\nPress 'b' to go back")
    choice = get_user_input("Select a version or go back: ")
    if choice.lower() == 'b':
        return None
    if choice.isdigit() and 1 <= int(choice) <= len(chapter.versions):
        return chapter.versions[-(int(choice))]
    return None

# UI and Formatting Functions
def get_terminal_width() -> int:
    try:
        return shutil.get_terminal_size().columns
    except AttributeError:
        return 80  # Default width if unable to determine

def center_text(text: str, fill_char: str = '=') -> str:
    total_width = get_terminal_width()
    text_length = len(text)
    if text_length >= total_width:
        return text
    left_padding = (total_width - text_length) // 2
    right_padding = total_width - text_length - left_padding
    return f"{fill_char * left_padding}{text}{fill_char * right_padding}"

def draw_line(fill_char: str = '=') -> str:
    return fill_char * get_terminal_width()

# Update these functions to use the nowrap parameter
def print_centered_text(text: str, fill_char: str = '='):
    logger.log(center_text(text, fill_char), nowrap=True)

def print_line(fill_char: str = '='):
    logger.log(draw_line(fill_char), nowrap=True)

def get_chapter_history(story: Story, current_chapter: Chapter, n: int = 6) -> str:
    history = []
    chapter = current_chapter
    chapters = deque(maxlen=n)

    while chapter and len(chapters) < n:
        chapters.appendleft(chapter)
        chapter = story.get_chapter_by_id(chapter.parent_id) if chapter.parent_id else None

    if chapter:
        history.append(f"[ Summary of story before start of recent chapter history ]\n{chapter.story_summary}\n")

    for ch in chapters:
        latest_version = ch.get_latest_version()
        history.append(draw_line('='))
        history.append(center_text(f" {ch.title} ", '='))
        history.append(draw_line('='))
        history.append(parse_content(latest_version.content))
        history.append(draw_line('-'))
        history.append(ch.what_changed)
        history.append(draw_line('-'))
        history.append(latest_version.call_to_action)
        history.append(draw_line('-'))
        history.append(f"[ User chose: {ch.user_choice} ]" if ch.user_choice else "[ Root chapter ]")
        history.append(draw_line('='))

    return "\n\n".join(history)

def refine_story_idea(initial_idea: str) -> Dict:
    original_idea = initial_idea
    user_feedback = ""
    latest_description = ""
    while True:
        prompt = f"""
        Original idea: {original_idea}
        Latest description: {latest_description}
        User feedback: {user_feedback}

        Based on the original idea, the latest description (if any), and the user's feedback,
        provide a refined description of the story idea. If this is the first iteration,
        focus on expanding and clarifying the original idea.
        """
        
        story_process = process_story_idea(prompt)
        
        clear_console()
        logger.log(f"\nRefined Description:\n{story_process['rehashed_description']}\n")
        logger.log("Does this description accurately capture your story idea?")
        logger.log("Enter 'y' to accept, or provide feedback to further refine the description.")
        
        user_response = get_user_input("Your response: ").strip().lower()
        
        if user_response == 'y':
            logger.log("\nGreat! The description is finalized.")
            return story_process
        else:
            latest_description = story_process['rehashed_description']
            user_feedback = user_response

# Main Editor Function
@log_function
def editor_mode():
    print("Welcome to the Story Editor!")

    # List existing stories
    existing_stories = list_existing_stories()
    if existing_stories:
        logger.log("\nExisting stories:")
        for i, story_title in enumerate(existing_stories, 1):
            logger.log(f"{i}. {story_title}")
        logger.log("\nEnter a number to load a story, or 'n' to start a new story")
    else:
        logger.log("No existing stories found. Starting a new story.")
        story = None
        choice = 'n'

    if existing_stories:
        choice = get_user_input("Your choice: ")
    
    if choice.lower() == 'n':
        story = None
    elif choice.isdigit() and 1 <= int(choice) <= len(existing_stories):
        story_title = existing_stories[int(choice) - 1]
        story = load_story(story_title)
        if story:
            logger.log(f"\nLoaded existing story: {story.title}")
            update_chapter_ids(story)
            save_story(story)
        else:
            logger.log(f"\nFailed to load story: {story_title}")
            return
    else:
        logger.log("Invalid choice. Starting a new story.")
        story = None

    if not story:
        # Create a new story
        initial_idea = get_user_input("Enter your story idea: ")
        story_process = refine_story_idea(initial_idea)

        logger.log("\nStory Title Options:")
        for index, title in enumerate(story_process["story_title_options"], 1):
            logger.log(f"{index}. {title}")
        logger.log("\nOr enter your own title")

        title_choice = get_user_input("Select a story title or enter your own: ")
        
        if title_choice.isdigit() and 1 <= int(title_choice) <= len(story_process["story_title_options"]):
            selected_title = story_process["story_title_options"][int(title_choice) - 1]
        else:
            selected_title = title_choice.strip()

        story = Story(selected_title, story_process["rehashed_description"], initial_idea)

    if story:
        if story.last_active_chapter_id:
            current_chapter = story.get_chapter_by_id(story.last_active_chapter_id)
            if current_chapter:
                current_chapter_id = current_chapter.id
                story_summary = current_chapter.story_summary
            else:
                current_chapter = story.get_last_chapter()
                current_chapter_id = current_chapter.id if current_chapter else None
                story_summary = story.description
        else:
            current_chapter = story.get_last_chapter()
            current_chapter_id = current_chapter.id if current_chapter else None
            story_summary = story.description
    else:
        current_chapter_id = None
        story_summary = story.description

    story_modified = False

    while True:
        if current_chapter_id:
            current_chapter = story.get_chapter_by_id(current_chapter_id)
            current_version = current_chapter.get_latest_version()
        else:
            current_chapter = None
            current_version = None

        clear_console()
        if current_chapter and current_version:
            # Display story summary
            print_centered_text(" Story Summary ")
            logger.log(story_summary)
            print_line()
            logger.log("\n")

            # Display current chapter data
            logger.silent = True
            logger.log(f"ID: {current_chapter.id}")
            logger.log(f"Title: {current_chapter.title}")
            logger.log(f"Version: {current_version.id}")
            logger.log(f"Created at: {current_version.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.silent = False
            print_centered_text(f" {current_chapter.title} ")
            logger.log("\n")
            logger.log(parse_content(current_version.content))
            logger.log("\n")
            print_centered_text(" What Next? ", fill_char='-')
            logger.log("\n")
            logger.log(f"{current_version.call_to_action}")
            logger.log("\n")
            print_centered_text(" Choices ", fill_char='-')

            # Display AI-suggested choices
            for i, choice in enumerate(parse_choices(current_version.choices), 1):
                logger.log(f"{i}. (*) {choice}")

            # Display existing child chapters and their choices
            existing_children = current_version.children
            for i, child in enumerate(existing_children, len(current_version.choices) + 1):
                chapter_id = child.id
                if chapter_id.isdigit():
                    chapter_display = f"Chapter {chapter_id}"
                else:
                    chapter_display = f"Chapter {chapter_id[:-1]}{chapter_id[-1].lower()}"
                logger.log(f"{i}. ({chapter_display}) {child.user_choice}")

            total_choices = len(current_version.choices) + len(existing_children)
            
            logger.log("\nOr enter your own idea")
            logger.log("\nPress b to go back")
            logger.log("Press m to open menu")
            logger.log("Press e to expand the current chapter")
            logger.log("Press vm to view version menu")
            print_line()

            choice = get_user_input("Select a choice, enter your own idea, or navigate: ").strip().lower()
            
            if choice in ['b', 'm', 'vm', 'e']:
                if choice == 'b':
                    # Go back to the previous chapter
                    if current_chapter.parent_id:
                        parent_chapter = story.get_chapter_by_id(current_chapter.parent_id)
                        if parent_chapter:
                            current_chapter_id = parent_chapter.id
                            # Update the story summary to the parent chapter's summary
                            story_summary = parent_chapter.story_summary
                        else:
                            logger.log("Error: Parent chapter not found.")
                    else:
                        logger.log("You're already at the start of the story.")
                elif choice == 'm':
                    # Show menu of all chapters
                    clear_console()
                    logger.log("\nAll Chapters:")
                    logger.log(f"Title: {story.title}")
                    logger.log("-" * 40)
                    display_chapter_tree(story.root_chapters, current_version=current_version)
                    logger.log("-" * 40)
                    back_choice = get_user_input("Enter the ID of the chapter you want to go to (or press Enter to stay): ")
                    if back_choice:
                        selected_chapter = story.get_chapter_by_id(back_choice)
                        if selected_chapter and selected_chapter.id == back_choice:
                            current_chapter_id = back_choice
                            story_summary = selected_chapter.story_summary
                        else:
                            logger.log(f"Chapter with ID {back_choice} not found.")
                elif choice == 'vm':
                    # Show version menu
                    clear_console()
                    selected_version = display_version_menu(current_chapter)
                    if selected_version:
                        current_version = selected_version
                elif choice == 'e':
                    # Expand the current chapter
                    new_version = expand_chapter_content(current_chapter, story)
                    logger.log("\nChapter expanded successfully!")
                    story_modified = True
                    # Update the current version to the new expanded version
                    current_version = new_version
                    # Don't update the story summary here
            elif choice.isdigit() and 1 <= int(choice) <= total_choices:
                choice_index = int(choice) - 1
                if choice_index < len(current_version.choices):
                    # AI-suggested choice
                    selected_choice = current_version.choices[choice_index]
                    existing_child = next((child for child in current_version.children if child.user_choice == selected_choice), None)
                    if existing_child:
                        current_chapter_id = existing_child.id
                        story_summary = generate_summary(current_chapter.story_summary, existing_child.get_latest_version().content, story.initial_idea)
                    else:
                        # Generate new chapter for AI-suggested choice
                        new_chapter_id = generate_chapter_id(current_chapter.id, story)
                        chapter_history = get_chapter_history(story, current_chapter)
                        logger.log("Generating new chapter content for AI-suggested choice...")
                        chapter_content = generate_chapter_content(
                            selected_title=story.title,
                            selected_chapter_title=f"Chapter {new_chapter_id}",
                            chapter_number=new_chapter_id,
                            story_summary=story_summary,
                            initial_idea=story.initial_idea,
                            chapter_history=chapter_history,
                            user_choice=selected_choice
                        )
                        if "error" in chapter_content:
                            logger.log(f"Error occurred while generating chapter content: {chapter_content['error']}")
                            continue  # Skip to the next iteration of the loop

                        logger.log("New chapter content generated successfully for AI-suggested choice.")
                        new_chapter = Chapter(
                            new_chapter_id,
                            chapter_content["chapter_title"],  # Use the AI-generated title
                            story_summary,
                            selected_choice,
                            current_chapter.id
                        )
                        new_chapter.add_version(
                            parse_content(chapter_content["content"]),
                            chapter_content["call_to_action"],
                            chapter_content["choices"],
                            chapter_content["what_changed"]
                        )
                        story.add_chapter(new_chapter, current_chapter.id)
                        current_chapter_id = new_chapter.id
                        story_modified = True
                        logger.log(f"New chapter {new_chapter_id} added to the story for AI-suggested choice.")
                else:
                    # Existing child chapter
                    existing_child = existing_children[choice_index - len(current_version.choices)]
                    current_chapter_id = existing_child.id
                    story_summary = generate_summary(current_chapter.story_summary, existing_child.get_latest_version().content, story.initial_idea)
            else:
                # Handle custom choice (generate new chapter)
                new_chapter_id = generate_chapter_id(current_chapter.id, story)
                custom_choice = choice.strip()
                chapter_history = get_chapter_history(story, current_chapter)
                logger.log("Generating new chapter content for custom choice...")
                chapter_content = generate_chapter_content(
                    selected_title=story.title,
                    selected_chapter_title=f"Chapter {new_chapter_id}",
                    chapter_number=new_chapter_id,
                    story_summary=story_summary,
                    initial_idea=story.initial_idea,
                    chapter_history=chapter_history,
                    user_choice=custom_choice
                )
                if "error" in chapter_content:
                    logger.log(f"Error occurred while generating chapter content: {chapter_content['error']}")
                    continue  # Skip to the next iteration of the loop

                logger.log("New chapter content generated successfully for custom choice.")
                new_chapter = Chapter(
                    new_chapter_id,
                    chapter_content["chapter_title"],  # Use the AI-generated title
                    story_summary,
                    custom_choice,
                    current_chapter.id
                )
                new_chapter.add_version(
                    parse_content(chapter_content["content"]),
                    chapter_content["call_to_action"],
                    chapter_content["choices"],
                    chapter_content["what_changed"]
                )
                story.add_chapter(new_chapter, current_chapter.id)
                current_chapter_id = new_chapter.id
                story_modified = True
                logger.log(f"New chapter {new_chapter_id} added to the story for custom choice.")

        else:
            # Handle the case when there's no current chapter (start of the story)
            new_chapter_id = generate_chapter_id(None, story)
            chapter_history = "This is the first chapter of the story."
            chapter_content = generate_chapter_content(
                selected_title=story.title,
                selected_chapter_title=f"Chapter {new_chapter_id}",
                chapter_number=new_chapter_id,
                story_summary=story_summary,
                initial_idea=story.initial_idea,
                chapter_history=chapter_history,
                user_choice=None
            )
            new_chapter = Chapter(
                new_chapter_id,
                chapter_content["chapter_title"],  # Use the AI-generated title
                story_summary,
                None,
                None
            )
            new_chapter.add_version(
                parse_content(chapter_content["content"]),
                chapter_content["call_to_action"],
                chapter_content["choices"],
                chapter_content["what_changed"]
            )
            story.add_chapter(new_chapter, None)
            current_chapter_id = new_chapter.id
            story_modified = True

        # Update the story summary only when a new chapter is added
        if story_modified and current_chapter_id != story.last_active_chapter_id:
            story_summary = generate_summary(story_summary, current_version.content if current_version else "", story.initial_idea)

        # Update the last_active_chapter_id before saving
        story.last_active_chapter_id = current_chapter_id

        # Save the story only if it has been modified
        if story_modified:
            save_story(story)
            story_modified = False  # Reset the flag after saving

# Initialization Function
def initialize_log():
    if not (os.getenv("C_LOG") == "False"):
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("")
    logger.silent = True
    logger.log(f"{'='*50}")
    logger.log(f"New session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"{'='*50}\n")
    logger.silent = False
    # clear the console
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Set up exception handling to log all unhandled exceptions
    sys.excepthook = logger.log_exception

def update_chapter_ids(story: Story):
    def update_recursive(chapters: List[Chapter], parent_id: Optional[str] = None):
        for i, chapter in enumerate(chapters):
            if parent_id is None:
                chapter.id = str(i + 1)
            else:
                if i == 0:
                    chapter.id = str(int(parent_id) + 1)
                else:
                    chapter.id = f"{int(parent_id) + 1}{chr(ord('a') + i - 1)}"
            chapter.parent_id = parent_id
            latest_version = chapter.get_latest_version()
            if latest_version:
                update_recursive(latest_version.children, chapter.id)

    update_recursive(story.root_chapters)

if __name__ == "__main__":
    initialize_log()
    editor_mode()
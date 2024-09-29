import os
import json
from typing import List, Dict, Optional
from strictjson import strict_json
from dotenv import load_dotenv
from openai import OpenAI
import shlex
import ast  # Add this at the top of the file with other imports
import traceback

load_dotenv()

# Initialize OpenAI with OpenRouter base URL and API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# MODEL = "mistralai/mistral-large"
MODEL = "openai/gpt-4o-mini"

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Set up a permanent folder for saving stories
STORIES_FOLDER = os.path.expanduser("~/Documents/my_stories")
os.makedirs(STORIES_FOLDER, exist_ok=True)

class Chapter:
    def __init__(self, id: str, title: str, content: str, call_to_action: str, choices: List[str], story_summary: str, user_choice: Optional[str] = None, parent_id: Optional[str] = None):
        self.id = id
        self.title = title
        self.content = content
        self.call_to_action = call_to_action
        self.choices = choices
        self.story_summary = story_summary
        self.user_choice = user_choice
        self.parent_id = parent_id
        self.children: List[Chapter] = []

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "call_to_action": self.call_to_action,
            "choices": self.choices,
            "story_summary": self.story_summary,
            "user_choice": self.user_choice,
            "parent_id": self.parent_id,
            "children": [child.to_dict() for child in self.children]
        }

class Story:
    def __init__(self, title: str, description: str, initial_idea: str):
        self.title = title
        self.description = description
        self.initial_idea = initial_idea
        self.root_chapters: List[Chapter] = []

    def add_chapter(self, chapter: Chapter, parent_id: Optional[str] = None):
        if parent_id is None:
            self.root_chapters.append(chapter)
        else:
            self._add_chapter_recursive(self.root_chapters, chapter, parent_id)

    def _add_chapter_recursive(self, chapters: List[Chapter], new_chapter: Chapter, parent_id: str):
        for chapter in chapters:
            if chapter.id == parent_id:
                chapter.children.append(new_chapter)
                return True
            if self._add_chapter_recursive(chapter.children, new_chapter, parent_id):
                return True
        return False

    def get_chapter_by_id(self, chapter_id: str) -> Optional[Chapter]:
        return self._get_chapter_recursive(self.root_chapters, chapter_id)

    def _get_chapter_recursive(self, chapters: List[Chapter], chapter_id: str) -> Optional[Chapter]:
        for chapter in chapters:
            if chapter.id == chapter_id:
                return chapter
            found = self._get_chapter_recursive(chapter.children, chapter_id)
            if found:
                return found
        return None

    def to_dict(self):
        return {
            "title": self.title,
            "description": self.description,
            "initial_idea": self.initial_idea,
            "chapters": [chapter.to_dict() for chapter in self.root_chapters]
        }

def get_user_input(prompt: str) -> str:
    return input(prompt)

def process_story_idea(story_idea: str) -> Dict:
    return strict_json(
        system_prompt="You are a helpful assistant that processes story ideas and generates title options.",
        user_prompt=f"Process this story idea and provide a rehashed description and 3 title options: {story_idea}",
        output_format={
            "rehashed_description": "A reworded description of the story idea",
            "story_title_options": "Array of 3 potential story titles"
        },
        model=MODEL,
        **{"client": client}
    )

def generate_chapter_content(selected_title: str, selected_chapter_title: str, chapter_number: int, story_summary: str, initial_idea: str, user_choice: Optional[str] = None) -> Dict:
    user_choice_prompt = f"\n\nUser chose: {user_choice}" if user_choice else ""
    return strict_json(
        system_prompt=f"You are a creative writer tasked with writing a short chapter. The chapter number is {chapter_number}.",
        user_prompt=f'Write a chapter titled "{selected_chapter_title}" for the story "{selected_title}". The chapter should be roughly 500 characters. Include a call to action and at least two choices for the reader to continue the story. Initial story idea: {initial_idea}\n\nStory summary so far: {story_summary}{user_choice_prompt}',
        output_format={
            "content": "The content of the chapter in the format of an array of paragraphs.",
            "call_to_action": "A compelling call to action for the reader.",
            "choices": "An array of at least two possible choices for the reader to continue the story."
        },
        model=MODEL,
        **{"client": client}
    )

def process_custom_choice(custom_choice: str, call_to_action: str) -> str:
    return strict_json(
        system_prompt="You are a helpful assistant that rephrases user choices into story continuations.",
        user_prompt=f'Rephrase this custom choice as a continuation of the story, based on the call to action: "{call_to_action}". Custom choice: "{custom_choice}"',
        output_format={
            "rephrased_choice": "A rephrased version of the custom choice that fits the story context"
        },
        model=MODEL,
        **{"client": client}
    )["rephrased_choice"]

def generate_safe_filename(title: str) -> str:
    return "".join([c for c in title.replace(' ', '_') if c.isalnum() or c in ('-', '_')]).rstrip() + '.json'

def save_story(story: Story):
    safe_filename = generate_safe_filename(story.title)
    story_file = os.path.join(STORIES_FOLDER, safe_filename)
    
    try:
        with open(story_file, 'w', encoding='utf-8') as f:
            json.dump(story.to_dict(), f, indent=2, ensure_ascii=False)
        
        quoted_path = shlex.quote(story_file)
        print(f"\nStory saved successfully to: {quoted_path}")
    except Exception as e:
        print(f"Error saving story: {e}")
        traceback.print_exc()

def load_story(story_title: str) -> Optional[Story]:
    safe_filename = generate_safe_filename(story_title)
    story_file = os.path.join(STORIES_FOLDER, safe_filename)
    
    if not os.path.exists(story_file):
        print(f"No story file found: {story_file}")
        return None

    try:
        with open(story_file, 'r', encoding='utf-8') as f:
            story_data = json.load(f)
        
        story = Story(story_data['title'], story_data['description'], story_data['initial_idea'])
        for chapter_data in story_data['chapters']:
            story.root_chapters.append(_load_chapter_recursive(chapter_data))
        return story
    except Exception as e:
        print(f"Error loading story: {e}")
        traceback.print_exc()
        return None

def _load_chapter_recursive(chapter_data: Dict) -> Chapter:
    chapter = Chapter(
        chapter_data['id'],
        chapter_data['title'],
        chapter_data['content'],
        chapter_data['call_to_action'],
        chapter_data['choices'],
        chapter_data['story_summary'],
        chapter_data.get('user_choice'),  # Use .get() to handle cases where user_choice might not exist in older saves
        chapter_data['parent_id']
    )
    for child_data in chapter_data['children']:
        chapter.children.append(_load_chapter_recursive(child_data))
    return chapter

def generate_summary(previous_summary: str, new_content: str, initial_idea: str) -> str:
    return strict_json(
        system_prompt="You are a helpful assistant that summarizes story progress.",
        user_prompt=f"Given the following information, provide a concise summary of the story so far, including the initial idea:\n\nInitial idea: {initial_idea}\n\nPrevious summary: {previous_summary}\n\nNew content: {new_content}",
        output_format={
            "summary": "A concise summary of the story progress, including key events and decisions"
        },
        model=MODEL,
        **{"client": client}
    )["summary"]

def increment_suffix(suffix: str) -> str:
    if suffix == 'z':
        return 'aa'
    elif len(suffix) > 1 and suffix[-1] == 'z':
        return suffix[:-1] + 'a'
    return suffix[:-1] + chr(ord(suffix[-1]) + 1)

def generate_chapter_id(parent_id: Optional[str], story: Story) -> str:
    if parent_id is None:
        # Generate a unique root chapter ID (e.g., 1a, 2a, 3a, ...)
        existing_ids = [chapter.id for chapter in story.root_chapters]
        numbers = [int(id[:-1]) for id in existing_ids if id[:-1].isdigit()]
        next_number = max(numbers, default=0) + 1
        return f"{next_number}a"
    else:
        parent_chapter = story.get_chapter_by_id(parent_id)
        if not parent_chapter:
            return "1a"  # Fallback ID if parent not found
        
        # Find the highest sibling ID
        sibling_ids = [child.id for child in parent_chapter.children]
        if not sibling_ids:
            # First child of the parent
            parent_number = int(parent_id[:-1])
            return f"{parent_number + 1}a"
        
        # Get the last sibling's numeric part and suffix
        last_sibling_number = int(sibling_ids[-1][:-1])
        last_sibling_suffix = sibling_ids[-1][-1]
        
        # If it's 'a', increment the suffix
        if last_sibling_suffix == 'a':
            return f"{last_sibling_number}b"
        else:
            # If it's not 'a', increment the number and use 'a'
            return f"{last_sibling_number + 1}a"

def display_chapter_tree(chapters: List[Chapter], indent: str = ""):
    for chapter in chapters:
        print(f"{indent}{chapter.id}: {chapter.title}")
        display_chapter_tree(chapter.children, indent + "  ")

def parse_content(content):
    if isinstance(content, list):
        return '\n\n'.join(content)
    return content

def list_existing_stories():
    stories = []
    for filename in os.listdir(STORIES_FOLDER):
        if filename.endswith('.json'):
            story_title = filename[:-5].replace('_', ' ')
            stories.append(story_title)
    return stories

def editor_mode():
    print("Welcome to the Story Editor!")

    # List existing stories
    existing_stories = list_existing_stories()
    if existing_stories:
        print("\nExisting stories:")
        for i, story_title in enumerate(existing_stories, 1):
            print(f"{i}. {story_title}")
        print("\nEnter a number to load a story, or 'n' to start a new story")

        choice = get_user_input("Your choice: ")
        
        if choice.lower() == 'n':
            story = None
        elif choice.isdigit() and 1 <= int(choice) <= len(existing_stories):
            story_title = existing_stories[int(choice) - 1]
            story = load_story(story_title)
            if story:
                print(f"\nLoaded existing story: {story.title}")
            else:
                print(f"\nFailed to load story: {story_title}")
                return
        else:
            print("Invalid choice. Starting a new story.")
            story = None
    else:
        print("No existing stories found. Starting a new story.")
        story = None

    if not story:
        # Create a new story
        initial_idea = get_user_input("Enter your story idea: ")
        story_process = process_story_idea(initial_idea)

        print("\nRehashed Description:")
        print(story_process["rehashed_description"])

        print("\nStory Title Options:")
        for index, title in enumerate(story_process["story_title_options"], 1):
            print(f"{index}. {title}")
        print("4. Enter your own title")

        title_choice = get_user_input("Select a story title or enter your own (enter the number or type your title): ")
        
        if title_choice.isdigit() and 1 <= int(title_choice) <= 3:
            selected_title = story_process["story_title_options"][int(title_choice) - 1]
        else:
            selected_title = title_choice.strip()

        story = Story(selected_title, story_process["rehashed_description"], initial_idea)

    current_chapter_id = None
    story_summary = story.description
    story_modified = False

    while True:
        if current_chapter_id:
            current_chapter = story.get_chapter_by_id(current_chapter_id)
        else:
            current_chapter = None

        if current_chapter:
            # Display current chapter data
            print("\nChapter Data:")
            print("=" * 40)
            print(f"ID: {current_chapter.id}")
            print(f"Title: {current_chapter.title}")
            print("-" * 40)
            print("Content:")
            print(current_chapter.content)
            print("-" * 40)
            print(f"Call to Action: {current_chapter.call_to_action}")
            print("Choices:")
            
            # Display existing paths and new paths
            for i, choice in enumerate(current_chapter.choices, 1):
                child = next((child for child in current_chapter.children if child.user_choice == choice), None)
                if child:
                    print(f"{i}. ({child.id}) {choice}")
                else:
                    print(f"{i}. (*) {choice}")
            
            print("\nOr enter your own idea")
            print("Press b to go back")
            print("Press m to open menu")
            print("=" * 40)

            # User selects a choice, enters their own idea, or navigates
            choice = get_user_input("Select a choice, enter your own idea, or navigate: ")
            
            if choice.lower() == 'b':
                # Go back to the previous chapter
                if current_chapter.parent_id:
                    current_chapter_id = current_chapter.parent_id
                else:
                    print("You're already at the start of the story.")
            elif choice.lower() == 'm':
                # Show menu of all chapters
                print("\nAll Chapters:")
                print(f"Title: {story.title}")
                print("-" * 40)
                display_chapter_tree(story.root_chapters)
                print("-" * 40)
                back_choice = get_user_input("Enter the ID of the chapter you want to go to (or press Enter to stay): ")
                if back_choice:
                    current_chapter_id = back_choice if story.get_chapter_by_id(back_choice) else current_chapter_id
            elif choice.isdigit() and 1 <= int(choice) <= len(current_chapter.choices):
                choice_index = int(choice) - 1
                selected_choice = current_chapter.choices[choice_index]
                existing_child = next((child for child in current_chapter.children if child.user_choice == selected_choice), None)
                
                if existing_child:
                    # Follow existing path
                    current_chapter_id = existing_child.id
                else:
                    # Create new path
                    new_chapter_id = generate_chapter_id(current_chapter.id, story)
                    chapter_content = generate_chapter_content(
                        selected_title=story.title,
                        selected_chapter_title=f"Chapter {new_chapter_id}",
                        chapter_number=new_chapter_id,
                        story_summary=story_summary,
                        initial_idea=story.initial_idea,
                        user_choice=selected_choice
                    )
                    new_chapter = Chapter(
                        new_chapter_id,
                        f"Chapter {new_chapter_id}",
                        parse_content(chapter_content["content"]),
                        chapter_content["call_to_action"],
                        chapter_content["choices"],
                        story_summary,
                        selected_choice,
                        current_chapter.id
                    )
                    story.add_chapter(new_chapter, current_chapter.id)
                    current_chapter_id = new_chapter.id
                    story_modified = True  # Set the flag to True as we've added a new chapter
            else:
                # Handle custom choice
                new_chapter_id = generate_chapter_id(current_chapter.id, story)
                custom_choice = choice.strip()
                chapter_content = generate_chapter_content(
                    selected_title=story.title,
                    selected_chapter_title=f"Chapter {new_chapter_id}",
                    chapter_number=new_chapter_id,
                    story_summary=story_summary,
                    initial_idea=story.initial_idea,
                    user_choice=custom_choice
                )
                new_chapter = Chapter(
                    new_chapter_id,
                    f"Chapter {new_chapter_id}",
                    parse_content(chapter_content["content"]),
                    chapter_content["call_to_action"],
                    chapter_content["choices"],
                    story_summary,
                    custom_choice,
                    current_chapter.id
                )
                story.add_chapter(new_chapter, current_chapter.id)
                current_chapter_id = new_chapter.id
                story_modified = True  # Set the flag to True as we've added a new chapter
        else:
            # Handle the case when there's no current chapter (start of the story)
            new_chapter_id = generate_chapter_id(None, story)
            chapter_content = generate_chapter_content(
                selected_title=story.title,
                selected_chapter_title=f"Chapter {new_chapter_id}",
                chapter_number=new_chapter_id,
                story_summary=story_summary,
                initial_idea=story.initial_idea
            )
            new_chapter = Chapter(
                new_chapter_id,
                f"Chapter {new_chapter_id}",
                parse_content(chapter_content["content"]),
                chapter_content["call_to_action"],
                chapter_content["choices"],
                story_summary,
                None,
                None
            )
            story.add_chapter(new_chapter, None)
            current_chapter_id = new_chapter.id
            story_modified = True  # Set the flag to True as we've added a new chapter

        # Save the story only if it has been modified
        if story_modified:
            save_story(story)
            story_modified = False  # Reset the flag after saving

        # Update the story summary
        story_summary = generate_summary(story_summary, story.get_chapter_by_id(current_chapter_id).content, story.initial_idea)

if __name__ == "__main__":
    editor_mode()
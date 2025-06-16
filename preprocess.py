import json
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from llm_helper import llm
import re

def process_posts(raw_file_path, processed_file_path = "data/processed_posts.json"):
    enriched_posts = []
    with open(raw_file_path, encoding = 'utf-8') as file:
        posts = json.load(file)
        for post in posts:
            metadata = extract_metadata(post['text'])
            post_with_metadata = post | metadata
            enriched_posts.append(post_with_metadata)

    unified_tags = get_unified_tags(enriched_posts)
    for post in enriched_posts:
        current_tags = post['tags']
        new_tags = {unified_tags[tag] for tag in current_tags}
        post['tags'] = list(new_tags)

    with open(processed_file_path, encoding="utf-8", mode="w") as outfile:
        json.dump(enriched_posts, outfile, indent = 4)
        
    

def clean_text(text):
    return text.encode("utf-8", "ignore").decode("utf-8")


def extract_metadata(post):
    template = '''
    You are given a LinkedIn post. Extract:
    - `line_count`: number of non-empty lines
    - `language`: either "English" or "Hinglish"
    - `tags`: an array of at most 2 text tags (keywords)

    ❗IMPORTANT❗
    - Output must be **ONLY** a valid JSON object with these three keys.
    - Do **not** include any explanation, code, markdown, or formatting.
    - Return only the JSON.

    Here is the post:
    {post}
    '''
    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    cleaned_post = clean_text(post)
    response = chain.invoke(input={'post': cleaned_post})


    try:
        json_parser = JsonOutputParser()
        cleaned_response = re.sub(r"```(?:json|javascript)?\s*([\s\S]*?)```", r"\1", response.content).strip()
        res = json_parser.parse(response.content)
    except OutputParserException as e:
        print("Failed to parse JSON. Raw response was:\n", response.content)
        raise OutputParserException("Invalid JSON format from LLM.") from e
    return res


def get_unified_tags(posts_with_metadata):
    unique_tags = set()
    for post in posts_with_metadata:
        unique_tags.update(post['tags']) 

    unique_tags_list = ','.join(unique_tags)

    template = '''I will give you a list of tags. You need to unify tags with the following requirements,
    1. Tags are unified and merged to create a shorter list. 
       Example 1: "Jobseekers", "Job Hunting" can be all merged into a single tag "Job Search". 
       Example 2: "Motivation", "Inspiration", "Drive" can be mapped to "Motivation"
       Example 3: "Personal Growth", "Personal Development", "Self Improvement" can be mapped to "Self Improvement"
       Example 4: "Scam Alert", "Job Scam" etc. can be mapped to "Scams"
    2. Each tag should be follow title case convention. example: "Motivation", "Job Search"
    3. Output should be a JSON object, No preamble
    3. Output should have mapping of original tag and the unified tag. 
       For example: {{"Jobseekers": "Job Search",  "Job Hunting": "Job Search", "Motivation": "Motivation}}
    
    Here is the list of tags: 
    {tags}
    '''
    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"tags": str(unique_tags_list)})
    try:
        json_parser = JsonOutputParser()
        res = json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Context too big. Unable to parse jobs.")
    return res


    
if __name__ == "__main__":
    process_posts("data/raw_posts.json", "data/processed_posts.json")
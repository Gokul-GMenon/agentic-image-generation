
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator
from io import BytesIO

# For other models
from google.adk.agents.loop_agent import LoopAgent
from google.adk.events import Event, EventActions
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import agent_tool, FunctionTool

# Import libraries from the Agent Development Kit
from pathlib import Path
from openai import OpenAI
import mimetypes, requests
from google.adk.agents import Agent, SequentialAgent, LlmAgent, BaseAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import datetime, os

key = os.getenv('OPENAI_KEY')

session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()

res = '1024'
img_model = 'dall-e-3'


# MODULE TO SAVE THE GENERATED IMAGE THROUGH PROXY SERVER
def save_image(image_link, prompt):
    # Obtaining the image from proxy server
    # Replace with your proxy server's public IP or domain
    proxy_server_url = "https://proxy-flask-repo.onrender.com/proxy_image"
    params = {"url": image_link}

    try:
        response = requests.get(proxy_server_url, params=params)
        response.raise_for_status()
        print("Status Code:", response.status_code)
        print("Content-Type:", response.headers.get("Content-Type"))
        with open(f"{prompt}.png", "wb") as f:
            f.write(response.content)
        print(f"✅ {prompt}.png saved successfully!!")
    except Exception as e:
        print(f"❌ Failed to download via proxy: {e}")
    
    print()

# IMAGE GENERATION MODULE (PROMPT -> IMAGE) USING DAL-E MODELS
def generate_image(user_prompt: str) -> dict:
    
    f"""Generates an image of the required description

    This function takes a description of the required image and generates it. It uses the openAI {img_model} model to generate a {res}x{res} image.
    
    Args:
        user_prompt: A accurate description of the required image

    Returns:
        A dictionary containing the URL of the generated image.
    
    """

    print("generation prompt - ", user_prompt)

    client = OpenAI(api_key=key)

    # Generate the image
    response = client.images.generate(
        model= img_model,
        prompt= user_prompt,
        size= f"{res}x{res}",
        n=1,
    )

    # Get the image URL
    image_url = response.data[0].url

    print(f"Image URL: {image_url}")

    return {'image_url': image_url}

def obtain_as_binary(image_url):

    print(f"Image URL: {image_url}")

    # Obtaining the image from proxy server
    # Replace with your proxy server's public IP or domain
    proxy_server_url = "https://proxy-flask-repo.onrender.com/proxy_image"
    params = {"url": image_url}

    try:
        response = requests.get(proxy_server_url, params=params)
        response.raise_for_status()
        print("Status Code:", response.status_code)
        print("Content-Type:", response.headers.get("Content-Type"))
        
        print(f"Image obtained successfully")

        return response.content
        
    except Exception as e:
        print(f"❌ Failed to download image via proxy: {e}")
        return 'No binary'

generation_prompt = ""
inferred_description = ""

LOOP_FLAG = None

# BIG LLM INSTACE IN THE FORM OF GPT 4.1
llm_instance_main = LiteLlm(
    model="gpt-4.1",
    api_key=key,
)

# MINI LLM INSTACE IN THE FORM OF GPT 4-O MINI
llm_instance_mini = LiteLlm(
    model="gpt-4o-mini",
    api_key=key,
)

base_generation_instruction = f"""
Obtain the required instructions for generating the image from state['enhanced_prompt'] if it exists. These instructions should be given as argument to the provided tool - 'generate_image'. 

If state['enhanced_prompt'] does not exist, then, you wil listen to all the original user requirements on the image to be generated. You will then draft a very specific prompt tailored for a {img_model} model ({res}x{res} resolution) so that it can generate an image that obeys all user requirements. Afterwards, These instructions should be passed as it is as an argument to the provided 'generate_image' tool. 

The final output should be exclusively the link provided by the tool and nothing else.
"""

# Initial agent whose takes the original user prompt, makes it more descriptive for the said model. Then calls tool to generate the image.
text_to_image_agent = Agent(
    name="PromptToImageGenerator",
    model=llm_instance_mini,
    description="This agent understands the requested image by the user and generates it accordingly.",
    instruction=base_generation_instruction,
    tools=[generate_image],
    output_key="generated_image_link" # Saves the enhanced_prompt or the boolean True
)

# To convert the image link obtained from previous tool into binary
class ImageBinaryAgent(BaseAgent): # Example custom agent
    name: str = "ObtainImageAsBinary"
    description: str = "Generates image from state['generated_image_link']."

    # ... internal logic ...
    async def _run_async_impl(self, ctx): # Simplified run logic
        image_url = ctx.session.state.get("generated_image_link")

        # Obtain image as binary
        image_bytes = obtain_as_binary(image_url)

        yield Event(author=self.name, content=types.Content(parts=[types.Part.from_bytes(data=image_bytes, mime_type="image/png")]))

# For image to description agent

image_to_text_instructions = f"""
Your job is to analyze an AI generated image and generate a detailed accurate description. To obtain the image, you should run "ObtainImageAsBinary" tool provided to you.
The image you have obtained is an AI generated image by the {img_model} model of resolution {res}x{res}. Your task is to create one brief accurate description of the entire image so that those instructions alone will enable another {img_model} model to recreate the exact same image. This description should point out all visible details of the image as well as general details like style and all. Your output will include just the description and not any pre text or follow up text. Also the entire description you create will contain no sub sections.
"""

image_agent = ImageBinaryAgent()
image_tool = agent_tool.AgentTool(agent=image_agent) # Wrap the agent

# Agent to understand the image created by DAL-E 2 and create a description of the image generated
image_to_text_agent = Agent(
    name="ImageToDescription",
    model=llm_instance_main,
    description=f"You will be provided with an AI generated image by the model {img_model} of resolution {res}x{res}. Your task is to create an accurate description of the entire image which will enable another {img_model} model to recreate the exact same image. Include nothing else but the instructions to recreate the image in your output (No follow up text or pretexts). Store the result in 'inferred_description'.",
    instruction=image_to_text_instructions,
    tools=[image_tool], # Include the AgentTool
    output_key='inferred_description'
)

# For cross checker agent

# ... internal logic ...
def obtain_image_descriptions() -> dict:
    
    """
    Retrieves both of the requried and generated image descriptions for comparison

    Returns:
        A dictionary containing a string which explains the original required image's and generated image's descriptions.
    """

    # original prompt is obtained at the point where image generation tool call is done
    global generation_prompt, inferred_description

    out = f"""
The original user requested image had the following description:
{generation_prompt}

The image generated by the model can be described as follows:
{inferred_description}

You can use these descriptions to cross check the images created and see it they match.
"""
    print("Comparison info - ",out)
    # Provide the info to the llm
    return {'image_descriptions_to_compare': out}

cross_checker_instructions = f"""
You will start by calling the 'obtain_image_descriptions' tool provided to you. It will provide two image descriptions. 
You will check the similarity of both descriptions and generate a score in the range of 1-100. 

If the score is greater than 95, then your output will strictly be the word 'True' and nothing else.

If that is not the case (score less than 95), then, your output will be set of instructions to {img_model} ({res}x{res} image) for generating the originally intended image. This new set of instructions will cover all the correct areas of the previous image and the parts that it missed. This particular output you generate will be passed directly as instructions to the model.

No need to report this score to the user.
"""

# This agent will take the original model prompt and the newly created propt by the previous agent to cross check and give a score to decide what should be called.
cross_checker = Agent(
    name="CompareImageDescription",
    model=llm_instance_mini,
    description="This agent will obtain two image descriptions and generates a similarity score between 1-100. Returns 'True' if score generated is greater than 95 else returns a more accurate image generation prompt. ",
    instruction=cross_checker_instructions,
    tools=[obtain_image_descriptions],
    output_key="enhanced_prompt" # Saves the enhanced_prompt or the boolean True
)

# For the loop agent

class CheckCondition(BaseAgent): # Custom agent to check state
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        print("\nChecking condition....\n\n")
        status = ctx.session.state.get("enhanced_prompt")
        is_done = (status == "True")
        yield Event(author=self.name, actions=EventActions(escalate=is_done)) # Escalate if done

refinement_loop = LoopAgent(
    name="RefineImageGeneration",
    max_iterations=5,
    sub_agents=[text_to_image_agent, image_to_text_agent, cross_checker, CheckCondition(name="StopChecker")]
)


AGENT_APP_NAME = 'agent_basic'

def send_query_to_agent(agent, query):
    global generation_prompt, inferred_description
    session = session_service.create_session(app_name=AGENT_APP_NAME,
                                            user_id='user')
    
    input_part = types.Part(text=query)
            
    content = types.Content(role='user', parts=[input_part])

    runner = Runner(app_name=AGENT_APP_NAME,
                    agent=agent,
                    artifact_service=artifact_service,
                    session_service=session_service)
    
    image_links = []
    image_paths = set()

    events = runner.run(user_id='user',
                        session_id=session.id,
                        new_message=content,)

    for _, event in enumerate(events):
        is_final_response = event.is_final_response()
        function_calls = event.get_function_calls()
        # function_responses = event.get_function_responses()

        # If the event represents a final response. Now we have to verify which agent is giving the final response
        if is_final_response:

            if event.content:
                # if event.author == ''
                print()
                final_response = event.content.parts[0].text
                print(f'------------Author:{event.author}-------Final Response------------\n\n {final_response}')
                print()

                if event.author == 'PromptToImageGenerator':
                    image_links.append(final_response)

                # To obtain the generated image's description
                elif event.author == 'ImageToDescription':
                    if event.content.parts[0].text:
                        inferred_description = event.content.parts[0].text        
            else:
                # Assuming that execution has ended

                for i, link in enumerate(image_links):
                    # Saving the images
                    save_image(link, f'img_{str(i+1)}')
                    image_paths.add(f'img_{str(i+1)}.png')
    
        elif function_calls:
            for function_call in function_calls:
                print(f'Calling agent: {event.author}')
                print(f'Call Function: {function_call.name}')
                print(f'Argument: {function_call.args}')

                if event.author == 'PromptToImageGenerator':
                    # Saving the original prompt that is used to generate the image
                    generation_prompt = function_call.args['user_prompt']

    return image_paths


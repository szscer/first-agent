from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
import math
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI

import numpy as np
import matplotlib.pyplot as plt

from typing import List

import torch
from self_attention import SelfAttention

# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def get_hypotenuse(arg1:float, arg2:float) -> float: #it's import to specify the return type
    #Keep this format for the description / args / args description but feel free to modify the tool
    """A tool that does nothing yet 
    Args:
        arg1: length of side of right triangle (int)
        arg2: length of other side of right triangle (int)
    """
    try:
        #make sure we have float types
        a = float(arg1)
        b = float(arg2)
        # get length of hypotenuse
        return math.hypot(a, b)
    except Exception as e:
          return f"Error fetching hyptenuse for sides '{a}' and {b}"
          

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

@tool
def matplotlib_plot(x: List, y: List) -> None:
    """
    A tool that generates a line plot. 
    
    Args:
    x: x-axis input
    y: y axis input
    
    """
    # Create the plot
    plt.plot(x, y)
    
    # Add labels and title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Simple Plot Example")

    # Display the plot
    plt.show()

@tool
def fourier_transform(signal: np.ndarray, sample_rate:float) -> None:
    """
    Compute the Fourier Transform of a given signal using numpy.fft.
    
    Args:
    signal: The input signal in time domain
    sample_rate: The sampling rate of the signal
    
    """
    n = len(signal)
    fft_values = np.fft.fft(signal)  # Compute FFT
    freq_values = np.fft.fftfreq(n, d=1/sample_rate)  # Frequency bins
    magnitude = np.abs(fft_values)  # Get 
    
    # Plot the spectrum
    plt.figure(figsize=(8, 4))
    plt.plot(freq_values[:len(freqs)//2], magnitude[:len(magnitude)//2])  # Only positive frequencies
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Fourier Transform using numpy.fft')
    plt.grid()
    plt.show()

    return None

@tool
def attention_scores(embedding_matrix: torch.Tensor) -> torch.Tensor :
    """A tool that computes attention scores
    Args:
        embedding_matrix: token encoding matrix
    """ 
    ## set the seed for the random number generator
    torch.manual_seed(42)
    ## create a basic self-attention ojbect
    selfAttention = SelfAttention(d_model=encodings_matrix.shape[0],
                               row_dim=0,
                               col_dim=1)
    ## calculate basic attention for the token encodings
    return selfAttention(encodings_matrix)
    
    

final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
max_tokens=2096,
#temperature=0.5,
temperature=0.1,
#model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
model_id='HuggingFaceH4/zephyr-7b-beta',
custom_role_conversions=None,
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer, get_current_time_in_timezone, attention_scores, fourier_transform], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()
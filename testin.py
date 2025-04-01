import openai
import time

# Set your OpenAI API key
#openai.api_key = "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"  # Replace with your actual API key

def make_request(prompt, model="gpt-3.5-turbo", max_tokens=100, retries=5, delay=60):
    """
    Function to make a request to the OpenAI API with retry logic for handling RateLimitError.

    Parameters:
    - prompt (str): The input text to generate a response for.
    - model (str): The OpenAI model to use (default is 'text-davinci-003').
    - max_tokens (int): Maximum number of tokens in the response.
    - retries (int): Number of retry attempts for rate limit errors.
    - delay (int): Delay in seconds between retries.

    Returns:
    - dict: The API response if successful, None otherwise.
    """
    for attempt in range(retries):
        try:
            print(f"Attempt {attempt + 1} of {retries}...")
            response = openai.Completion.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens
            )
            return response  # Return the API response if successful
        except openai.error.RateLimitError:
            print(f"Rate limit exceeded. Retrying in {delay} seconds...")
            time.sleep(delay)  # Wait before retrying
        except openai.error.OpenAIError as e:
            print(f"An error occurred: {e}")
            break  # Exit the loop for other errors
    print("Failed after all retries.")
    return None

# Define the prompt
prompt = "Explain the importance of error handling in software development."

# Make a request to the OpenAI API
response = make_request(prompt)

# Check and display the response
if response:
    print("Response from OpenAI API:")
    print(response['choices'][0]['text'].strip())  # Print the generated text
else:
    print("Unable to get a response from the OpenAI API.")
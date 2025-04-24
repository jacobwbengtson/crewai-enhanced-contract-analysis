from crewai import LLM as CrewAILLM
from typing import Dict, Any, List


class CrewAIAdapter:
    """Adapter to use CrewAI LLM with the existing system."""

    def __init__(self, model_name="openai/gpt-4", temperature=0.7, max_tokens=4000,
                 top_p=1.0, frequency_penalty=0, presence_penalty=0, stop=None, seed=None):
        """Initialize CrewAI LLM with configuration."""
        # Extract provider and model name
        if "/" in model_name:
            provider, model = model_name.split("/", 1)
        else:
            provider, model = "openai", model_name

        # Initialize CrewAI LLM
        self.llm = CrewAILLM(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            seed=seed
        )

        # Store configuration
        self.config = {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop,
            "seed": seed
        }

    def call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the CrewAI LLM with the given prompts."""
        try:
            # Format messages in the expected CrewAI format
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Call the CrewAI LLM
            response = self.llm.generate(messages=messages)

            # Return the generated text
            return response
        except Exception as e:
            error_msg = f"Error calling CrewAI LLM: {str(e)}"
            print(error_msg)
            return error_msg


class OpenAICompatWrapper:
    """
    Make CrewAI LLM compatible with the OpenAI client interface used in the system.
    This class mimics the necessary parts of the OpenAI API client.
    """

    def __init__(self, crewai_adapter: CrewAIAdapter):
        """Initialize with a CrewAIAdapter."""
        self.adapter = crewai_adapter

        # Create namespaces to mimic OpenAI client structure
        self.chat = ChatCompletions(self.adapter)

        # Set API key and base URL as instance attributes
        # to be compatible with the existing code
        self.api_key = "crewai-adapter"
        self.base_url = None

    def configure(self, api_key=None, base_url=None):
        """Configure the adapter (for compatibility)."""
        if api_key:
            self.api_key = api_key
        if base_url:
            self.base_url = base_url


class ChatCompletions:
    """Mock of OpenAI chat completions namespace."""

    def __init__(self, adapter: CrewAIAdapter):
        """Initialize with a CrewAIAdapter."""
        self.adapter = adapter

    def create(self, model: str, messages: List[Dict[str, str]], max_tokens: int = None, **kwargs) -> Any:
        """Mimic the OpenAI chat completions create method."""
        # Extract system and user messages
        system_prompt = "You are a helpful assistant."
        user_prompt = ""

        for message in messages:
            if message["role"] == "system":
                system_prompt = message["content"]
            elif message["role"] == "user":
                user_prompt = message["content"]

        # Call the adapter
        response_text = self.adapter.call_llm(system_prompt, user_prompt)

        # Create a response object that mimics OpenAI's response format
        response = MockCompletionResponse(response_text)
        return response


class MockCompletionResponse:
    """Mock of OpenAI chat completion response."""

    def __init__(self, content: str):
        """Initialize with content."""
        self.choices = [MockChoice(content)]


class MockChoice:
    """Mock of OpenAI chat completion choice."""

    def __init__(self, content: str):
        """Initialize with content."""
        self.message = MockMessage(content)
        self.index = 0
        self.finish_reason = "stop"


class MockMessage:
    """Mock of OpenAI chat completion message."""

    def __init__(self, content: str):
        """Initialize with content."""
        self.content = content
        self.role = "assistant"


def get_openai_compatible_client(
        model_name="openai/gpt-4",
        temperature=0.7,
        max_tokens=4000,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        seed=None
):
    """
    Create and return an OpenAI-compatible client using CrewAI LLM.
    This function creates an instance that can be used as a drop-in replacement
    for the OpenAI client in the existing code.
    """
    adapter = CrewAIAdapter(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        seed=seed
    )

    return OpenAICompatWrapper(adapter)
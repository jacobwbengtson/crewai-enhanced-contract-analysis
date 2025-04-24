import os
import yaml
from typing import Dict, Any, List
from crewai import Agent, Task, Crew, LLM

# Import existing components
from main import ChromaDBStorage, OpenAIDocumentProcessor


class CrewAIIntegration:
    """Integration class to connect existing system with CrewAI."""

    def __init__(self,
                 agents_file: str = "agents.yaml",
                 tasks_file: str = "tasks.yaml",
                 model: str = "openai/gpt-4",
                 temperature: float = 0.7,
                 max_tokens: int = 4000):
        """Initialize with configuration files."""
        self.agents_file = agents_file
        self.tasks_file = tasks_file

        # Initialize CrewAI LLM
        self.llm = LLM(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )

        # Load agent and task configurations
        self.agent_configs = self._load_yaml(agents_file)
        self.task_configs = self._load_yaml(tasks_file)

        # Initialize storage and processor
        self.storage = ChromaDBStorage(
            db_path="./chromadb",
            chunks_collection="document_chunks",
            summaries_collection="document_summaries"
        )

        # Use OPENAI_API_KEY from environment
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("WARNING: OPENAI_API_KEY not found in environment variables.")

        # Initialize processor
        self.processor = OpenAIDocumentProcessor(
            model_name=model.split('/')[-1] if '/' in model else model,
            max_length=5000,
            api_key=api_key
        )

        # Create agents and tasks
        self.agents = self._create_agents()
        self.tasks = self._create_tasks()

    def _load_yaml(self, file_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return {}

    def _create_agents(self) -> Dict[str, Agent]:
        """Create CrewAI agents from configuration."""
        agents = {}

        for agent_id, config in self.agent_configs.items():
            # Create agent
            agent = Agent(
                role=config.get("role", f"Agent {agent_id}"),
                goal=config.get("goal", "Process legal documents"),
                backstory=config.get("backstory", "An expert in legal document analysis"),
                verbose=config.get("verbose", True),
                allow_delegation=config.get("allow_delegation", True),
                llm=self.llm
            )

            agents[agent_id] = agent

        return agents

    def _create_tasks(self) -> Dict[str, Task]:
        """Create CrewAI tasks from configuration."""
        tasks = {}

        for task_id, config in self.task_configs.items():
            # Get the assigned agent
            agent_id = config.get("agent")
            if agent_id not in self.agents:
                print(f"Warning: Agent '{agent_id}' not found for task '{task_id}'")
                continue

            # Create task
            task = Task(
                description=config.get("description", f"Task {task_id}"),
                expected_output=config.get("expected_output", "Task output"),
                agent=self.agents[agent_id]
            )

            tasks[task_id] = task

        return tasks

    def run_task(self, task_id: str, context: Dict[str, Any] = None) -> str:
        """Run a task with optional context variables."""
        if task_id not in self.tasks:
            return f"Error: Task '{task_id}' not found"

        task = self.tasks[task_id]

        # Apply context variables to task description
        if context:
            task_desc = task.description
            try:
                task.description = task_desc.format(**context)
            except KeyError as e:
                print(f"Warning: Missing context variable {e} for task '{task_id}'")

        # Create and run a crew with just this task
        crew = Crew(
            agents=[task.agent],
            tasks=[task],
            verbose=True,
            process=Crew.Process.SEQUENTIAL
        )

        result = crew.kickoff()

        # Restore original task description
        if context:
            task.description = task_desc

        return result

    def run_workflow(self, task_ids: List[str], context: Dict[str, Any] = None) -> Dict[str, str]:
        """Run multiple tasks as a workflow with shared context."""
        # Collect tasks
        workflow_tasks = []

        for task_id in task_ids:
            if task_id not in self.tasks:
                print(f"Warning: Task '{task_id}' not found")
                continue

            task = self.tasks[task_id]

            # Apply context variables to task description
            if context:
                task_desc = task.description
                try:
                    task.description = task_desc.format(**context)
                except KeyError as e:
                    print(f"Warning: Missing context variable {e} for task '{task_id}'")

            workflow_tasks.append(task)

        if not workflow_tasks:
            return {"error": "No valid tasks found for workflow"}

        # Create and run crew
        crew = Crew(
            agents=[task.agent for task in workflow_tasks],
            tasks=workflow_tasks,
            verbose=True,
            process=Crew.Process.SEQUENTIAL
        )

        results = crew.kickoff()

        # Restore original task descriptions
        if context:
            for i, task_id in enumerate(task_ids):
                if task_id in self.tasks:
                    original_desc = self.task_configs[task_id].get("description", "")
                    self.tasks[task_id].description = original_desc

        # Format results
        if isinstance(results, str):
            return {task_ids[-1]: results}
        elif isinstance(results, list):
            return {task_id: result for task_id, result in zip(task_ids, results) if task_id in self.tasks}
        else:
            return {"result": results}


# Example usage
def main():
    """Example of using the CrewAI integration."""
    # Initialize integration
    integration = CrewAIIntegration()

    # Show available agents and tasks
    print("\nAvailable Agents:")
    for agent_id in integration.agents:
        print(f"  - {agent_id}")

    print("\nAvailable Tasks:")
    for task_id in integration.tasks:
        print(f"  - {task_id}")

    # Example: Run a task
    doc_id = "doc_001"  # Replace with an actual document ID
    print(f"\nRunning 'summarize_document' task for document {doc_id}...")
    result = integration.run_task("summarize_document", {"doc_id": doc_id})
    print(f"\nSummary result:\n{result}")

    # Example: Run a workflow
    print(f"\nRunning document analysis workflow for {doc_id}...")
    workflow_result = integration.run_workflow(
        ["summarize_document", "analyze_document", "extract_legal_definitions"],
        {"doc_id": doc_id}
    )

    for task_id, task_result in workflow_result.items():
        print(f"\n--- Result from {task_id} ---")
        print(task_result[:500] + "..." if len(task_result) > 500 else task_result)


if __name__ == "__main__":
    main()
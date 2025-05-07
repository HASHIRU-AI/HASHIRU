# In tool_manager.py
from default_tools.test_cost.agent_creator_tool import AgentCreator

def test_agent_creation():
    creator = AgentCreator()
    response = creator.run(
        agent_name="costbenefit-test-agent",
        system_prompt="You are an expert assistant helping with cost-based model selection.",
        description="Tests agent creation using cost-benefit logic."
    )
    print("\nTest Output:")
    print(response)

if __name__ == "__main__":
    test_agent_creation()

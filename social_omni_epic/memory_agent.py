"""Memory injection into Sotopia's LLMAgent.

Option A (used in Phase 1): bake the memory text into a `custom_template`
string at agent construction. `sotopia.agents.LLMAgent` already accepts a
`custom_template` kwarg and `sotopia.generation_utils.generate.agenerate`
uses simple .replace()-based substitution that ignores unmatched braces, so
literal text inserted into the template is rendered verbatim in the prompt.

Option B (kept on file for Phase 2 mid-episode reflection): subclass
LLMAgent and override `aact` to inject memory into the inbox-derived history
each turn. Not used right now.
"""
from typing import Optional


# Verbatim copy of sotopia's default action template
# (sotopia/generation_utils/generate.py:422-436). If Sotopia upgrades this
# template, re-vendor this string.
DEFAULT_ACTION_TEMPLATE = """
                Imagine you are {agent}, your task is to act/speak as {agent} would, keeping in mind {agent}'s social goal.
                You can find {agent}'s goal (or background) in the 'Here is the context of the interaction' field.
                Note that {agent}'s goal is only visible to you.
                You should try your best to achieve {agent}'s goal in a way that align with their character traits.
                Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
                {history}.
                You are at Turn #{turn_number}. Your available action types are
                {action_list}.
                Note: You can "leave" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.

                Please only generate a JSON string including the action type and the argument.
                Your action should follow the given format:
                {format_instructions}
            """


def build_custom_template_with_memory(memory_text: str) -> Optional[str]:
    """Return a custom_template string with memory text spliced in, or None
    if memory_text is empty (so LLMAgent falls back to the default template).

    The memory block is inserted BEFORE the {history} block so the agent
    reads its prior-experience notes before scanning the live dialogue.
    """
    if not memory_text or not memory_text.strip():
        return None
    memory_block = (
        "\n                === Lessons from prior similar interactions "
        "(visible only to you) ===\n"
        f"                {memory_text.strip()}\n"
        "                === End of lessons ===\n"
    )
    # Insert immediately before the "{history}." line in the default template.
    spliced = DEFAULT_ACTION_TEMPLATE.replace(
        "{history}.",
        memory_block + "                {history}.",
    )
    return spliced

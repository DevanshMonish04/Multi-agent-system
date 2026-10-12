from agents import (
    built_search_agent,
    built_reader_agent,
    writer_chain,
    critic_chain
)


def run_research_pipeline(topic: str) -> dict:

    state = {}

    # ============================================================
    # STEP 1 - SEARCH AGENT
    # ============================================================

    print("\n" + "=" * 50)
    print("Step 1 - Search agent is working...")
    print("=" * 50)

    search_agent = built_search_agent()

    search_result = search_agent.invoke({
        "messages": [
            (
                "user",
                f"Find recent and reliable information on the topic: {topic}."
            )
        ]
    })

    state["search_results"] = search_result["messages"][-1].content

    print("\nSearch Result:\n")
    print(state["search_results"])


    # ============================================================
    # STEP 2 - READER AGENT
    # ============================================================

    print("\n" + "=" * 50)
    print("Step 2 - Reader agent is working...")
    print("=" * 50)

    reader_agent = built_reader_agent()

    reader_result = reader_agent.invoke({
        "messages": [
            (
                "user",
                f"""
Research topic: {topic}

Below are the complete search results from the Search Agent.

Search Results:
{state["search_results"]}

Instructions:

1. Identify the most relevant and reliable URL.
2. Use the scrape_url tool to scrape that URL.
3. Read the scraped webpage content.
4. Return a detailed summary of the actual scraped content.
5. Include the selected URL in your response.

IMPORTANT:
Do not simply recommend a website.
You must actually use the scrape_url tool.
"""
            )
        ]
    })

    state["scraped_content"] = reader_result["messages"][-1].content

    print("\nScraped Content:\n")
    print(state["scraped_content"])


    # ============================================================
    # STEP 3 - WRITER
    # ============================================================

    print("\n" + "=" * 50)
    print("Step 3 - Writer agent is working...")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )

    report_response = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })

    state["report"] = report_response.content

    print("\nReport:\n")
    print(state["report"])


    # ============================================================
    # STEP 4 - CRITIC
    # ============================================================

    print("\n" + "=" * 50)
    print("Step 4 - Critic agent is working...")
    print("=" * 50)

    feedback_response = critic_chain.invoke({
        "report": state["report"]
    })

    state["feedback"] = feedback_response.content

    print("\nCritic Report:\n")
    print(state["feedback"])            


    # ============================================================
    # RETURN FINAL STATE
    # ============================================================

    return state


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    topic = input("Enter a topic to research: ")

    run_research_pipeline(topic)
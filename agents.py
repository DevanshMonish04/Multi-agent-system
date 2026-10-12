from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from tools import web_search, scrape_url

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)


# ============================================================
# 1. SEARCH AGENT
# ============================================================

def built_search_agent():

    search_prompt = """
You are a web research search agent.

Your job is to search the web using the web_search tool.

IMPORTANT RULES:

1. You MUST use the web_search tool.
2. Preserve the actual URLs returned by the tool.
3. Do NOT remove or hide URLs.
4. Do NOT provide only a general summary.
5. Return the search results with:
   - Title
   - URL
   - Relevant snippet
6. Return at least 3 relevant results when available.

Your output MUST contain the actual URLs returned
by the web_search tool.
"""

    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=search_prompt
    )


# ============================================================
# 2. READER AGENT
# ============================================================

def built_reader_agent():

    reader_prompt = """
You are a web research reader agent.

Your job is to read search results and extract information
from the most relevant webpage.

Follow these steps:

1. Find the most relevant and reliable URL from the search results.
2. You MUST call the scrape_url tool using that URL.
3. Read the actual content returned by the scrape_url tool.
4. Summarize the important factual information from the
   scraped webpage.
5. Return the selected URL.
6. Return the actual scraped information and summary.

IMPORTANT:
- Do NOT merely recommend a website.
- You MUST use the scrape_url tool.
- Do NOT invent information.
- If the URL cannot be scraped, clearly state that.
"""

    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=reader_prompt
    )


# ============================================================
# 3. WRITER CHAIN
# ============================================================

writer_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
You are an expert research writer.

Write clear, detailed, factual and professional research reports.

Use ONLY the information provided in the research.

Do not invent facts, statistics, sources, URLs,
companies, dates, or examples.

If URLs are present in the research, preserve them
in the Sources section.
"""
    ),

    (
        "human",
        """
Write a detailed research report on the topic below.

Topic:
{topic}

Research Gathered:
{research}

Structure the report as:

1. Introduction

2. Key Findings
   - Minimum 3 well-explained points

3. Conclusion

4. Sources
   - List all URLs actually used for research
   - Do not invent URLs
   - Include the selected webpage URL from the Reader Agent

Be detailed, factual and professional.
"""
    )
])


writer_chain = writer_prompt | llm


# ============================================================
# 4. CRITIC CHAIN
# ============================================================

critic_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
You are a sharp and constructive research critic.

Review research reports strictly.

Be honest, specific and useful.

Pay particular attention to:

- Accuracy
- Research depth
- Source quality
- URLs and citations
- Factual support
- Organization
- Missing information
- Unsupported claims
"""
    ),

    (
        "human",
        """
Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
...
"""
    )
])


critic_chain = critic_prompt | llm
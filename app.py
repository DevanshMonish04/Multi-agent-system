import streamlit as st

from pipeline import run_research_pipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .agent-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 15px;
    }

    .report-box {
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #ddd;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 AI Research Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Multi-Agent Research System powered by Gemini + Tavily'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Research System")

    st.markdown(
        """
        ### Agents

        🔎 **Search Agent**  
        Finds relevant web information.

        📖 **Reader Agent**  
        Scrapes and analyzes a selected webpage.

        ✍️ **Writer Agent**  
        Generates the research report.

        🧐 **Critic Agent**  
        Reviews the final report.
        """
    )

    st.divider()

    st.info(
        "Enter a topic and let the multi-agent system "
        "research it automatically."
    )


# ============================================================
# TOPIC INPUT
# ============================================================

st.subheader("🔍 Research Topic")

topic = st.text_input(
    "Enter a topic",
    placeholder="Example: Artificial Intelligence"
)


# ============================================================
# START RESEARCH
# ============================================================

if st.button(
    "🚀 Start Research",
    type="primary",
    use_container_width=True
):

    if not topic.strip():

        st.warning("Please enter a research topic.")

    else:

        # Store result in session state
        st.session_state["research_result"] = None

        # Progress display
        progress = st.progress(0)

        status = st.empty()

        try:

            # ------------------------------------------------
            # SEARCH
            # ------------------------------------------------

            status.info("🔎 Search Agent is searching the web...")
            progress.progress(20)

            result = run_research_pipeline(topic)

            progress.progress(100)

            status.success(
                "✅ Research completed successfully!"
            )

            st.session_state["research_result"] = result

        except Exception as e:

            progress.empty()

            status.error(
                "❌ An error occurred while running the research pipeline."
            )

            st.exception(e)


# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.get("research_result"):

    result = st.session_state["research_result"]

    st.divider()

    st.header("📊 Research Results")


    # ========================================================
    # SEARCH RESULTS
    # ========================================================

    with st.expander(
        "🔎 Search Agent Results",
        expanded=False
    ):

        st.markdown(
            result.get(
                "search_results",
                "No search results available."
            )
        )


    # ========================================================
    # SCRAPED CONTENT
    # ========================================================

    with st.expander(
        "📖 Reader Agent - Scraped Content",
        expanded=False
    ):

        st.markdown(
            result.get(
                "scraped_content",
                "No scraped content available."
            )
        )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    st.subheader("📄 Research Report")

    st.markdown(
        '<div class="report-box">',
        unsafe_allow_html=True
    )

    st.markdown(
        result.get(
            "report",
            "No report generated."
        )
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # CRITIC
    # ========================================================

    st.subheader("🧐 Critic Evaluation")

    st.markdown(
        result.get(
            "feedback",
            "No critic feedback available."
        )
    )


    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    st.divider()

    report_text = result.get(
        "report",
        ""
    )

    if report_text:

        st.download_button(
            label="⬇️ Download Research Report",
            data=report_text,
            file_name="research_report.txt",
            mime="text/plain",
            use_container_width=True
        )   
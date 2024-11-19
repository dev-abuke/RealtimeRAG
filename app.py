import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict

from pipeline.rag import RAGPipeline
from pipeline.retriever import VectorDBRetriever
from pipeline.embeddings import CrossEncoderModelSingleton
from pipeline import constants

def initialize_rag_pipeline():
    """Initialize the RAG pipeline with necessary components."""
    retriever = VectorDBRetriever(
        cross_encoder_model=CrossEncoderModelSingleton()
    )
    
    return RAGPipeline(
        retriever=retriever
    )

def display_source_documents(sources: List[Dict]):
    """Display source documents in an expandable format."""
    for i, source in enumerate(sources, 1):
        with st.expander(f"📄 Source {i}: {source['headline']}", expanded=False):
            # Create two columns for metadata
            col2, col3 = st.columns(2)
            
            with col2:
                st.markdown("**Date Metadata:**")
                st.write("**Created Date:**", source['created_at'])
                st.write("**Updated Date:**", source['updated_at'])

            with col3:
                st.markdown("**Other Metadata:**")
                st.write("**Author:**", source['author'])
                st.write("**Symbols:**", ", ".join(source['symbols']))
            
            # Display content
            st.markdown("**Content:**")
            st.markdown(source['content'])
            
            # Display URL if available
            if source['url'] != 'N/A':
                st.markdown(f"[Read full article]({source['url']})")

def main():
    st.set_page_config(
        page_title="Financial News RAG System",
        page_icon="📈",
        layout="wide"
    )
    
    # Initialize session state
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = initialize_rag_pipeline()
    
    # Header
    st.title("📈 Financial News Research Assistant")
    st.markdown("""
    Ask questions about financial news and get AI-powered answers backed by real-time sources.
    """)
    
    # Sidebar for filters
    st.sidebar.header("Search Filters")
    
    # Date filter
    st.sidebar.subheader("Date Range")
    date_filter = st.sidebar.radio(
        "Select date range:",
        ["All time", "Past day", "Past week", "Past month", "Custom"]
    )
    
    if date_filter == "Custom":
        date_from = st.sidebar.date_input("From date")
        date_to = st.sidebar.date_input("To date")
    else:
        date_to = datetime.now()
        if date_filter == "Past day":
            date_from = date_to - timedelta(days=1)
        elif date_filter == "Past week":
            date_from = date_to - timedelta(weeks=1)
        elif date_filter == "Past month":
            date_from = date_to - timedelta(days=30)
        else:
            date_from = None
            date_to = None
    
    # Symbol filter
    symbols = st.sidebar.text_input(
        "Stock Symbols (comma-separated)",
        help="Enter stock symbols like: AAPL, GOOGL, MSFT"
    ).upper()
    symbols = [s.strip() for s in symbols.split(",")] if symbols else None
    
    # Analysis mode
    analysis_mode = st.sidebar.checkbox(
        "Enable Market Analysis Mode",
        help="Provides deeper market analysis instead of simple answers"
    )
    
    # Scoring mechanism selection
    scoring_method = st.sidebar.radio(
        "Select Scoring Mechanism:",
        ["Weighted Average", "Harmonic Mean", "Geometric Mean"]
    )

    show_detailed_score =  st.sidebar.checkbox("Show detailed scores table")
    
    # Main interface
    question = st.text_input(
        "Enter your question:",
        placeholder="E.g., What was the latest FDA announcement about Sangamo Therapeutics?"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        num_sources = st.number_input(
            "Number of sources:",
            min_value=1,
            max_value=10,
            value=3
        )
        use_time_decay = st.checkbox(
            "Use Time Decay for Reranking",
            value=True,
            help="Enable this to apply time decay to the relevance scores."
        )
    
    if st.button("Search", type="primary"):
        if question:
            try:
                with st.spinner("Searching and analyzing..."):
                    # Get response from RAG pipeline
                    response = st.session_state.rag_pipeline.query_by_filters(
                        question=question,
                        symbols=symbols,
                        date_from=date_from if date_from else None,
                        date_to=date_to if date_to else None,
                        analysis_mode=analysis_mode,
                        limit=num_sources,
                        use_time_decay=use_time_decay,
                        scoring_method=scoring_method
                    )
                    
                    # Display answer
                    st.markdown("### 💡 Answer")
                    st.markdown(response.answer)
                    
                    # Display sources
                    st.markdown("### 📚 Sources")
                    if response.sources:
                        display_source_documents(response.sources)
                        
                        # Display relevance metrics for each source
                        st.markdown("### 📊 Source Relevance Metrics")
                        
                        # Prepare data for visualization
                        all_scores_data = []
                        import altair as alt
                        
                        for i, source in enumerate(response.sources, 1):
                            # Add each score type for this source
                            all_scores_data.extend([
                                {
                                    "Source": f"Source {i}: {source['headline'][:30]}...",
                                    "Score Type": "Weighted",
                                    "Score": source.get('weighted_score', 0)
                                },
                                {
                                    "Source": f"Source {i}: {source['headline'][:30]}...",
                                    "Score Type": "Harmonic",
                                    "Score": source.get('harmonic_score', 0)
                                },
                                {
                                    "Source": f"Source {i}: {source['headline'][:30]}...",
                                    "Score Type": "Geometric",
                                    "Score": source.get('geometric_score', 0)
                                },
                                {
                                    "Source": f"Source {i}: {source['headline'][:30]}...",
                                    "Score Type": "Decay",
                                    "Score": source.get('decay_score', 0)
                                },
                                {
                                    "Source": f"Source {i}: {source['headline'][:30]}...",
                                    "Score Type": "Similarity",
                                    "Score": source.get('original_score', 0)
                                },
                                {
                                    "Source": f"Source {i}: {source['headline'][:30]}...",
                                    "Score Type": "Cross-Encoder",
                                    "Score": source.get('rerank_score', 0)
                                }
                            ])
                        
                        # Create DataFrame
                        scores_df = pd.DataFrame(all_scores_data)
                        
                        # Create grouped bar chart using Altair
                        chart = alt.Chart(scores_df).mark_bar().encode(
                            x=alt.X('Source:N', title='Sources'),
                            y=alt.Y('Score:Q', title='Score Value'),
                            color=alt.Color('Score Type:N', title='Scoring Method'),
                            xOffset='Score Type:N'  # This creates the grouping
                        ).properties(
                            width=800,
                            height=400,
                            title='Comparison of Scoring Methods Across Sources'
                        ).configure_axis(
                            labelAngle=-45  # Angle the labels for better readability
                        )
                        
                        # Display the chart
                        st.altair_chart(chart, use_container_width=True)
                        
                        # Optionally add a table view of the scores
                        if show_detailed_score:
                            st.dataframe(
                                scores_df.pivot(
                                    index='Source',
                                    columns='Score Type',
                                    values='Score'
                                ).round(3)
                            )
                    else:
                        st.warning("No relevant sources found for your query.")
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a question.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    💡 **Tips:**
    - Use specific questions for better results
    - Enable Market Analysis Mode for deeper insights
    - Filter by date and symbols for targeted research
    """)

if __name__ == "__main__":
    main()
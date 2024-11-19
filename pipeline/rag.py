from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from pipeline.retriever import VectorDBRetriever
from pipeline.models import NewsArticle, EmbeddedChunkedArticle
from pipeline import constants

@dataclass
class RAGResponse:
    """Data class for RAG pipeline response."""
    answer: str
    sources: List[Dict]
    rerank_scores: Optional[List[float]] = None
    query: Optional[str] = None

class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation."""
    
    def __init__(
        self,
        retriever: VectorDBRetriever,
        model_name: str = constants.LLM_MODEL_NAME,
        openai_api_key: str = constants.OPENAI_API_KEY,
        temperature: float = 0.7,
        streaming: bool = False
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            retriever: Initialized QdrantVectorDBRetriever
            model_name: Name of the LLM model
            openai_api_key: OpenAI API key
            temperature: LLM temperature
            streaming: Whether to enable streaming responses
        """
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=openai_api_key,
            temperature=temperature,
            streaming=streaming,
            base_url=constants.LLM_BASE_URL
        )
        
        # Initialize prompts
        self._init_prompts()
        
        # Create the chain
        self.chain = self._create_chain()
    
    def _init_prompts(self):
        """Initialize prompt templates."""
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful financial news assistant. 
            Use the provided context to answer questions accurately, short, very crisp and concise.
            Your primary objective is to provide clear and accurate responses to user queries while considering 
            the provided context. Multiple contexts are provided under separated by Context: delimiter.
            Your answer MUST be 600 characters or less. You MUST adhere to the following guidelines: 
                1. Rely solely on the provided context or the relevant component of the conversation history 
                    to formulate your response.
                2. If the context lacks essential details for a complete answer, kindly acknowledge it. 
                3. If your response involves code examples, ensure you format them using triple backticks, 
                    specifying the programming language. 
                4. Avoid generating pseudo-code or unrelated information. 
                5. Strive for clarity and brevity, summarizing your response within 200 characters or less. 
                6. Focus on factual information from the provided context
                7. Cite specific details and sources when possible
                8. Maintain objectivity and avoid speculation
                9. Clearly indicate if information is insufficient
            """),
            ("user", """Use the following pieces of context to answer the question.
            Context:
            {context}
            
            Question: {question}
            
            Provide a clear, well-structured answer using the context provided.""")
        ])
        
        self.market_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a market analysis expert.
            Analyze the provided financial news context to provide insights and implications.
            Focus on key developments, market impact, and potential outcomes.
            
            Guidelines:
            - Highlight key market developments
            - Analyze potential implications
            - Consider industry context
            - Remain objective and evidence-based
            """),
            ("user", """Based on the following financial news context:
            {context}
            
            Question: {question}
            
            Provide a thorough analysis with specific supporting evidence.""")
        ])
    
    def _format_context(self, articles: List[EmbeddedChunkedArticle]) -> str:
        """
        Format retrieved articles into context string.
        
        Args:
            articles: List of retrieved articles
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, article in enumerate(articles, 1):
            part = f"Source {i}:\n"
            if hasattr(article, 'headline') and article.headline:
                part += f"Headline: {article.headline}\n"
            if hasattr(article, 'created_at') and article.created_at:
                date_str = article.created_at
                if isinstance(date_str, datetime):
                    date_str = date_str.strftime("%Y-%m-%d %H:%M:%S")
                part += f"Date: {date_str}\n"
            part += f"Content: {article.text}\n"
            if hasattr(article, 'rerank_score'):
                part += f"Relevance Score: {article.rerank_score:.3f}\n"
            context_parts.append(part)
        
        return "\n\n".join(context_parts)
    
    def _create_chain(self):
        """Create the RAG chain."""
        # Define a retrieval function
        def retrieve_and_format(question: str) -> str:
            # Get retrieved articles
            results = self.retriever.search(
                query=question,
                limit=3,  # Adjust as needed
                return_all=True
            )
            
            # Format the context
            return self._format_context(results)
        
        chain = (
            RunnableParallel(
                {
                    "context": lambda x: retrieve_and_format(x),
                    "question": RunnablePassthrough()
                }
            )
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def query(
        self,
        question: str,
        analysis_mode: bool = False,
        limit: int = 3,
        return_sources: bool = True
    ) -> RAGResponse:
        """
        Query the RAG pipeline.
        
        Args:
            question: User question
            analysis_mode: Whether to use market analysis prompt
            limit: Number of sources to retrieve
            return_sources: Whether to return source documents
            
        Returns:
            RAGResponse with answer and sources
        """
        try:
            logger.info(f"Processing query: {question}")
            
            # Get retrieved articles with scores
            retrieved_results = self.retriever.search(
                query=question,
                limit=limit,
                return_all=False,
            )

            logger.info(f"Retrieved {len(retrieved_results)} articles and results: {retrieved_results}")
            
            # Use appropriate prompt based on mode
            current_prompt = self.market_analysis_prompt if analysis_mode else self.qa_prompt
            
            # Generate answer
            context = self._format_context(retrieved_results)

            logger.info(f"The Context is: {context}")
            answer = current_prompt.format(
                context=context,
                question=question
            )
            answer = self.llm.invoke(answer).content
            
            # Prepare sources
            sources = []
            if return_sources:
                for article in retrieved_results:
                    source = {
                        "headline": getattr(article, 'headline', 'N/A'),
                        "url": getattr(article, 'url', 'N/A'),
                        "created_at": getattr(article, 'created_at', 'N/A'),
                        "symbols": getattr(article, 'symbols', []),
                        "author": getattr(article, 'author', 'N/A'),
                        "content": article.text,
                        "rerank_score": getattr(article, 'rerank_score', None),
                        "original_score": article.score
                    }
                    sources.append(source)
            
            return RAGResponse(
                answer=answer,
                sources=sources,
                rerank_scores=[getattr(a, 'rerank_score', None) for a in retrieved_results],
                query=question
            )
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            raise
    def query_by_filters(
            self,
            question: str,
            symbols: Optional[List[str]] = None,
            date_from: Optional[datetime] = None,
            date_to: Optional[datetime] = None,
            analysis_mode: bool = False,
            limit: int = 3,
            use_time_decay: bool = False,
            scoring_method: str = "Weighted Average"
        ) -> RAGResponse:
            """
            Query with filters.
            
            Args:
                question: User question
                symbols: List of stock symbols to filter
                date_from: Start date filter
                date_to: End date filter
                analysis_mode: Whether to use market analysis prompt
                limit: Number of sources to retrieve
                
            Returns:
                RAGResponse with filtered results
            """
            try:
                retrieved_results = self.retriever.search_by_filters(
                    query=question,
                    symbols=symbols,
                    date_from=date_from,
                    date_to=date_to,
                    limit=limit,
                    return_all=False,
                    return_scores=False,
                    use_time_decay=use_time_decay,
                    scoring_method=scoring_method
                )

                logger.info(f"Retrieved {len(retrieved_results)} articles")
                
                current_prompt = self.market_analysis_prompt if analysis_mode else self.qa_prompt
                
                context = self._format_context(retrieved_results)

                logger.info(f"The Context is: {context}")
                answer = current_prompt.format(
                    context=context,
                    question=question
                )
                answer = self.llm.invoke(answer).content
                sources = [{
                    "headline": getattr(article, 'headline', 'N/A'),
                    "url": getattr(article, 'url', 'N/A'),
                    "created_at": getattr(article, 'created_at', 'N/A'),
                    "symbols": getattr(article, 'symbol', []),
                    "author": getattr(article, 'author', 'N/A'),
                    "content": article.text,
                    "rerank_score": getattr(article, 'rerank_score', None),
                    "weighted_score": getattr(article, 'weighted_avg_score', None),
                    "harmonic_score": getattr(article, 'harmonic_mean_score', None),
                    "geometric_score": getattr(article, 'geometric_mean_score', None),
                    "decay_score": getattr(article, 'decay_score', None),
                    "original_score": article.score,
                    "updated_at": article.updated_at,
                    "article_id": article.article_id
                } for article in retrieved_results]
                
                return RAGResponse(
                    answer=answer,
                    sources=sources,
                    rerank_scores=[getattr(a, 'rerank_score', None) for a in retrieved_results],
                    query=question
                )
                
            except Exception as e:
                logger.error(f"Error in filtered RAG pipeline: {str(e)}")
                raise

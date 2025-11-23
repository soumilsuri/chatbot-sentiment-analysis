# Chatbot with Sentiment Analysis

A sophisticated chatbot application that conducts conversations with users and performs sentiment analysis on the entire conversation. Built with Google Gemini API for intelligent responses and Hugging Face transformers for sentiment analysis.

## How to Run

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Google Gemini API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Assignment-chatbot-sentiment-analysis
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```bash
   # Copy the example file
   # On Windows (PowerShell)
   Copy-Item .env.example .env
   
   # On macOS/Linux
   cp .env.example .env
   ```
   
   Edit `.env` and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```
   
   The application will open in your default web browser at `http://localhost:8501`

## Chosen Technologies

### Core Technologies

- **Python 3.8+**: Primary programming language
- **Streamlit**: Web framework for building the interactive UI
- **Google Gemini API**: Powers the chatbot's conversational intelligence
- **Hugging Face Transformers**: Provides sentiment analysis capabilities
- **PyTorch**: Deep learning framework for model inference

### Libraries & Dependencies

- `streamlit`: Modern web app framework for Python
- `google-generativeai`: Official Google Gemini API client
- `transformers`: Hugging Face library for NLP models
- `torch`: PyTorch for model inference
- `python-dotenv`: Environment variable management
- `pandas`: Data handling utilities
- `plotly`: Interactive data visualization for charts and graphs

## Explanation of Sentiment Logic

### Models Used

#### 1. Sentiment Analysis Model
The application uses **`cardiffnlp/twitter-roberta-base-sentiment-latest`**, a RoBERTa-based model fine-tuned on Twitter data for sentiment analysis. 

**Why this model?**
- **Niche Task Specialization**: Specifically designed for social media and conversational text, making it ideal for chatbot interactions.
- **Lightweight & Efficient**: Offers high accuracy without the heavy computational overhead of larger LLMs.
- **Cost-Effective**: Runs locally using Hugging Face Transformers, eliminating per-token costs associated with external APIs.
- **Three-Class Classification**: Provides clear Negative, Neutral, and Positive labels.

#### 2. Emotion Analysis Model
For Tier 2 features, the application utilizes **`j-hartmann/emotion-english-distilroberta-base`**.

**Capabilities:**
- Detects 7 distinct emotions: Joy, Sadness, Anger, Fear, Surprise, Disgust, and Neutral.
- Enables multi-dimensional emotional profiling of the conversation.
- Supports the generation of radar charts and emotional intensity tracking.

### Analysis Process

1. **Text Preparation**: Only **user messages** are analyzed for sentiment, as sentiment analysis reflects the user's emotional state and satisfaction level. The chatbot's responses are excluded since they don't represent user sentiment.

2. **Tokenization**: The user messages are concatenated and tokenized using the model's tokenizer with:
   - Maximum length: 512 tokens (handles most conversations)
   - Truncation for longer conversations
   - Proper padding for batch processing

3. **Inference**: 
   - The model processes the tokenized input
   - Softmax activation provides probability scores for each sentiment class
   - The highest confidence label is selected

4. **Output Formatting**:
   - Primary label: Negative, Neutral, or Positive
   - Confidence score: Probability of the selected label
   - Explanation: Contextual description based on confidence level
   - Formatted output: "Overall conversation sentiment: [Label] – [Explanation]"

### Sentiment Interpretation

- **Positive**: Indicates satisfaction, positive engagement, or favorable interaction
- **Negative**: Indicates dissatisfaction, concerns, complaints, or negative feedback
- **Neutral**: Indicates balanced, mixed, or neither strongly positive nor negative sentiment

The confidence score helps assess the strength of the sentiment classification. Higher confidence (>70%) indicates stronger sentiment, while lower confidence suggests more ambiguous sentiment.

## Status of Tier 2 Implementation

**Current Status**: FULLY IMPLEMENTED

All Tier 2 features have been successfully implemented:

- **Statement-Level Sentiment Analysis**: Each user message is analyzed individually
- **Real-Time Sentiment Display**: Sentiment badges appear next to messages (toggleable in sidebar)
- **Multi-Dimensional Emotion Analysis**: Full emotion detection with 7 emotion categories
- **Emotion Radar Chart**: Visual representation of emotion distribution
- **Mood Trend Visualization**: Interactive charts showing sentiment/emotion progression
- **Conversation Summary**: Automatic summarization with mood trajectory
- **Key Emotional Moments**: Extraction of high-intensity emotional moments
- **Dual Analysis Modes**: Switch between basic sentiment and comprehensive emotion analysis
- **Export Functionality**: Export conversation and analysis results to PDF, CSV, or JSON
- **AI Summary & Suggestions**: Generates intelligent summaries and actionable next steps using Gemini

## Tests

To test the application, you can use the built-in **Test Zone** or perform manual testing.

### Automated Scenarios (Test Zone)

1. Open the application sidebar.
2. Locate the **Test Zone** section.
3. Select a predefined scenario (e.g., "Positive Flow", "Negative Flow", "Emotional Rollercoaster").
4. Click **Run Scenario**.
5. The system will simulate the conversation and immediately display the sentiment analysis results.

### Manual Testing

1. Start a conversation with various sentiment expressions.
2. Try positive messages: "I love this service!", "Great job!"
3. Try negative messages: "This is terrible", "I'm disappointed"
4. Try mixed conversations to see how overall sentiment is calculated.
5. Use the sentiment analysis button to verify results.

## Highlights of Innovations

- **Test Zone**: A built-in simulation tool to instantly verify sentiment analysis logic without manual typing.
- **Export Functionality**: Ability to export comprehensive analysis reports in PDF, CSV, and JSON formats.
- **Real-Time Analysis**: Instant sentiment feedback on every message sent by the user.
- **Emotion Analysis**: Goes beyond basic positive/negative sentiment to detect specific emotions like Joy, Sadness, Anger, and Fear.
- **AI-Powered Insights**: Generates comprehensive summaries and actionable suggestions using Google Gemini.

## Project Structure

```
Assignment-chatbot-sentiment-analysis/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── chatbot.py           # Gemini API integration
│   ├── sentiment.py         # Sentiment & emotion analysis module
│   ├── conversation.py      # Conversation history management
│   ├── visualization.py     # Chart and graph generation
│   ├── summary.py           # Conversation summarization
│   ├── export.py            # Export functionality (PDF, CSV, JSON)
│   ├── test_scenarios.py    # Predefined test scenarios
│   └── utils.py             # Utility functions
├── app.py                   # Streamlit main application
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```

"""
Streamlit application for Chatbot with Sentiment Analysis.
Includes Tier 1 and Tier 2 features: basic sentiment, emotion analysis, visualizations, and summaries.
"""

import streamlit as st
from datetime import datetime
from src.chatbot import Chatbot
from src.sentiment import SentimentAnalyzer
from src.conversation import ConversationManager
from src.visualization import (
    create_emotion_radar_chart, 
    create_mood_trend_chart,
    create_sentiment_distribution_chart,
    get_sentiment_color,
    get_emotion_color
)
from src.summary import ConversationSummarizer
from src.export import ConversationExporter
from src.alerts import SentimentAlertManager
from src.utils import load_environment_variables
from src.test_scenarios import TEST_SCENARIOS

st.set_page_config(
    page_title="Chatbot with Sentiment Analysis",
    page_icon="💬",
    layout="wide"
)

if 'conversation' not in st.session_state:
    st.session_state.conversation = ConversationManager()

if 'chatbot' not in st.session_state:
    load_environment_variables()
    try:
        st.session_state.chatbot = Chatbot()
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {str(e)}")
        st.session_state.chatbot = None

if 'sentiment_analyzer' not in st.session_state:
    try:
        with st.spinner("Loading sentiment and emotion analysis models..."):
            st.session_state.sentiment_analyzer = SentimentAnalyzer()
    except Exception as e:
        st.error(f"Failed to load sentiment analyzer: {str(e)}")
        st.session_state.sentiment_analyzer = None

if 'sentiment_result' not in st.session_state:
    st.session_state.sentiment_result = None

if 'statement_sentiments' not in st.session_state:
    st.session_state.statement_sentiments = []

if 'statement_emotions' not in st.session_state:
    st.session_state.statement_emotions = []

if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = 'basic'

if 'show_real_time' not in st.session_state:
    st.session_state.show_real_time = True

if 'alert_manager' not in st.session_state:
    st.session_state.alert_manager = SentimentAlertManager(threshold=30.0)

if 'exporter' not in st.session_state:
    st.session_state.exporter = ConversationExporter()

if 'alert_triggered' not in st.session_state:
    st.session_state.alert_triggered = None


def display_message_with_sentiment(role: str, content: str, message_index: int = -1):
    if role == 'user':
        with st.chat_message("user"):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(content)
            with col2:
                if st.session_state.show_real_time and message_index >= 0 and message_index < len(st.session_state.statement_sentiments):
                    sentiment = st.session_state.statement_sentiments[message_index]
                    color = get_sentiment_color(sentiment['label'])
                    st.markdown(
                        f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em;">{sentiment["label"].upper()}</span>',
                        unsafe_allow_html=True
                    )
                    
                    if message_index < len(st.session_state.statement_emotions):
                        emotion = st.session_state.statement_emotions[message_index]
                        emotion_color = get_emotion_color(emotion['label'])
                        st.markdown(
                            f'<span style="background-color: {emotion_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; margin-top: 4px; display: block;">{emotion["label"].upper()}</span>',
                            unsafe_allow_html=True
                        )
    else:
        with st.chat_message("assistant"):
            st.write(content)


def main():
    st.title("💬 Chatbot with Sentiment & Emotion Analysis")
    st.markdown("---")
    
    if st.session_state.chatbot is None or st.session_state.sentiment_analyzer is None:
        st.error("Application not properly initialized. Please check your API keys and try refreshing.")
        return
    
    with st.sidebar:
        st.header("Controls")
        
        st.subheader("Analysis Mode")
        analysis_mode = st.radio(
            "Choose analysis type:",
            ["Basic Sentiment", "Emotion Analysis"],
            index=0 if st.session_state.analysis_mode == 'basic' else 1,
            key="mode_selector"
        )
        st.session_state.analysis_mode = 'basic' if analysis_mode == "Basic Sentiment" else 'emotion'
        
        st.session_state.show_real_time = st.checkbox(
            "Show real-time sentiment badges",
            value=st.session_state.show_real_time
        )
        
        st.markdown("---")
        
        st.subheader("⚠️ Alert Settings")
        alert_enabled = st.checkbox("Enable sentiment alerts", value=st.session_state.alert_manager.alert_enabled)
        st.session_state.alert_manager.alert_enabled = alert_enabled
        
        if alert_enabled:
            threshold = st.slider(
                "Alert Threshold (Score)",
                min_value=0,
                max_value=100,
                value=int(st.session_state.alert_manager.threshold),
                help="Alerts when sentiment score drops below this value"
            )
            st.session_state.alert_manager.set_threshold(float(threshold))
        
        st.markdown("---")
        
        st.subheader("💾 Session Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Session", width='stretch'):
                try:
                    filepath = st.session_state.conversation.save_to_file()
                    st.success(f"Saved to {filepath}")
                except Exception as e:
                    st.error(f"Error saving: {str(e)}")
        
        with col2:
            if st.button("📂 Load Session", width='stretch'):
                sessions = ConversationManager.list_saved_sessions()
                if sessions:
                    session_options = {f"{s['session_id']} ({s['message_count']} msgs)": s['filepath'] for s in sessions}
                    selected = st.selectbox("Select session", list(session_options.keys()), key="session_selector")
                    if st.button("Load", key="load_btn"):
                        try:
                            st.session_state.conversation = ConversationManager.load_from_file(session_options[selected])
                            st.session_state.statement_sentiments = []
                            st.session_state.statement_emotions = []
                            st.session_state.sentiment_result = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading: {str(e)}")
                else:
                    st.info("No saved sessions found")
        
        st.markdown("---")
        
        st.subheader("🧪 Test Zone")
        scenario_names = list(TEST_SCENARIOS.keys())
        selected_scenario = st.selectbox("Select Scenario", scenario_names)
        
        if st.button("▶️ Run Scenario", width='stretch'):
            scenario = TEST_SCENARIOS[selected_scenario]
            
            # Reset state
            st.session_state.conversation.clear()
            st.session_state.chatbot.reset()
            st.session_state.statement_sentiments = []
            st.session_state.statement_emotions = []
            st.session_state.sentiment_result = None
            
            # Load messages
            for msg in scenario['messages']:
                st.session_state.conversation.add_message(msg['role'], msg['content'])
                
                # If it's a user message, run analysis for real-time stats
                if msg['role'] == 'user':
                    if st.session_state.show_real_time:
                        sentiment = st.session_state.sentiment_analyzer.analyze_statement(msg['content'])
                        st.session_state.statement_sentiments.append(sentiment)
                        
                        if st.session_state.analysis_mode == 'emotion':
                            emotion = st.session_state.sentiment_analyzer.analyze_emotion(msg['content'])
                            st.session_state.statement_emotions.append(emotion)
            
            st.success(f"Loaded scenario: {selected_scenario}")
            st.rerun()

        st.markdown("---")
        
        if st.button("🗑️ Clear Conversation", width='stretch'):
            st.session_state.conversation.clear()
            st.session_state.chatbot.reset()
            st.session_state.sentiment_result = None
            st.session_state.statement_sentiments = []
            st.session_state.statement_emotions = []
            st.rerun()
        
        st.markdown("---")
        total_messages = st.session_state.conversation.get_message_count()
        user_messages = st.session_state.conversation.get_user_message_count()
        st.markdown(f"**Total Messages:** {total_messages}")
        st.markdown(f"**User Messages:** {user_messages}")
        
        if not st.session_state.conversation.is_empty():
            button_text = "📊 Analyze Emotion" if st.session_state.analysis_mode == 'emotion' else "📊 Analyze Sentiment"
            if st.button(button_text, width='stretch', type="primary"):
                with st.spinner("Analyzing conversation..."):
                    user_messages_list = [msg['content'] for msg in st.session_state.conversation.get_history() if msg['role'] == 'user']
                    
                    if st.session_state.analysis_mode == 'emotion':
                        emotion_results = st.session_state.sentiment_analyzer.analyze_emotions_all_statements(user_messages_list)
                        emotion_summary = st.session_state.sentiment_analyzer.get_emotion_summary(emotion_results)
                        
                        sentiment_results = st.session_state.sentiment_analyzer.analyze_all_statements(user_messages_list)
                        user_messages_text = st.session_state.conversation.get_conversation_text()
                        overall_sentiment = st.session_state.sentiment_analyzer.analyze_conversation(user_messages_text)
                        
                        st.session_state.sentiment_result = {
                            'type': 'emotion',
                            'emotion_summary': emotion_summary,
                            'emotion_results': emotion_results,
                            'sentiment_results': sentiment_results,
                            'overall_sentiment': overall_sentiment
                        }
                    else:
                        sentiment_results = st.session_state.sentiment_analyzer.analyze_all_statements(user_messages_list)
                        user_messages_text = st.session_state.conversation.get_conversation_text()
                        overall_sentiment = st.session_state.sentiment_analyzer.analyze_conversation(user_messages_text)
                        
                        st.session_state.sentiment_result = {
                            'type': 'basic',
                            'sentiment_results': sentiment_results,
                            'overall_sentiment': overall_sentiment
                        }
                st.rerun()
    
    if st.session_state.alert_triggered:
        alert = st.session_state.alert_triggered
        severity_colors = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }
        severity_icon = severity_colors.get(alert['severity'], '⚠️')
        
        st.warning(
            f"{severity_icon} **Sentiment Alert ({alert['severity'].upper()})**: "
            f"Sentiment score dropped to {alert['score']:.0f}/100 (threshold: {alert['threshold']:.0f})"
        )
        st.session_state.alert_triggered = None
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Conversation")
        
        if st.session_state.conversation.is_empty():
            st.info("👋 Start a conversation by typing a message below!")
        else:
            user_msg_index = 0
            for msg in st.session_state.conversation.get_history():
                if msg['role'] == 'user':
                    display_message_with_sentiment(msg['role'], msg['content'], user_msg_index)
                    user_msg_index += 1
                else:
                    display_message_with_sentiment(msg['role'], msg['content'], -1)
        
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            st.session_state.conversation.add_message('user', user_input)
            
            if st.session_state.show_real_time:
                sentiment = st.session_state.sentiment_analyzer.analyze_statement(user_input)
                st.session_state.statement_sentiments.append(sentiment)
                
                alert = st.session_state.alert_manager.check_statement_sentiment(sentiment, user_input)
                if alert:
                    st.session_state.alert_triggered = alert
                
                if st.session_state.analysis_mode == 'emotion':
                    emotion = st.session_state.sentiment_analyzer.analyze_emotion(user_input)
                    st.session_state.statement_emotions.append(emotion)
            
            with st.spinner("Thinking..."):
                conversation_history = st.session_state.conversation.get_history()
                bot_response = st.session_state.chatbot.get_response(user_input, conversation_history)
            
            st.session_state.conversation.add_message('assistant', bot_response)
            st.session_state.sentiment_result = None
            
            st.rerun()
    
    with col2:
        # Quick stats sidebar
        if not st.session_state.conversation.is_empty() and st.session_state.statement_sentiments:
            st.header("📈 Quick Stats")
            
            # Sentiment distribution
            sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
            for sent in st.session_state.statement_sentiments:
                if sent['label'] in sentiment_counts:
                    sentiment_counts[sent['label']] += 1
            
            for label, count in sentiment_counts.items():
                if count > 0:
                    color = get_sentiment_color(label)
                    st.markdown(f'<span style="color: {color}; font-weight: bold;">{label.capitalize()}: {count}</span>', 
                              unsafe_allow_html=True)
    
    # Display comprehensive analysis results
    if st.session_state.sentiment_result:
        st.markdown("---")
        
        result = st.session_state.sentiment_result
        
        if result['type'] == 'emotion':
            st.header("🎭 Emotion Analysis Results")
            
            # Overall sentiment with score
            overall = result['overall_sentiment']
            score = overall.get('score', 50)
            
            col_score1, col_score2 = st.columns([2, 1])
            with col_score1:
                st.markdown(f"### Overall Sentiment: {overall['label'].capitalize()}")
            with col_score2:
                st.metric("Sentiment Score", f"{score}/100")
            
            st.progress(overall['confidence'], text=f"Confidence: {overall['confidence']:.1%}")
            
            # Emotion radar chart
            st.subheader("Emotion Distribution")
            emotion_summary = result['emotion_summary']
            fig_radar = create_emotion_radar_chart(emotion_summary)
            st.plotly_chart(fig_radar, width='stretch')
            
            # Mood trend chart
            st.subheader("Mood Trend Over Time")
            fig_trend = create_mood_trend_chart(
                result['sentiment_results'],
                result['emotion_results']
            )
            st.plotly_chart(fig_trend, width='stretch')
            
            # Statement-level results
            with st.expander("📋 Statement-Level Analysis"):
                for i, (msg, sent, emo) in enumerate(zip(
                    [m['content'] for m in st.session_state.conversation.get_history() if m['role'] == 'user'],
                    result['sentiment_results'],
                    result['emotion_results']
                )):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.text(f"Message {i+1}: {msg[:50]}...")
                    with col2:
                        st.markdown(f"**Sentiment:** {sent['label']}")
                    with col3:
                        st.markdown(f"**Emotion:** {emo['label']}")
        
        else:
            # Basic sentiment analysis
            st.header("📊 Sentiment Analysis Results")
            
            overall = result['overall_sentiment']
            sentiment_label = overall['label'].capitalize()
            score = overall.get('score', 50)
            
            # Color coding
            if sentiment_label == 'Positive':
                color = "🟢"
            elif sentiment_label == 'Negative':
                color = "🔴"
            else:
                color = "🟡"
            
            col_sent1, col_sent2 = st.columns([2, 1])
            with col_sent1:
                st.markdown(f"### {color} {overall['formatted_output']}")
            with col_sent2:
                st.metric("Sentiment Score", f"{score}/100")
            
            st.progress(overall['confidence'], text=f"Confidence: {overall['confidence']:.1%}")
            
            # Mood trend chart
            st.subheader("Mood Trend Over Time")
            fig_trend = create_mood_trend_chart(result['sentiment_results'])
            st.plotly_chart(fig_trend, width='stretch')
            
            # Sentiment distribution
            st.subheader("Sentiment Distribution")
            fig_dist = create_sentiment_distribution_chart(result['sentiment_results'])
            st.plotly_chart(fig_dist, width='stretch')
            
            # Statement-level results
            with st.expander("📋 Statement-Level Analysis"):
                user_messages = [m['content'] for m in st.session_state.conversation.get_history() if m['role'] == 'user']
                for i, (msg, sent) in enumerate(zip(user_messages, result['sentiment_results'])):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"Message {i+1}: {msg}")
                    with col2:
                        sent_color = get_sentiment_color(sent['label'])
                        st.markdown(
                            f'<span style="background-color: {sent_color}; color: white; padding: 4px 12px; border-radius: 4px;">{sent["label"].upper()} ({sent["confidence"]:.0%})</span>',
                            unsafe_allow_html=True
                        )
        
        # Conversation Summary
        st.markdown("---")
        st.header("📝 Conversation Summary")
        
        summarizer = ConversationSummarizer(st.session_state.chatbot)
        conversation_history = st.session_state.conversation.get_history()
        
        if result['type'] == 'emotion':
            summary = summarizer.generate_summary(
                conversation_history,
                result['sentiment_results'],
                result.get('emotion_results', [])
            )
            mood_trajectory = summarizer.generate_mood_trajectory(
                result['sentiment_results'],
                result.get('emotion_results', [])
            )
        else:
            summary = summarizer.generate_summary(
                conversation_history,
                result['sentiment_results']
            )
            mood_trajectory = summarizer.generate_mood_trajectory(result['sentiment_results'])
        
        st.markdown(f"**Summary:** {summary}")
        st.markdown(f"**Mood Trajectory:** {mood_trajectory}")
        
        # Key moments
        key_moments = summarizer.extract_key_moments(
            conversation_history,
            result['sentiment_results'],
            result.get('emotion_results', [])
        )
        
        if key_moments:
            with st.expander("🔑 Key Emotional Moments"):
                for moment in key_moments:
                    st.markdown(f"**Message {moment['position']}:** {moment['message'][:100]}...")
                    st.markdown(f"- Sentiment: {moment['sentiment']} (confidence: {moment['confidence']:.0%})")
                    if 'emotion' in moment:
                        st.markdown(f"- Emotion: {moment['emotion']} (confidence: {moment['emotion_confidence']:.0%})")
                    st.markdown("---")
        
        # AI Summary & Suggestions
        if st.button("✨ Generate AI Summary & Suggestions", type="primary"):
            with st.spinner("Generating insights with Gemini..."):
                ai_summary = summarizer.generate_ai_summary(conversation_history)
                st.markdown("### 🤖 AI Insights")
                st.markdown(ai_summary)
        
        # Export functionality
        st.markdown("---")
        st.header("📥 Export Results")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            if st.button("📄 Export to PDF", width='stretch'):
                try:
                    pdf_bytes = st.session_state.exporter.export_to_pdf(
                        st.session_state.conversation.get_history(),
                        result,
                        result.get('sentiment_results', [])
                    )
                    st.download_button(
                        label="Download PDF",
                        data=pdf_bytes,
                        file_name=f"conversation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Error exporting PDF: {str(e)}")
        
        with col_exp2:
            if st.button("📊 Export to CSV", width='stretch'):
                try:
                    csv_data = st.session_state.exporter.export_to_csv(
                        st.session_state.conversation.get_history(),
                        result.get('sentiment_results', [])
                    )
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"conversation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error exporting CSV: {str(e)}")
        
        with col_exp3:
            if st.button("📋 Export to JSON", width='stretch'):
                try:
                    json_data = st.session_state.exporter.export_to_json(
                        st.session_state.conversation.get_history(),
                        result,
                        {
                            'session_id': st.session_state.conversation.session_id,
                            'export_date': datetime.now().isoformat()
                        }
                    )
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"conversation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Error exporting JSON: {str(e)}")


if __name__ == "__main__":
    main()

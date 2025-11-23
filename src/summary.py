"""
Conversation summary and mood trajectory analysis module.
"""

from typing import List, Dict, Optional
from src.chatbot import Chatbot


class ConversationSummarizer:
    def __init__(self, chatbot: Optional[Chatbot] = None):
        self.chatbot = chatbot
    
    def generate_summary(self, conversation_history: List[Dict[str, str]], 
                        sentiment_results: Optional[List[Dict[str, any]]] = None,
                        emotion_results: Optional[List[Dict[str, any]]] = None) -> str:
        if not conversation_history:
            return "No conversation to summarize."
        
        user_messages = [msg['content'] for msg in conversation_history if msg['role'] == 'user']
        
        if not user_messages:
            return "No user messages to summarize."
        
        if len(user_messages) <= 3:
            summary = " ".join(user_messages)
        else:
            summary_parts = [
                user_messages[0],
                user_messages[len(user_messages) // 2] if len(user_messages) > 2 else "",
                user_messages[-1]
            ]
            summary = " ".join([part for part in summary_parts if part])
        
        if sentiment_results:
            dominant_sentiment = self._get_dominant_sentiment(sentiment_results)
            summary += f" [Overall sentiment: {dominant_sentiment}]"
        
        return summary
    
    def generate_mood_trajectory(self, sentiment_results: List[Dict[str, any]],
                                emotion_results: Optional[List[Dict[str, any]]] = None) -> str:
        if not sentiment_results:
            return "No sentiment data available for trajectory analysis."
        
        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
        sentiment_values = [sentiment_map.get(r['label'], 0) * r['confidence'] for r in sentiment_results]
        
        if len(sentiment_values) == 1:
            label = sentiment_results[0]['label']
            return f"The conversation maintained a {label} tone throughout."
        
        first_half = sentiment_values[:len(sentiment_values)//2]
        second_half = sentiment_values[len(sentiment_values)//2:]
        
        first_avg = sum(first_half) / len(first_half) if first_half else 0
        second_avg = sum(second_half) / len(second_half) if second_half else 0
        
        if second_avg > first_avg + 0.2:
            trajectory = "improved"
            description = "The conversation showed a positive shift, with sentiment becoming more favorable over time."
        elif second_avg < first_avg - 0.2:
            trajectory = "declined"
            description = "The conversation showed a negative shift, with sentiment becoming less favorable over time."
        else:
            trajectory = "stable"
            description = "The conversation maintained a relatively stable emotional tone throughout."
        
        if emotion_results:
            dominant_emotions = [r['label'] for r in emotion_results]
            emotion_counts = {}
            for emotion in dominant_emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            most_common = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None
            if most_common:
                description += f" The dominant emotion was {most_common}."
        
        return description
    
    def extract_key_moments(self, conversation_history: List[Dict[str, str]],
                           sentiment_results: List[Dict[str, any]],
                           emotion_results: Optional[List[Dict[str, any]]] = None) -> List[Dict[str, str]]:
        key_moments = []
        user_messages = [msg for msg in conversation_history if msg['role'] == 'user']
        
        # Find messages with strong sentiment
        for i, (msg, sent_result) in enumerate(zip(user_messages, sentiment_results)):
            if sent_result['confidence'] > 0.7:  # Strong sentiment
                moment = {
                    'message': msg['content'],
                    'sentiment': sent_result['label'],
                    'confidence': sent_result['confidence'],
                    'position': i + 1
                }
                
                if emotion_results and i < len(emotion_results):
                    moment['emotion'] = emotion_results[i]['label']
                    moment['emotion_confidence'] = emotion_results[i]['confidence']
                
                key_moments.append(moment)
        
        return key_moments
    
    def _get_dominant_sentiment(self, sentiment_results: List[Dict[str, any]]) -> str:
        sentiment_counts = {}
        for result in sentiment_results:
            label = result['label']
            sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
        
        return max(sentiment_counts.items(), key=lambda x: x[1])[0] if sentiment_counts else 'neutral'
    
    def generate_ai_summary(self, conversation_history: List[Dict[str, str]]) -> str:
        if not self.chatbot:
            return self.generate_summary(conversation_history)
        
        # Format conversation for summary
        conversation_text = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in conversation_history
        ])
        
        prompt = f"""Please analyze the following conversation and provide:
1. A concise summary of the main topics and user sentiment.
2. Actionable suggestions or next steps for the user based on the conversation.

Conversation:
{conversation_text}

Output Format:
**Summary:** [Your summary here]

**Suggestions:**
- [Suggestion 1]
- [Suggestion 2]
..."""
        
        try:
            summary = self.chatbot.get_response(prompt, [])
            return summary
        except Exception as e:
            return f"Error generating AI summary: {str(e)}. Using extractive summary instead."



"""
Telegram client for QIE Risk Oracle sentiment analysis.

This module implements the Telethon client for connecting to QIE Telegram channels,
filtering messages for risk-related keywords, and preprocessing messages for sentiment analysis.
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import List, Optional, Set, Dict, Any
from dataclasses import dataclass

from telethon import TelegramClient, events
from telethon.tl.types import Message

from config import get_config


@dataclass
class TelegramMessage:
    """Represents a processed Telegram message."""
    message_id: int
    text: str
    timestamp: datetime
    user_id: int
    channel: str
    sentiment_score: Optional[float] = None
    is_anomaly: bool = False


class TelegramMessageProcessor:
    """Handles Telegram message processing and filtering."""
    
    # Risk-related keywords from requirements
    RISK_KEYWORDS = {
        "hack", "exploit", "down", "scam", "selling", "dump",
        "crash", "collapse", "panic", "fear", "liquidation",
        "rugpull", "rug pull", "exit scam", "ponzi", "fraud"
    }
    
    def __init__(self, api_id: int, api_hash: str, phone_number: str):
        """Initialize the Telegram client.
        
        Args:
            api_id: Telegram API ID
            api_hash: Telegram API hash
            phone_number: Phone number for authentication
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone_number = phone_number
        self.client: Optional[TelegramClient] = None
        self.logger = logging.getLogger(__name__)
        self.message_handlers: List[callable] = []
        
    async def connect(self) -> None:
        """Establish connection to Telegram."""
        try:
            self.client = TelegramClient('qie_risk_oracle', self.api_id, self.api_hash)
            await self.client.start(phone=self.phone_number)
            self.logger.info("Successfully connected to Telegram")
        except Exception as e:
            self.logger.error(f"Failed to connect to Telegram: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Telegram."""
        if self.client:
            await self.client.disconnect()
            self.logger.info("Disconnected from Telegram")
    
    def add_message_handler(self, handler: callable) -> None:
        """Add a message handler function.
        
        Args:
            handler: Function to call when a filtered message is received
        """
        self.message_handlers.append(handler)
    
    def filter_message(self, text: str) -> bool:
        """Check if message contains risk-related keywords.
        
        Args:
            text: Message text to filter
            
        Returns:
            True if message contains risk keywords, False otherwise
        """
        if not text:
            return False
            
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Check for exact keyword matches
        for keyword in self.RISK_KEYWORDS:
            if keyword in text_lower:
                return True
                
        return False
    
    def preprocess_message(self, text: str) -> str:
        """Clean and preprocess message text.
        
        Args:
            text: Raw message text
            
        Returns:
            Cleaned and preprocessed text
        """
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove emojis (basic pattern)
        text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    async def listen_to_channels(self, channel_usernames: List[str]) -> None:
        """Start listening to specified Telegram channels.
        
        Args:
            channel_usernames: List of channel usernames to monitor
        """
        if not self.client:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        self.logger.info(f"Starting to listen to channels: {channel_usernames}")
        
        @self.client.on(events.NewMessage(chats=channel_usernames))
        async def message_handler(event):
            """Handle new messages from monitored channels."""
            try:
                message = event.message
                if not message.text:
                    return
                
                # Filter message for risk keywords
                if not self.filter_message(message.text):
                    return
                
                # Preprocess the message
                processed_text = self.preprocess_message(message.text)
                if not processed_text:
                    return
                
                # Create TelegramMessage object
                telegram_msg = TelegramMessage(
                    message_id=message.id,
                    text=processed_text,
                    timestamp=message.date,
                    user_id=message.sender_id or 0,
                    channel=event.chat.username or str(event.chat_id)
                )
                
                # Call all registered handlers
                for handler in self.message_handlers:
                    try:
                        await handler(telegram_msg)
                    except Exception as e:
                        self.logger.error(f"Error in message handler: {e}")
                
                self.logger.debug(f"Processed message from {telegram_msg.channel}: {processed_text[:100]}...")
                
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
        
        # Keep the client running
        await self.client.run_until_disconnected()
    
    async def get_recent_messages(self, channel_username: str, limit: int = 100) -> List[TelegramMessage]:
        """Get recent messages from a channel for testing/analysis.
        
        Args:
            channel_username: Channel username to fetch from
            limit: Maximum number of messages to fetch
            
        Returns:
            List of processed TelegramMessage objects
        """
        if not self.client:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        messages = []
        try:
            async for message in self.client.iter_messages(channel_username, limit=limit):
                if not message.text:
                    continue
                
                # Filter and preprocess
                if self.filter_message(message.text):
                    processed_text = self.preprocess_message(message.text)
                    if processed_text:
                        telegram_msg = TelegramMessage(
                            message_id=message.id,
                            text=processed_text,
                            timestamp=message.date,
                            user_id=message.sender_id or 0,
                            channel=channel_username
                        )
                        messages.append(telegram_msg)
            
            self.logger.info(f"Fetched {len(messages)} filtered messages from {channel_username}")
            
        except Exception as e:
            self.logger.error(f"Error fetching messages from {channel_username}: {e}")
        
        return messages


class TelegramConfig:
    """Configuration for Telegram client."""
    
    def __init__(self):
        config = get_config()
        self.api_id = config.api.telegram_api_id
        self.api_hash = config.api.telegram_api_hash
        self.phone_number = config.api.telegram_phone_number
        
        # QIE-related channels to monitor (these would be actual QIE channels)
        self.monitored_channels = [
            "qie_pulse_official",  # Main QIE channel
            "qie_developers",      # Developer discussions
            "qie_community",       # Community discussions
            "crypto_alerts",       # General crypto alerts
        ]
    
    def validate(self) -> bool:
        """Validate that all required configuration is present."""
        required_fields = [self.api_id, self.api_hash, self.phone_number]
        return all(field for field in required_fields)


# Factory function for easy instantiation
def create_telegram_processor() -> TelegramMessageProcessor:
    """Create a configured TelegramMessageProcessor instance."""
    config = TelegramConfig()
    
    if not config.validate():
        raise ValueError("Missing required Telegram configuration. Check your .env file.")
    
    return TelegramMessageProcessor(
        api_id=config.api_id,
        api_hash=config.api_hash,
        phone_number=config.phone_number
    )
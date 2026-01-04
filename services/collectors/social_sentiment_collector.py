"""
Social Sentiment Data Collector
Implements Twitter Academic Research API and Reddit API clients for historical social sentiment data
"""

import asyncio
import aiohttp
import praw
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import time
import re
from urllib.parse import urlencode

from config import get_config
from logging_config import get_data_logger, get_performance_logger

@dataclass
class SocialSentimentData:
    """Data structure for social sentiment information"""
    platform: str  # "twitter" or "reddit"
    post_id: str
    text: str
    timestamp: datetime
    author: str
    engagement_score: float  # likes, retweets, upvotes, etc.
    contains_keywords: List[str]
    sentiment_score: Optional[float] = None
    source: str = "unknown"

@dataclass
class MarketCrashEvent:
    """Market crash event configuration for data collection"""
    name: str
    start_date: datetime
    end_date: datetime
    keywords: List[str]
    description: str

class TwitterAcademicCollector:
    """Twitter Academic Research API client for historical tweets"""
    
    def __init__(self):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        self.base_url = "https://api.twitter.com/2"
        self.bearer_token = self.config.api.twitter_bearer_token
        
        if not self.bearer_token:
            raise ValueError("Twitter Bearer Token not configured")
        
        # Risk-related keywords from requirements
        self.risk_keywords = [
            "hack", "exploit", "down", "scam", "selling", "dump",
            "crash", "collapse", "panic", "fear", "liquidation",
            "rugpull", "exit scam", "ponzi", "bubble", "correction"
        ]
        
        # Market crash events for historical analysis
        self.crash_events = [
            MarketCrashEvent(
                name="COVID-19 Crash",
                start_date=datetime(2020, 3, 1),
                end_date=datetime(2020, 3, 31),
                keywords=["covid", "coronavirus", "pandemic", "lockdown", "crash"],
                description="March 2020 COVID-19 market crash"
            ),
            MarketCrashEvent(
                name="Terra Luna Collapse",
                start_date=datetime(2022, 5, 1),
                end_date=datetime(2022, 5, 31),
                keywords=["terra", "luna", "ust", "depeg", "algorithmic", "stablecoin"],
                description="May 2022 Terra Luna ecosystem collapse"
            ),
            MarketCrashEvent(
                name="FTX Collapse",
                start_date=datetime(2022, 11, 1),
                end_date=datetime(2022, 11, 30),
                keywords=["ftx", "alameda", "sbf", "bankrupt", "liquidity", "withdrawal"],
                description="November 2022 FTX exchange collapse"
            )
        ]
    
    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers for Twitter API"""
        return {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }
    
    def _build_search_query(self, keywords: List[str], additional_filters: str = "") -> str:
        """Build Twitter search query with keywords and filters"""
        # Combine risk keywords with event-specific keywords
        all_keywords = self.risk_keywords + keywords
        
        # Create OR query for keywords
        keyword_query = " OR ".join(f'"{keyword}"' for keyword in all_keywords)
        
        # Add crypto-related context
        crypto_context = "(bitcoin OR crypto OR cryptocurrency OR btc OR eth OR ethereum)"
        
        # Combine with additional filters
        query = f"({keyword_query}) {crypto_context} lang:en -is:retweet"
        
        if additional_filters:
            query += f" {additional_filters}"
        
        return query
    
    async def get_historical_tweets(self, crash_event: MarketCrashEvent, 
                                   max_results: int = 100) -> List[SocialSentimentData]:
        """Get historical tweets for a specific market crash event"""
        start_time = time.time()
        
        try:
            # Build search query
            query = self._build_search_query(crash_event.keywords)
            
            # Format dates for Twitter API
            start_time_str = crash_event.start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            end_time_str = crash_event.end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            
            url = f"{self.base_url}/tweets/search/all"
            params = {
                "query": query,
                "start_time": start_time_str,
                "end_time": end_time_str,
                "max_results": min(max_results, 500),  # Twitter API limit
                "tweet.fields": "created_at,author_id,public_metrics,text",
                "expansions": "author_id",
                "user.fields": "username"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self._get_headers(), params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        results = []
                        tweets = data.get("data", [])
                        users = {user["id"]: user for user in data.get("includes", {}).get("users", [])}
                        
                        for tweet in tweets:
                            # Extract engagement metrics
                            metrics = tweet.get("public_metrics", {})
                            engagement_score = (
                                metrics.get("like_count", 0) + 
                                metrics.get("retweet_count", 0) * 2 +  # Weight retweets more
                                metrics.get("reply_count", 0)
                            )
                            
                            # Find matching keywords
                            text_lower = tweet["text"].lower()
                            matching_keywords = [
                                keyword for keyword in (self.risk_keywords + crash_event.keywords)
                                if keyword.lower() in text_lower
                            ]
                            
                            # Get author username
                            author_id = tweet.get("author_id", "unknown")
                            author_username = users.get(author_id, {}).get("username", f"user_{author_id}")
                            
                            sentiment_data = SocialSentimentData(
                                platform="twitter",
                                post_id=tweet["id"],
                                text=tweet["text"],
                                timestamp=datetime.fromisoformat(tweet["created_at"].replace("Z", "+00:00")),
                                author=author_username,
                                engagement_score=float(engagement_score),
                                contains_keywords=matching_keywords,
                                source="twitter_academic"
                            )
                            results.append(sentiment_data)
                        
                        duration_ms = (time.time() - start_time) * 1000
                        self.performance_logger.log_inference_time("twitter_historical_tweets", duration_ms, True)
                        self.data_logger.log_data_collection("twitter", crash_event.name, len(results), True)
                        
                        return results
                    
                    elif response.status == 429:
                        # Rate limit exceeded
                        raise Exception("Twitter API rate limit exceeded")
                    else:
                        error_text = await response.text()
                        raise Exception(f"Twitter API error {response.status}: {error_text}")
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("twitter_historical_tweets", duration_ms, False)
            self.data_logger.log_data_collection("twitter", crash_event.name, 0, False, str(e))
            raise
    
    async def get_recent_tweets(self, keywords: List[str] = None, 
                               max_results: int = 100) -> List[SocialSentimentData]:
        """Get recent tweets with risk-related keywords"""
        start_time = time.time()
        
        try:
            # Use provided keywords or default risk keywords
            search_keywords = keywords or self.risk_keywords
            query = self._build_search_query(search_keywords)
            
            url = f"{self.base_url}/tweets/search/recent"
            params = {
                "query": query,
                "max_results": min(max_results, 100),  # Recent search limit
                "tweet.fields": "created_at,author_id,public_metrics,text",
                "expansions": "author_id",
                "user.fields": "username"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self._get_headers(), params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        results = []
                        tweets = data.get("data", [])
                        users = {user["id"]: user for user in data.get("includes", {}).get("users", [])}
                        
                        for tweet in tweets:
                            # Extract engagement metrics
                            metrics = tweet.get("public_metrics", {})
                            engagement_score = (
                                metrics.get("like_count", 0) + 
                                metrics.get("retweet_count", 0) * 2 +
                                metrics.get("reply_count", 0)
                            )
                            
                            # Find matching keywords
                            text_lower = tweet["text"].lower()
                            matching_keywords = [
                                keyword for keyword in search_keywords
                                if keyword.lower() in text_lower
                            ]
                            
                            # Get author username
                            author_id = tweet.get("author_id", "unknown")
                            author_username = users.get(author_id, {}).get("username", f"user_{author_id}")
                            
                            sentiment_data = SocialSentimentData(
                                platform="twitter",
                                post_id=tweet["id"],
                                text=tweet["text"],
                                timestamp=datetime.fromisoformat(tweet["created_at"].replace("Z", "+00:00")),
                                author=author_username,
                                engagement_score=float(engagement_score),
                                contains_keywords=matching_keywords,
                                source="twitter_recent"
                            )
                            results.append(sentiment_data)
                        
                        duration_ms = (time.time() - start_time) * 1000
                        self.performance_logger.log_inference_time("twitter_recent_tweets", duration_ms, True)
                        self.data_logger.log_data_collection("twitter", "recent", len(results), True)
                        
                        return results
                    
                    else:
                        error_text = await response.text()
                        raise Exception(f"Twitter API error {response.status}: {error_text}")
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("twitter_recent_tweets", duration_ms, False)
            self.data_logger.log_data_collection("twitter", "recent", 0, False, str(e))
            raise

class RedditCollector:
    """Reddit API client for r/cryptocurrency historical posts"""
    
    def __init__(self):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # Initialize Reddit API client
        self.reddit = praw.Reddit(
            client_id=self.config.api.reddit_client_id,
            client_secret=self.config.api.reddit_client_secret,
            user_agent=self.config.api.reddit_user_agent
        )
        
        # Risk-related keywords (same as Twitter)
        self.risk_keywords = [
            "hack", "exploit", "down", "scam", "selling", "dump",
            "crash", "collapse", "panic", "fear", "liquidation",
            "rugpull", "exit scam", "ponzi", "bubble", "correction"
        ]
        
        # Target subreddits
        self.target_subreddits = [
            "cryptocurrency",
            "CryptoCurrency", 
            "Bitcoin",
            "ethereum",
            "CryptoMarkets"
        ]
    
    def _contains_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """Check if text contains any of the specified keywords"""
        text_lower = text.lower()
        matching_keywords = []
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                matching_keywords.append(keyword)
        
        return matching_keywords
    
    def get_historical_posts(self, crash_event: MarketCrashEvent, 
                           subreddit_name: str = "cryptocurrency",
                           limit: int = 100) -> List[SocialSentimentData]:
        """Get historical Reddit posts for a specific market crash event"""
        start_time = time.time()
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            results = []
            
            # Search for posts with crash event keywords
            search_query = " OR ".join(crash_event.keywords)
            
            # Get posts from the time period
            for submission in subreddit.search(search_query, sort="new", time_filter="all", limit=limit):
                post_date = datetime.fromtimestamp(submission.created_utc)
                
                # Filter by date range
                if not (crash_event.start_date <= post_date <= crash_event.end_date):
                    continue
                
                # Check for risk keywords in title and text
                full_text = f"{submission.title} {submission.selftext}"
                matching_keywords = self._contains_keywords(full_text, self.risk_keywords + crash_event.keywords)
                
                if matching_keywords:  # Only include posts with risk keywords
                    # Calculate engagement score
                    engagement_score = (
                        submission.score +  # Upvotes - downvotes
                        submission.num_comments * 2  # Weight comments more
                    )
                    
                    sentiment_data = SocialSentimentData(
                        platform="reddit",
                        post_id=submission.id,
                        text=full_text,
                        timestamp=post_date,
                        author=str(submission.author) if submission.author else "deleted",
                        engagement_score=float(max(0, engagement_score)),  # Ensure non-negative
                        contains_keywords=matching_keywords,
                        source="reddit_historical"
                    )
                    results.append(sentiment_data)
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("reddit_historical_posts", duration_ms, True)
            self.data_logger.log_data_collection("reddit", f"{subreddit_name}_{crash_event.name}", len(results), True)
            
            return results
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("reddit_historical_posts", duration_ms, False)
            self.data_logger.log_data_collection("reddit", f"{subreddit_name}_{crash_event.name}", 0, False, str(e))
            raise
    
    def get_recent_posts(self, subreddit_name: str = "cryptocurrency", 
                        keywords: List[str] = None, limit: int = 100) -> List[SocialSentimentData]:
        """Get recent Reddit posts with risk-related keywords"""
        start_time = time.time()
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            search_keywords = keywords or self.risk_keywords
            results = []
            
            # Get recent posts
            for submission in subreddit.new(limit=limit):
                # Check for risk keywords in title and text
                full_text = f"{submission.title} {submission.selftext}"
                matching_keywords = self._contains_keywords(full_text, search_keywords)
                
                if matching_keywords:  # Only include posts with risk keywords
                    # Calculate engagement score
                    engagement_score = (
                        submission.score +
                        submission.num_comments * 2
                    )
                    
                    sentiment_data = SocialSentimentData(
                        platform="reddit",
                        post_id=submission.id,
                        text=full_text,
                        timestamp=datetime.fromtimestamp(submission.created_utc),
                        author=str(submission.author) if submission.author else "deleted",
                        engagement_score=float(max(0, engagement_score)),
                        contains_keywords=matching_keywords,
                        source="reddit_recent"
                    )
                    results.append(sentiment_data)
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("reddit_recent_posts", duration_ms, True)
            self.data_logger.log_data_collection("reddit", f"{subreddit_name}_recent", len(results), True)
            
            return results
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("reddit_recent_posts", duration_ms, False)
            self.data_logger.log_data_collection("reddit", f"{subreddit_name}_recent", 0, False, str(e))
            raise

class SocialSentimentCollector:
    """Main social sentiment data collector combining Twitter and Reddit"""
    
    def __init__(self):
        self.twitter = TwitterAcademicCollector()
        self.reddit = RedditCollector()
        self.config = get_config()
        self.data_logger = get_data_logger()
        
        # Market crash events for historical analysis
        self.crash_events = self.twitter.crash_events
    
    async def collect_crash_event_data(self, crash_event: MarketCrashEvent, 
                                     max_tweets: int = 100, 
                                     max_reddit_posts: int = 100) -> Dict[str, List[SocialSentimentData]]:
        """Collect social sentiment data for a specific market crash event"""
        results = {
            "twitter": [],
            "reddit": []
        }
        
        try:
            # Collect Twitter data
            twitter_data = await self.twitter.get_historical_tweets(crash_event, max_tweets)
            results["twitter"] = twitter_data
        except Exception as e:
            self.data_logger.log_data_collection("twitter", crash_event.name, 0, False, str(e))
        
        try:
            # Collect Reddit data from multiple subreddits
            reddit_data = []
            for subreddit in self.reddit.target_subreddits:
                try:
                    subreddit_posts = self.reddit.get_historical_posts(
                        crash_event, subreddit, max_reddit_posts // len(self.reddit.target_subreddits)
                    )
                    reddit_data.extend(subreddit_posts)
                except Exception as e:
                    self.data_logger.log_data_collection("reddit", f"{subreddit}_{crash_event.name}", 0, False, str(e))
                    continue
            
            results["reddit"] = reddit_data
        except Exception as e:
            self.data_logger.log_data_collection("reddit", crash_event.name, 0, False, str(e))
        
        return results
    
    async def collect_all_crash_events_data(self, max_tweets_per_event: int = 100,
                                          max_reddit_posts_per_event: int = 100) -> Dict[str, Dict[str, List[SocialSentimentData]]]:
        """Collect social sentiment data for all major market crash events"""
        all_results = {}
        
        for crash_event in self.crash_events:
            try:
                event_data = await self.collect_crash_event_data(
                    crash_event, max_tweets_per_event, max_reddit_posts_per_event
                )
                all_results[crash_event.name] = event_data
            except Exception as e:
                self.data_logger.log_data_collection("social_sentiment", crash_event.name, 0, False, str(e))
                all_results[crash_event.name] = {"twitter": [], "reddit": []}
        
        return all_results
    
    async def collect_recent_sentiment_data(self, keywords: List[str] = None,
                                          max_tweets: int = 100,
                                          max_reddit_posts: int = 100) -> Dict[str, List[SocialSentimentData]]:
        """Collect recent social sentiment data"""
        results = {
            "twitter": [],
            "reddit": []
        }
        
        try:
            # Collect recent Twitter data
            twitter_data = await self.twitter.get_recent_tweets(keywords, max_tweets)
            results["twitter"] = twitter_data
        except Exception as e:
            self.data_logger.log_data_collection("twitter", "recent", 0, False, str(e))
        
        try:
            # Collect recent Reddit data from multiple subreddits
            reddit_data = []
            for subreddit in self.reddit.target_subreddits:
                try:
                    subreddit_posts = self.reddit.get_recent_posts(
                        subreddit, keywords, max_reddit_posts // len(self.reddit.target_subreddits)
                    )
                    reddit_data.extend(subreddit_posts)
                except Exception as e:
                    self.data_logger.log_data_collection("reddit", f"{subreddit}_recent", 0, False, str(e))
                    continue
            
            results["reddit"] = reddit_data
        except Exception as e:
            self.data_logger.log_data_collection("reddit", "recent", 0, False, str(e))
        
        return results
    
    def filter_by_keywords(self, data: List[SocialSentimentData], 
                          keywords: List[str]) -> List[SocialSentimentData]:
        """Filter social sentiment data by specific keywords"""
        filtered_data = []
        
        for item in data:
            text_lower = item.text.lower()
            matching_keywords = [
                keyword for keyword in keywords
                if keyword.lower() in text_lower
            ]
            
            if matching_keywords:
                # Update the contains_keywords field
                item.contains_keywords = list(set(item.contains_keywords + matching_keywords))
                filtered_data.append(item)
        
        return filtered_data
    
    def validate_data(self, data: List[SocialSentimentData]) -> List[SocialSentimentData]:
        """Validate and clean social sentiment data"""
        valid_data = []
        
        for item in data:
            # Basic validation
            if not item.text or len(item.text.strip()) < 10:  # Too short
                continue
            if item.engagement_score < 0:  # Invalid engagement
                continue
            if not item.contains_keywords:  # No relevant keywords
                continue
            
            # Clean text (remove excessive whitespace, etc.)
            item.text = re.sub(r'\s+', ' ', item.text.strip())
            
            valid_data.append(item)
        
        return valid_data
    
    def get_crash_events(self) -> List[MarketCrashEvent]:
        """Get list of configured market crash events"""
        return self.crash_events
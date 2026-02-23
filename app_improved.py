import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import praw
import googleapiclient.discovery
import googleapiclient.errors
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
from datetime import datetime, timedelta
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote
import requests
from googleapiclient.discovery import build
from flask import Flask, render_template, request, jsonify
import openai
import json
from transformers import pipeline
import tempfile
import shutil
from werkzeug.utils import secure_filename
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize as None - will be set up through UI
reddit = None

# Hardcoded Google API credentials
GOOGLE_API_KEY = "AIzaSyDhVYI_4Z-6UIu3iZp5GC7Hk1BwcsyMGF0"
GOOGLE_CSE_ID = "964673bf82dba4a9c"

# Define the scope for YouTube API
SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']

app = Flask(__name__)
app.secret_key = os.urandom(24)  # More secure secret key

# Create a temporary directory for storing uploaded files
TEMP_DIR = tempfile.mkdtemp()

def cleanup_temp_files():
    """Clean up temporary files older than 1 hour"""
    current_time = datetime.now()
    for filename in os.listdir(TEMP_DIR):
        filepath = os.path.join(TEMP_DIR, filename)
        file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
        if current_time - file_modified > timedelta(hours=1):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Error removing temporary file {filepath}: {e}")

def init_reddit():
    """Initialize Reddit API with credentials from session"""
    if 'reddit_credentials' not in session:
        return None
    
    creds = session['reddit_credentials']
    return praw.Reddit(
        client_id=creds['client_id'],
        client_secret=creds['client_secret'],
        user_agent=creds['user_agent'],
        username=creds['username'],
        password=creds['password']
    )

def authenticate_youtube():
    """Authenticate and return the YouTube API service using temporary credentials"""
    if 'youtube_client_secret' not in session:
        return None

    creds = None
    token_path = os.path.join(TEMP_DIR, f"token_{session.get('user_id', 'default')}.pickle")
    
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
            
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            client_secret_path = os.path.join(TEMP_DIR, f"client_secret_{session.get('user_id', 'default')}.json")
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
            creds = flow.run_local_server(port=0)
            
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
            
    return googleapiclient.discovery.build('youtube', 'v3', credentials=creds)

def scrape_content(search_term, num_results=10, start=1):
    """Fetch search results using Google Custom Search API with hardcoded credentials"""
    try:
        service = build(
            "customsearch", 
            "v1", 
            developerKey=GOOGLE_API_KEY
        )
        
        res = service.cse().list(
            q=search_term, 
            cx=GOOGLE_CSE_ID, 
            num=num_results, 
            start=start
        ).execute()

        results = []
        if 'items' in res:
            for item in res['items']:
                results.append((item['title'], item['link']))

        if not results:
            results = [("No relevant results found", "#")]

        return results
    except Exception as e:
        print(f"An error occurred: {e}")
        return [("Unable to fetch results at the moment. Please try again later.", "#")]

@app.route('/auth', methods=['GET'])
def auth():
    cleanup_temp_files()  # Clean up old files
    return render_template('auth.html')

@app.route('/submit_credentials', methods=['POST'])
def submit_credentials():
    try:
        # Get Reddit credentials
        reddit_creds = {
            'client_id': request.form['reddit_client_id'],
            'client_secret': request.form['reddit_client_secret'],
            'user_agent': 'social_media_recommendation:v1.0',
            'username': request.form['reddit_username'],
            'password': request.form['reddit_password']
        }
        
        # Test Reddit credentials
        try:
            test_reddit = praw.Reddit(**reddit_creds)
            test_reddit.user.me()  # This will fail if credentials are invalid
            session['reddit_credentials'] = reddit_creds
        except Exception as e:
            return jsonify({'error': f'Invalid Reddit credentials: {str(e)}'}), 400

        # Handle YouTube client secret file
        if 'youtube_client_secret' not in request.files:
            return jsonify({'error': 'No YouTube client secret file provided'}), 400
            
        file = request.files['youtube_client_secret']
        if file.filename == '':
            return jsonify({'error': 'No YouTube client secret file selected'}), 400
            
        if file:
            # Generate unique user ID if not exists
            if 'user_id' not in session:
                session['user_id'] = os.urandom(16).hex()
                
            # Save client secret file
            filename = f"client_secret_{session['user_id']}.json"
            filepath = os.path.join(TEMP_DIR, filename)
            file.save(filepath)
            session['youtube_client_secret'] = True
            
            # Schedule file deletion
            deletion_time = datetime.now() + timedelta(hours=1)
            session['file_deletion_time'] = deletion_time.timestamp()

        return jsonify({'success': True, 'message': 'Credentials saved successfully'})
        
    except Exception as e:
        return jsonify({'error': f'Error saving credentials: {str(e)}'}), 500

# Fetch Reddit history data
def fetch_reddit_history(limit=10):
    """Fetch Reddit history using session-based credentials"""
    reddit = init_reddit()
    if not reddit:
        return []
        
    try:
        user = reddit.user.me()
        history = []

        # 1️⃣ Saved posts/comments
        for item in user.saved(limit=limit):
            history.append(process_reddit_item(item, signal_type="saved"))

        # 2️⃣ Upvoted posts/comments
        for item in user.upvoted(limit=limit):
            history.append(process_reddit_item(item, signal_type="upvoted"))

        return history

    except Exception as e:
        print(f"Error fetching Reddit history: {str(e)}")
        return []
def process_reddit_item(item, signal_type):
    timestamp = datetime.utcfromtimestamp(item.created_utc).strftime('%Y-%m-%d %H:%M:%S')

    return {
        'type': 'Comment' if isinstance(item, praw.models.Comment) else 'Submission',
        'title': item.submission.title if isinstance(item, praw.models.Comment) else item.title,
        'content': item.body if isinstance(item, praw.models.Comment) else item.selftext,
        'url': item.permalink if isinstance(item, praw.models.Comment) else item.url,
        'subreddit': item.subreddit.display_name,
        'timestamp': timestamp,
        'signal': signal_type   #  Important for ML weighting
    }
# fetch YT playlist created by user of watch later video
def fetch_youtube_playlists(youtube, max_results=5):
    try:
        playlists = youtube.playlists().list(
            part="snippet",
            mine=True,
            maxResults=max_results
        ).execute()

        playlist_data = []

        for playlist in playlists.get("items", []):
            playlist_id = playlist["id"]

            items = youtube.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=5
            ).execute()

            for video in items.get("items", []):
                playlist_data.append({
                    'title': video['snippet']['title'],
                    'url': f"https://www.youtube.com/watch?v={video['snippet']['resourceId']['videoId']}",
                    'channel': video['snippet']['channelTitle'],
                    'timestamp': video['snippet']['publishedAt'],
                    'type': 'YouTube',
                    'signal': 'playlist'
                })

        return playlist_data

    except Exception as e:
        print(f"Error fetching playlists: {str(e)}")
        return []

# Fetch YouTube liked videos
def fetch_youtube_liked_videos(youtube, max_results=10):
    try:
        request = youtube.videos().list(
            part="snippet",
            myRating="like",
            maxResults=max_results
        )
        response = request.execute()
        liked_videos = []
        for item in response.get('items', []):
            video = {
                'title': item['snippet']['title'],
                'url': f"https://www.youtube.com/watch?v={item['id']}",
                'channel': item['snippet']['channelTitle'],
                'timestamp': item['snippet']['publishedAt'],  # Use actual timestamp
                'type': 'YouTube'
            }
            liked_videos.append(video)
        return liked_videos
    except googleapiclient.errors.HttpError as e:
        print(f"An error occurred: {e}")
        return []

# Function to categorize search terms using K-Means clustering
def categorize_search_terms(search_history, num_categories=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    search_terms = []
    for entry in search_history:
        title = entry.get('title', '')
        content = entry.get('content', '')
        search_terms.append(title + " " + content)
    X = vectorizer.fit_transform(search_terms)
    num_clusters = min(len(search_terms), num_categories)
    if num_clusters < 1:
        return pd.DataFrame(columns=['search_term', 'cluster', 'time']), {}
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    terms = vectorizer.get_feature_names_out()
    cluster_terms = {}
    for i in range(num_clusters):
        center_terms = kmeans.cluster_centers_[i].argsort()[-5:][::-1]
        cluster_terms[i] = [terms[idx] for idx in center_terms]
    categorized_data = pd.DataFrame({
        'search_term': search_terms,
        'cluster': kmeans.labels_,
        'time': [entry.get('timestamp', '') for entry in search_history]
    })
    return categorized_data, cluster_terms

# Function to recommend search terms based on cluster frequency
def recommend_search_terms(categorized_data):
    recommendations = {}
    grouped = categorized_data.groupby('cluster')['search_term'].apply(lambda x: x.value_counts())
    for cluster, terms in grouped.groupby(level=0):
        recommendations[cluster] = terms.index.get_level_values(1).tolist()
    return recommendations

# Modify the index route to handle authentication first
@app.route('/')
def index():
    # First check if user is authenticated
    if 'reddit_credentials' not in session or 'youtube_client_secret' not in session:
        return redirect(url_for('auth'))
        
    try:
        # Initialize clients with session credentials
        reddit = init_reddit()
        youtube = authenticate_youtube()
        
        if not reddit or not youtube:
            flash('Authentication failed. Please log in again.', 'error')
            return redirect(url_for('auth'))
            
        # Try to fetch data
        reddit_data = fetch_reddit_history(limit=5)
        youtube_liked_data = fetch_youtube_liked_videos(youtube, max_results=5)
        youtube_playlist_data = fetch_youtube_playlists(youtube)

        # If we have data, show the main page
        if reddit_data or youtube_liked_data:
            return render_template('index.html', 
                                reddit_data=reddit_data, 
                                youtube_data=youtube_liked_data)
        
        # If no data and no preferences, show preferences
        if 'user_interests' not in session:
            flash('No content found. Please set your preferences first.', 'info')
            return redirect(url_for('preferences'))
            
        # If we have preferences but no data, show recommendations based on preferences
        flash('No content found. Showing recommendations based on your preferences.', 'info')
        return redirect(url_for('recommendations'))
        
    except Exception as e:
        print(f"Error in index route: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('auth'))

@app.route('/process', methods=['POST'])
def process():
    try:
        # Fetch data from Reddit and YouTube
        reddit_data = fetch_reddit_history(limit=10)
        youtube = authenticate_youtube()
        youtube_liked_data = fetch_youtube_liked_videos(youtube, max_results=5)
        
        # If no data and no preferences, redirect to preferences
        if not reddit_data and not youtube_liked_data:
            if 'user_interests' not in session:
                flash('No content found. Please set your preferences first.', 'info')
                return redirect(url_for('preferences'))
            # If we have preferences, redirect to recommendations
            flash('No content found. Showing recommendations based on your preferences.', 'info')
            return redirect(url_for('recommendations'))
            
        # If we have data, proceed with recommendations
        combined_data = reddit_data + youtube_liked_data + youtube_playlist_data
        categorized_results, cluster_terms = categorize_search_terms(combined_data, num_categories=5)
        recommendations = recommend_search_terms(categorized_results)
        results = {}
        for cluster, terms in recommendations.items():
            results[cluster] = {}
            for term in terms:
                results[cluster][term] = scrape_content(term)
        return render_template('results.html', results=results, cluster_terms=cluster_terms)
        
    except Exception as e:
        print(f"Error in process route: {str(e)}")
        if 'user_interests' not in session:
            flash('An error occurred. Please set your preferences.', 'error')
            return redirect(url_for('preferences'))
        flash('An error occurred. Showing recommendations based on your preferences.', 'error')
        return redirect(url_for('recommendations'))

@app.route('/preferences')
def preferences():
    # Check authentication first
    if 'reddit_credentials' not in session or 'youtube_client_secret' not in session:
        return redirect(url_for('auth'))
        
    try:
        reddit = init_reddit()
        youtube = authenticate_youtube()
        
        if not reddit or not youtube:
            flash('Authentication failed. Please log in again.', 'error')
            return redirect(url_for('auth'))
            
        return render_template('preferences.html')
        
    except Exception as e:
        print(f"Error in preferences route: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('auth'))

@app.route('/save_preferences', methods=['POST'])
def save_preferences():
    interests = json.loads(request.form.get('interests', '[]'))
    if len(interests) < 3:
        return jsonify({'error': 'Please select at least 3 interests'}), 400
    
    session['user_interests'] = interests
    return jsonify({'success': True})

@app.route('/recommendations')
def recommendations():
    # Check authentication first
    if 'reddit_credentials' not in session or 'youtube_client_secret' not in session:
        return redirect(url_for('auth'))
        
    try:
        reddit = init_reddit()
        youtube = authenticate_youtube()
        
        if not reddit or not youtube:
            flash('Authentication failed. Please log in again.', 'error')
            return redirect(url_for('auth'))
            
        # Try to fetch data
        reddit_data = fetch_reddit_history(limit=5)
        youtube_liked_data = fetch_youtube_liked_videos(youtube, max_results=5)
        
        # If we have data, use data-based recommendations
        if reddit_data or youtube_liked_data:
            combined_data = reddit_data + youtube_liked_data
            categorized_results, cluster_terms = categorize_search_terms(combined_data, num_categories=5)
            recommendations = recommend_search_terms(categorized_results)
            results = {}
            for cluster, terms in recommendations.items():
                results[cluster] = {}
                for term in terms:
                    results[cluster][term] = scrape_content(term)
            return render_template('results.html', 
                                results=results, 
                                cluster_terms=cluster_terms)
        
        # If no data but have preferences, use preference-based recommendations
        if 'user_interests' in session:
            user_interests = session['user_interests']
            recommendations = get_default_recommendations_by_interests(user_interests)
            return render_template('recommendations.html', 
                                recommendations=recommendations)
            
        # If no data and no preferences, redirect to preferences with a message
        flash('No content or preferences found. Please set your preferences first.', 'info')
        return redirect(url_for('preferences'))
        
    except Exception as e:
        print(f"Error in recommendations: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('auth'))

#@app.route('/visualize')


# Load OpenAI API Key

@app.route('/visualize', methods=['GET'])

def visualize():
    youtube = authenticate_youtube()
    reddit_data = fetch_reddit_history(limit=5)
    youtube_liked_data = fetch_youtube_liked_videos(youtube, max_results=5)
    combined_data = reddit_data + youtube_liked_data

    # Categorize data into clusters
    categorized_results, cluster_terms = categorize_search_terms(combined_data, num_categories=5)

    # Calculate cluster distribution for visualization
    cluster_distribution = categorized_results['cluster'].value_counts().to_dict()

    # Detect genres using BERT-based zero-shot classification
    genres = detect_genres_bert(cluster_terms)

    return render_template(
        'visualization.html', 
        cluster_distribution=cluster_distribution, 
        genres=genres,
        cluster_terms=cluster_terms  # ✅ Ensure this is passed
    )


def detect_genres_bert(cluster_terms):
    """
    Assign genres to clusters using DistilBERT for zero-shot classification.
    
    Args:
        cluster_terms (dict): Dictionary where keys are cluster IDs and values are lists of terms.
    
    Returns:
        dict: Dictionary mapping cluster IDs to genres.
    """
    if not cluster_terms:
        return {}

    # Initialize the zero-shot classification pipeline using a pretrained model
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    # Expanded list of possible genres
    possible_labels = [
        "Technology", "Entertainment", "Sports", "Education", "News", 
        "Health", "Business", "Lifestyle", "Science", "Travel", 
        "Finance", "Gaming", "Politics", "History", "Art"
    ]
    
    cluster_genres = {}
    
    # Convert numeric keys to strings if they aren't already
    cluster_terms = {str(k): v for k, v in cluster_terms.items()}
    
    for cluster_id, terms in cluster_terms.items():
        if not terms:  # Skip if terms list is empty
            continue
            
        # Combine the terms from each cluster into a single string
        cluster_text = " ".join(str(term) for term in terms)
        
        # Use zero-shot classification to predict the genre
        result = classifier(cluster_text, candidate_labels=possible_labels)
        
        # Get the genre with the highest probability score
        genre = result['labels'][0]
        
        # Store as integer key for consistency
        cluster_genres[int(cluster_id)] = genre
    
    return cluster_genres


def get_time_of_day(timestamp):
    """Convert timestamp to time of day category.
    
    Args:
        timestamp: Can be either a string timestamp or datetime object
    """
    try:
        if isinstance(timestamp, str):
            # Try YouTube format first (ISO format)
            try:
                hour = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ').hour
            except ValueError:
                # Try Reddit format
                hour = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').hour
        elif isinstance(timestamp, datetime):
            hour = timestamp.hour
        else:
            return 'unknown'

        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    except Exception as e:
        print(f"Error parsing timestamp {timestamp}: {str(e)}")
        return 'unknown'

def get_recommendations_for_time(filtered_data, cluster_terms, genres, current_time):
    """Generate recommendations based on time-based patterns."""
    try:
        # Group similar content
        categorized_results, _ = categorize_search_terms(filtered_data, num_categories=3)
        
        # Get all unique clusters
        unique_clusters = categorized_results['cluster'].unique()
        
        recommendations = {}
        for cluster_id in unique_clusters:
            # Get the genre and terms for this cluster
            genre = genres.get(int(cluster_id), "Unknown")
            terms = cluster_terms.get(int(cluster_id), [])
            
            # Clean and enhance search query
            if terms:
                # Filter out potentially problematic terms
                clean_terms = [term for term in terms if len(term) > 2 and term.lower() not in ['the', 'and', 'or', 'but']]
                
                # Add genre to search if available and not "Unknown"
                if genre and genre != "Unknown":
                    search_query = f"{genre} {' '.join(clean_terms[:3])}"
                else:
                    search_query = ' '.join(clean_terms[:3])
                
                # Add "recommendations" to make results more relevant
                search_query = f"{search_query} recommendations"
                
                try:
                    # Get recommendations using Google Custom Search
                    search_results = scrape_content(
                        search_query,
                        num_results=5
                    )
                    
                    # Only add to recommendations if we got valid results
                    if search_results and not all(result[0] == "Unable to fetch results at the moment. Please try again later." for result in search_results):
                        recommendations[int(cluster_id)] = {
                            'genre': genre,
                            'terms': clean_terms[:5],  # Show up to 5 clean terms
                            'results': search_results,
                            'search_query': search_query
                        }
                except Exception as e:
                    print(f"Error fetching results for cluster {cluster_id}: {str(e)}")
                    continue
        
        return recommendations
    except Exception as e:
        print(f"Error generating recommendations: {str(e)}")
        return {}

@app.route('/load_more_time_recommendations', methods=['GET'])
def load_more_time_recommendations():
    cluster_id = request.args.get('cluster_id')
    search_query = request.args.get('search_query')
    page = int(request.args.get('page', 1))
    
    if not cluster_id or not search_query:
        return jsonify({'error': 'Missing parameters'}), 400

    try:
        # Get more results using the same search query
        start_index = (page - 1) * 5 + 1  # 5 results per page
        results = scrape_content(
            search_query,
            num_results=5,
            start=start_index
        )
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/time_based_filter', methods=['GET'])
def time_based_filter():
    youtube = authenticate_youtube()
    reddit_data = fetch_reddit_history(limit=20)
    youtube_liked_data = fetch_youtube_liked_videos(youtube, max_results=10)
    combined_data = reddit_data + youtube_liked_data

    # Get current time of day
    current_time = get_time_of_day(datetime.now())

    # First try to get time-specific data
    filtered_data = []
    for item in combined_data:
        if 'timestamp' in item:
            try:
                if isinstance(item['timestamp'], str):
                    time_category = get_time_of_day(item['timestamp'])
                else:
                    time_category = get_time_of_day(datetime.now())
                
                if time_category == current_time:
                    filtered_data.append(item)
            except Exception as e:
                print(f"Error processing timestamp: {str(e)}")

    # Cold start handling: If no time-specific data, use different strategies
    if not filtered_data:
        print("Cold start detected: No data for current time period")
        
        # Strategy 1: Use recent interactions regardless of time
        if combined_data:
            filtered_data = combined_data[:5]
            print("Using recent interactions as fallback")
        
        # Strategy 2: Use popular categories and trending content
        default_recommendations = get_default_recommendations(current_time)
        
        return render_template(
            'time_based_visualization.html',
            cluster_distribution={},
            genres={},
            cluster_terms={},
            current_time=current_time,
            filtered_content=[],
            results=default_recommendations,
            is_cold_start=True
        )

    # Normal flow if we have time-specific data
    try:
        categorized_results, cluster_terms = categorize_search_terms(filtered_data, num_categories=3)
        
        if not isinstance(cluster_terms, dict):
            cluster_terms = dict(enumerate(cluster_terms))
        
        cluster_distribution = categorized_results['cluster'].value_counts().to_dict()
        genres = detect_genres_bert(cluster_terms)
        results = get_recommendations_for_time(filtered_data, cluster_terms, genres, current_time)

        # If clustering didn't produce good results, fall back to default recommendations
        if not results:
            results = get_default_recommendations(current_time)

        # Organize content by clusters
        filtered_content = []
        unique_clusters = sorted(categorized_results['cluster'].unique())
        
        for cluster_id in unique_clusters:
            cluster_items = []
            cluster_mask = categorized_results['cluster'] == cluster_id
            cluster_indices = cluster_mask[cluster_mask].index
            
            for idx in cluster_indices:
                item = filtered_data[idx]
                content_item = {
                    'title': item.get('title', 'Untitled'),
                    'url': item.get('url', '#'),
                    'content': item.get('content', ''),
                    'type': item.get('type', 'Unknown'),
                    'subreddit': item.get('subreddit', '') if 'subreddit' in item else None
                }
                cluster_items.append(content_item)
            
            if cluster_items:
                filtered_content.append({
                    'cluster_id': int(cluster_id),
                    'content_list': cluster_items
                })

    except Exception as e:
        print(f"Error processing data: {str(e)}")
        results = get_default_recommendations(current_time)
        cluster_distribution = {}
        cluster_terms = {}
        genres = {}
        filtered_content = []

    return render_template(
        'time_based_visualization.html',
        cluster_distribution=cluster_distribution,
        genres=genres,
        cluster_terms=cluster_terms,
        current_time=current_time,
        filtered_content=filtered_content,
        results=results,
        is_cold_start=False
    )

def get_default_recommendations(time_period):
    """Generate default recommendations for cold start situations."""
    time_based_genres = {
        'morning': [
            ("Technology", "Latest tech news and productivity tools"),
            ("Education", "Online courses and learning resources"),
            ("Health", "Morning workout and wellness tips")
        ],
        'afternoon': [
            ("Entertainment", "Popular media and trending content"),
            ("Technology", "Tech reviews and updates"),
            ("Business", "Industry news and professional development")
        ],
        'evening': [
            ("Entertainment", "Movies and TV shows"),
            ("Gaming", "Popular games and gaming news"),
            ("Lifestyle", "Evening activities and relaxation")
        ],
        'night': [
            ("Entertainment", "Relaxing content and media"),
            ("Education", "Night learning and reading"),
            ("Gaming", "Gaming communities and discussions")
        ]
    }

    recommendations = {}
    for i, (genre, description) in enumerate(time_based_genres.get(time_period, time_based_genres['afternoon'])):
        search_query = f"{genre} {description} recommendations"
        search_results = scrape_content(
            search_query,
            num_results=5
        )
        
        if search_results and not all(result[0] == "Unable to fetch results at the moment. Please try again later." for result in search_results):
            recommendations[i] = {
                'genre': genre,
                'terms': [description.lower(), genre.lower(), 'recommended'],
                'results': search_results,
                'search_query': search_query
            }
    
    return recommendations

@app.route('/new_search_results', methods=['GET'])
def new_search_results():
    search_term = request.args.get('search_term', '')
    page = int(request.args.get('page', 1))

    # Google Custom Search API allows `start` to fetch paginated results
    start_index = (page - 1) * 10 + 1  # Adjust based on API requirements
    results = scrape_content(
        search_term, 
        num_results=10, 
        start=start_index
    )

    return jsonify({'results': results})

def get_default_recommendations_by_interests(interests):
    """Generate default recommendations based on user's selected interests"""
    recommendations = []
    
    # Define some default content categories
    default_categories = {
        "Technology": ["Programming tutorials", "Tech news", "Gadget reviews"],
        "Gaming": ["Game reviews", "Gaming tips", "Esports news"],
        "Entertainment": ["Movie reviews", "TV show recommendations", "Music playlists"],
        "Education": ["Online courses", "Study tips", "Educational resources"],
        "Sports": ["Sports news", "Workout tips", "Game highlights"],
        "Business": ["Business news", "Entrepreneurship tips", "Career advice"],
        "Art": ["Art tutorials", "Creative inspiration", "Design tips"],
        "Science": ["Science news", "Scientific discoveries", "Research updates"],
        "Health": ["Health tips", "Wellness advice", "Fitness guides"],
        "Travel": ["Travel guides", "Destination reviews", "Travel tips"]
    }
    
    # Generate recommendations based on selected interests
    for interest in interests:
        if interest in default_categories:
            for topic in default_categories[interest]:
                search_results = scrape_content(
                    f"{interest} {topic}",
                    num_results=3
                )
                for title, link in search_results:
                    recommendations.append({
                        'interest': interest,
                        'title': title,
                        'link': link
                    })
    
    return recommendations

@app.route('/check_auth')
def check_auth():
    """Check if both Reddit and YouTube are authenticated and valid"""
    try:
        is_reddit_auth = 'reddit_credentials' in session
        is_youtube_auth = 'youtube_client_secret' in session
        
        # Verify the credentials are still valid
        if is_reddit_auth and is_youtube_auth:
            reddit = init_reddit()
            youtube = authenticate_youtube()
            
            if not reddit or not youtube:
                # Clear invalid credentials
                session.pop('reddit_credentials', None)
                session.pop('youtube_client_secret', None)
                is_reddit_auth = False
                is_youtube_auth = False
        
        # Check if files should be deleted
        if 'file_deletion_time' in session:
            if datetime.now().timestamp() > session['file_deletion_time']:
                cleanup_temp_files()
                session.pop('youtube_client_secret', None)
                session.pop('file_deletion_time', None)
                is_youtube_auth = False
        
        return jsonify({
            'reddit_authenticated': is_reddit_auth,
            'youtube_authenticated': is_youtube_auth
        })
    except Exception as e:
        print(f"Error checking auth status: {str(e)}")
        return jsonify({
            'reddit_authenticated': False,
            'youtube_authenticated': False
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

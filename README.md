# Social Media Content Recommendation System

A Flask-based web application that provides personalized content recommendations based on your Reddit and YouTube activity.

## Features

- Reddit and YouTube integration
- Content clustering and categorization
- Time-based recommendations
- Interest-based recommendations
- Secure credential management
- Modern, responsive UI

## Prerequisites

- Python 3.8 or higher
- Reddit API credentials (client ID and secret)
- YouTube API credentials (client_secret.json file)
- Google Custom Search API credentials
- A server with HTTPS support (for production)

## Local Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory:
```env
FLASK_SECRET_KEY=your_secret_key_here
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_cse_id
```

5. Run the development server:
```bash
python app_main.py
```


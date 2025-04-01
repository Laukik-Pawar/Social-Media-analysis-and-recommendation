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

## Production Deployment

### Option 1: Deploy to Heroku

1. Install Heroku CLI and login:
```bash
heroku login
```

2. Create a new Heroku app:
```bash
heroku create your-app-name
```

3. Add buildpacks:
```bash
heroku buildpacks:add heroku/python
```

4. Configure environment variables:
```bash
heroku config:set FLASK_SECRET_KEY=your_secret_key_here
heroku config:set GOOGLE_API_KEY=your_google_api_key
heroku config:set GOOGLE_CSE_ID=your_google_cse_id
```

5. Deploy:
```bash
git push heroku main
```

### Option 2: Deploy to a Linux Server

1. Install required packages:
```bash
sudo apt-get update
sudo apt-get install python3-pip python3-venv nginx
```

2. Clone the repository and set up the environment:
```bash
git clone <repository-url>
cd <repository-name>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn
```

3. Create a systemd service file `/etc/systemd/system/recommendation-app.service`:
```ini
[Unit]
Description=Recommendation App
After=network.target

[Service]
User=your_user
WorkingDirectory=/path/to/your/app
Environment="PATH=/path/to/your/app/venv/bin"
ExecStart=/path/to/your/app/venv/bin/gunicorn -w 4 -b 127.0.0.1:8000 app_main:app

[Install]
WantedBy=multi-user.target
```

4. Configure Nginx `/etc/nginx/sites-available/recommendation-app`:
```nginx
server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

5. Enable and start services:
```bash
sudo ln -s /etc/nginx/sites-available/recommendation-app /etc/nginx/sites-enabled
sudo systemctl start recommendation-app
sudo systemctl enable recommendation-app
sudo systemctl restart nginx
```

### Option 3: Deploy using Docker

1. Create a Dockerfile:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=app_main.py
ENV FLASK_ENV=production

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app_main:app"]
```

2. Build and run the Docker container:
```bash
docker build -t recommendation-app .
docker run -d -p 5000:5000 --env-file .env recommendation-app
```

## Security Considerations

1. Always use HTTPS in production
2. Set secure session cookies
3. Implement rate limiting
4. Regularly update dependencies
5. Monitor server logs
6. Back up data regularly
7. Use environment variables for sensitive data

## Maintenance

1. Monitor application logs:
```bash
heroku logs --tail  # For Heroku
journalctl -u recommendation-app  # For Linux server
docker logs container_id  # For Docker
```

2. Update dependencies regularly:
```bash
pip install -r requirements.txt --upgrade
```

3. Clean up temporary files:
- The application automatically cleans up temporary files older than 1 hour
- Monitor disk space usage regularly

## Troubleshooting

1. If the application fails to start:
- Check all environment variables are set
- Verify API credentials are valid
- Check server logs for errors

2. If authentication fails:
- Verify Reddit API credentials
- Check YouTube API quota limits
- Ensure client_secret.json is properly formatted

3. If recommendations are not showing:
- Check API rate limits
- Verify user has sufficient activity history
- Check clustering algorithm parameters

## Support

For issues and support, please:
1. Check the troubleshooting guide
2. Review server logs
3. Open an issue in the repository
4. Contact the development team 
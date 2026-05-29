# 🎵 Music Lyrics & Theme Analyzer Bot

An intelligent tool that fetches song lyrics and provides deep thematic analysis using AI. Choose between a sleek **Web Interface** or a classic **CLI Terminal** experience.

---

## ✨ Features

- **Lyric Fetching**: Automatically retrieves clean, annotation-free lyrics from the Genius API.
- **AI Analysis**: Powered by NVIDIA NIM (Llama 3.1), providing structured insights into:
  - Overall Mood & Vibe
  - Top 3 Core Themes
  - Meaningful Summary
- **Dual Mode**:
  - **Web UI**: Modern, responsive dark-mode dashboard.
  - **CLI**: Fast, terminal-based interaction.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+
- A [Genius API Token](https://genius.com/api-clients)
- An [NVIDIA NIM API Key](https://build.nvidia.com) (Free tier available)

### 2. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd music-theme-bot

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory and add your keys:
```env
GENIUS_API_TOKEN=your_genius_token_here
NIM_API_KEY=your_nvidia_nim_key_here
```

---

## 🎮 Usage

### Option A: Web Interface (Recommended)
Launch the web server and open the UI in your browser:
```bash
python web_server.py
```
📍 Visit: **[http://localhost:8000](http://localhost:8000)**

### Option B: CLI Mode
Interact directly with the bot via your terminal:
```bash
python main.py
```

---

## 📁 Project Structure

- `main.py`: Entry point for the CLI application.
- `web_server.py`: FastAPI backend serving the Web UI.
- `static/`: Contains the frontend HTML/CSS/JS.
- `services/`: Core logic for Genius API and AI Analysis.
- `requirements.txt`: Python package dependencies.
- `GEMINI.md`: Deep technical documentation for developers.

---

## 🛠 Tech Stack

- **Backend**: Python, FastAPI, Uvicorn
- **Frontend**: Vanilla HTML5/CSS3/JS, Marked.js
- **APIs**: Genius API, NVIDIA NIM (OpenAI-compatible)

---

## ⚖️ License
MIT License. Feel free to use and modify!

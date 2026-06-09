GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* Global Font Override */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0f111a;
    color: #e2e8f0;
}

/* Beautiful Animated Gradient Header */
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
}

.main-title {
    font-size: 3rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #4f46e5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.05em;
    animation: pulse 6s infinite alternate;
}

.sub-title {
    font-size: 1.1rem !important;
    color: #94a3b8 !important;
    margin-bottom: 2rem !important;
    font-weight: 400 !important;
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(30, 41, 59, 0.45);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
}

.glass-card:hover {
    transform: translateY(-5px);
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.15);
}

/* Micro-animations and buttons */
div.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: white !important;
    border: none !important;
    padding: 10px 24px !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
}

div.stButton > button:hover {
    transform: scale(1.03) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
    background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%) !important;
}

div.stButton > button:active {
    transform: scale(0.98) !important;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #0f111a;
}
::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #4f46e5;
}

/* Tab styling overrides */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 2px solid rgba(255, 255, 255, 0.05) !important;
    gap: 8px !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    background-color: transparent !important;
    border-radius: 8px 8px 0px 0px !important;
    padding: 12px 20px !important;
    transition: all 0.3s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #e2e8f0 !important;
    background-color: rgba(255, 255, 255, 0.03) !important;
}

.stTabs [aria-selected="true"] {
    color: #6366f1 !important;
    border-bottom: 3px solid #6366f1 !important;
}

/* Interactive elements */
.source-tag {
    display: inline-block;
    padding: 2px 8px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #a5b4fc;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 6px;
}

/* Custom styles for Flashcards */
.flashcard-container {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 20px;
    padding: 40px 30px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    min-height: 250px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    transition: all 0.5s ease;
}

.flashcard-topic {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #818cf8;
    margin-bottom: 15px;
}

.flashcard-content {
    font-size: 1.4rem;
    font-weight: 600;
    color: #f1f5f9;
    line-height: 1.4;
    margin-bottom: 25px;
}

.flashcard-hint {
    font-size: 0.9rem;
    font-style: italic;
    color: #94a3b8;
}

/* Styled lists in Quiz */
.quiz-option {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
}

.quiz-option:hover {
    background: rgba(99, 102, 241, 0.1);
    border-color: rgba(99, 102, 241, 0.3);
    transform: translateX(5px);
}

@keyframes pulse {
    0% { filter: hue-rotate(0deg); }
    100% { filter: hue-rotate(15deg); }
}
</style>
"""

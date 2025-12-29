import os
import json
import logging
import datetime

logger = logging.getLogger(__name__)

# Define file paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
ACHIEVEMENTS_PATH = os.path.join(DATA_DIR, 'achievements.json')
EXERCISES_PATH = os.path.join(DATA_DIR, 'exercises.json')
JOURNEYS_PATH = os.path.join(DATA_DIR, 'journeys.json')
PROGRESS_PATH = os.path.join(DATA_DIR, 'progress.json')
USERS_PATH = os.path.join(DATA_DIR, 'users.json')
INTERACTION_LOG_PATH = os.path.join(DATA_DIR, 'interaction_log.json')
RESPONSES_PATH = os.path.join(DATA_DIR, 'responses.json')
HOPE_JAR_PATH = os.path.join(DATA_DIR, 'hope_jar.json')


def load_json_file(path, default_value):
    """Helper function to read a JSON file safely."""
    if not os.path.exists(path):
        return default_value
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except (json.JSONDecodeError, Exception):
        logger.exception('Failed to load or parse JSON file: %s', path)
        return default_value

def save_json_file(path, data):
    """Helper function to write data to a JSON file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception('Failed to save JSON file: %s', path)

def load_responses():
    """Load responses from the JSON file."""
    return load_json_file(RESPONSES_PATH, {})

def load_exercises_file():
    return load_json_file(EXERCISES_PATH, [])

def load_users():
    return load_json_file(USERS_PATH, {})

def save_users(users):
    save_json_file(USERS_PATH, users)

def save_interaction(user_input, response_text, sentiment_label):
    """Saves an interaction to the log file."""
    entry = {
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'input': user_input,
        'result': response_text,
        'sentiment_label': sentiment_label
    }
    log_data = load_json_file(INTERACTION_LOG_PATH, [])
    log_data.append(entry)
    save_json_file(INTERACTION_LOG_PATH, log_data)

def track_progress(record):
    """Saves a progress record."""
    progress_data = load_json_file(PROGRESS_PATH, [])
    progress_data.append(record)
    save_json_file(PROGRESS_PATH, progress_data)

def award_badges_for_user(user):
    """Simple badge awarding based on streak length."""
    badges = set(user.get('badges', []))
    streak = int(user.get('streak', 0))
    if streak >= 3:
        badges.add('3-day-streak')
    if streak >= 7:
        badges.add('7-day-streak')
    if streak >= 30:
        badges.add('30-day-streak')
    user['badges'] = sorted(list(badges))
    return user

def daily_checkin(user):
    """Handles the daily check-in logic."""
    today = datetime.datetime.utcnow().date()
    last = None
    if user.get('last_checkin'):
        try:
            last = datetime.datetime.fromisoformat(user.get('last_checkin')).date()
        except Exception:
            last = None

    if last == today:
        return user, 'already_checked_in'

    if last == (today - datetime.timedelta(days=1)):
        user['streak'] = int(user.get('streak', 0)) + 1
    else:
        user['streak'] = 1
    
    user['last_checkin'] = datetime.datetime.utcnow().isoformat()
    return user, 'checked_in'

def add_hope_note(user_id, content):
    """Adds a note to the user's Hope Jar."""
    data = load_json_file(HOPE_JAR_PATH, {})
    if user_id not in data:
        data[user_id] = []
    
    note = {
        'id': str(datetime.datetime.utcnow().timestamp()),
        'content': content,
        'date': datetime.datetime.utcnow().isoformat()
    }
    data[user_id].append(note)
    save_json_file(HOPE_JAR_PATH, data)
    return note

def get_random_hope_note(user_id):
    """Retrieves a random note from the jar."""
    data = load_json_file(HOPE_JAR_PATH, {})
    user_notes = data.get(user_id, [])
    
    # Default notes if empty
    if not user_notes:
        return {
            'content': "تذكر أنك أقوى مما تعتقد. 🌟",
            'date': datetime.datetime.utcnow().isoformat(),
            'is_default': True
        }
    
    import random
    return random.choice(user_notes)

def complete_quest(user, quest_id):
    """Handles quest completion logic."""
    progress = user.get('progress', {})
    if quest_id in progress:
        return user # Already completed

    progress[quest_id] = True
    user['progress'] = progress

    journeys = load_json_file(JOURNEYS_PATH, {})
    quest_xp = 0
    completed_journey_id = None

    for journey_data in journeys.values():
        for quest in journey_data.get('quests', []):
            if quest['id'] == quest_id:
                quest_xp = quest.get('xp', 0)
                all_quests_in_journey = {q['id'] for q in journey_data.get('quests', [])}
                if all_quests_in_journey.issubset(set(user['progress'].keys())):
                    completed_journey_id = journey_data.get('id')
                break
    
    user['xp'] = user.get('xp', 0) + quest_xp

    if completed_journey_id:
        user_achievements = set(user.get('achievements', []))
        if completed_journey_id not in user_achievements:
            user_achievements.add(completed_journey_id)
            user['achievements'] = sorted(list(user_achievements))

    return user

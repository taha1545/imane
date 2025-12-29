from flask import Blueprint, render_template, request, jsonify
from . import services
from . import ml
from . import utils
import uuid
import datetime
import random

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint to receive user input and return a response.
    """
    try:
        data = request.get_json(force=True)
        user_input = data.get('message', '')
        context = data.get('context', []) 
        active_topic = data.get('active_topic', None)
        
        if not user_input:
            return jsonify({'response': "يرجى إدخال رسالة.", 'sentiment_label': 'محايد', 'active_topic': active_topic})

        # 1. Analyze sentiment and risk
        sentiment, probabilities, risk_level = ml.analyzer.analyze_sentiment(user_input, context)

        # 2. Generate a response
        response_text, new_topic = ml.analyzer.generate_response(sentiment, risk_level, user_input=user_input, conversation_context=context, active_topic=active_topic)
        
        # 3. Save the interaction
        services.save_interaction(user_input, response_text, sentiment)
        
        return jsonify({
            'response': response_text,
            'sentiment_label': sentiment,
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'risk_level': risk_level,
            'active_topic': new_topic,
            'disclaimer': ml.APP_DISCLAIMER
        })
        
    except Exception as e:
        ml.logger.exception('Failed during chat processing')
        return jsonify({'response': "حدث خطأ غير متوقع. يرجى المحاولة لاحقاً.", 'sentiment_label': 'محايد', 'error': str(e)}), 500

@bp.route('/save_interaction', methods=['POST'])
def save_interaction_route():
     try:
         payload = request.get_json()
         if not payload:
             return {'status': 'error', 'error': 'Invalid JSON payload'}, 400
         services.save_interaction(
             payload.get('input', ''),
             payload.get('result', ''),
             payload.get('sentiment_label')
         )
         return {'status': 'ok'}
     except Exception as e:
         return {'status': 'error', 'error': str(e)}, 500

@bp.route('/exercises', methods=['GET'])
def exercises_route():
    ex = services.load_exercises_file()
    return jsonify({'exercises': ex})

@bp.route('/user/create', methods=['POST'])
def create_user():
    try:
        payload = request.get_json(force=True)
        username = (payload.get('username') or '').strip()
        if not username:
            return jsonify({'status': 'error', 'error': 'username required'}), 400
        users = services.load_users()
        user_id = str(uuid.uuid4())
        users[user_id] = {'username': username, 'created': datetime.datetime.utcnow().isoformat() + 'Z'}
        users[user_id].update({'streak': 0, 'last_checkin': None, 'badges': []})
        services.save_users(users)
        return jsonify({'status': 'ok', 'user_id': user_id, 'username': username})
    except Exception as e:
        ml.logger.exception('Failed to create user')
        return jsonify({'status': 'error', 'error': str(e)}), 500

@bp.route('/progress', methods=['GET'])
def get_progress():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'error': 'user_id required'}), 400
    try:
        arr = services.load_json_file(services.PROGRESS_PATH, [])
        user_records = [r for r in arr if r.get('user_id') == user_id]
        return jsonify({'status': 'ok', 'records': user_records})
    except Exception as e:
        ml.logger.exception('Failed to load exercises')
        return jsonify({'status': 'error', 'error': str(e)}), 500

@bp.route('/track_progress', methods=['POST'])
def track_progress():
    try:
        payload = request.get_json(force=True)
        user_id = payload.get('user_id')
        action = payload.get('action') # e.g., 'completed_exercise', 'check_in'
        
        if not user_id or not action:
             return jsonify({'status': 'error', 'error': 'Missing data'}), 400
             
        # Log to progress file
        data = services.load_json_file(services.PROGRESS_PATH, [])
        record = {
            'user_id': user_id,
            'action': action,
            'details': payload.get('details', {}),
            'timestamp': datetime.datetime.utcnow().isoformat()
        }
        data.append(record)
        services.save_json_file(services.PROGRESS_PATH, data)
        
        # Update User Streak/Badges (Simplified logic)
        users = services.load_users()
        if user_id in users:
            # Logic to update streak would go here
            services.save_users(users)
            
        return jsonify({'status': 'ok', 'record': record})
    except Exception as e:
        ml.logger.exception('Failed to track progress')
        return jsonify({'status': 'error', 'error': str(e)}), 500

@bp.route('/transmute', methods=['POST'])
def transmute_worry():
    """
    Takes a 'worry' text and returns a 'Phoenix Insight' (positive reframe).
    """
    try:
        payload = request.get_json(force=True)
        worry_text = payload.get('worry', '')
        user_id = payload.get('user_id')
        
        if not worry_text:
            return jsonify({'insight': "حتى الصمت يمكن حرقه. ابدأ من جديد."})

        # Generate Insight
        insight = ml.analyzer.generate_phoenix_insight(worry_text)
        
        # Optionally save this 'release' event (without saving the worry content for privacy if desired)
        # services.log_release(user_id) 
        
        return jsonify({'status': 'ok', 'insight': insight})
    except Exception as e:
        ml.logger.exception('Transmutation failed')
        return jsonify({'status': 'error', 'error': str(e)}), 500

@bp.route('/hope/add', methods=['POST'])
def add_hope_note():
    try:
        payload = request.get_json(force=True)
        user_id = payload.get('user_id')
        content = payload.get('content')
        
        if not user_id or not content:
            return jsonify({'status': 'error', 'error': 'Missing data'}), 400
            
        note = services.add_hope_note(user_id, content)
        return jsonify({'status': 'ok', 'note': note})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@bp.route('/hope/shake', methods=['GET'])
def shake_hope_jar():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'error': 'user_id required'}), 400
        
    note = services.get_random_hope_note(user_id)
    return jsonify({'status': 'ok', 'note': note})
    try:
        payload = request.get_json(force=True)
        record = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            'user_id': payload.get('user_id'),
            'exercise_id': payload.get('exercise_id'),
            'completed': bool(payload.get('completed', False)),
            'note': payload.get('note', '')
        }
        if not record.get('user_id'):
            return jsonify({'status': 'error', 'error': 'user_id required'}), 400
        
        services.track_progress(record)
        return jsonify({'status': 'ok', 'record': record})
    except Exception as e:
        ml.logger.exception('Failed to track progress')
        return jsonify({'status': 'error', 'error': str(e)}), 500

@bp.route('/daily_quote', methods=['GET'])
def daily_quote():
    q = random.choice(ml.MOTIVATIONAL_QUOTES) if ml.MOTIVATIONAL_QUOTES else "أنت تستحق لحظة لطف مع نفسك اليوم."
    return jsonify({'quote': q})

@bp.route('/resources', methods=['GET'])
def get_resources():
    return jsonify(ml.YOUTH_RESOURCES)

@bp.route('/journey', methods=['GET'])
def get_journey_data():
    journey_data = services.load_json_file(services.JOURNEYS_PATH, {})
    return jsonify(journey_data)

@bp.route('/user/data', methods=['GET'])
def get_user_data():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'error': 'user_id required'}), 400
    
    users = services.load_users()
    user = users.get(user_id)
    if not user:
        return jsonify({'status': 'error', 'error': 'user not found'}), 404
        
    achievements_def = services.load_json_file(services.ACHIEVEMENTS_PATH, {})
    user['achievements_details'] = [achievements_def[ach_id] for ach_id in user.get('achievements', []) if ach_id in achievements_def]

    return jsonify({'status': 'ok', 'user': user})

@bp.route('/quest/complete', methods=['POST'])
def complete_quest_route():
    try:
        payload = request.get_json(force=True)
        user_id = payload.get('user_id')
        quest_id = payload.get('quest_id')

        if not user_id or not quest_id:
            return jsonify({'status': 'error', 'error': 'user_id and quest_id are required'}), 400

        users = services.load_users()
        user = users.get(user_id)
        if not user:
            return jsonify({'status': 'error', 'error': 'user not found'}), 404

        user = services.complete_quest(user, quest_id)
        
        services.save_users(users)
        
        # This part needs to be adjusted in the services
        new_achievement = None 
        
        return jsonify({'status': 'ok', 'user': user, 'new_achievement': new_achievement})

    except Exception as e:
        ml.logger.exception('Failed to complete quest')
        return jsonify({'status': 'error', 'error': str(e)}), 500

@bp.route('/daily_checkin', methods=['POST'])
def daily_checkin():
    try:
        payload = request.get_json(force=True)
        user_id = payload.get('user_id')
        if not user_id:
            return jsonify({'status': 'error', 'error': 'user_id required'}), 400
        
        users = services.load_users()
        user = users.get(user_id)
        if not user:
            return jsonify({'status': 'error', 'error': 'user not found'}), 404

        user, message = services.daily_checkin(user)
        
        users[user_id] = user
        services.save_users(users)

        return jsonify({'status': 'ok', 'message': message, 'streak': user['streak'], 'badges': user.get('badges', [])})
    except Exception as e:
        ml.logger.exception('daily_checkin failed')
        return jsonify({'status': 'error', 'error': str(e)}), 500
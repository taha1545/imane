import logging
import re
import difflib
import random
import importlib
import numpy as np
from . import utils
from . import services

# --- Conditional Imports for ML & Audio ---
librosa = None
sf = None
try:
    import librosa
    import soundfile as sf
except ImportError:
    pass

TORCH_AVAILABLE = False
torch = None
try:
    if importlib.util.find_spec("torch") is not None:
        torch = importlib.import_module("torch")
        TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

AutoModelForSequenceClassification = None
AutoTokenizer = None
try:
    if importlib.util.find_spec("transformers") is not None and TORCH_AVAILABLE:
        mod = importlib.import_module("transformers")
        AutoModelForSequenceClassification = getattr(mod, "AutoModelForSequenceClassification", None)
        AutoTokenizer = getattr(mod, "AutoTokenizer", None)
except ImportError:
    pass

# --- Global Config and Data ---
logger = logging.getLogger(__name__)

if TORCH_AVAILABLE and torch is not None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

# Load data from services
responses_data = services.load_responses()
GREETING_RESPONSES = responses_data.get("greetings", {})
INTRO_RESPONSES = responses_data.get("intro", {})
EMOTION_RESPONSES = responses_data.get("emotions", {})
SPECIAL_RESPONSES = responses_data.get("special", {})
MOTIVATIONAL_QUOTES = responses_data.get("motivational_quotes", [])
PHRASE_BANK = responses_data.get("phrase_bank", {})
YOUTH_RESOURCES = responses_data.get("youth_resources", {})
THERAPEUTIC_INTERVENTIONS = responses_data.get("therapeutic_interventions", {})

# Constants from original file
CRISIS_KEYWORDS = [
    "انتحر", "أريد أن أنتحر", "أُنهِي حياتي", "قتل نفسي", "suicide", "end my life", "موت", "اموت"
]
APP_DISCLAIMER = (
     "ملاحظة: هذا التطبيق ليس بديلاً عن العلاج النفسي المحترف. إذا كنت تفكر في إيذاء نفسك، "
     "اتصل بالأرقام الطارئة أو بخط الدعم النفسي المحلي فوراً. يمكنني تقديم دعم لحظي وإجراءات تهدئة قصيرة."
)
CRISIS_RESPONSE_AR = "إذا كنت تفكر في إيذاء نفسك، اتصل بالأرقام الطارئة أو بخط الدعم فوراً. حياتك غالية ومهمة."
EMOTION_POLARITY = {
    "سعيد": 0.8, "مبسوط": 0.9, "فرحان": 1.0, "متفائل": 0.7, "سعيدة": 0.8, "مبسوطة": 0.9, "فرحانة": 1.0,
    "حزين": -0.8, "مكتئب": -0.9, "متضايق": -0.4, "محبط": -0.7, "يأس": -1.0, "حزينة": -0.8, "مكتئبة": -0.9, "متضايقة": -0.4,
    "غاضب": -0.8, "عصبي": -0.7, "مستفز": -0.6, "غاضبة": -0.8, "عصبية": -0.7,
    "قلق": -0.6, "متوتر": -0.7, "خايف": -0.5, "خائف": -0.5, "قلقة": -0.6, "متوترة": -0.7, "خائفة": -0.5,
    "وحدة": -0.8, "وحيد": -0.8, "وحيدة": -0.8,
    "تعبان": -0.4, "مرهق": -0.5, "تعبانة": -0.4, "مرهقة": -0.5,
    "رائع": 0.9, "ممتاز": 1.0, "جميل": 0.8, "مذهل": 0.9,
    "سيء": -0.7, "فاشل": -0.9, "مشكلة": -0.5, "صعب": -0.4,
}

# Topics and Keywords
TOPIC_KEYWORDS = {
    "دراسة": ["دراسة", "مدرسة", "جامعة", "امتحان", "اختبار", "مذاكرة", "درجات", "معلم", "دكتور"],
    "عمل": ["عمل", "شغل", "وظيفة", "مدير", "راتب", "دوام", "زملاء", "ترقية", "مشروع"],
    "علاقات": ["حب", "زواج", "خطوبة", "حبيبي", "حبيبتي", "زوجي", "زوجتي", "صديق", "صاحب", "خيانة", "فراق"],
    "عائلة": ["أمي", "أبي", "أخ", "أخت", "عائلتي", "أهل", "بيت"],
    "نفس": ["شكلي", "وزني", "جسمي", "شخصيتي", "ثقة", "نفسي"]
}

TOPIC_OVERRIDE_KEYWORDS = {
    "work_stress": ["كره العمل", "اكره وظيفتي", "مديري", "شغل", "دوام", "استقالة", "عمل", "وظيفة", "زملاء العمل", "ضغط عمل", "تعبت من الشغل", "كرهت الشغل"],
    "divorce": ["طلاق", "انفصال", "زوجي السابق", "طليقي", "طليقتي", "محاكم", "خلع", "نفقة", "حضانة"],
    "social_issues": ["مجتمع", "نفاق", "ناس", "علاقات اجتماعية", "وحدة وسط الناس", "كذب", "مجاملات", "نفاق اجتماعي", "الناس تغيرت"]
}

TOPIC_RESPONSES = {
    "دراسة": [
        "الدراسة قد تكون مصدراً كبيراً للضغط. تذكر أن درجاتك لا تحدد قيمتك كإنسان.",
        "من الطبيعي أن تشعر بالتوتر تجاه الدراسة. هل حاولت تقسيم المهام إلى أجزاء صغيرة؟",
        "التعليم رحلة طويلة، والمهم هو ما تتعلمه وليس فقط النتيجة النهائية."
    ],
    "عمل": [
        "بيئة العمل قد تكون مليئة بالتحديات. صحتك النفسية أهم من أي وظيفة.",
        "التوازن بين العمل والحياة الشخصية أمر حيوي. هل تأخذ وقتاً كافياً لنفسك؟",
        "مشاكل العمل مؤقتة، لكن مهاراتك وقدرتك على التجاوز هي الباقية."
    ],
    "علاقات": [
        "العلاقات الإنسانية معقدة وتؤثر فينا بعمق.",
        "من حقك أن تكون في علاقات تمنحك الاحترام والتقدير.",
        "الألم الناتج عن العلاقات يأخذ وقتاً للشفاء، كن صبوراً مع نفسك."
    ],
    "عائلة": [
        "العائلة هي الجذور، وأحياناً تكون هي مصدر الألم. هذا شعور معقد جداً.",
        "التعامل مع الأهل قد يحتاج إلى حدود صحية للحفاظ على سلامك النفسي.",
        "لا بأس أن تختلف مع عائلتك، هذا جزء من استقلالك الشخصي."
    ],
    "نفس": [
        "أنت كافٍ كما أنت. لا تدع أي معايير خارجية تهز ثقتك بنفسك.",
        "علاقتك بنفسك هي أهم علاقة في حياتك. كن لطيفاً معها.",
        "كلنا لدينا شكوك، لكن هذا لا يقلل من روعتك."
    ]
}

class AdvancedSentimentAnalyzer:
    def __init__(self, model_name="aubmindlab/bert-base-arabertv2", max_context_turns=5):
        self.tokenizer = None
        self.model = None
        self.max_context_turns = max_context_turns
        self.labels = {
            0: "حزين",  # Mapped from "حزن/اكتئاب"
            1: "قلق",   # Mapped from "قلق/توتر"
            2: "محايد/أخرى", # "تعلق/حب تملك" -> Generic for now
            3: "محايد/أخرى", # "قلة ثقة بالنفس" -> Generic for now
            4: "محايد/أخرى",
        }

        if not model_name:
            logger.info("No model_name provided — using lexicon fallback only.")
            return

        if not AutoTokenizer or not AutoModelForSequenceClassification:
            logger.warning("transformers or required classes unavailable — using lexicon fallback only.")
            return

        logger.info("Loading model %s on %s...", model_name, device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
            if TORCH_AVAILABLE and torch is not None and hasattr(self.model, 'to'):
                self.model = self.model.to(device)
        except Exception:
            logger.exception("Failed to load ML model; continuing with fallback lexicon approach.")
            self.tokenizer = None
            self.model = None
            return

    def generate_phoenix_insight(self, worry_text):
        """
        Generates a 'Phoenix Insight' - a reframing of a worry into wisdom.
        Uses rule-based patterns or falls back to general wisdom.
        """
        worry_lower = worry_text.lower()
        
        # 1. Check for specific worry themes
        if any(w in worry_lower for w in ["فشل", "أفشل", "خسارة", "ما أنجح", "fail"]):
            return "الفشل ليس نقيض النجاح، بل هو جزء منه. كل عثرة هي درس يقربك خطوة نحو هدفك."
        
        if any(w in worry_lower for w in ["وحدة", "وحيد", "أحد", "lonely", "alone"]):
            return "الوحدة فرصة لاكتشاف ذاتك ومصادقتها. أنت بصحبة نفسك، وهي صحبة تستحق التقدير."
            
        if any(w in worry_lower for w in ["خوف", "خائف", "مرعوب", "fear", "scared"]):
            return "الخوف مجرد ظل لشيء لم يحدث بعد. واجهه بالنور، وسيختفي."
            
        if any(w in worry_lower for w in ["مستقبل", "بكرة", "غدا", "future"]):
            return "المستقبل يُبنى بقرارات اليوم. ركز على لحظتك الحالية، فهي كل ما تملك."
            
        if any(w in worry_lower for w in ["حب", "فراق", "تركني", "love", "breakup"]):
            return "القلب الذي ينكسر يفسح مجالاً لنور جديد ليدخله. أنت تستحق الحب، ابدأ بحب نفسك."

        if any(w in worry_lower for w in ["موت", "فقدان", "death", "loss"]):
            return "الألم ثمن الحب. من نحبهم يتركون بصمة في أرواحنا لا تمحى، فهم يعيشون فينا."

        # 2. Random Wisdom
        wisdoms = [
            "كل نار تحرق غابة، تترك تربة خصبة لحياة جديدة. همومك اليوم هي سماد حكمتك غداً.",
            "العنقاء لا تطير إلا بعد أن تحترق. هذا الألم هو بداية تحولك لنسخة أقوى.",
            "لا يمكنك منع طيور الهم من التحليق فوق رأسك، لكن يمكنك منعها من التعشيش في شعرك.",
            "حتى أظلم الليالي ستنتهي بشروق الشمس. هذا القانون الكوني ينطبق على حياتك أيضاً.",
            "أنت لست ما حدث لك، أنت ما تختار أن تكونه بعد ما حدث."
        ]
        
        return random.choice(wisdoms)

    def analyze_sentiment(self, user_input, conversation_context):
        if not getattr(self, 'model', None):
            return self.fallback_sentiment(user_input), None, self.check_for_risk(user_input)

        context_list = [self.preprocess_arabic_text(c) for c in (conversation_context or [])]
        context_list.append(self.preprocess_arabic_text(user_input))
        relevant_context = context_list[-self.max_context_turns:]
        contextual_input = " [SEP] ".join(relevant_context)

        try:
            encoding = self.tokenizer.encode_plus(
                contextual_input, max_length=512, padding='max_length', truncation=True, return_tensors='pt'
            )
        except Exception:
             return self.fallback_sentiment(user_input), None, self.check_for_risk(user_input)

        if not TORCH_AVAILABLE or torch is None:
            return self.fallback_sentiment(user_input), None, self.check_for_risk(user_input)

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = getattr(outputs, 'logits', None)
        if logits is None:
            return self.fallback_sentiment(user_input), None, self.check_for_risk(user_input)

        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class_id = int(np.argmax(probabilities))
        predicted_sentiment = self.labels.get(predicted_class_id, "محايد/أخرى")
        
        # Override with topic detection
        for topic, keywords in TOPIC_OVERRIDE_KEYWORDS.items():
            if any(k in user_input for k in keywords):
                predicted_sentiment = topic
                break
                
        risk_level = self.check_for_risk(user_input)
        return predicted_sentiment, probabilities, risk_level

    def fallback_sentiment(self, message):
        text = message or ""
        
        # Check for Topic Overrides first
        for topic, keywords in TOPIC_OVERRIDE_KEYWORDS.items():
            if any(k in text for k in keywords):
                return topic

        norm = utils.normalize_text(text)
        if not norm:
            return "محايد/أخرى"

        words_original = norm.split()
        words_stemmed = [utils.simple_stem(w) for w in words_original]
        norm_polarity = {utils.simple_stem(utils.normalize_text(k)): v for k, v in EMOTION_POLARITY.items()}

        score = 0.0
        matches = 0
        detected_emotions = []

        for i, w_stem in enumerate(words_stemmed):
            polarity = norm_polarity.get(w_stem, 0)
            if polarity != 0:
                matches += 1
                # Find the emotion key for this stem
                for emotion, val in EMOTION_POLARITY.items():
                    if utils.simple_stem(utils.normalize_text(emotion)) == w_stem:
                        detected_emotions.append(emotion)
                        break

                if i > 0 and words_original[i - 1] in ("لا", "ليس", "لم", "لن", "غير", "ما"):
                    score -= polarity * 1.5
                else:
                    score += polarity
        
        if matches == 0:
            return "محايد/أخرى"
        
        # If we found specific emotion words, prioritize the most frequent or strongest
        if detected_emotions:
            # Simple majority vote or just return the last one for now
            # Better: Map the detected emotion to our core categories
            last_detected = detected_emotions[-1]
            return last_detected

        return "محايد/أخرى"

    def check_for_risk(self, text):
        for word in CRISIS_KEYWORDS:
            if word in text.lower():
                return "خطورة عالية - يرجى طلب المساعدة"
        return "خطورة منخفضة"

    def preprocess_arabic_text(self, text):
        return text.replace('\n', ' ').strip()

    def _handle_intro_questions(self, text):
        """
        Detects questions about the bot's identity, capabilities, or general trust-building chat.
        """
        t = text.lower()
        
        # 1. Identity (Who are you?)
        if any(x in t for x in ["من انت", "من أنت", "عرفني", "مين معي", "مين انت", "اسمك", "عرف عن نفسك", "شكون انت"]):
            return random.choice(INTRO_RESPONSES.get("self", ["أنا رفيقك الذكي."]))
            
        # 2. Capabilities (What do you do?)
        if any(x in t for x in ["ماذا تفعل", "شو بتعمل", "وظيفتك", "فايدتك", "عملك", "ايش تسوي", "شنو دير", "تطبيق ايش", "فائدة التطبيق"]):
            return random.choice(INTRO_RESPONSES.get("capabilities", ["أنا هنا لمساعدتك."]))
            
        # 3. Trust (Are you human? Privacy?)
        if any(x in t for x in ["روبوت", "انسان", "بشر", "آلة", "تخزين", "خصوصية", "سرية", "تفضحني", "حقيقي"]):
            return random.choice(INTRO_RESPONSES.get("trust", ["أنا هنا لأسمعك بسرية تامة."]))
            
        # 4. General Small Talk (Openers)
        if any(x in t for x in ["نتعرف", "ندردش", "نسولف", "نتكلم", "احكي", "ملل", "زهقان", "طفشان", "ضايج"]):
            return random.choice(INTRO_RESPONSES.get("general_chat", ["أنا جاهز للحديث عن أي شيء!"]))
            
        return None

    def _handle_greeting(self, normalized_text):
        for key, resp in GREETING_RESPONSES.items():
            if key in normalized_text:
                return resp
        return None

    def _handle_phrase_bank(self, normalized_text):
        if not (isinstance(PHRASE_BANK, dict) and PHRASE_BANK):
            return None
        try:
            keys = list(PHRASE_BANK.keys())
            match = difflib.get_close_matches(normalized_text, keys, n=1, cutoff=0.6) # Increased cutoff
            if match:
                return PHRASE_BANK.get(match[0])
        except Exception:
            logger.warning("Error during difflib matching in phrase bank.")
        return None

    def _detect_topic(self, text):
        text = text.lower()
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(k in text for k in keywords):
                return topic
        return None

    def _map_sentiment_to_core(self, sentiment):
        # Maps varied sentiment strings to core keys in EMOTION_RESPONSES
        if not sentiment: return "محايد/أخرى"
        
        s = sentiment.lower()
        
        # Check for direct topic overrides first
        if s in TOPIC_OVERRIDE_KEYWORDS:
            return s
            
        if any(x in s for x in ["سعيد", "مبسوط", "فرحان", "متفائل", "رائع", "ممتاز", "جميل", "مذهل", "سعيدة", "مبسوطة", "فرحانة"]):
            return "سعيد"
        if any(x in s for x in ["حزين", "مكتئب", "متضايق", "محبط", "يأس", "سيء", "فاشل", "مشكلة", "صعب", "حزن", "حزينة", "مكتئبة", "متضايقة", "وحيدة"]):
            return "حزين"
        if any(x in s for x in ["غاضب", "عصبي", "مستفز", "غاضبة", "عصبية"]):
            return "غاضب"
        if any(x in s for x in ["قلق", "متوتر", "خايف", "خائف", "قلقة", "متوترة", "خائفة"]):
            return "قلق"
        if any(x in s for x in ["وحدة", "وحيد"]):
            return "وحدة" # Note: 'وحدة' is also mapped to 'حزين' above? No, 'وحدة' has its own key in EMOTION_POLARITY but handled as 'حزين' in logic usually? 
            # In EMOTION_RESPONSES (json), is there a "وحدة" key?
            # Let's check responses.json again.
        if any(x in s for x in ["تعبان", "مرهق"]):
            return "تعبان"
        
        return "محايد/أخرى"

    def generate_response(self, sentiment, risk_level, user_input=None, conversation_context=None, active_topic=None):
        if "خطورة عالية" in risk_level:
            return CRISIS_RESPONSE_AR, "crisis"

        user_text = (user_input or "").strip()
        norm = utils.normalize_text(user_text)

        # 0. Check for Continuation
        CONTINUATION_KEYWORDS = ["تابع", "كمل", "بعدين", "اكمل", "المزيد", "استمر", "ثم", "وبعدين", "شنو بعد"]
        if active_topic and any(k in norm for k in CONTINUATION_KEYWORDS):
             # Try to provide more content for the active topic
             if active_topic in TOPIC_RESPONSES:
                 # Get a response that hasn't been used recently if possible (random for now)
                 return random.choice(TOPIC_RESPONSES[active_topic]), active_topic
             elif active_topic in INTRO_RESPONSES:
                 return random.choice(INTRO_RESPONSES[active_topic]), active_topic

        # 1. Greetings & Intro (Priority 1)
        if norm:
            greeting_resp = self._handle_greeting(norm)
            if greeting_resp:
                return greeting_resp, "greeting"

            # Check for Intro/Identity questions
            intro_resp = self._handle_intro_questions(user_text) 
            if intro_resp:
                return intro_resp, "intro"

            phrase_bank_resp = self._handle_phrase_bank(norm)
            if phrase_bank_resp:
                return phrase_bank_resp, "general"

        # 2. Analyze Context & Sentiment
        core_emotion = self._map_sentiment_to_core(sentiment)
        topic = self._detect_topic(user_text)
        
        # If no new topic detected, keep the old one (unless sentiment changed drastically)
        final_topic = topic if topic else (active_topic if active_topic else "general")
        
        is_long_message = len(user_text.split()) > 7

        # 3. Construct Response Components
        emotion_data = EMOTION_RESPONSES.get(core_emotion, EMOTION_RESPONSES.get("محايد/أخرى"))
        
        # Validation (Mirroring)
        validation_pool = emotion_data.get("validation", [])
        validation = random.choice(validation_pool) if validation_pool else "أسمعك بوضوح."

        # Topic Specific Response
        topic_response = ""
        if topic and topic in TOPIC_RESPONSES:
            topic_response = random.choice(TOPIC_RESPONSES[topic])
        elif active_topic and not topic: # If no new topic, but we have an active one, maybe reinforce it?
            # Optional: Don't repeat topic response unless asked, just validate emotion
            pass

        # Question (Exploration) - Only if message is short or random chance
        question = ""
        question_pool = emotion_data.get("questions", [])
        if question_pool:
            if not is_long_message or random.random() < 0.3:
                question = random.choice(question_pool)

        # Therapeutic Intervention (Optional)
        intervention = ""
        if random.random() < 0.2 and core_emotion in ["قلق", "حزين", "غاضب"]:
             if "grounding" in THERAPEUTIC_INTERVENTIONS:
                 intervention = random.choice(THERAPEUTIC_INTERVENTIONS["grounding"])
        elif random.random() < 0.1 and "reframing" in THERAPEUTIC_INTERVENTIONS:
             intervention = random.choice(THERAPEUTIC_INTERVENTIONS["reframing"])

        # 4. Assemble Final Response
        # Strategy: Vary the combination to avoid repetition
        
        parts = []
        
        # Always validate emotion first
        parts.append(validation)

        # If we have a specific topic response, use it
        if topic_response:
            parts.append(topic_response)
        
        # Add intervention or question (usually not both to avoid overwhelming)
        if intervention:
            parts.append(intervention)
        elif question:
            parts.append(question)
            
        return " ".join(parts), final_topic


# Initialize the analyzer once when the module is loaded
try:
    analyzer = AdvancedSentimentAnalyzer()
except Exception as e:
    logger.error(f"Failed to initialize AdvancedSentimentAnalyzer: {e}")
    analyzer = AdvancedSentimentAnalyzer(model_name=None)
document.addEventListener('DOMContentLoaded', () => {
    // --- Elements ---
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const navItems = document.querySelectorAll('.nav-item');
    const sections = document.querySelectorAll('.content-section');
    const menuToggle = document.getElementById('menu-toggle');
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('overlay');
    const dailyQuoteBtn = document.getElementById('daily-quote-btn');
    const modal = document.getElementById('modal');
    const modalBody = document.getElementById('modal-body');
    const closeModal = document.querySelector('.close-modal');

    // --- State ---
    let chatHistory = [];
    let userId = localStorage.getItem('imma_user_id');
    let currentTopic = null;
    
    // --- Initialization ---
    if (!userId) {
        createUser();
    } else {
        loadUserData();
    }
    
    // --- Event Listeners ---
    
    // Navigation
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = item.getAttribute('data-target');
            
            // Active class for nav
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
            
            // Show section
            sections.forEach(section => section.classList.remove('active'));
            document.getElementById(targetId).classList.add('active');
            
            // Mobile: Close sidebar
            if (window.innerWidth <= 768) {
                toggleSidebar();
            }

            // Load data if needed
            if (targetId === 'exercises-section') loadExercises();
            if (targetId === 'resources-section') loadResources();
            if (targetId === 'progress-section') loadProgress();
        });
    });

    // --- Hope Jar Logic ---
    const addNoteBtn = document.getElementById('add-note-btn');
    const shakeJarBtn = document.getElementById('shake-jar-btn');
    const addNoteModal = document.getElementById('add-note-modal');
    const closeNoteModal = document.getElementById('close-note-modal');
    const saveNoteBtn = document.getElementById('save-note-btn');
    const hopeNoteInput = document.getElementById('hope-note-input');
    const noteDisplay = document.getElementById('note-display');
    const noteContent = document.getElementById('note-content');
    const noteDate = document.getElementById('note-date');
    const jar = document.getElementById('the-hope-jar');

    if (addNoteBtn) {
        addNoteBtn.addEventListener('click', () => {
            addNoteModal.style.display = 'flex';
        });
    }

    if (closeNoteModal) {
        closeNoteModal.addEventListener('click', () => {
            addNoteModal.style.display = 'none';
        });
    }

    if (saveNoteBtn) {
        saveNoteBtn.addEventListener('click', async () => {
            const content = hopeNoteInput.value.trim();
            if (!content) return;

            try {
                const res = await fetch('/hope/add', {
                    method: 'POST',
                    body: JSON.stringify({ user_id: userId, content: content })
                });
                const data = await res.json();
                if (data.status === 'ok') {
                    addNoteModal.style.display = 'none';
                    hopeNoteInput.value = '';
                    showModal("تم حفظ لحظتك السعيدة! 🌟");
                }
            } catch (e) {
                console.error(e);
            }
        });
    }

    if (shakeJarBtn) {
        shakeJarBtn.addEventListener('click', async () => {
            if (!jar) return;
            jar.classList.add('shaking');
            noteDisplay.classList.add('hidden');
            
            setTimeout(async () => {
                jar.classList.remove('shaking');
                try {
                    const res = await fetch(`/hope/shake?user_id=${userId}`);
                    const data = await res.json();
                    if (data.status === 'ok' && data.note) {
                        noteContent.innerText = data.note.content;
                        noteDate.innerText = new Date(data.note.date).toLocaleDateString('ar-EG');
                        noteDisplay.classList.remove('hidden');
                    }
                } catch (e) {
                    console.error(e);
                }
            }, 1000);
        });
    }

    // --- Calm Room Logic (Synthesized Audio) ---
    class SynthesizedSound {
        constructor() {
            this.ctx = new (window.AudioContext || window.webkitAudioContext)();
            this.sources = {};
            this.gains = {};
        }

        createNoiseBuffer() {
            const bufferSize = this.ctx.sampleRate * 2; // 2 seconds
            const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
            const data = buffer.getChannelData(0);
            for (let i = 0; i < bufferSize; i++) {
                data[i] = Math.random() * 2 - 1;
            }
            return buffer;
        }

        createPinkNoiseBuffer() {
            const bufferSize = this.ctx.sampleRate * 2;
            const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
            const data = buffer.getChannelData(0);
            let b0, b1, b2, b3, b4, b5, b6;
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0.0;
            for (let i = 0; i < bufferSize; i++) {
                const white = Math.random() * 2 - 1;
                b0 = 0.99886 * b0 + white * 0.0555179;
                b1 = 0.99332 * b1 + white * 0.0750759;
                b2 = 0.96900 * b2 + white * 0.1538520;
                b3 = 0.86650 * b3 + white * 0.3104856;
                b4 = 0.55000 * b4 + white * 0.5329522;
                b5 = -0.7616 * b5 - white * 0.0168981;
                data[i] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362;
                data[i] *= 0.11; // (roughly) compensate for gain
                b6 = white * 0.115926;
            }
            return buffer;
        }

        playRain(volume) {
            if (this.sources['rain']) {
                this.gains['rain'].gain.value = volume;
                return;
            }
            const buffer = this.createPinkNoiseBuffer();
            const source = this.ctx.createBufferSource();
            source.buffer = buffer;
            source.loop = true;
            const gain = this.ctx.createGain();
            gain.gain.value = volume;
            source.connect(gain);
            gain.connect(this.ctx.destination);
            source.start();
            this.sources['rain'] = source;
            this.gains['rain'] = gain;
        }

        playForest(volume) {
             // Forest = Wind (Pink Noise) + Chirps (Oscillators)
             // Simplified: Just Wind for now
            if (this.sources['forest']) {
                this.gains['forest'].gain.value = volume;
                return;
            }
            const buffer = this.createPinkNoiseBuffer();
            const source = this.ctx.createBufferSource();
            source.buffer = buffer;
            source.loop = true;
            
            // High pass filter for wind
            const filter = this.ctx.createBiquadFilter();
            filter.type = 'highpass';
            filter.frequency.value = 800;

            const gain = this.ctx.createGain();
            gain.gain.value = volume;
            
            source.connect(filter);
            filter.connect(gain);
            gain.connect(this.ctx.destination);
            source.start();
            this.sources['forest'] = source;
            this.gains['forest'] = gain;
        }

        playOcean(volume) {
             // Ocean = Brownish noise (Low passed Pink)
            if (this.sources['ocean']) {
                this.gains['ocean'].gain.value = volume;
                return;
            }
            const buffer = this.createPinkNoiseBuffer();
            const source = this.ctx.createBufferSource();
            source.buffer = buffer;
            source.loop = true;
            
            const filter = this.ctx.createBiquadFilter();
            filter.type = 'lowpass';
            filter.frequency.value = 400;

            const gain = this.ctx.createGain();
            gain.gain.value = volume;
            
            source.connect(filter);
            filter.connect(gain);
            gain.connect(this.ctx.destination);
            source.start();
            this.sources['ocean'] = source;
            this.gains['ocean'] = gain;
        }

        playFire(volume) {
             // Fire = Crackling (Random impulses)
             // Simplified: Brown noise (heavily filtered)
             if (this.sources['fire']) {
                this.gains['fire'].gain.value = volume;
                return;
            }
            const buffer = this.createPinkNoiseBuffer();
            const source = this.ctx.createBufferSource();
            source.buffer = buffer;
            source.loop = true;
            
            const filter = this.ctx.createBiquadFilter();
            filter.type = 'lowpass';
            filter.frequency.value = 150;

            const gain = this.ctx.createGain();
            gain.gain.value = volume;
            
            source.connect(filter);
            filter.connect(gain);
            gain.connect(this.ctx.destination);
            source.start();
            this.sources['fire'] = source;
            this.gains['fire'] = gain;
        }
    }

    const calmSynth = new SynthesizedSound();

    document.querySelectorAll('.volume-slider').forEach(slider => {
        slider.addEventListener('input', (e) => {
            const soundType = e.target.getAttribute('data-sound');
            const volume = parseFloat(e.target.value) / 100;
            
            if (calmSynth.ctx.state === 'suspended') {
                calmSynth.ctx.resume();
            }

            if (soundType === 'rain') calmSynth.playRain(volume);
            if (soundType === 'forest') calmSynth.playForest(volume);
            if (soundType === 'ocean') calmSynth.playOcean(volume);
            if (soundType === 'fire') calmSynth.playFire(volume);
        });
    });

    // Breathing Animation
    const startBreathBtn = document.getElementById('start-breath-btn');
    const breathingCircle = document.getElementById('breathing-circle');
    let isBreathing = false;

    if (startBreathBtn) {
        startBreathBtn.addEventListener('click', () => {
            if (isBreathing) {
                isBreathing = false;
                startBreathBtn.innerText = "ابدأ التمرين";
                breathingCircle.classList.remove('inhale', 'exhale');
                breathingCircle.innerText = "تنفس";
                return;
            }

            isBreathing = true;
            startBreathBtn.innerText = "إيقاف";
            breathCycle();
        });
    }

    function breathCycle() {
        if (!isBreathing) return;
        
        // Inhale (4s)
        breathingCircle.classList.add('inhale');
        breathingCircle.classList.remove('exhale');
        breathingCircle.innerText = "شهيق";
        
        setTimeout(() => {
            if (!isBreathing) return;
            // Hold (4s) - optional, skipping for simplicity or adding pause
             breathingCircle.innerText = "أحبس";
             
             setTimeout(() => {
                if (!isBreathing) return;
                // Exhale (4s)
                breathingCircle.classList.remove('inhale');
                breathingCircle.classList.add('exhale');
                breathingCircle.innerText = "زفير";
                
                setTimeout(() => {
                     if (!isBreathing) return;
                     breathCycle();
                }, 4000);
             }, 4000);
        }, 4000);
    }

    function showModal(content) {
        const m = document.getElementById('modal');
        const b = document.getElementById('modal-body');
        b.innerHTML = content;
        m.style.display = 'flex';
    }

    // Mobile Sidebar
    menuToggle.addEventListener('click', toggleSidebar);
    overlay.addEventListener('click', toggleSidebar);

    function toggleSidebar() {
        sidebar.classList.toggle('open');
        overlay.classList.toggle('active');
    }

    // Chat
    sendButton.addEventListener('click', () => sendMessage());
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    // Modal
    closeModal.addEventListener('click', () => modal.style.display = 'none');
    window.addEventListener('click', (e) => {
        if (e.target === modal) modal.style.display = 'none';
    });
    
    if (dailyQuoteBtn) {
        dailyQuoteBtn.addEventListener('click', async () => {
            try {
                const res = await fetch('/daily_quote');
                const data = await res.json();
                showModal(data.quote || "كن لطيفاً مع نفسك.");
            } catch (e) {
                console.error(e);
            }
        });
    }

    // --- Game Logic ---
    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas ? canvas.getContext('2d') : null;
    let gameInterval;
    let bubbles = [];
    let score = 0;
    const scoreDisplay = document.getElementById('score');
    
    // Words to put in bubbles (Stressors)
    const stressors = ["قلق", "خوف", "توتر", "حزن", "تعب", "ضغط", "أرق", "تفكير", "وحدة"];
    // Words after pop (Relief)
    const relief = ["راحة", "أمان", "هدوء", "سعادة", "قوة", "أمل", "ثقة", "نور", "حب"];

    window.startGame = function() {
        if (!canvas) return;
        score = 0;
        if(scoreDisplay) scoreDisplay.innerText = score;
        bubbles = [];
        
        // Resize canvas
        canvas.width = canvas.parentElement.offsetWidth;
        canvas.height = canvas.parentElement.offsetHeight;

        if (gameInterval) clearInterval(gameInterval);
        
        // Game Loop
        gameInterval = setInterval(updateGame, 30);
        
        // Click listener
        canvas.removeEventListener('mousedown', handleGameClick); // Prevent duplicates
        canvas.addEventListener('mousedown', handleGameClick);
        canvas.removeEventListener('touchstart', handleGameClick);
        canvas.addEventListener('touchstart', handleGameClick);
    };

    function createBubble() {
        const radius = 25 + Math.random() * 20;
        const x = Math.random() * (canvas.width - radius * 2) + radius;
        const y = canvas.height + radius;
        const speed = 1 + Math.random() * 2;
        const text = stressors[Math.floor(Math.random() * stressors.length)];
        
        bubbles.push({
            x, y, radius, speed, text, 
            popped: false, 
            popTimer: 0,
            color: `hsla(${Math.random() * 360}, 70%, 70%, 0.8)`
        });
    }

    function updateGame() {
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Spawn bubbles randomly
        if (Math.random() < 0.03) createBubble();
        
        // Update and draw bubbles
        for (let i = bubbles.length - 1; i >= 0; i--) {
            let b = bubbles[i];
            
            if (!b.popped) {
                b.y -= b.speed;
                
                // Draw Bubble
                ctx.beginPath();
                ctx.arc(b.x, b.y, b.radius, 0, Math.PI * 2);
                ctx.fillStyle = b.color;
                ctx.fill();
                ctx.strokeStyle = "#fff";
                ctx.lineWidth = 2;
                ctx.stroke();
                
                // Draw Text
                ctx.fillStyle = "#fff";
                ctx.font = "14px Tajawal";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(b.text, b.x, b.y);
                
                // Remove if off screen
                if (b.y + b.radius < 0) {
                    bubbles.splice(i, 1);
                }
            } else {
                // Popped animation (expanding text)
                b.popTimer++;
                ctx.fillStyle = "#4CAF50"; // Green for relief
                ctx.font = "bold 18px Tajawal";
                ctx.textAlign = "center";
                ctx.fillText(b.text, b.x, b.y - b.popTimer); // Float up
                
                if (b.popTimer > 20) {
                    bubbles.splice(i, 1);
                }
            }
        }
    }

    function handleGameClick(e) {
        const rect = canvas.getBoundingClientRect();
        // Handle touch or mouse
        const clientX = e.clientX || (e.touches ? e.touches[0].clientX : 0);
        const clientY = e.clientY || (e.touches ? e.touches[0].clientY : 0);
        
        const clickX = clientX - rect.left;
        const clickY = clientY - rect.top;
        
        bubbles.forEach(b => {
            if (!b.popped) {
                const dist = Math.sqrt((clickX - b.x) ** 2 + (clickY - b.y) ** 2);
                if (dist < b.radius) {
                    b.popped = true;
                    b.text = relief[Math.floor(Math.random() * relief.length)];
                    score++;
                    if(scoreDisplay) scoreDisplay.innerText = score;
                    // Optional: Play sound here
                }
            }
        });
    }

    // Handle Resize
    window.addEventListener('resize', () => {
        if(canvas && canvas.parentElement) {
             canvas.width = canvas.parentElement.offsetWidth;
             canvas.height = canvas.parentElement.offsetHeight;
        }
    });

    // --- Functions ---

    async function createUser() {
        // Create a guest user automatically
        try {
            const res = await fetch('/user/create', {
                method: 'POST',
                body: JSON.stringify({ username: 'زائر' })
            });
            const data = await res.json();
            if (data.status === 'ok') {
                userId = data.user_id;
                localStorage.setItem('imma_user_id', userId);
                loadUserData();
            }
        } catch (e) {
            console.error('Failed to create user', e);
        }
    }

    async function loadUserData() {
        if (!userId) return;
        try {
            const res = await fetch(`/user/data?user_id=${userId}`);
            const data = await res.json();
            if (data.status === 'ok' && data.user) {
                document.getElementById('username-display').innerText = data.user.username;
                document.getElementById('streak-display').innerText = `${data.user.streak || 0} يوم تتابع`;
                
                // Also update stats page
                const streakStat = document.getElementById('streak-stat');
                if (streakStat) streakStat.innerText = data.user.streak || 0;
                
                const badgeCount = document.getElementById('badges-count');
                if (badgeCount) badgeCount.innerText = (data.user.badges || []).length;
            }
        } catch (e) {
            console.error('Failed to load user data', e);
        }
    }

    function addMessage(message, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);
        
        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        contentDiv.innerHTML = message;
        
        const timeSpan = document.createElement('span');
        timeSpan.classList.add('message-time');
        const now = new Date();
        timeSpan.innerText = `${now.getHours()}:${now.getMinutes().toString().padStart(2, '0')}`;
        
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timeSpan);
        
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;

        // Button Listeners
        contentDiv.querySelectorAll('.chat-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const text = btn.getAttribute('data-input');
                userInput.value = text;
                sendMessage();
            });
        });
    }

    /* --- Advanced Sound Generation Engine (Web Audio API) --- */
    class SoundEngine {
        constructor() {
            this.ctx = null;
            this.masterGain = null;
            this.oscillators = [];
            this.noiseNode = null;
            this.isPlaying = false;
            this.currentMood = 'neutral';
        }

        init() {
            if (!this.ctx) {
                const AudioContext = window.AudioContext || window.webkitAudioContext;
                this.ctx = new AudioContext();
                this.masterGain = this.ctx.createGain();
                this.masterGain.connect(this.ctx.destination);
                this.masterGain.gain.value = 0.3; // General Volume
                this.isPlaying = true; // Auto-enable on init
            }
        }

        async toggle() {
            if (!this.ctx) this.init();
            
            if (this.ctx.state === 'suspended') {
                await this.ctx.resume();
            }

            if (this.isPlaying) {
                this.stopAll();
                this.isPlaying = false;
                return false;
            } else {
                this.isPlaying = true;
                this.playMood(this.currentMood);
                return true;
            }
        }

        stopAll() {
            this.oscillators.forEach(osc => {
                try { osc.stop(); osc.disconnect(); } catch(e){}
            });
            this.oscillators = [];
            
            if (this.noiseNode) {
                try { this.noiseNode.disconnect(); } catch(e){}
                this.noiseNode = null;
            }
        }
        
        playMood(mood) {
            if (!this.ctx) this.init();
            
            // Auto-resume if suspended (browser policy)
            if (this.ctx && this.ctx.state === 'suspended') {
                this.ctx.resume();
            }

            if (!this.isPlaying) return;
            
            this.stopAll();
            this.currentMood = mood;

            if (['قلق', 'متوتر', 'خائف', 'anxious'].some(s => mood.includes(s))) {
                // Calm Ocean Drone
                this.createDrone(150, 'sine', 0.1);
                this.createDrone(152, 'sine', 0.1); // Binaural beat 2Hz
                this.createNoise('lowpass', 400, 0.15); // Ocean swoosh
            } 
            else if (['حزين', 'مكتئب', 'وحدة', 'sad'].some(s => mood.includes(s))) {
                // Warm Comfort (Fire crackle approximation + low hum)
                this.createDrone(100, 'triangle', 0.05);
                this.createNoise('lowpass', 800, 0.1); 
            }
            else if (['غاضب', 'عصبي', 'angry'].some(s => mood.includes(s))) {
                // Grounding Earth (Deep low freqs)
                this.createDrone(60, 'sawtooth', 0.03); // Deep rumble
                this.createDrone(120, 'sine', 0.1);
                this.createNoise('lowpass', 200, 0.2); // Wind
            }
            else {
                // Neutral/Happy (Uplifting)
                this.createDrone(300, 'sine', 0.05);
                this.createDrone(450, 'sine', 0.05); // Major 5th
            }
        }

        createDrone(freq, type, vol) {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = type;
            osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
            gain.gain.setValueAtTime(0, this.ctx.currentTime);
            gain.gain.linearRampToValueAtTime(vol, this.ctx.currentTime + 2); // Fade in
            
            osc.connect(gain);
            gain.connect(this.masterGain);
            osc.start();
            this.oscillators.push(osc);
        }

        createNoise(filterType, filterFreq, vol) {
            const bufferSize = 2 * this.ctx.sampleRate;
            const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
            const data = buffer.getChannelData(0);
            
            // Brown noise generation
            let lastOut = 0;
            for (let i = 0; i < bufferSize; i++) {
                const white = Math.random() * 2 - 1;
                data[i] = (lastOut + (0.02 * white)) / 1.02;
                lastOut = data[i];
                data[i] *= 3.5; 
            }

            const noise = this.ctx.createBufferSource();
            noise.buffer = buffer;
            noise.loop = true;
            
            const filter = this.ctx.createBiquadFilter();
            filter.type = filterType;
            filter.frequency.value = filterFreq;
            
            const gain = this.ctx.createGain();
            gain.gain.value = vol;

            noise.connect(filter);
            filter.connect(gain);
            gain.connect(this.masterGain);
            noise.start();
            this.noiseNode = noise;
        }
    }

    const soundEngine = new SoundEngine();

    /* --- UI Functions --- */

    window.toggleSound = function() {
        soundEngine.toggle().then(isActive => {
            const btn = document.getElementById('soundToggle');
            if (btn) {
                const icon = btn.querySelector('i');
                if (isActive) {
                    btn.classList.add('active');
                    if(icon) icon.className = 'fas fa-volume-up';
                } else {
                    btn.classList.remove('active');
                    if(icon) icon.className = 'fas fa-volume-mute';
                }
            }
        });
    };

    function updateAtmosphere(sentiment) {
        console.log("Updating atmosphere for sentiment:", sentiment);
        
        // Reset class list but keep necessary ones if any (none currently for body)
        document.body.className = ''; 
        
        // Define gradients map for direct style injection (The "Radical" Fix)
        const gradients = {
            'anxious': 'linear-gradient(135deg, #e0f7fa, #4dd0e1, #26c6da)',
            'sad': 'linear-gradient(135deg, #fff3e0, #ffb74d, #ff9800)',
            'angry': 'linear-gradient(135deg, #e8f5e9, #81c784, #4caf50)',
            'neutral': 'linear-gradient(135deg, #f5f5f5, #e0e0e0, #cfcfcf)',
            'work_stress': 'linear-gradient(135deg, #f3e5f5, #ce93d8, #ab47bc)',
            'divorce': 'linear-gradient(135deg, #fce4ec, #f48fb1, #ec407a)',
            'social_issues': 'linear-gradient(135deg, #e0f2f1, #80cbc4, #26a69a)'
        };

        const chatTints = {
            'anxious': 'rgba(224, 247, 250, 0.9)',
            'sad': 'rgba(255, 243, 224, 0.9)',
            'angry': 'rgba(232, 245, 233, 0.9)',
            'neutral': 'rgba(255, 255, 255, 0.85)',
            'work_stress': 'rgba(243, 229, 245, 0.9)',
            'divorce': 'rgba(252, 228, 236, 0.9)',
            'social_issues': 'rgba(224, 242, 241, 0.9)'
        };

        let chosenMood = 'neutral';

        // Logic to determine mood key
        if (['قلق', 'متوتر', 'خائف', 'anxious'].some(s => sentiment.includes(s))) {
            chosenMood = 'anxious';
        } else if (['حزين', 'مكتئب', 'وحدة', 'sad'].some(s => sentiment.includes(s))) {
            chosenMood = 'sad';
        } else if (['غاضب', 'عصبي', 'angry'].some(s => sentiment.includes(s))) {
            chosenMood = 'angry';
        } else if (sentiment.includes('work_stress')) {
            chosenMood = 'work_stress';
        } else if (sentiment.includes('divorce')) {
            chosenMood = 'divorce';
        } else if (sentiment.includes('social_issues')) {
            chosenMood = 'social_issues';
        }

        // Apply Class for specific CSS overrides (fonts, borders, etc.)
        document.body.classList.add(`mood-${chosenMood}`);

        // DIRECTLY set background to ensure it changes (Overrides any CSS specificity issues)
        document.body.style.background = gradients[chosenMood];
        document.body.style.backgroundSize = "400% 400%"; // Ensure animation keeps working
        
        // Also tint the chat interface itself
        const chatWrapper = document.querySelector('.chat-wrapper');
        const sidebar = document.querySelector('.sidebar');
        if (chatWrapper) {
            chatWrapper.style.background = chatTints[chosenMood];
            chatWrapper.style.borderColor = "rgba(255,255,255,0.6)";
        }
        if (sidebar) {
            sidebar.style.background = chatTints[chosenMood];
        }

        // Play Sound
        if (soundEngine) {
             // Map sentiment to simple sound moods
             let soundMood = 'neutral';
             if (chosenMood === 'anxious' || chosenMood === 'work_stress' || chosenMood === 'social_issues') soundMood = 'anxious';
             if (chosenMood === 'sad' || chosenMood === 'divorce') soundMood = 'sad';
             if (chosenMood === 'angry') soundMood = 'angry';
             
             soundEngine.playMood(soundMood);
        }
    }

    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        addMessage(text, 'user');
        chatHistory.push(text);
        userInput.value = '';

        // Add loading indicator
        const loadingId = 'loading-' + Date.now();
        const loadingDiv = document.createElement('div');
        loadingDiv.id = loadingId;
        loadingDiv.classList.add('message', 'bot-message');
        loadingDiv.innerHTML = '<div class="message-content"><i class="fas fa-ellipsis-h"></i></div>';
        chatBox.appendChild(loadingDiv);
        chatBox.scrollTop = chatBox.scrollHeight;

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: text,
                    context: chatHistory.slice(0, -1),
                    user_id: userId,
                    active_topic: currentTopic // Send current topic for context
                })
            });
            
            const data = await response.json();
            
            // Remove loading
            document.getElementById(loadingId).remove();
            
            addMessage(data.response, 'bot');
            chatHistory.push(data.response);
            
            // Update current topic if changed
            if (data.active_topic) {
                currentTopic = data.active_topic;
            }

            // --- INNOVATION: Update Atmosphere ---
            if (data.sentiment_label) {
                updateAtmosphere(data.sentiment_label);
            }
            
        } catch (error) {
            document.getElementById(loadingId).remove();
            addMessage("عذراً، حدث خطأ في الاتصال. 😔", 'bot');
            console.error(error);
        }
    }

    // --- Worry Burner Logic ---
    const burnBtn = document.getElementById('burn-btn');
    const worryInput = document.getElementById('worry-input');
    const parchmentArea = document.getElementById('parchment-area');
    const phoenixMessage = document.getElementById('phoenix-message');
    const phoenixText = document.getElementById('phoenix-text');
    const resetBurnerBtn = document.getElementById('reset-burner');
    const fireCanvas = document.getElementById('fire-canvas');
    let fireCtx = fireCanvas ? fireCanvas.getContext('2d') : null;
    let particles = [];
    let isBurning = false;

    if (burnBtn) {
        burnBtn.addEventListener('click', async () => {
            const worry = worryInput.value.trim();
            if (!worry) {
                showModal("اكتب ما يثقل صدرك أولاً.");
                return;
            }

            // Start Burning Effect
            isBurning = true;
            parchmentArea.classList.add('burning');
            startFireAnimation();
            
            // Play Fire Sound (using SoundEngine noise if available or creating new)
            // For simplicity, we can use a generated crackle
            if(soundEngine && soundEngine.ctx) {
                soundEngine.createNoise('lowpass', 1000, 0.5); // Crackle approximation
            }

            // Call Backend for Transmutation
            try {
                const res = await fetch('/transmute', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ worry: worry, user_id: userId })
                });
                const data = await res.json();
                
                // Wait for animation (3 seconds)
                setTimeout(() => {
                    stopFireAnimation();
                    parchmentArea.style.opacity = '0';
                    setTimeout(() => {
                        parchmentArea.classList.add('hidden');
                        parchmentArea.style.display = 'none'; // Ensure it's gone
                        burnBtn.style.display = 'none';
                        
                        phoenixMessage.classList.remove('hidden');
                        phoenixText.innerText = data.insight || "كل نهاية هي بداية جديدة. أنت أقوى مما تظن.";
                        
                        // Sound effect: Phoenix rise (chime) - Optional
                    }, 1000);
                }, 3000);

            } catch (e) {
                console.error(e);
                isBurning = false;
                stopFireAnimation();
                parchmentArea.classList.remove('burning');
                showModal("حدث خطأ في الاتصال. حاول مرة أخرى.");
            }
        });
    }

    if (resetBurnerBtn) {
        resetBurnerBtn.addEventListener('click', () => {
            phoenixMessage.classList.add('hidden');
            parchmentArea.style.display = 'block';
            parchmentArea.classList.remove('hidden');
            setTimeout(() => parchmentArea.style.opacity = '1', 10);
            parchmentArea.classList.remove('burning');
            worryInput.value = '';
            burnBtn.style.display = 'block';
        });
    }

    function startFireAnimation() {
        if (!fireCanvas) return;
        fireCanvas.width = fireCanvas.parentElement.offsetWidth;
        fireCanvas.height = fireCanvas.parentElement.offsetHeight;
        
        particles = [];
        for(let i=0; i<100; i++) createParticle();
        
        animateFire();
    }

    function createParticle() {
        particles.push({
            x: Math.random() * fireCanvas.width,
            y: fireCanvas.height,
            size: Math.random() * 20 + 10,
            speedY: Math.random() * 5 + 2,
            life: Math.random() * 100 + 50,
            color: `rgba(255, ${Math.random()*150}, 0,`
        });
    }

    function animateFire() {
        if (!isBurning) return;
        fireCtx.clearRect(0, 0, fireCanvas.width, fireCanvas.height);
        
        for (let i = 0; i < particles.length; i++) {
            let p = particles[i];
            p.y -= p.speedY;
            p.size *= 0.95;
            p.life--;
            
            fireCtx.beginPath();
            fireCtx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            fireCtx.fillStyle = p.color + (p.life/100) + ')';
            fireCtx.fill();
            
            if (p.life <= 0 || p.size < 0.5) {
                particles.splice(i, 1);
                createParticle(); // Respawn
            }
        }
        
        // Add random new particles
        if(Math.random() < 0.5) createParticle();
        
        requestAnimationFrame(animateFire);
    }

    function stopFireAnimation() {
        isBurning = false;
        if(fireCtx) fireCtx.clearRect(0, 0, fireCanvas.width, fireCanvas.height);
    }

    // --- Data Loading Functions ---

    async function loadExercises() {
        const list = document.getElementById('exercises-list');
        if (list.childElementCount > 1) return; // Already loaded (assuming spinner is 1)
        
        try {
            const res = await fetch('/exercises');
            const data = await res.json();
            list.innerHTML = '';
            
            if (data.exercises && data.exercises.length > 0) {
                data.exercises.forEach(ex => {
                    const card = document.createElement('div');
                    card.className = 'card';
                    card.innerHTML = `
                        <h3><i class="fas fa-seedling"></i> ${ex.title || 'تمرين'}</h3>
                        <p>${ex.description || ''}</p>
                        <button class="chat-btn" style="margin-top:10px" onclick="alert('ابدأ التمرين: ${ex.title}')">ابدأ الآن</button>
                    `;
                    list.appendChild(card);
                });
            } else {
                list.innerHTML = '<p>لا توجد تمارين متاحة حالياً.</p>';
            }
        } catch (e) {
            list.innerHTML = '<p>فشل تحميل التمارين.</p>';
        }
    }

    async function loadResources() {
        const list = document.getElementById('resources-list');
        if (list.childElementCount > 0) return;
        
        try {
            const res = await fetch('/resources');
            const data = await res.json();
            
            // Assuming resources is a dict/object based on your python code
            // Or if it's a list, adapt accordingly. 
            // Based on earlier read, it's a dict. Let's handle generic key-values or list.
            
            // If it's empty
            if (Object.keys(data).length === 0) {
                list.innerHTML = '<p>لا توجد مصادر متاحة.</p>';
                return;
            }

            for (const [key, value] of Object.entries(data)) {
                 const card = document.createElement('div');
                    card.className = 'card';
                    // Check if value is string or object
                    let content = typeof value === 'string' ? value : JSON.stringify(value);
                    card.innerHTML = `
                        <h3>${key}</h3>
                        <p>${content}</p>
                    `;
                    list.appendChild(card);
            }
        } catch (e) {
            list.innerHTML = '<p>فشل تحميل المصادر.</p>';
        }
    }

    async function loadProgress() {
        // Simple placeholder for progress list
        const list = document.getElementById('progress-list');
        // Ideally fetch from /progress?user_id=...
        if (!userId) return;
        
        try {
            const res = await fetch(`/progress?user_id=${userId}`);
            const data = await res.json();
            list.innerHTML = '';
            if (data.records && data.records.length > 0) {
                data.records.reverse().forEach(rec => {
                    const li = document.createElement('li');
                    li.innerHTML = `
                        <strong>${new Date(rec.timestamp).toLocaleDateString('ar')}</strong>: 
                        ${rec.note || 'نشاط مكتمل'} 
                        ${rec.completed ? '✅' : ''}
                    `;
                    list.appendChild(li);
                });
            } else {
                list.innerHTML = '<li>لا يوجد سجل نشاط بعد. ابدأ محادثة أو تمريناً!</li>';
            }
        } catch (e) {
            console.error(e);
        }
    }

    function showModal(content) {
        modalBody.innerHTML = content;
        modal.style.display = 'flex';
    }
});

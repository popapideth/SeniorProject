const AudioContextClass = window.AudioContext || window.webkitAudioContext;
const audioCtx = AudioContextClass ? new AudioContextClass() : null;
let masterGain = null;

if (audioCtx) {
	masterGain = audioCtx.createGain();
	masterGain.gain.value = 1.0; 
	masterGain.connect(audioCtx.destination);
}


const completeSound = new Audio('/static/sounds/ding-101492.mp3');
completeSound.preload = 'auto';
completeSound.crossOrigin = 'anonymous';

const allCompleteSound = new Audio('/static/sounds/wow-423653.mp3');
allCompleteSound.preload = 'auto';
allCompleteSound.crossOrigin = 'anonymous';
try {
	if (audioCtx && typeof audioCtx.createMediaElementSource === 'function') {
		const src1 = audioCtx.createMediaElementSource(completeSound);
		const src2 = audioCtx.createMediaElementSource(allCompleteSound);
		src1.connect(masterGain);
		src2.connect(masterGain);
	}
} catch (e) {
	// createMediaElementSource may throw if used multiple times; ignore and fallback to element playback
	console.warn('WebAudio hookup failed:', e);
}

// Expose helpers to control gain and resume audio context from a user gesture
window.enableAudio = async function() {
	try {
		if (audioCtx && audioCtx.state === 'suspended') await audioCtx.resume();
		return true;
	} catch (e) {
		console.warn('enableAudio failed', e);
		return false;
	}
};

window.setMasterGain = function(v) {
	if (masterGain) masterGain.gain.value = v;
	else console.warn('No masterGain available');
};

// Convenience: expose current sound objects
window.completeSound = completeSound;
window.allCompleteSound = allCompleteSound;

// Load audio (best-effort)
try { completeSound.load(); } catch (e) {}
try { allCompleteSound.load(); } catch (e) {}
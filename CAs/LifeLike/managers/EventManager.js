// we treat is as a static class for handling event detection and sending signals

// Define trigger keys
const PLAY_PAUSE_KEY = ' ';
const NEXT_FRAME_KEY = '.';

// THIS WAS THE FIRST SCRIPT 
// Flags for event coordination
var running = false;
var oneFrame = false;
var newRuleString = false;
var ruleString = "";
var forcedUpdate = () => { return; }; // Value will be added when defined

// Keyboard event callback
function keyListener(e) {
    switch (e.key) {
        case PLAY_PAUSE_KEY:
            running = !running;
            break;
        case NEXT_FRAME_KEY:
            oneFrame = true;
            forcedUpdate(); // Force immediate update
            break;
        default:
    }
}

// THE ENDING SCRIPT THING
document.getElementsByTagName("body")[0].onkeypress = keyListener;

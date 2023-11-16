// we treat is as a static class for handling event detection and sending signals

/* pseudocode from tyler while he talks about static classes and stuff
class EventManager {
    // private
    constructor(str_) { str = str_; }

    str = "";
    eventManager;

    static getInstance() {
        if (eventManager === null) {
            eventManager = EventManager("hi");
        }
        return this.eventManager;
    }

    getString() {
        return this.str;
    }
}; 
*/


export class EventManager {
    constructor() { }
    static oneFrame = false;
    static running = true;
    static newRuleString = false;
    static ruleString = "";
    static forcedUpdate = () => { return; }; // Value will be added when defined. 
    // ^ this guy is bad because every time you call it, another copy is made. -tyler
    // that's why its called static and it makes a lot of sense. 
    static PLAY_PAUSE_KEY = 'k';
    static NEXT_FRAME_KEY = '.';
}

// Keyboard event callback
export function keyListener(e) {
    switch (e.key) {
        case EventManager.PLAY_PAUSE_KEY:
            EventManager.running = !EventManager.running;
            break;
        case EventManager.NEXT_FRAME_KEY:
            EventManager.oneFrame = true;
            EventManager.forcedUpdate(); // Force immediate update
            break;
        default:
    }
}
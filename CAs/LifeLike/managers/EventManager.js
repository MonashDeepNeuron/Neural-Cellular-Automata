// we treat is as a static class for handling event detection and sending signals

export class EventManager {
    constructor() { }

    // essentially global variables that handle event related things
    static oneFrame = false;
    static running = true;
    static newRuleString = false;
    static ruleString = "";
    static forcedUpdate = () => { return; };  // anonymous func. problem child. 

    // key bindings
    static PLAY_PAUSE_KEY = 'k';
    static NEXT_FRAME_KEY = '.';

    // static methods
    static playPause() {
        console.log(EventManager.running)
        EventManager.running = !EventManager.running
    };
    static moveOneFrame() {
        EventManager.oneFrame = true;
        EventManager.forcedUpdate()
    }
    static keyListener(e) {
        // console.log("th", this.PLAY_PAUSE_KEY);
        // console.log("em", EventManager.PLAY_PAUSE_KEY);
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
    } // blah blah blah something static method
}
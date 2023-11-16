// we treat is as a static class for handling event detection and sending signals

export default class EventManager {
    constructor() { };

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
    static playPause() { EventManager.running = !EventManager.running };
    static moveOneFrame() { EventManager.oneFrame = true; EventManager.forcedUpdate(); }
    static keyListener(e) {
        // console.log("th", this.PLAY_PAUSE_KEY);
        // console.log("em", EventManager.PLAY_PAUSE_KEY);
        switch (e.key) {
            case EventManager.PLAY_PAUSE_KEY:
                EventManager.playPause()
                break;
            case EventManager.NEXT_FRAME_KEY:
                EventManager.moveOneFrame()
                break;
            default:
        }
    } // blah blah blah something static method

    static updateRuleString() {
        const inputText = document.getElementById('simulationInput').value;
        EventManager.newRuleString = true
        EventManager.ruleString = inputText
        console.log(inputText)
        EventManager.forcedUpdate()
    };
}
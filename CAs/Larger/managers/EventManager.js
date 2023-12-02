// static class used to handle and process events
// it uses the default constructor

export default class EventManager {
    // event related global variables
    static oneFrame = false;
    static running = true;
    static newRuleString = false;
    static ruleString = "/2";
    static updateInterval = 50;
    static currentTimer = 0; // Update interval
    static updateLoop = () => {};

    // key bindings
    static PLAY_PAUSE_KEY = 'k';
    static NEXT_FRAME_KEY = '.';

    // static methods
    static forcedUpdate = () => { return; };  // anonymous func. problem child.

    static playPause() {
        EventManager.running = !EventManager.running
    };

    static moveOneFrame() {
        EventManager.oneFrame = true;
        EventManager.forcedUpdate();
    };

    static keyListener(e) {
        switch (e.key) {
            case EventManager.PLAY_PAUSE_KEY:
                EventManager.playPause()
                break;
            case EventManager.NEXT_FRAME_KEY:
                EventManager.moveOneFrame()
                break;
            default:
        }
    };

    static updateRuleString() {
        const inputText = document.getElementById('simulationInput').value;
        EventManager.newRuleString = true
        EventManager.ruleString = inputText
        EventManager.forcedUpdate()
    };

    static updateSpeed() {
        const inputSpeed = document.getElementById('speedInputBox').value;
        const newUpdateInterval = 50 + (2 * (100 - inputSpeed));
        EventManager.updateInterval = newUpdateInterval;
    }

    static bindEvents(){
        document.getElementById('play').addEventListener('click', EventManager.playPause);  // play pause button
        document.getElementById('next').addEventListener('click', EventManager.moveOneFrame); // move one frame button
        document.getElementsByTagName("body")[0].addEventListener("keydown", EventManager.keyListener); // key presses
        document.getElementById('submitInput').addEventListener('click', EventManager.updateRuleString); // new rule string input button
        document.getElementById('speedInput').addEventListener('click', () => {
            EventManager.updateSpeed();
            clearInterval(EventManager.currentTimer);
            EventManager.currentTimer = setInterval(EventManager.updateLoop, EventManager.updateInterval)
        }); // change speed
    }
}
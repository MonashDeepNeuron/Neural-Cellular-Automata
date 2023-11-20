// static class used to handle and process events
// it uses the default constructor

export default class EventManager {
    // event related global variables
    static oneFrame = false;
    static running = true;
    static newRuleString = false;
    static ruleString = "/2";
    static updateInterval = 50;

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
        console.log(e.key)
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
        console.log(inputText)
        EventManager.newRuleString = true
        EventManager.ruleString = inputText
        EventManager.forcedUpdate()
    };

    static updateSpeed() {
        const inputSpeed = document.getElementById('speedInputBox').value;
        console.log(inputSpeed);
        const newUpdateInterval = 50 + (2 * (100 - inputSpeed));
        EventManager.updateInterval = newUpdateInterval;
    }
}
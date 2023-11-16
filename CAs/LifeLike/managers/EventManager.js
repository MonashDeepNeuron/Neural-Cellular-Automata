// static class used to handle and process events
// it uses the default constructor

export default class EventManager {
    // event related global variables
    static oneFrame = false;
    static running = true;
    static newRuleString = true;
    static ruleString = "/2";

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
}
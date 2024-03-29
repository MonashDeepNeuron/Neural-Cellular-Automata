// static class used to handle and process events
// it uses the default constructor

export default class EventManager {
    // event related global variables
    static running = true;
    static newRuleString = false;
    static resetTemplate = false;
    static ruleString = "R5,S33-57,B34-45,NM"; // Start with Conway's life // Temporarily removed C2 as second entry
    static updateInterval = 50;
    static templateNo = 0;
    static loopID = 0; // Update interval
    static updateLoop = () => { }; // Set in versions of life.js
    static getRule = () => { }; // Caters for different interface setups

    // key bindings
    static PLAY_PAUSE_KEY = 'k';
    static NEXT_FRAME_KEY = '.';

    static playPause() {
        if (this.running) {
            // Pause
            clearInterval(this.loopID);
            this.loopID = null;
        } else {
            // Play
            this.loopID = setInterval(this.updateLoop, this.updateInterval);
        };

        this.running = !this.running;
    };

    static moveOneFrame() {
        if (this.loopID != null) {
            return;
        }
        this.updateLoop(); // Forced update
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


    static setUpdateLoop(newUpdateLoop) {
        this.updateLoop = newUpdateLoop;

    }


    static updateRuleString() {
        EventManager.newRuleString = true
        EventManager.ruleString = EventManager.getRule();
        EventManager.updateLoop()
    };

    static updateSpeed() {
        const inputSpeed = document.getElementById('speedInputBox').value;
        const newUpdateInterval = 50 + (2 * (100 - inputSpeed));
        EventManager.updateInterval = newUpdateInterval;
    }

    static resetCanvas() {
        EventManager.templateNo = document.getElementById('templateSelect').value
        EventManager.resetTemplate = true;
        console.log(`Resetting canvas... Template:${EventManager.templateNo}`);
        EventManager.updateLoop();
    }

    static bindEvents() {
        document.getElementById('play').addEventListener('click', EventManager.playPause);  // play pause button
        document.getElementById('next').addEventListener('click', EventManager.moveOneFrame); // move one frame button
        document.getElementsByTagName("body")[0].addEventListener("keydown", EventManager.keyListener); // key presses
        document.getElementById('submitInput').addEventListener('click', EventManager.updateRuleString); // new rule string input button
        document.getElementById('reset').addEventListener('click', EventManager.resetCanvas);
        document.getElementById('speedInput').addEventListener('click', () => {
            EventManager.updateSpeed();
            clearInterval(EventManager.loopID);
            EventManager.loopID = setInterval(EventManager.updateLoop, EventManager.updateInterval)
        }); // change speed
    }
}
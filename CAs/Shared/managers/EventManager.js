// static class used to handle and process events
// it uses the default constructor

export default class EventManager {
    // event related global variables
    static running = true;
    static newRuleString = false;
    static resetTemplate = false;
    static randomiseGrid = false;
    static ruleString = ""; // Start with Conway's life // Temporarily removed C2 as second entry
    static updateInterval = 500; // Ignore this preset number, should be reset immediately at the top of each script
    static templateNo = 1;
    static loopID = 0; // Update interval
    static updateLoop = () => { }; // Set in versions of life.js
    static getRule = () => { }; // Caters for different interface setups
    
    static framesPerUpdateLoop = 1;
    static cycles = 0; // The current number of update cycles since reset
    static MAX_FPS = 30;

    // key bindings
    static PLAY_PAUSE_KEY = 'k';
    static NEXT_FRAME_KEY = '.';

    static playPause() {
        if (EventManager.running) {
            // Pause
            clearInterval(EventManager.loopID);
            EventManager.loopID = null;
        } else {
            // Play
            EventManager.loopID = setInterval(EventManager.updateLoop, EventManager.updateInterval);
        };

        EventManager.running = !EventManager.running;
    };

    static moveOneFrame() {
        if (EventManager.loopID != null) {
            return;
        }
        EventManager.updateLoop(); // Forced update
    };


    static randomise() {
        EventManager.randomiseGrid = true;
        EventManager.moveOneFrame();
    }


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
        EventManager.updateLoop = newUpdateLoop;

    }


    static updateRuleString() {
        EventManager.newRuleString = true
        EventManager.ruleString = EventManager.getRule();
        EventManager.updateLoop()
    };


    static updateSpeed() {
        const inputSpeed = document.getElementById('speedInputBox').value;
        if (inputSpeed > EventManager.MAX_FPS) {
            EventManager.framesPerUpdateLoop = Math.round(inputSpeed/EventManager.MAX_FPS);
            EventManager.updateInterval = 1000/(inputSpeed/EventManager.framesPerUpdateLoop);
        } else {
            EventManager.framesPerUpdateLoop = 1;
            EventManager.updateInterval = 1000/(inputSpeed);
        }
        document.getElementById("framesDisplayed").innerHTML = `Displays every <b>${EventManager.framesPerUpdateLoop}</b> frames`;
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
        document.getElementById('randomise').addEventListener('click', EventManager.randomise); //Randomise the grid 
        document.getElementsByTagName("body")[0].addEventListener("keydown", EventManager.keyListener); // key presses
        document.getElementById('submitInput').addEventListener('click', EventManager.updateRuleString); // new rule string input button
        document.getElementById('reset').addEventListener('click', EventManager.resetCanvas);
        document.getElementById('speedInput').addEventListener('click', () => {
            EventManager.updateSpeed();
            clearInterval(EventManager.loopID);
            EventManager.loopID = setInterval(EventManager.updateLoop, EventManager.updateInterval)
        }); // change speed
    }

    static incrementCycleCount(){
        this.cycles = this.cycles +1;
    }

    static updateCyclesDisplay(){
        document.getElementById('cycleCounter').innerText = "Update Cycles:" + this.cycles;
    }

    static resetCycleCount(){
        this.cycles = 0;
        document.getElementById('cycleCounter').innerText = "Update Cycles:" + this.cycles;
    }


}
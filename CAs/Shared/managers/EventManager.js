// static class used to handle and process events
// it uses the default constructor

// NOTE: do not refer to this class as 'this' in this file
// Many functions are used as event triggered functions and for some reason this
// does not work well for these applications.
// I have no idea why but switching to 'EventManager' on these functions solved the issue
// Potentially a script language thing (as opposed to complied)

export default class EventManager {
    // event related global variables
    static running = false;
    static newRuleString = false;
    static resetTemplate = false;
    static randomiseGrid = false;
    static ruleString = ""; // Start with Conway's life // Temporarily removed C2 as second entry
    static templateNo = 1;
    static loopID = null; // Update interval
    static updateLoop = () => { }; // Set in versions of life.js
    static getRule = () => { }; // Caters for different interface setups
    
    
    static updateInterval = 500; // Ignore EventManager preset number, should be reset immediately at the top of each script
    static currentSpeed = 60; // Speed in fps
    static framesPerUpdateLoop = 1;
    static skipEvenFrames = false;
    static cycles = 0; // The current number of update cycles since reset
    static MAX_FPS = 50;

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
        EventManager.updateLoop(EventManager.currentSpeed);
    };

    static submitSpeed() {
        EventManager.currentSpeed = document.getElementById('speedInputBox').value;
        EventManager.updateSpeed(EventManager.currentSpeed);
        if (EventManager.running){
            clearInterval(EventManager.loopID);
            EventManager.loopID = setInterval(EventManager.updateLoop, EventManager.updateInterval)
        }
    }

    static updateSpeed(inputSpeed) {
        EventManager.framesPerUpdateLoop = Math.ceil(inputSpeed/EventManager.MAX_FPS);
        EventManager.updateInterval = 1000/(inputSpeed/EventManager.framesPerUpdateLoop);
        
        if (EventManager.skipEvenFrames) {
            EventManager.framesPerUpdateLoop = EventManager.framesPerUpdateLoop*2;
        }
        
        document.getElementById("framesDisplayed").innerHTML = `Displays every <b>${EventManager.framesPerUpdateLoop}</b> updates (updates per sec: ${Math.round(1000*EventManager.framesPerUpdateLoop/EventManager.updateInterval)})`;
    }

    static resetCanvas() {
        EventManager.templateNo = document.getElementById('templateSelect').value
        EventManager.resetTemplate = true;
        console.log(`Resetting canvas... Template:${EventManager.templateNo}`);
        EventManager.updateLoop();
    }


    static evenFramesCheckboxClicked() {
        EventManager.skipEvenFrames = document.getElementById("skipEvenCheckbox").checked;
        console.log(`Checkbox = ${EventManager.skipEvenFrames}`);
        EventManager.updateSpeed(EventManager.currentSpeed);
    }

    static bindEvents() {
        document.getElementById('play').addEventListener('click', EventManager.playPause);  // play pause button
        document.getElementById('next').addEventListener('click', EventManager.moveOneFrame); // move one frame button
        document.getElementById('randomise').addEventListener('click', EventManager.randomise); //Randomise the grid 
        document.getElementsByTagName("body")[0].addEventListener("keydown", EventManager.keyListener); // key presses
        document.getElementById('submitInput').addEventListener('click', EventManager.updateRuleString); // new rule string input button
        document.getElementById('reset').addEventListener('click', EventManager.resetCanvas);
        document.getElementById('speedInput').addEventListener('click', EventManager.submitSpeed); // change speed
        document.getElementById('skipEvenCheckbox').addEventListener('change', EventManager.evenFramesCheckboxClicked);
    }

    static incrementCycleCount(){
        EventManager.cycles = EventManager.cycles +1;
    }

    static updateCyclesDisplay(){
        document.getElementById('cycleCounter').innerText = "Update Cycles:" + EventManager.cycles;
    }

    static resetCycleCount(){
        EventManager.cycles = 0;
        document.getElementById('cycleCounter').innerText = "Update Cycles:" + EventManager.cycles;
    }


}
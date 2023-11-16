// imports
import run from "./life.js";
import EventManager from "./managers/EventManager.js";

// Setting up event bindings
document.getElementById('play').addEventListener('click', EventManager.playPause);  // play pause button
document.getElementById('next').addEventListener('click', EventManager.moveOneFrame); // move one frame button
document.getElementsByTagName("body")[0].addEventListener("keydown", EventManager.keyListener); // key presses
document.getElementById('submitInput').addEventListener('click', EventManager.updateRuleString); // new rule string input button

// run cellular automata
run()
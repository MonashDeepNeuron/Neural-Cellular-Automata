const UPDATE_INTERVAL = 1000;
function updateLoop() {
    console.log("Counter: " + counter);
};
setInterval(updateLoop, UPDATE_INTERVAL);
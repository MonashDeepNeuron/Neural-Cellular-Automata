import { createSlice } from "@reduxjs/toolkit";

const initialState = {
    running: false,
    framesPerUpdateLoop: 1,
    step: 0,
    template: 8,
};

const webGPUSlice = createSlice({
    name: "webGPU",
    initialState,
    reducers: {

        changeTemplate: (state, action) => {
            state.template = action.payload;
        },

        toggleRunning: (state) => {
            state.running = !state.running;
        },

        resetStep(state) {
            state.step = 0;
        },

        incrementStep(state) {
            state.step += 1;
        },

        setFramesPerUpdateLoop(state, action) {
            state.framesPerUpdateLoop = action.payload;
        },
    },
});

export const {
    changeTemplate,
    toggleRunning,
    resetStep,
    incrementStep,
    setFramesPerUpdateLoop,
} = webGPUSlice.actions;

export default webGPUSlice.reducer;

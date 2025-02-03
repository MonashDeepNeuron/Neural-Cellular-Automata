import { createSlice } from '@reduxjs/toolkit';

const initialState = {
	running: false,
	framesPerSecond: 40,
	step: 0,
	template: 8
};

const webGPUSlice = createSlice({
	name: 'webGPU',
	initialState,
	reducers: {
		changeTemplate: (state, action) => {
			state.template = action.payload;
		},

		toggleRunning: state => {
			state.running = !state.running;
		},

		resetStep(state) {
			state.step = 0;
		},

		incrementStep(state) {
			state.step += 1;
		},

		setFramesPerSecond(state, action) {
			console.log('frames', action.payload);
			state.framesPerSecond = action.payload;
		}
	}
});

export const { changeTemplate, toggleRunning, resetStep, incrementStep, setFramesPerSecond } = webGPUSlice.actions;

export default webGPUSlice.reducer;

// src/store/store.js
import { configureStore } from '@reduxjs/toolkit';
import webGPUReducer from './webGPUSlice';

const store = configureStore({
	reducer: {
		webGPU: webGPUReducer
	}
});

export default store;

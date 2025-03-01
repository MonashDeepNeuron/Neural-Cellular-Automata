import { configureStore } from '@reduxjs/toolkit';
import webGPUReducer from './webGPUSlice';

const store = configureStore({
	reducer: {
		webGPU: webGPUReducer
	}
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

export default store;

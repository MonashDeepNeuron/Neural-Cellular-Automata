import React from "react";
import WebGPUCanvas from "./components/Canvas.jsx";
import store from './store/store';
import { Provider } from "react-redux";
import './index.css';

const App = () => {

  return (
    <Provider store={store}>
      <main >
        <WebGPUCanvas />
      </main>
    </Provider>
  );
};

export default App;

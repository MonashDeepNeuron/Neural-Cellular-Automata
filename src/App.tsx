import React from "react";
import WebGPUCanvas from "./components/Canvas.jsx";

const App = () => {
  return (
    <main>
      <h1>
        chloe is trying
      </h1>
      <div style={{ width: "100vw", height: "100vh" }}>
        <WebGPUCanvas />
      </div>
    </main>

  );
};

export default App;

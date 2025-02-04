import React from "react";
import { Link } from "react-router-dom";
import "../assets/styles/style.css"; // Ensure correct path to CSS

const SimulatorHome = () => {
  return (
    <div className="container">
      {/* Navbar */}
      <div className="navbar">
        <a href="https://www.deepneuron.org/">
          <img 
            src="/Pages/Images/mdn_logo.png" 
            alt="Deep Neuron Logo" 
            style={{ height: "20px", verticalAlign: "middle" }} 
          />
        </a>
        <Link to="/">Home</Link>
        <Link to="/cellular-automata">Cellular Automata</Link>
        <Link to="/simulator-home">Simulator</Link>
        <Link to="/troubleshooting">Troubleshooting</Link>
        <Link to="/contact">Contact Us</Link>
      </div>

      {/* Title */}
      <div className="text-column">
        <h1>Simulator</h1>
      </div>

      {/* Warning Message */}
      <div className="warning centre">
        <p>
          <b>Warning:</b> This website contains content that may 
          <u> flash at high frequencies</u>. If you or someone viewing this is sensitive to this type of content, please use discretion when choosing frame-rates.
        </p>
        <button onClick={(e) => e.target.parentElement.style.display = "none"}>
          Dismiss
        </button>
      </div>

      {/* Model Selection Panel */}
      <div className="controlPanel">
        <h3>Select a model:</h3>
        <div className="panelRow controlGroup">
          <a href="/CAs/ConwaysLife/life.html" className="button" id="classic_conway">Classic Conway</a>
          <a href="/CAs/LifeLike/life.html" className="button" id="life_like">Life Like</a>
          <a href="/CAs/Larger/life.html" className="button" id="larger">Larger</a>
        </div>
        <div className="panelRow controlGroup">
          <a href="/CAs/Continuous/life.html" className="button" id="continuous">Continuous</a>
          <a href="/CAs/GCA/life.html" className="button" id="growing_nca">G-NCA</a>
        </div>
      </div>

      {/* Navigation to Troubleshooting */}
      <p style={{ textAlign: "center", marginTop: "20px" }}>
        <Link to="/troubleshooting" style={{ fontWeight: "bold" }}>
          See Next: Trouble Shooting
        </Link>
      </p>
    </div>
  );
};

export default SimulatorHome;

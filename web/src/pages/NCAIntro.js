import React from "react";
import { Link } from "react-router-dom";
import "../assets/styles/style.css"; // Ensure correct path to CSS

const NCAIntro = () => {
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

      {/* Main Content */}
      <div className="centre">
        <h1>Neural Cellular Automata</h1>
        <h2>Introduction</h2>
        <h2>The Cellular Automata Perspective</h2>
        <h2>The Neural Network Perspective</h2>
      </div>
    </div>
  );
};

export default NCAIntro;

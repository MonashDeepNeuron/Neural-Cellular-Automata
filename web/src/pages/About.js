import React from "react";
import "../assets/styles/style.css"; // Import global styles

function About() {
  return (
    <div className="centre">
      <h1>About Us</h1>
      <h2>Who are we?</h2>
      <p>
        We are a project team under Monash DeepNeuron, an Engineering/IT student team, run by Monash
        University students. NCA is one of many research projects, which you can read more about{" "}
        <a href="https://www.deepneuron.org/">here</a>!
      </p>
      <h2>Project Objectives</h2>
      <ol>
        <li>What are NCA? How is NCA different from other CA and Neural Networks?</li>
        <li>What can NCA be used for? Does NCA provide an advantage over other similar architectures?</li>
        <li>How can NCA be improved?</li>
      </ol>
      <p>As a result of answering these questions, we aim to produce a research paper.</p>

      <h2>Project Updates</h2>
      <p>
        Watch <a href="/keeping-up">this</a> page for project updates!
      </p>

      <h2>Join Us!</h2>
      <p>
        Are you a Monash Engineering or IT student, interested in working on this project? Register
        your email here to receive emails about when new positions open up! First-year or Masters,
        all are welcome.
      </p>
    </div>
  );
}

export default About;



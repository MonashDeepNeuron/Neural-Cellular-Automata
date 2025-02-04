import React from "react";
import "../assets/styles/style.css"; // Ensure the correct path to CSS

function Contact() {
  return (
    <div className="centre">
      <h1>Contact Us</h1>
      <p>
        We would love to hear from you! If you have any questions, feedback, or inquiries, please reach out to us.
      </p>
      <p>You can contact us directly at:</p>
      <p>
        <a href="https://www.deepneuron.org/contact-us" target="_blank" rel="noopener noreferrer">
          Deep Neuron Contact Page
        </a>
      </p>
    </div>
  );
}

export default Contact;

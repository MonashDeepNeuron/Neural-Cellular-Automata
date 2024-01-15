
console.log('Setup started');

document.getElementById("output").innerText = `Setting up test. Please wait...`;

const suite = new Benchmark.Suite("My Perf. test");

// COMPLETE
suite.on("complete", (event) => {
  const suite = event.currentTarget;
  const fastestOption = suite.filter("fastest").map("name");
  console.log(`The fastest option is ${fastestOption}`);

  document.getElementById("output").innerText = 
          document.getElementById("output").innerText 
          + "\n" + `The fastest option is ${fastestOption}`;
}); 

// CYCLE
suite.on("cycle", (event) => {
  const benchmark = event.target;
  console.log(benchmark.toString());
  document.getElementById("output").innerText = 
          document.getElementById("output").innerText 
          + "\n" + benchmark.toString();
});

console.log('Begin testing');

document.getElementById("output").innerText = `Running benchmark...`;
suite
    .add("RegExp#test", () => /o/.test("Hello World!"))
    .add("String#indexOf", () => "Hello World!".indexOf("o") > -1)
    .run();


console.log(suite);





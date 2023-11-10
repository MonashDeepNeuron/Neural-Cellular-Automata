<!-- <svelte:head>
  <script async src="./life.js"></script>
</svelte:head> -->

<head>
  <meta charset="utf-8" />
  <title>WebGPU Life</title>
</head>
<div class="text-column">
  <h1>Game of Life</h1>
</div>

<body>
  <canvas width="512" height="512" />

  <script type="module">
    import test from "/Users/nyankyaw/Documents/UNI/YEAR4/MDN/Neural-Cellular-Automata/My-App/src/routes/life/life.js";
    // ignore my path lmao
    test();
  </script>
</body>


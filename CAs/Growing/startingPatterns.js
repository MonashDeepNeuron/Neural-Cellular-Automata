const spaceship1 = {
    name: "Conway's life glider",
    pattern: [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1,
        0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0,
        0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    width: 16,
    height: 16,
    minGrid: 16,
    rule: '1,1,1,1,9,1,1,1,1,',
    activation: 'if (x == 3. || x == 11. || x == 12.){\nreturn 1.;\n}\nreturn 0.;'
};



const worms = {
    name: "Worms",
    pattern: null,
    width: 16,
    height: 16,
    minGrid: 1024,
    rule: '0.68,-0.9,0.68,-0.9,-0.66,-0.9,0.68,-0.9,0.68,',
    activation: 'return -1./pow(2., (0.6*pow(x, 2.)))+1.;',
};

const waves = {
    name: "Waves (experimental)",
    pattern: null,
    width: 16,
    height: 16,
    minGrid: 1024,
    rule: '0.565, -0.716, 0.565, -0.716, 0.627 , -0.716, 0.565, -0.716, 0.565,',
    activation: 'return abs(1.2*x);',
};

const B29 = {
    name: "Conway's life B29 Glider",
    pattern: [
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    width: 32,
    height: 8,
    minGrid: 32,
    rule: '1,1,1,1,9,1,1,1,1,',
    activation: 'if (x == 3. || x == 11. || x == 12.){\nreturn 1.;\n}\nreturn 0.;'
};

const slimeGrid = {
    name: "Slime Grid",
    pattern: null,
    width: 0,
    height: 0,
    minGrid: 1024,
    rule: "0.8,-0.85,0.8,-0.85,-0.2,-0.85,0.8,-0.85,0.8",
    activation: 'return -1./(0.89*pow(x, 2.)+1.)+1.;'
}




export default [
    spaceship1,
    worms,
    B29,
    waves,
    slimeGrid
];
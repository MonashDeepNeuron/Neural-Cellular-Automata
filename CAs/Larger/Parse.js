// TODO: rulestring validation function.

export const NEIGHBOURHOOD_MAP = {
    M: 0,
    N: 1,
    C: 2,
};

export function parseRuleString(ruleString) {

    ruleString = ruleString.replace(/\s/g, "").toUpperCase(); // Cut out white space and assert uppercase

    console.log("Getting RULE...");
    // Larger than life rulestring
    // Rulestring is given as Rr, Cc, Ss Bb, Nn where:
    // r is the radius in units
        // c is the number of possible states NOTE: Temporarily excluded bc i don't understand
    // s is ranges in the form a-b, c-d, e, ... conditions of survival
    // b is ranges in the form a-b, c-d, e, ... conditions of birth
    // n is the type of neighbourhood (eg. square, cicular?)
    // See https://conwaylife.com/wiki/Larger_than_Life

  // TODO: this function returns null with invalid input, needs to be accounted for.

  // ruleString is given by the user. it is a string
  // Output format 
    // [r, c, no. srange, su, sl, su, sl, ... , no. brange, bu, bl, bu, bl, ... , n]
    // For the sake of efficiency, for S and B,
    // if the number is specifically just the number, rather than a range, this 
    // is denoted through the use of negatives
        // eg. 2-5, 7, 11-13, ...,
        // push 2, 5, -7, 11, 13, ...,
    // TODO: BUG - When no given birth or survival cases given, results in error
    if (ruleString.length < 8) {
        return null;
    }

    let ruleList = []

    let i = 0;
    if (ruleString[i] != 'R'){
        console.log("ERROR");
        return null;
    }

    let nextNumber = () => {
        //console.log(`Parsing at ${i}`);
        while (!isDigit(ruleString[i])){
            i++;
        }

        let numStart = i;
        while (isDigit(ruleString[i])){
            i++;
        }
        //console.log(`Extracted ${ruleString.substring(numStart, i)}`)
        return Number(ruleString.substring(numStart, i));
    }


    i++;

    ruleList.push(nextNumber()); // Radius
    ruleList.push(nextNumber()); // C (multiple states) 
    if (ruleList[1] < 2){
        ruleList[1] = 2; // Lowest possiple number of states allowable
    }
    ruleList.push(0); // Save a space for neighbourhood type


    // Parse survival cases

    while (ruleString[i] != 'S'){
        i++;
    }
    ruleList.push(0);
    let sConditionCountIndex = ruleList.length-1;

    // Rule list should be 4 long now
    // See how many characters are occupies by survival rule
    let lastS = i; // Index of the start of last set of numbers
    // 2-4, 34-36, B3-4, ...
    //    ^ -----

    while (ruleString[lastS] != 'B'){
        lastS++;
    }
    lastS-= 3; 
    while (ruleString[lastS] != ',' && ruleString[lastS] != 'S'){
        lastS--;
    }
    
    console.log(`Last S range at index ${lastS}, current index at ${i}`);

    // eg. 2-5, 7, 11-13, ...,
    // push 2, 5, 7, 7, 11, 13, ...,
    while (i <= lastS) {
        ruleList.push(nextNumber());
        ruleList[sConditionCountIndex]++;

        while (ruleString[i] != '-' && ruleString[i] != ','){
            //console.log(`Rejected ${ruleString[i]} in survive parse`);
            i++;
        }

        if (ruleString[i] == '-'){
            i++; // Exclude dash to prevent misinterpretation as negative
            ruleList.push(nextNumber());
            ruleList[sConditionCountIndex]++;
        } else {
            ruleList.push(ruleList[ruleList.length -1]);
            ruleList[sConditionCountIndex]++;
        }
    } // NOTE: this will over-push by one number i.e. it will include the 
     // number refered to by lastS

    //console.log(`Update after finding survivials: ${ruleList}`);
    
    ruleList.push(0);
    let bConditionCountIndex = ruleList.length-1;

    // Rule list should be 3 long now
    // See how many characters are occupies by survival rule
    let lastB = i; // Index of the start of last set of numbers
    // 2-4, 34-36, B3-4, ...
    //    ^ -----

    while (ruleString[lastB] != 'N'){
        lastB++;
    }
    lastB-= 3; 
    while (ruleString[lastB] != ',' && ruleString[lastB] != 'B'){
        lastB--;
    }

    
    //console.log(`Last B range at index ${lastB}`);

    // eg. 2-5, 7, 11-13, ...,
    // push 2, 5, -7, 11, 13, ...,
    while (i <= lastB) {
        ruleList.push(nextNumber());
        ruleList[bConditionCountIndex]++;

        while (ruleString[i] != '-' && ruleString[i] != ','){
            i++;
        }

        if (ruleString[i] == '-'){
            ruleList.push(nextNumber());
            ruleList[bConditionCountIndex]++;
        } else {
            ruleList.push(ruleList[ruleList.length -1]);
            ruleList[bConditionCountIndex]++;
        }
    } // NOTE: this may over-push by one number i.e. it will include the 
     // number refered to by lastB
    
    
    // from i: ,Nn, S..., B...,
    // Assume n is a single character

    while (ruleString[i] != 'N'){
        i++;
    }
    i++;

    ruleList[2] = NEIGHBOURHOOD_MAP[ruleString[i]];
    
    i++;

    let RULE = new Uint32Array(ruleList.length);
    for (let i = 0; i < ruleList.length; i++){
        RULE[i] = ruleList[i];
    }

        console.log(ruleString);
        console.log(ruleList);
    return RULE
}

export function displayRule(ruleString){
    console.log(`Displaying ${ruleString}`);
    
    let i = 0;
    while (ruleString[i] != 'R'){
        i++;
    }
    i++;

    let R = "";

    while (ruleString[i] != ','){
        R += ruleString[i];
        i++;
    }

    document.getElementById("simulationInputR").value = R;

    while (ruleString[i] != 'C'){
        i++;
    }
    i++;

    let C = "";

    while (ruleString[i] != ','){
        C += ruleString[i];
        i++;
    }

    document.getElementById("simulationInputC").value = C;

    while (ruleString[i] != 'S'){
        i++;
    }
    i++;


    let sStart = i;

    while (ruleString[i] != 'B'){
        i++;
    }

    while (ruleString[i] != ','){
        i--;
    }
    
    
    let S = ruleString.substring(sStart, i);
    document.getElementById("simulationInputS").value = S;

    while (ruleString[i] != 'B'){
        i++;
    }
    i++;


    let bStart = i;

    while (ruleString[i] != 'N'){
        i++;
    }

    while (ruleString[i] != ','){
        i--;
    }
    
    let B = ruleString.substring(bStart, i);

    document.getElementById("simulationInputB").value = B;

    while (ruleString[i] != 'N'){
        i++;
    }
    i++;

    document.getElementById("simulationInputN").value = ruleString[i];
}



function isDigit(c){
    return c >= '0' && c <= '9';
}



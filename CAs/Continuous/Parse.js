export function parseRuleString(ruleString) {
    console.log(ruleString);


    let RULE = new Float32Array(9)

    let parseIndex = 0;

    let nextNumber = () => {
        while (!isNumerical(ruleString[parseIndex])){
            parseIndex++;
        }

        let numStart = parseIndex;
        while (isNumerical(ruleString[parseIndex])){
            parseIndex++;
        }
        return Number(ruleString.substring(numStart, parseIndex));
    }
    
    for (let i = 0; i < 9; i++){
        RULE[i] = nextNumber();
    }
    return RULE
}



function isNumerical(c){ // Digits and "."
    return (c >= '0' && c <= '9') || (c == '.');
}


export function displayRule(ruleString){
    let parseIndex = 0;

    let nextNumber = () => {
        while (!isNumerical(ruleString[parseIndex])){
            parseIndex++;
        }

        let numStart = parseIndex;
        while (isNumerical(ruleString[parseIndex])){
            parseIndex++;
        }
        return Number(ruleString.substring(numStart, parseIndex));
    }
    
    for (let i = 1; i < 10; i++){
        document.getElementById("kernel" + i).value = nextNumber();
    }
}

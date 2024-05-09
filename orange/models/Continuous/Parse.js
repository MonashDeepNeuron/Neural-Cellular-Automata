
/** 
 * Interprets the rulestring according to a made up notation of 
 * "float, float, float ... (total 9 floats in string format)"
 * @param {String} ruleString 
 * @returns {Uint32Array}
 * @todo The existence of this feels like a legacy feature from previous 
 * implementations. All implementations can probably be re-done as 
 * float/integer arrays instead of strings. This would be much more direct
 */
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


/** 
 * 100% sure this is redundant.
 * Pretty sure this exists somewhere as a library function
 * Pretty sure what this is being used for is also redundant
 * */
function isNumerical(c){ // Digits and "."
    return (c >= '0' && c <= '9') || (c == '.') || (c == '-');
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
